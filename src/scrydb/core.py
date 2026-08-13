"""
scrydb — lexical, semantic, and hybrid search built on SQLite.

Public surface
--------------
    Index.open(path)                      -- open/create an index
    index.add_model(model)                -- opt in to dense search
    index.index_documents(source, ...)    -- index a JSONL path or any
                                              iterable of dict-like rows
    index.index_queries(source, ...)      -- same, for the query side
    index.search(query, mode=...)         -- one query in, ranked list out
    index.batch_search(queries=..., mode=...)  -- many queries -> Run
    index.documents[id] / index.queries[id]    -- mapping-style lookups
    index.document_embeddings[id] / index.query_embeddings[id]
                                           -- mapping-style embedding lookups
                                              (``_binary`` variants for the
                                              ubinary-quantized vectors)
    run.write_trec(path, tag=...)         -- TREC run file
    run.to_dataframe()                    -- pandas, for analysis/eval

``search``/``batch_search`` take ``mode`` ("lexical", "semantic", or
"hybrid") and ``rerank`` (``False``, ``"hamming"``, ``"cosine"``, or
``True`` as a synonym for ``"cosine"``), which together select any of
the six standard strategies:

    BM25                             mode="lexical",  rerank=False
    BM25 >> Hamming/Binary           mode="lexical",  rerank="hamming"
    BM25 >> Cosine/Full              mode="lexical",  rerank="cosine"
    Hamming/Binary                   mode="semantic", rerank=False
    Hamming/Binary >> Cosine/Full    mode="semantic", rerank="cosine"
    Hybrid/RRF                       mode="hybrid"

Requires: numpy, tqdm, pandas (for ``to_dataframe``),
sentence-transformers (only exercised when a model is attached or
precomputed embeddings are indexed/searched).
"""

from __future__ import annotations

import json
import platform
import re
import sqlite3
from collections.abc import Iterable, Mapping
from importlib import resources
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from tqdm import tqdm


# ===========================================================================
# Locating the compiled hamming_distance() SQLite extension
# ===========================================================================

def _bundled_extension_filename() -> str:
    """Filename of the compiled extension for the current platform.

    The extension is compiled from ``ext/hamming.c`` at install time (see
    this package's ``setup.py``), producing a shared library named for the
    current platform's loader conventions.
    """
    system = platform.system()
    if system == "Darwin":
        return "hamming.dylib"
    if system == "Linux":
        return "hamming.so"
    raise RuntimeError(
        f"The hamming_distance SQLite extension is not supported on {system!r}. "
        "Only Linux and macOS are supported; pass hamming_ext_path=None to "
        "Index.open()/Index() to disable Hamming-distance (binary/hex) search "
        "and use lexical/cosine search only."
    )


def _bundled_hamming_path() -> "Path | None":
    """Path to the compiled extension shipped inside this package, or
    ``None`` if it hasn't been built for the current platform."""
    try:
        filename = _bundled_extension_filename()
    except RuntimeError:
        return None
    try:
        ext_dir = resources.files("scrydb.ext")
    except (ModuleNotFoundError, FileNotFoundError):
        return None
    candidate = ext_dir / filename
    if candidate.is_file():
        return Path(str(candidate))
    return None


# ===========================================================================
# Query text helpers
# ===========================================================================

# Characters with special meaning inside an FTS5 MATCH query (phrase
# quoting, column filters, grouping, NEAR/prefix syntax, ...). A stray
# occurrence in ordinary text — most commonly a trailing "?" on a
# natural-language question — is not valid FTS5 syntax and raises
# sqlite3.OperationalError. Replaced with a space so words on either side
# don't get glued together.

# _FTS5_SPECIAL_CHARS_RE = re.compile(r'[\[\]\'\.\/\-\,\\"(){}:^*?~&!;%#$=@+<>|`]')


# def _sanitize(text: str) -> str:
#     # FTS5's AND/OR/NOT/NEAR operators are only recognized in upper case;
#     # lower-casing first means a stray "AND" etc. in the input is treated
#     # as an ordinary search term instead of a boolean operator. The default
#     # FTS5 tokenizers are case-insensitive, so this doesn't change matching.
#     cleaned = _FTS5_SPECIAL_CHARS_RE.sub(" ", text.lower())
#     return " ".join(cleaned.split())


# def _as_or_query(text: str) -> str:
#     """Sanitize *text* and OR-join its terms: the safe default for
#     free-form natural-language queries."""
#     return " OR ".join(_sanitize(text).split())

# Allowlist, not a blocklist: keep only word characters as terms and
# discard everything else. We don't need to know what FTS5 considers
# "special" — nothing outside \w ever reaches the query string.
_WORD_RE = re.compile(r"\w+", re.UNICODE)


def _terms(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def _quote(term: str) -> str:
    # Turn `term` into an FTS5 string literal. Inside double quotes the
    # only character with syntactic meaning is `"` itself, escaped by
    # doubling — everything else is inert, so this is safe regardless of
    # what FTS5's grammar does or doesn't treat as special.
    return '"' + term.replace('"', '""') + '"'


def _as_or_query(text: str) -> str:
    """Turn arbitrary free-form text into a safe FTS5 MATCH expression
    that OR-matches any of its words."""
    terms = _terms(text)
    if not terms:
        return ""  # MATCH '' is itself a syntax error — caller must check
    return " OR ".join(_quote(t) for t in terms)


# ===========================================================================
# Result types
# ===========================================================================

class SearchResult(Mapping):
    """One ranked hit.

    Supports both attribute access (``r.id``, ``r.document``) and
    dict-like access (``r["id"]``, ``r.get("cosine_similarity")``,
    ``dict(r)``), so results drop straight into pandas/JSON without a
    conversion step.
    """

    __slots__ = ("_data",)

    def __init__(self, **fields: Any):
        object.__setattr__(self, "_data", fields)

    def __getattr__(self, name: str) -> Any:
        try:
            return self._data[name]
        except KeyError:
            raise AttributeError(name) from None

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return f"SearchResult({self._data!r})"


class Run(dict):
    """Batch-search output: ``{query_id: [SearchResult, ...]}``.

    A thin ``dict`` subclass carrying its own evaluation-oriented
    conveniences, so batch results are self-contained instead of needing a
    separate free function that has to guess their shape.
    """

    # Recognized score fields, in priority order, and whether a *lower*
    # value means *better* (so it must be negated to become higher-is-better
    # for the TREC SCORE column). Rerank fields (cosine_similarity,
    # hamming_distance) must outrank the plain lexical "score": _rerank()
    # merges a reranked result's fields on top of its base-stage fields
    # (``{**base_extra, "hamming_distance": dist}``), so a BM25 → rerank
    # result still carries its original lexical "score" alongside the new
    # rerank field. Most eval tools (trec_eval, pytrec_eval, ir_measures,
    # ...) rank purely by the written SCORE column and ignore row/rank
    # order, so picking the stale "score" here would silently drop the
    # rerank stage from evaluation while still *looking* reranked in the
    # RANK column.
    _SCORE_FIELDS = (
        ("rrf_score", False),
        ("cosine_similarity", False),
        ("hamming_distance", True),
        ("score", False),
    )

    def write_trec(
        self,
        path: "str | Path",
        tag: str = "run",
        top_k: "int | None" = None,
    ) -> None:
        """Write this run to *path* in the standard 6-column TREC format:
        ``QID Q0 DOCID RANK SCORE TAG``."""
        path = Path(path)
        with path.open("w", encoding="utf-8") as f:
            for qid, results in self.items():
                rows = results[:top_k] if top_k is not None else results
                n = len(rows)
                for rank, result in enumerate(rows, start=1):
                    score = self._score(result, rank, n)
                    f.write(f"{qid} Q0 {result['id']} {rank} {score} {tag}\n")

    def to_dataframe(self):
        """Flatten this run into a ``pandas.DataFrame`` with one row per
        (query, result) pair — handy for `sklearn`/`pytrec_eval`-style
        analysis. Requires ``pandas``."""
        import pandas as pd

        rows = []
        for qid, results in self.items():
            for rank, result in enumerate(results, start=1):
                rows.append({"query_id": qid, "rank": rank, **result})
        return pd.DataFrame(rows)

    @classmethod
    def _score(cls, result: SearchResult, fallback_rank: int, fallback_size: int) -> float:
        for field, negate in cls._SCORE_FIELDS:
            value = result.get(field)
            if value is not None:
                return -value if negate else value
        # No recognized score field — synthesize one that preserves the
        # existing (best-first) ordering.
        return float(fallback_size - fallback_rank + 1)


# ===========================================================================
# Retrieval models — pluggable, registered via Index.add_model()
# ===========================================================================

class SentenceEmbedding:
    """Dense-retrieval model backed by a ``sentence-transformers`` model.

    Pass an instance to :meth:`Index.add_model` to enable
    ``mode="semantic"``/``mode="hybrid"`` search and on-the-fly embedding
    during :meth:`Index.index_documents`/:meth:`Index.index_queries`.

    This is the extension point for future retrieval families: a
    ``ColbertModel`` or ``SpladeModel`` would implement the same
    ``encode_documents``/``encode_queries`` interface and drop in via
    ``add_model()`` without any change to ``Index``.
    """

    def __init__(
        self,
        model_name: str = "mixedbread-ai/mxbai-embed-large-v1",
        truncate_dim: "int | None" = 512,
        query_prompt: str = "Represent this query for searching relevant documents: {query}",
    ):
        self._model_name = model_name
        self._truncate_dim = truncate_dim
        self._query_prompt = query_prompt
        self._model = None  # lazily loaded

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name, truncate_dim=self._truncate_dim)
        return self._model

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self.model.encode(list(texts)), dtype=np.float32)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        prompted = [self._query_prompt.format(query=t) for t in texts]
        return np.asarray(self.model.encode(prompted), dtype=np.float32)


# ===========================================================================
# Read-only mapping views over the `documents` / `queries` tables
# ===========================================================================

class _PayloadTable(Mapping):
    """Read-only ``{id: payload_dict}`` view over a ``(id, payload)`` table."""

    def __init__(self, conn: sqlite3.Connection, table: str):
        self._conn = conn
        self._table = table

    def __getitem__(self, item_id: str) -> dict:
        cur = self._conn.execute(
            f"SELECT payload FROM {self._table} WHERE id = ?", (str(item_id),)
        )
        row = cur.fetchone()
        if row is None:
            raise KeyError(item_id)
        return json.loads(row[0])

    def __iter__(self) -> Iterator[str]:
        cur = self._conn.execute(f"SELECT id FROM {self._table}")
        for (item_id,) in cur:
            yield item_id

    def __len__(self) -> int:
        return self._conn.execute(f"SELECT COUNT(*) FROM {self._table}").fetchone()[0]

    def __repr__(self) -> str:
        return f"<{self._table}: {len(self)} items>"


class _EmbeddingTable(Mapping):
    """Read-only ``{id: np.ndarray}`` view over an ``(id, embedding)`` table.

    *dtype* is ``np.uint8`` for ``ubinary``-quantized tables (``embeddings``
    / ``query_embeddings``) and ``np.float32`` for full-precision tables
    (``embeddings_full`` / ``query_embeddings_full``).
    """

    def __init__(self, conn: sqlite3.Connection, table: str, dtype: "np.dtype"):
        self._conn = conn
        self._table = table
        self._dtype = dtype

    def __getitem__(self, item_id: str) -> np.ndarray:
        cur = self._conn.execute(
            f"SELECT embedding FROM {self._table} WHERE id = ?", (str(item_id),)
        )
        row = cur.fetchone()
        if row is None:
            raise KeyError(item_id)
        return np.frombuffer(row[0], dtype=self._dtype)

    def __iter__(self) -> Iterator[str]:
        cur = self._conn.execute(f"SELECT id FROM {self._table}")
        for (item_id,) in cur:
            yield item_id

    def __len__(self) -> int:
        return self._conn.execute(f"SELECT COUNT(*) FROM {self._table}").fetchone()[0]

    def __repr__(self) -> str:
        return f"<{self._table}: {len(self)} items>"


# ===========================================================================
# Source normalization — accept a path or any iterable of dict-like rows,
# so the caller never has to name a storage format.
# ===========================================================================

def _rows_from(source: "str | Path | Iterable[dict]") -> Iterator[dict]:
    """Yield dict rows from *source*.

    *source* may be a path to a UTF-8 JSONL file, or any iterable of
    dict-like rows — a list of dicts, a generator, a HuggingFace
    ``Dataset``/streaming dataset (both iterate as dict rows already), a
    pandas ``DataFrame.to_dict("records")``, and so on. The format is
    inferred from the object itself rather than from which method was
    called.
    """
    if isinstance(source, (str, Path)):
        path = Path(source)
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
        return
    if isinstance(source, Iterable):
        yield from source
        return
    raise TypeError(
        f"Unsupported source: {source!r}. Pass a path to a JSONL file or an "
        "iterable of dict-like rows."
    )

# ===========================================================================
# Index
# ===========================================================================

class Index:
    """A document + query store supporting lexical (FTS5/BM25) and, once a
    model is attached, dense/hybrid search.

    Open with :meth:`Index.open`; it doubles as a context manager::

        with Index.open("idx.db") as index:
            index.add_model(SentenceEmbedding())
            index.index_documents("corpus.jsonl", id_field="docid", text_field="text")
            results = index.search("...", mode="hybrid", rerank=True)
    """

    def __init__(self, db_path: "str | Path" = "idx.db", hamming_ext_path: "str | None" = "auto"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self._model: "SentenceEmbedding | None" = None

        if hamming_ext_path is not None:
            self._load_hamming_extension(hamming_ext_path)

        self._create_tables()

    @classmethod
    def open(cls, db_path: "str | Path" = "idx.db", hamming_ext_path: "str | None" = "auto") -> "Index":
        """Open (creating if needed) an index at *db_path*. Mirrors
        ``sqlite3.connect``/``pathlib.Path.open``: the one obvious entry
        point.

        *hamming_ext_path* controls the ``hamming_distance()`` SQLite
        extension used for binary/hex vector search:

        - ``"auto"`` (default) -- use the copy of ``hamming.so``/
          ``hamming.dylib`` compiled for this platform and bundled inside
          the ``scrydb`` package at install time.
        - an explicit path -- load that compiled extension file instead
          (e.g. a custom build).
        - ``None`` -- skip loading it; Hamming-distance search is
          unavailable but lexical (BM25) and cosine-rerank search still
          work.
        """
        return cls(db_path=db_path, hamming_ext_path=hamming_ext_path)

    def __enter__(self) -> "Index":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"Index({self.db_path!r}, documents={len(self.documents)}, queries={len(self.queries)})"

    # -- configuration -------------------------------------------------

    def add_model(self, model: SentenceEmbedding) -> "Index":
        """Register a retrieval model, enabling ``mode="semantic"``/
        ``mode="hybrid"`` search and on-the-fly embedding during indexing.
        Returns ``self`` so it can be chained onto :meth:`open`."""
        self._model = model
        return self

    # -- collections -----------------------------------------------------

    @property
    def documents(self) -> _PayloadTable:
        """Read-only mapping ``{doc_id: document_dict}``."""
        return _PayloadTable(self.conn, "documents")

    @property
    def queries(self) -> _PayloadTable:
        """Read-only mapping ``{query_id: query_dict}``."""
        return _PayloadTable(self.conn, "queries")

    @property
    def document_embeddings(self) -> _EmbeddingTable:
        """Read-only mapping ``{doc_id: float32 ndarray}`` of full-precision
        document embeddings (requires ``store_full_embeddings=True``)."""
        return _EmbeddingTable(self.conn, "embeddings_full", np.float32)

    @property
    def document_embeddings_binary(self) -> _EmbeddingTable:
        """Read-only mapping ``{doc_id: uint8 ndarray}`` of ``ubinary``-
        quantized document embeddings."""
        return _EmbeddingTable(self.conn, "embeddings", np.uint8)

    @property
    def query_embeddings(self) -> _EmbeddingTable:
        """Read-only mapping ``{query_id: float32 ndarray}`` of full-precision
        query embeddings (requires ``store_full_embeddings=True``)."""
        return _EmbeddingTable(self.conn, "query_embeddings_full", np.float32)

    @property
    def query_embeddings_binary(self) -> _EmbeddingTable:
        """Read-only mapping ``{query_id: uint8 ndarray}`` of ``ubinary``-
        quantized query embeddings."""
        return _EmbeddingTable(self.conn, "query_embeddings", np.uint8)

    # -- indexing ----------------------------------------------------------

    def index_documents(
        self,
        source: "str | Path | Iterable[dict]",
        id_field: str = "id",
        text_field: str = "text",
        embedding_field: str = "emb",
        batch_size: int = 1000,
        limit: "int | None" = None,
        store_full_embeddings: bool = True,
    ) -> None:
        """Index documents for lexical (and, if applicable, dense) search.

        *source* is a JSONL path or any iterable of dict-like rows (see
        :func:`_rows_from`) — the caller never names the storage format.

        Per row, dense indexing happens automatically when either:

        - the row already has an ``embedding_field`` (default ``"emb"``)
          — that vector is stored as-is, no model is called, so
          precomputed embeddings (e.g. shipped with a benchmark dataset)
          are used verbatim; or
        - a model has been registered via :meth:`add_model` — the row's
          ``text_field`` is encoded on the fly.

        If neither applies, only the lexical (BM25) index is built for
        that row.
        """
        self._index_rows(
            _rows_from(source),
            table="documents",
            id_field=id_field,
            text_field=text_field,
            embedding_field=embedding_field,
            batch_size=batch_size,
            limit=limit,
            store_full_embeddings=store_full_embeddings,
            is_query=False,
        )

    def index_queries(
        self,
        source: "str | Path | Iterable[dict]",
        id_field: str = "id",
        text_field: str = "text",
        embedding_field: str = "emb",
        batch_size: int = 1000,
        limit: "int | None" = None,
        store_full_embeddings: bool = True,
    ) -> None:
        """Index test queries, mirroring :meth:`index_documents`.

        Storing a query's own precomputed embedding (rather than
        re-encoding its text locally) matters for reproducing published
        results, since it guarantees the query vector came from the exact
        same model/pipeline that produced the corpus embeddings — this
        happens automatically whenever rows carry ``embedding_field``.
        """
        self._index_rows(
            _rows_from(source),
            table="queries",
            id_field=id_field,
            text_field=text_field,
            embedding_field=embedding_field,
            batch_size=batch_size,
            limit=limit,
            store_full_embeddings=store_full_embeddings,
            is_query=True,
        )

    def _index_rows(
        self,
        rows: Iterator[dict],
        *,
        table: str,
        id_field: str,
        text_field: str,
        embedding_field: str,
        batch_size: int,
        limit: "int | None",
        store_full_embeddings: bool,
        is_query: bool,
    ) -> None:
        payload_batch: list = []
        text_batch: list = []  # (id, text) for FTS (documents only)
        emb_batch: list = []
        full_emb_batch: list = []
        pending_encode: list = []  # (id, text) rows needing the attached model

        desc = "Indexing queries" if is_query else "Indexing documents"
        pbar = tqdm(desc=desc, unit="rows", unit_scale=True, total=limit)

        def flush():
            if is_query:
                self._flush_queries(payload_batch, emb_batch, full_emb_batch)
            else:
                self._flush_documents(payload_batch, text_batch)
                self._flush_embeddings(emb_batch, full_emb_batch)
            payload_batch.clear()
            text_batch.clear()
            emb_batch.clear()
            full_emb_batch.clear()

        n = 0
        for row in rows:
            if limit is not None and n >= limit:
                break
            if id_field not in row:
                raise KeyError(f"Missing id field: {id_field!r}")

            item_id = str(row[id_field])
            payload = {k: v for k, v in row.items() if k != embedding_field}
            payload_batch.append((item_id, json.dumps(payload, ensure_ascii=False)))

            text = row.get(text_field)
            if not is_query:
                if text_field not in row:
                    raise KeyError(f"Missing text field: {text_field!r}")
                text_batch.append((item_id, str(text)))

            emb = row.get(embedding_field)
            if emb is not None:
                self._append_embedding(
                    item_id, np.asarray(emb, dtype=np.float32), emb_batch, full_emb_batch, store_full_embeddings
                )
            elif self._model is not None and text is not None:
                pending_encode.append((item_id, str(text)))

            n += 1
            if len(payload_batch) >= batch_size:
                if pending_encode:
                    self._encode_and_append(pending_encode, emb_batch, full_emb_batch, store_full_embeddings, is_query)
                    pending_encode.clear()
                flush()
            pbar.update(1)

        if pending_encode:
            self._encode_and_append(pending_encode, emb_batch, full_emb_batch, store_full_embeddings, is_query)
        if payload_batch or emb_batch or full_emb_batch:
            flush()
        pbar.close()

    def _encode_and_append(self, pending, emb_batch, full_emb_batch, store_full, is_query) -> None:
        ids = [item_id for item_id, _ in pending]
        texts = [text for _, text in pending]
        vectors = self._model.encode_queries(texts) if is_query else self._model.encode_documents(texts)
        for item_id, vec in zip(ids, vectors):
            self._append_embedding(item_id, np.asarray(vec, dtype=np.float32), emb_batch, full_emb_batch, store_full)

    @staticmethod
    def _append_embedding(item_id, vector: np.ndarray, emb_batch, full_emb_batch, store_full) -> None:
        from sentence_transformers import quantize_embeddings

        binary = quantize_embeddings(vector[None, :], precision="ubinary")
        emb_batch.append((item_id, sqlite3.Binary(binary.tobytes())))
        if store_full:
            full_emb_batch.append((item_id, sqlite3.Binary(vector.tobytes())))

    # -- interactive search --------------------------------------------------

    def search(
        self,
        query: str,
        mode: str = "lexical",
        top_k: int = 10,
        rerank: "bool | str" = False,
        rerank_depth: int = 200,
        candidate_limit: int = 50,
        rrf_k: int = 60,
        raw: bool = False,
    ) -> list[SearchResult]:
        """Run one query, return a ranked list of :class:`SearchResult`.

        Parameters
        ----------
        mode:
            ``"lexical"`` (default, BM25 over FTS5), ``"semantic"``
            (Hamming/Binary dense search, requires :meth:`add_model` or
            precomputed embeddings), or ``"hybrid"`` (Reciprocal Rank
            Fusion of both).
        rerank:
            Adds a second-stage rerank over the top *rerank_depth*
            candidates from ``mode``. ``False`` (default) -- no rerank.
            ``"hamming"`` -- rerank with binary embeddings + Hamming
            distance (only meaningful when ``mode="lexical"``, giving
            BM25 >> Hamming/Binary). ``"cosine"`` (or ``True``, kept as a
            synonym for backward compatibility) -- rerank with
            full-precision embeddings + cosine similarity, requires
            full-precision embeddings to be stored (the default during
            indexing). With ``mode="hybrid"``, *rerank* applies to the
            semantic side before fusion.
        raw:
            ``mode="lexical"`` only. If ``True``, *query* is passed to
            FTS5's ``MATCH`` unmodified (advanced syntax the caller wrote
            themselves). If ``False`` (default), it is sanitized and its
            terms OR-joined, which is safe for arbitrary natural-language
            input (e.g. a trailing "?" no longer raises a syntax error).
        """
        rerank_with = self._normalize_rerank(rerank)
        if mode == "lexical":
            base_limit = rerank_depth if rerank_with else top_k
            ranked = self._lexical_rank(query, limit=base_limit, raw=raw)
            if rerank_with:
                ranked = self._rerank(ranked, rerank_with, limit=top_k, query_text=query)
        elif mode == "semantic":
            ranked = self._semantic_rank(
                query_text=query, limit=top_k, rerank=rerank_with, rerank_depth=rerank_depth
            )
        elif mode == "hybrid":
            lexical = self._lexical_rank(query, limit=candidate_limit)
            semantic = self._semantic_rank(
                query_text=query, limit=candidate_limit, rerank=rerank_with, rerank_depth=rerank_depth
            )
            ranked = self._reciprocal_rank_fusion(lexical, semantic, rrf_k)
        else:
            raise ValueError(f"Unknown mode {mode!r}; expected 'lexical', 'semantic', or 'hybrid'.")
        return self._materialize(ranked, top_k)

    # -- batch evaluation --------------------------------------------------

    def batch_search(
        self,
        queries: "str | list[str] | None" = None,
        mode: str = "lexical",
        top_k: int = 10,
        rerank: "bool | str" = False,
        rerank_depth: int = 200,
        candidate_limit: int = 50,
        rrf_k: int = 60,
        raw: bool = False,
    ) -> Run:
        """Run many stored queries, return a :class:`Run`.

        Same ``mode``/``rerank``/``raw`` vocabulary as :meth:`search`, but
        driven by stored query ids instead of a literal query string.

        Parameters
        ----------
        queries:
            ``None`` (default) evaluates every query in :attr:`queries` —
            the common case for producing a TREC run. Otherwise a single
            id or a list of ids.

        For each query, a precomputed embedding stored via
        :meth:`index_queries` is preferred over re-encoding the query's
        text with the attached model, guaranteeing the same vector used
        when the run was originally built.
        """
        rerank_with = self._normalize_rerank(rerank)
        ids = self._resolve_query_ids(queries)
        run = Run()
        desc = f"batch_search(mode={mode!r}, rerank={rerank_with!r})"
        for query_id in tqdm(ids, desc=desc, unit="queries"):
            query_payload = self.queries.get(query_id) or {}
            query_text = query_payload.get("text")

            if mode == "lexical":
                if query_text is None:
                    raise ValueError(f"Query {query_id!r} has no stored 'text'; required for mode='lexical'.")
                base_limit = rerank_depth if rerank_with else top_k
                ranked = self._lexical_rank(query_text, limit=base_limit, raw=raw)
                if rerank_with:
                    ranked = self._rerank(
                        ranked, rerank_with, limit=top_k, query_text=query_text, query_id=query_id
                    )
            elif mode == "semantic":
                ranked = self._semantic_rank(
                    query_text=query_text,
                    query_id=query_id,
                    limit=top_k,
                    rerank=rerank_with,
                    rerank_depth=rerank_depth,
                )
            elif mode == "hybrid":
                if query_text is None:
                    raise ValueError(
                        f"Query {query_id!r} has no stored 'text'; required for mode='hybrid' (lexical stage)."
                    )
                lexical = self._lexical_rank(query_text, limit=candidate_limit)
                semantic = self._semantic_rank(
                    query_text=query_text,
                    query_id=query_id,
                    limit=candidate_limit,
                    rerank=rerank_with,
                    rerank_depth=rerank_depth,
                )
                ranked = self._reciprocal_rank_fusion(lexical, semantic, rrf_k)
            else:
                raise ValueError(f"Unknown mode {mode!r}; expected 'lexical', 'semantic', or 'hybrid'.")

            run[query_id] = self._materialize(ranked, top_k)
        return run

    def _resolve_query_ids(self, queries: "str | list[str] | None") -> list[str]:
        if queries is None:
            return list(self.queries)
        if isinstance(queries, str):
            return [queries]
        return [str(q) for q in queries]

    # -- text extras: snippets / highlighting -------------------------------

    def snippet(
        self,
        query: str,
        column: int = -1,
        pre: str = "<b>",
        post: str = "</b>",
        ellipsis: str = "...",
        max_tokens: int = 30,
        limit: int = 20,
    ) -> list[SearchResult]:
        """Short fragments of matching documents with match markup, via
        SQLite's ``snippet()``."""
        max_tokens = max(1, min(int(max_tokens), 64))
        q = _as_or_query(query)
        if not q:
            return []
        cur = self.conn.execute(
            """
            SELECT d.id, d.payload, snippet(documents_fts, ?, ?, ?, ?, ?) AS snippet_text
            FROM documents_fts JOIN documents d ON d.id = documents_fts.id
            WHERE documents_fts MATCH ? LIMIT ?
            """,
            (column, pre, post, ellipsis, max_tokens, q, limit),
        )
        return [
            SearchResult(id=doc_id, snippet=snippet_text, document=json.loads(payload))
            for doc_id, payload, snippet_text in cur.fetchall()
        ]

    def highlight(
        self,
        query: str,
        column: int = 0,
        pre: str = "<b>",
        post: str = "</b>",
        limit: int = 20,
    ) -> list[SearchResult]:
        """Full column text of matching documents with match markup, via
        SQLite's ``highlight()``."""
        q = _as_or_query(query)
        if not q:
            return []
        cur = self.conn.execute(
            """
            SELECT d.id, d.payload, highlight(documents_fts, ?, ?, ?) AS highlighted_text
            FROM documents_fts JOIN documents d ON d.id = documents_fts.id
            WHERE documents_fts MATCH ? LIMIT ?
            """,
            (column, pre, post, q, limit),
        )
        return [
            SearchResult(id=doc_id, highlight=highlighted_text, document=json.loads(payload))
            for doc_id, payload, highlighted_text in cur.fetchall()
        ]

    # -- ranking internals ---------------------------------------------------
    # Each of these returns an ordered (best-first) list of
    # (item_id, extra_fields_dict) pairs. Keeping the "who matched, with what
    # score" step separate from "attach the document payload" (_materialize)
    # is what lets `search`, `batch_search`, and the hybrid fusion stage all
    # share the same building blocks instead of three near-duplicate methods.

    def _lexical_rank(self, query: str, limit: int, b: float = 0.6, k1: float = 0.9, raw: bool = False):
        q = query if raw else _as_or_query(query)
        if not q:
            return []
        cur = self.conn.execute(
            """
            SELECT id, bm25(documents_fts, ?, ?) AS score
            FROM documents_fts WHERE documents_fts MATCH ?
            ORDER BY score LIMIT ?
            """,
            (float(b), float(k1), q, limit),
        )
        # FTS5's bm25() is lower-is-better; negate so "higher is better"
        # holds for every ranker's score, matching cosine/rrf.
        return [(doc_id, {"score": -score}) for doc_id, score in cur.fetchall()]

    def _semantic_rank(
        self,
        query_text: "str | None" = None,
        query_id: "str | None" = None,
        limit: int = 50,
        rerank: "str | None" = None,
        rerank_depth: int = 200,
        candidate_ids: "list[str] | None" = None,
    ):
        query_blob, query_vector = self._query_vector(query_text, query_id)
        distances = self._hamming_rank(query_blob, candidate_ids)
        base_ranked = [(doc_id, {"hamming_distance": dist}) for doc_id, dist in distances]

        if not rerank:
            return base_ranked[:limit]
        if rerank == "hamming":
            raise ValueError(
                "mode='semantic' is already Hamming/Binary-ranked; rerank='hamming' would "
                "be a no-op. Use rerank='cosine' for Hamming/Binary >> Cosine/Full."
            )
        return self._rerank(
            base_ranked,
            rerank,
            limit=limit,
            rerank_depth=rerank_depth,
            query_blob=query_blob,
            query_vector=query_vector,
        )

    @staticmethod
    def _normalize_rerank(rerank: "bool | str | None") -> "str | None":
        """Normalize the public ``rerank=`` argument (``False`` / ``None``
        / ``True`` / ``"hamming"`` / ``"cosine"``) to ``None`` or one of
        ``"hamming"``/``"cosine"``. ``True`` is kept as a backward-
        compatible synonym for ``"cosine"``, the only rerank previously
        supported."""
        if rerank is False or rerank is None:
            return None
        if rerank is True:
            return "cosine"
        if rerank in ("hamming", "cosine"):
            return rerank
        raise ValueError(f"Invalid rerank={rerank!r}; expected False, True, 'hamming', or 'cosine'.")

    def _rerank(
        self,
        base_ranked,
        rerank_with: str,
        limit: int,
        rerank_depth: "int | None" = None,
        query_text: "str | None" = None,
        query_id: "str | None" = None,
        query_blob: "bytes | None" = None,
        query_vector: "np.ndarray | None" = None,
    ):
        """Second-stage rerank of an ordered (doc_id, extra) list — the
        shared machinery behind BM25 >> Hamming/Binary, BM25 >> Cosine/
        Full, and Hamming/Binary >> Cosine/Full. *rerank_with* selects the
        embedding/distance used: ``"hamming"`` (binary embeddings +
        Hamming distance) or ``"cosine"`` (full-precision embeddings +
        cosine similarity). *base_ranked* is truncated to *rerank_depth*
        candidates before reranking, if given."""
        if not base_ranked:
            return []
        shortlist = base_ranked[:rerank_depth] if rerank_depth is not None else base_ranked
        base_extra = dict(shortlist)
        candidate_ids = [doc_id for doc_id, _ in shortlist]

        if query_blob is None:
            query_blob, query_vector = self._query_vector(query_text, query_id)

        if rerank_with == "hamming":
            reranked = self._hamming_rank(query_blob, candidate_ids)[:limit]
            return [
                (doc_id, {**base_extra.get(doc_id, {}), "hamming_distance": dist})
                for doc_id, dist in reranked
            ]
        elif rerank_with == "cosine":
            if query_vector is None:
                raise RuntimeError(
                    "rerank='cosine' requires a full-precision query embedding, but none is "
                    "stored/available for this query. Index with store_full_embeddings=True "
                    "(the default), or attach a model via add_model()."
                )
            reranked = self._cosine_rerank(query_vector, candidate_ids, limit)
            return [
                (doc_id, {**base_extra.get(doc_id, {}), "cosine_similarity": sim})
                for doc_id, sim in reranked
            ]
        else:
            raise ValueError(f"Unknown rerank_with {rerank_with!r}; expected 'hamming' or 'cosine'.")

    def _reciprocal_rank_fusion(self, lexical, semantic, rrf_k: int):
        lexical_ranks = {doc_id: i + 1 for i, (doc_id, _) in enumerate(lexical)}
        semantic_ranks = {doc_id: i + 1 for i, (doc_id, _) in enumerate(semantic)}
        lexical_extra = dict(lexical)
        semantic_extra = dict(semantic)

        scores: dict[str, float] = {}
        for doc_id in set(lexical_ranks) | set(semantic_ranks):
            score = 0.0
            if doc_id in lexical_ranks:
                score += 1.0 / (rrf_k + lexical_ranks[doc_id])
            if doc_id in semantic_ranks:
                score += 1.0 / (rrf_k + semantic_ranks[doc_id])
            scores[doc_id] = score

        ranked_ids = sorted(scores, key=lambda d: scores[d], reverse=True)
        return [
            (
                doc_id,
                {
                    "rrf_score": scores[doc_id],
                    "lexical_rank": lexical_ranks.get(doc_id),
                    "semantic_rank": semantic_ranks.get(doc_id),
                    **lexical_extra.get(doc_id, {}),
                    **semantic_extra.get(doc_id, {}),
                },
            )
            for doc_id in ranked_ids
        ]

    def _materialize(self, ranked, limit: int) -> list[SearchResult]:
        results = []
        for doc_id, extra in ranked[:limit]:
            document = self.documents.get(doc_id)
            if document is None:
                continue
            results.append(SearchResult(id=doc_id, document=document, **extra))
        return results

    def _query_vector(self, query_text: "str | None", query_id: "str | None"):
        """Return ``(binary_blob, full_vector)`` for a query: a stored
        precomputed embedding (looked up by *query_id*) if one exists,
        otherwise *query_text* encoded with the attached model."""
        if query_id is not None:
            binary_blob, full_vector = self._stored_query_embedding(query_id)
            if binary_blob is not None:
                return binary_blob, full_vector

        if query_text is None:
            raise ValueError(
                "No query embedding available: no stored embedding for this query id, "
                "and no query text to encode."
            )
        if self._model is None:
            raise RuntimeError(
                "Semantic/hybrid search requires a model — call index.add_model(...) first, "
                "or index this query's embedding via index_queries()."
            )
        from sentence_transformers import quantize_embeddings

        full_vector = self._model.encode_queries([query_text])[0].astype(np.float32)
        binary_blob = sqlite3.Binary(quantize_embeddings(full_vector[None, :], precision="ubinary").tobytes())
        return binary_blob, full_vector

    def _stored_query_embedding(self, query_id: str):
        binary_blob = None
        full_vector = None
        cur = self.conn.execute("SELECT embedding FROM query_embeddings WHERE id = ?", (query_id,))
        row = cur.fetchone()
        if row is not None:
            binary_blob = row[0]
        cur = self.conn.execute("SELECT embedding FROM query_embeddings_full WHERE id = ?", (query_id,))
        row = cur.fetchone()
        if row is not None:
            full_vector = np.frombuffer(row[0], dtype=np.float32)
        return binary_blob, full_vector

    def _hamming_rank(self, query_blob: bytes, candidate_ids: "list[str] | None"):
        """``(doc_id, hamming_distance)`` pairs, ascending (closest first).
        Computed inside SQLite by the ``hamming_distance()`` extension."""
        if candidate_ids is not None and not candidate_ids:
            return []
        if candidate_ids is not None:
            placeholders = ",".join("?" for _ in candidate_ids)
            cur = self.conn.execute(
                f"""
                SELECT id, hamming_distance(embedding, ?) AS dist FROM embeddings
                WHERE id IN ({placeholders}) ORDER BY dist ASC
                """,
                (query_blob, *candidate_ids),
            )
        else:
            cur = self.conn.execute(
                "SELECT id, hamming_distance(embedding, ?) AS dist FROM embeddings ORDER BY dist ASC",
                (query_blob,),
            )
        return [(row[0], row[1]) for row in cur.fetchall()]

    def _cosine_rerank(self, query_vector: np.ndarray, candidate_ids: list[str], top_k: int):
        """``(doc_id, cosine_similarity)`` pairs, descending. Second-stage
        rerank over full-precision embeddings for a short candidate list."""
        if not candidate_ids:
            return []
        placeholders = ",".join("?" for _ in candidate_ids)
        cur = self.conn.execute(
            f"SELECT id, embedding FROM embeddings_full WHERE id IN ({placeholders})", candidate_ids
        )
        rows = cur.fetchall()
        if not rows:
            return []
        ids = [row[0] for row in rows]
        vectors = np.stack([np.frombuffer(row[1], dtype=np.float32) for row in rows])
        query_norm = query_vector / (np.linalg.norm(query_vector) + 1e-12)
        vector_norms = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12)
        similarities = vector_norms @ query_norm
        order = np.argsort(-similarities)[:top_k]
        return [(ids[i], float(similarities[i])) for i in order]

    # -- storage internals -----------------------------------------------

    def _create_tables(self) -> None:
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS documents (id TEXT PRIMARY KEY, payload TEXT NOT NULL)"
        )
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS embeddings (id TEXT PRIMARY KEY, embedding BLOB NOT NULL)"
        )
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS embeddings_full (id TEXT PRIMARY KEY, embedding BLOB NOT NULL)"
        )
        self.conn.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts
            USING fts5(id UNINDEXED, text, tokenize='porter')
            """
        )
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS queries (id TEXT PRIMARY KEY, payload TEXT NOT NULL)"
        )
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS query_embeddings (id TEXT PRIMARY KEY, embedding BLOB NOT NULL)"
        )
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS query_embeddings_full (id TEXT PRIMARY KEY, embedding BLOB NOT NULL)"
        )
        self.conn.commit()

    def _load_hamming_extension(self, ext_path: str) -> None:
        if ext_path == "auto":
            resolved = _bundled_hamming_path()
            if resolved is None:
                raise RuntimeError(
                    "Could not find a compiled hamming_distance extension bundled "
                    f"for this platform ({platform.system()}). Reinstall scrydb (a C "
                    "compiler must be available at install time), or pass an "
                    "explicit hamming_ext_path=... to Index.open(), or pass "
                    "hamming_ext_path=None to disable Hamming-distance search."
                )
            ext_path = str(resolved)

        try:
            self.conn.enable_load_extension(True)
        except AttributeError as exc:
            raise RuntimeError(
                "This Python build's sqlite3 module does not support loading extensions "
                "(enable_load_extension is missing). On macOS, the system Python is usually "
                "built against Apple's SQLite, which disables extension loading. Install "
                "Python via Homebrew (`brew install python`) or via pyenv with "
                "`PYTHON_CONFIGURE_OPTS='--enable-loadable-sqlite-extensions' pyenv install "
                "<version>`, which links against a SQLite that allows it."
            ) from exc
        try:
            self.conn.load_extension(ext_path)
        except sqlite3.OperationalError as exc:
            raise RuntimeError(
                f"Failed to load hamming_distance extension from '{ext_path}'. Make sure it "
                "has been compiled for your platform (see hamming.c / build instructions) and "
                "that the path is correct."
            ) from exc
        finally:
            self.conn.enable_load_extension(False)

    def _flush_documents(self, payload_batch: list, text_batch: list) -> None:
        if not payload_batch:
            return
        with self.conn:
            self.conn.executemany(
                "INSERT OR REPLACE INTO documents (id, payload) VALUES (?, ?)", payload_batch
            )
            # FTS5 has no INSERT OR REPLACE, so delete first.
            ids = [row[0] for row in text_batch]
            if ids:
                placeholders = ",".join("?" for _ in ids)
                self.conn.execute(f"DELETE FROM documents_fts WHERE id IN ({placeholders})", ids)
                self.conn.executemany(
                    "INSERT INTO documents_fts (id, text) VALUES (?, ?)", text_batch
                )

    def _flush_embeddings(self, emb_batch: list, full_emb_batch: list) -> None:
        if not emb_batch and not full_emb_batch:
            return
        with self.conn:
            if emb_batch:
                self.conn.executemany(
                    "INSERT OR REPLACE INTO embeddings (id, embedding) VALUES (?, ?)", emb_batch
                )
            if full_emb_batch:
                self.conn.executemany(
                    "INSERT OR REPLACE INTO embeddings_full (id, embedding) VALUES (?, ?)", full_emb_batch
                )

    def _flush_queries(self, payload_batch: list, emb_batch: list, full_emb_batch: list) -> None:
        if not payload_batch and not emb_batch and not full_emb_batch:
            return
        with self.conn:
            if payload_batch:
                self.conn.executemany(
                    "INSERT OR REPLACE INTO queries (id, payload) VALUES (?, ?)", payload_batch
                )
            if emb_batch:
                self.conn.executemany(
                    "INSERT OR REPLACE INTO query_embeddings (id, embedding) VALUES (?, ?)", emb_batch
                )
            if full_emb_batch:
                self.conn.executemany(
                    "INSERT OR REPLACE INTO query_embeddings_full (id, embedding) VALUES (?, ?)", full_emb_batch
                )

    def close(self) -> None:
        self.conn.close()