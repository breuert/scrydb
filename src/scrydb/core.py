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
                                              (``_binary``/``_int8`` variants
                                              for the quantized vectors)
    run.write_trec(path, tag=...)         -- TREC run file
    run.to_dataframe()                    -- pandas, for analysis/eval

``search``/``batch_search`` take ``mode`` ("lexical", "semantic", or
"hybrid"), ``precision`` ("binary", "int8", or "float" -- which vector
representation ``mode="semantic"``/the semantic side of ``mode="hybrid"``
ranks with), and ``rerank`` (``False``, ``"binary"``, ``"int8"``,
``"float"``, or ``True`` as a synonym for ``"float"``; the legacy
``"hamming"``/``"cosine"`` spellings are accepted as aliases for
``"binary"``/``"float"``), which together select any of a dozen standard
strategies, e.g.:

    BM25                              mode="lexical",  rerank=False
    BM25 >> Binary/Hamming            mode="lexical",  rerank="binary"
    BM25 >> Int8                      mode="lexical",  rerank="int8"
    BM25 >> Float/Cosine              mode="lexical",  rerank="float"
    Binary/Hamming                    mode="semantic", precision="binary"
    Int8                              mode="semantic", precision="int8"
    Float/Cosine                      mode="semantic", precision="float"
    Binary >> Float/Cosine            mode="semantic", precision="binary", rerank="float"
    Hybrid/RRF                        mode="hybrid"

Vector search (all three precisions, plus quantization) is powered by the
`sqlite-vec <https://github.com/asg017/sqlite-vec>`_ SQLite extension,
loaded from the ``sqlite-vec`` PyPI package -- a pure pip dependency with
prebuilt wheels for Linux/macOS/Windows, so no C compiler is required at
install time.

Requires: numpy, tqdm, sqlite-vec, pandas (for ``to_dataframe``),
sentence-transformers (only exercised when a model is attached or
precomputed embeddings are indexed/searched).
"""

from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import sqlite_vec
from tqdm import tqdm


# ===========================================================================
# Vector precisions -- the three ways an embedding can be stored/searched
# via sqlite-vec's ``vec0`` virtual tables. Every precision-aware knob in
# the public API (``precision=``, ``rerank=``) normalizes down to one of
# these three strings.
# ===========================================================================

_PRECISIONS = ("binary", "int8", "float")

# Legacy/ergonomic spellings from the Hamming-distance-based v0.1.x API,
# kept working as aliases: "hamming" was always binary/ubinary search,
# "cosine" was always full-precision.
_PRECISION_ALIASES = {"hamming": "binary", "cosine": "float"}

# SQL expressions (one ``?`` placeholder each) that turn a *raw float32
# vector* blob into the form stored in each precision's vec0 column.
# Binary quantization is a per-dimension sign bit (scale-invariant); int8
# quantization maps a [-1, 1] range onto the 256 int8 buckets, so the
# vector is L2-normalized first via ``vec_normalize()`` -- otherwise
# embeddings with any dimension outside [-1, 1] would saturate.
_QUANTIZE_EXPR = {
    "binary": "vec_quantize_binary(vec_f32(?))",
    "int8": "vec_quantize_int8(vec_normalize(vec_f32(?)), 'unit')",
    "float": "vec_f32(?)",
}

# SQL expressions (one ``?`` placeholder each) that tag an *already
# quantized* blob (e.g. one fetched back out of a vec0 column) with the
# right element type for use as a MATCH operand -- a raw BLOB parameter is
# ambiguous to sqlite-vec without this.
_CAST_EXPR = {
    "binary": "vec_bit(?)",
    "int8": "vec_int8(?)",
    "float": "vec_f32(?)",
}

# ``vec0`` column definitions per precision, parameterized by dimension.
# int8/float use cosine distance so results are reported as a similarity
# (1 - distance); bit columns are always compared by Hamming distance.
_COLUMN_DEF = {
    "binary": lambda dim: f"embedding bit[{dim}]",
    "int8": lambda dim: f"embedding int8[{dim}] distance_metric=cosine",
    "float": lambda dim: f"embedding float[{dim}] distance_metric=cosine",
}

# The result field each precision's base/rerank ranking is reported under.
_SCORE_FIELD = {
    "binary": "hamming_distance",
    "int8": "int8_similarity",
    "float": "cosine_similarity",
}

# numpy dtype used to decode a precision's stored embedding blob back into
# an array (see _EmbeddingTable).
_NUMPY_DTYPE = {"binary": np.uint8, "int8": np.int8, "float": np.float32}


def _normalize_precision(value: str) -> str:
    value = _PRECISION_ALIASES.get(value, value)
    if value not in _PRECISIONS:
        raise ValueError(
            f"Invalid precision {value!r}; expected 'binary', 'int8', or 'float' "
            "(legacy aliases 'hamming'/'cosine' are also accepted)."
        )
    return value


def _score_from_distance(precision: str, distance: float):
    if precision == "binary":
        return int(round(distance))
    return 1.0 - distance  # cosine distance -> cosine similarity


# ===========================================================================
# Query text helpers
# ===========================================================================

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
    # int8_similarity, hamming_distance) must outrank the plain lexical
    # "score": _rerank_vec() merges a reranked result's fields on top of its
    # base-stage fields (``{**base_extra, "hamming_distance": dist}``), so a
    # BM25 → rerank result still carries its original lexical "score"
    # alongside the new rerank field. Most eval tools (trec_eval,
    # pytrec_eval, ir_measures, ...) rank purely by the written SCORE column
    # and ignore row/rank order, so picking the stale "score" here would
    # silently drop the rerank stage from evaluation while still *looking*
    # reranked in the RANK column.
    _SCORE_FIELDS = (
        ("rrf_score", False),
        ("cosine_similarity", False),
        ("int8_similarity", False),
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
    """Read-only ``{id: np.ndarray}`` view over a precision's ``vec0``
    embedding table, joined back to its parent ``documents``/``queries``
    table by rowid.

    *precision* is ``"binary"`` (packed bits, ``np.uint8``), ``"int8"``
    (``np.int8``), or ``"float"`` (full-precision, ``np.float32``). If the
    underlying ``vec0`` table hasn't been created yet (nothing has been
    indexed at that precision), this behaves as an empty mapping rather
    than raising.
    """

    def __init__(self, conn: sqlite3.Connection, collection: str, precision: str):
        self._conn = conn
        self._collection = collection
        self._precision = precision
        self._vec_table = f"vec_{collection}_{precision}"
        self._dtype = _NUMPY_DTYPE[precision]

    def _exists(self) -> bool:
        return self._conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (self._vec_table,)
        ).fetchone() is not None

    def __getitem__(self, item_id: str) -> np.ndarray:
        if not self._exists():
            raise KeyError(item_id)
        cur = self._conn.execute(
            f"""
            SELECT v.embedding FROM {self._vec_table} v
            JOIN {self._collection} d ON d.rowid = v.rowid
            WHERE d.id = ?
            """,
            (str(item_id),),
        )
        row = cur.fetchone()
        if row is None:
            raise KeyError(item_id)
        return np.frombuffer(row[0], dtype=self._dtype)

    def __iter__(self) -> Iterator[str]:
        if not self._exists():
            return
        cur = self._conn.execute(
            f"SELECT d.id FROM {self._collection} d JOIN {self._vec_table} v ON d.rowid = v.rowid"
        )
        for (item_id,) in cur:
            yield item_id

    def __len__(self) -> int:
        if not self._exists():
            return 0
        return self._conn.execute(
            f"SELECT COUNT(*) FROM {self._collection} d JOIN {self._vec_table} v ON d.rowid = v.rowid"
        ).fetchone()[0]

    def __repr__(self) -> str:
        return f"<{self._vec_table}: {len(self)} items>"


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

    Dense/hybrid search is powered by the ``sqlite-vec`` extension: vectors
    are stored (and searched) in up to three precisions per collection —
    ``binary`` (1 bit/dim, Hamming distance), ``int8`` (1 byte/dim, cosine),
    and ``float`` (full precision, cosine) — each backed by its own
    ``vec0`` virtual table.

    Open with :meth:`Index.open`; it doubles as a context manager::

        with Index.open("idx.db") as index:
            index.add_model(SentenceEmbedding())
            index.index_documents("corpus.jsonl", id_field="docid", text_field="text")
            results = index.search("...", mode="hybrid", rerank=True)
    """

    def __init__(self, db_path: "str | Path" = "idx.db", vec_ext_path: "str | None" = "auto"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self._model: "SentenceEmbedding | None" = None

        if vec_ext_path is not None:
            self._load_vec_extension(vec_ext_path)

        self._create_tables()

    @classmethod
    def open(cls, db_path: "str | Path" = "idx.db", vec_ext_path: "str | None" = "auto") -> "Index":
        """Open (creating if needed) an index at *db_path*. Mirrors
        ``sqlite3.connect``/``pathlib.Path.open``: the one obvious entry
        point.

        *vec_ext_path* controls the ``sqlite-vec`` extension used for
        binary/int8/float vector search:

        - ``"auto"`` (default) -- load the copy of the extension bundled
          inside the installed ``sqlite-vec`` pip package
          (``sqlite_vec.loadable_path()``), prebuilt for the current
          platform. No compiler needed.
        - an explicit path -- load that build of the extension instead
          (e.g. a custom/newer ``vec0`` build).
        - ``None`` -- skip loading it; semantic/hybrid search and rerank
          are unavailable, but lexical (BM25) search still works.
        """
        return cls(db_path=db_path, vec_ext_path=vec_ext_path)

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
        return _EmbeddingTable(self.conn, "documents", "float")

    @property
    def document_embeddings_binary(self) -> _EmbeddingTable:
        """Read-only mapping ``{doc_id: uint8 ndarray}`` of binary-quantized
        (packed-bit) document embeddings."""
        return _EmbeddingTable(self.conn, "documents", "binary")

    @property
    def document_embeddings_int8(self) -> _EmbeddingTable:
        """Read-only mapping ``{doc_id: int8 ndarray}`` of int8-quantized
        document embeddings (requires ``store_int8_embeddings=True``)."""
        return _EmbeddingTable(self.conn, "documents", "int8")

    @property
    def query_embeddings(self) -> _EmbeddingTable:
        """Read-only mapping ``{query_id: float32 ndarray}`` of full-precision
        query embeddings (requires ``store_full_embeddings=True``)."""
        return _EmbeddingTable(self.conn, "queries", "float")

    @property
    def query_embeddings_binary(self) -> _EmbeddingTable:
        """Read-only mapping ``{query_id: uint8 ndarray}`` of binary-quantized
        (packed-bit) query embeddings."""
        return _EmbeddingTable(self.conn, "queries", "binary")

    @property
    def query_embeddings_int8(self) -> _EmbeddingTable:
        """Read-only mapping ``{query_id: int8 ndarray}`` of int8-quantized
        query embeddings (requires ``store_int8_embeddings=True``)."""
        return _EmbeddingTable(self.conn, "queries", "int8")

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
        store_int8_embeddings: bool = False,
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
        that row. Binary-quantized embeddings are always stored whenever a
        vector is available; *store_full_embeddings*/*store_int8_embeddings*
        additionally store full-precision/int8-quantized copies (all
        quantization happens inside SQLite via ``sqlite-vec``).
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
            store_int8_embeddings=store_int8_embeddings,
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
        store_int8_embeddings: bool = False,
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
            store_int8_embeddings=store_int8_embeddings,
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
        store_int8_embeddings: bool,
        is_query: bool,
    ) -> None:
        precisions = ["binary"]
        if store_int8_embeddings:
            precisions.append("int8")
        if store_full_embeddings:
            precisions.append("float")

        payload_batch: list = []
        text_batch: list = []  # (id, text) for FTS (documents only)
        emb_batch: list = []  # (id, float32 ndarray)
        pending_encode: list = []  # (id, text) rows needing the attached model

        desc = "Indexing queries" if is_query else "Indexing documents"
        pbar = tqdm(desc=desc, unit="rows", unit_scale=True, total=limit)

        def flush():
            id_rowids = self._flush_payloads(table, payload_batch, None if is_query else text_batch)
            self._flush_vec_embeddings(table, id_rowids, emb_batch, precisions)
            payload_batch.clear()
            text_batch.clear()
            emb_batch.clear()

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
                emb_batch.append((item_id, np.asarray(emb, dtype=np.float32)))
            elif self._model is not None and text is not None:
                pending_encode.append((item_id, str(text)))

            n += 1
            if len(payload_batch) >= batch_size:
                if pending_encode:
                    self._encode_and_append(pending_encode, emb_batch, is_query)
                    pending_encode.clear()
                flush()
            pbar.update(1)

        if pending_encode:
            self._encode_and_append(pending_encode, emb_batch, is_query)
        if payload_batch or emb_batch:
            flush()
        pbar.close()

    def _encode_and_append(self, pending, emb_batch, is_query) -> None:
        ids = [item_id for item_id, _ in pending]
        texts = [text for _, text in pending]
        vectors = self._model.encode_queries(texts) if is_query else self._model.encode_documents(texts)
        for item_id, vec in zip(ids, vectors):
            emb_batch.append((item_id, np.asarray(vec, dtype=np.float32)))

    # -- interactive search --------------------------------------------------

    def search(
        self,
        query: str,
        mode: str = "lexical",
        top_k: int = 10,
        rerank: "bool | str" = False,
        precision: str = "binary",
        rerank_depth: int = 200,
        candidate_limit: int = 50,
        rrf_k: int = 60,
        raw: bool = False,
        bm25_b: float = 0.6,
        bm25_k1: float = 0.9,
    ) -> list[SearchResult]:
        """Run one query, return a ranked list of :class:`SearchResult`.

        Parameters
        ----------
        mode:
            ``"lexical"`` (default, BM25 over FTS5), ``"semantic"``
            (vector search at *precision*, requires :meth:`add_model` or
            precomputed embeddings), or ``"hybrid"`` (Reciprocal Rank
            Fusion of both).
        precision:
            Which vector representation ``mode="semantic"``/the semantic
            side of ``mode="hybrid"`` searches with: ``"binary"``
            (default -- 1 bit/dim, Hamming distance), ``"int8"`` (1
            byte/dim, cosine), or ``"float"`` (full precision, cosine).
            Legacy aliases ``"hamming"``/``"cosine"`` are also accepted.
            Ignored when ``mode="lexical"``.
        rerank:
            Adds a second-stage rerank over the top *rerank_depth*
            candidates from ``mode``. ``False`` (default) -- no rerank.
            Otherwise one of ``"binary"``, ``"int8"``, ``"float"`` (or the
            legacy ``"hamming"``/``"cosine"`` aliases, or ``True`` as a
            synonym for ``"float"``), which reranks the candidates by
            vector search at that precision. With ``mode="semantic"``,
            *rerank* must differ from *precision* (e.g.
            ``precision="binary", rerank="float"`` for Binary >>
            Float/Cosine). With ``mode="hybrid"``, *rerank* applies to the
            semantic side before fusion.
        raw:
            ``mode="lexical"`` only. If ``True``, *query* is passed to
            FTS5's ``MATCH`` unmodified (advanced syntax the caller wrote
            themselves). If ``False`` (default), it is sanitized and its
            terms OR-joined, which is safe for arbitrary natural-language
            input (e.g. a trailing "?" no longer raises a syntax error).
        bm25_b, bm25_k1:
            BM25 parameters passed through to FTS5's ``bm25()``, used
            whenever ``mode`` is ``"lexical"`` or ``"hybrid"``.
        """
        rerank_with = self._normalize_rerank(rerank)
        if mode == "lexical":
            base_limit = rerank_depth if rerank_with else top_k
            ranked = self._lexical_rank(query, limit=base_limit, b=bm25_b, k1=bm25_k1, raw=raw)
            if rerank_with:
                ranked = self._rerank_vec(ranked, rerank_with, limit=top_k, query_text=query)
        elif mode == "semantic":
            ranked = self._semantic_rank(
                query_text=query,
                limit=top_k,
                precision=_normalize_precision(precision),
                rerank=rerank_with,
                rerank_depth=rerank_depth,
            )
        elif mode == "hybrid":
            lexical = self._lexical_rank(query, limit=candidate_limit, b=bm25_b, k1=bm25_k1)
            semantic = self._semantic_rank(
                query_text=query,
                limit=candidate_limit,
                precision=_normalize_precision(precision),
                rerank=rerank_with,
                rerank_depth=rerank_depth,
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
        precision: str = "binary",
        rerank_depth: int = 200,
        candidate_limit: int = 50,
        rrf_k: int = 60,
        raw: bool = False,
        bm25_b: float = 0.6,
        bm25_k1: float = 0.9,
    ) -> Run:
        """Run many stored queries, return a :class:`Run`.

        Same ``mode``/``precision``/``rerank``/``raw``/``bm25_b``/``bm25_k1``
        vocabulary as :meth:`search`, but driven by stored query ids
        instead of a literal query string.

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
        precision_n = _normalize_precision(precision)
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
                ranked = self._lexical_rank(query_text, limit=base_limit, b=bm25_b, k1=bm25_k1, raw=raw)
                if rerank_with:
                    ranked = self._rerank_vec(
                        ranked, rerank_with, limit=top_k, query_text=query_text, query_id=query_id
                    )
            elif mode == "semantic":
                ranked = self._semantic_rank(
                    query_text=query_text,
                    query_id=query_id,
                    limit=top_k,
                    precision=precision_n,
                    rerank=rerank_with,
                    rerank_depth=rerank_depth,
                )
            elif mode == "hybrid":
                if query_text is None:
                    raise ValueError(
                        f"Query {query_id!r} has no stored 'text'; required for mode='hybrid' (lexical stage)."
                    )
                lexical = self._lexical_rank(query_text, limit=candidate_limit, b=bm25_b, k1=bm25_k1)
                semantic = self._semantic_rank(
                    query_text=query_text,
                    query_id=query_id,
                    limit=candidate_limit,
                    precision=precision_n,
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
        precision: str = "binary",
        rerank: "str | None" = None,
        rerank_depth: int = 200,
        candidate_ids: "list[str] | None" = None,
    ):
        stored, float_vector = self._query_material(query_text, query_id)
        base_ranked = self._vec_search("documents", precision, stored, float_vector, limit, candidate_ids)

        if not rerank:
            return base_ranked
        if rerank == precision:
            raise ValueError(
                f"mode='semantic' is already {precision!r}-ranked; rerank={rerank!r} would "
                "be a no-op. Use a different precision, e.g. rerank='float' for Binary >> "
                "Float/Cosine."
            )
        return self._rerank_vec(
            base_ranked, rerank, limit=limit, rerank_depth=rerank_depth, stored=stored, float_vector=float_vector
        )

    @staticmethod
    def _normalize_rerank(rerank: "bool | str | None") -> "str | None":
        """Normalize the public ``rerank=`` argument (``False`` / ``None``
        / ``True`` / ``"binary"`` / ``"int8"`` / ``"float"``, plus the
        legacy ``"hamming"``/``"cosine"`` aliases) to ``None`` or one of
        ``"binary"``/``"int8"``/``"float"``. ``True`` is kept as a
        backward-compatible synonym for ``"float"`` (the only rerank
        previously supported)."""
        if rerank is False or rerank is None:
            return None
        if rerank is True:
            return "float"
        return _normalize_precision(rerank)

    def _rerank_vec(
        self,
        base_ranked,
        rerank_with: str,
        limit: int,
        rerank_depth: "int | None" = None,
        stored: "dict | None" = None,
        float_vector: "np.ndarray | None" = None,
        query_text: "str | None" = None,
        query_id: "str | None" = None,
    ):
        """Second-stage rerank of an ordered (doc_id, extra) list — the
        shared machinery behind BM25 >> Binary, BM25 >> Int8, BM25 >>
        Float/Cosine, and e.g. Binary >> Float/Cosine. *rerank_with*
        selects the vec0 precision used. *base_ranked* is truncated to
        *rerank_depth* candidates before reranking, if given."""
        if not base_ranked:
            return []
        shortlist = base_ranked[:rerank_depth] if rerank_depth is not None else base_ranked
        base_extra = dict(shortlist)
        candidate_ids = [doc_id for doc_id, _ in shortlist]

        if stored is None:
            stored, float_vector = self._query_material(query_text, query_id)

        reranked = self._vec_search("documents", rerank_with, stored, float_vector, limit, candidate_ids)
        return [(doc_id, {**base_extra.get(doc_id, {}), **extra}) for doc_id, extra in reranked]

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

    # -- sqlite-vec search internals ----------------------------------------

    def _query_material(self, query_text: "str | None", query_id: "str | None"):
        """Return ``(stored, float_vector)`` for a query: *stored* is
        ``{"binary"|"int8"|"float": blob-or-None}`` of whatever precomputed
        embeddings are stored for *query_id* (via :meth:`index_queries`),
        and *float_vector* is a raw float32 ``np.ndarray`` — the stored
        full-precision embedding if present, else *query_text* encoded
        with the attached model, else ``None``.

        Storage is preferred over re-encoding so a query searched at
        ``precision="binary"`` can use its exact indexed binary vector
        without needing a float vector (or a model) at all."""
        stored = {"binary": None, "int8": None, "float": None}
        if query_id is not None:
            for precision in stored:
                stored[precision] = self._stored_embedding_blob("queries", precision, query_id)

        float_vector = None
        if stored["float"] is not None:
            float_vector = np.frombuffer(stored["float"], dtype=np.float32)
        elif query_text is not None and self._model is not None:
            float_vector = self._model.encode_queries([query_text])[0].astype(np.float32)

        if float_vector is None and all(v is None for v in stored.values()):
            if query_text is None:
                raise ValueError(
                    "No query embedding available: no stored embedding for this query id, "
                    "and no query text to encode."
                )
            raise RuntimeError(
                "Semantic/hybrid search requires a model — call index.add_model(...) first, "
                "or index this query's embedding via index_queries()."
            )
        return stored, float_vector

    def _stored_embedding_blob(self, collection: str, precision: str, item_id: str) -> "bytes | None":
        table = f"vec_{collection}_{precision}"
        if not self._table_exists(table):
            return None
        cur = self.conn.execute(
            f"SELECT v.embedding FROM {table} v JOIN {collection} d ON d.rowid = v.rowid WHERE d.id = ?",
            (str(item_id),),
        )
        row = cur.fetchone()
        return row[0] if row is not None else None

    def _query_expr(self, precision: str, stored: dict, float_vector: "np.ndarray | None"):
        """Return ``(param_value, sql_expr)`` for use as ``embedding MATCH
        {sql_expr}`` with ``param_value`` bound to its one ``?``: the
        stored blob for *precision* if available (tagged with
        :data:`_CAST_EXPR`, no requantization), else *float_vector*
        quantized on the fly (:data:`_QUANTIZE_EXPR`)."""
        blob = stored.get(precision)
        if blob is not None:
            return blob, _CAST_EXPR[precision]
        if float_vector is None:
            raise RuntimeError(
                f"No query embedding available at precision={precision!r}: no stored "
                "embedding for this query, and no float vector to derive it from "
                "(attach a model via add_model(), or index this query's embedding via "
                "index_queries())."
            )
        value = sqlite3.Binary(np.asarray(float_vector, dtype=np.float32).tobytes())
        return value, _QUANTIZE_EXPR[precision]

    def _vec_search(
        self,
        collection: str,
        precision: str,
        stored: dict,
        float_vector: "np.ndarray | None",
        limit: int,
        candidate_ids: "list[str] | None" = None,
    ):
        """``(item_id, {score_field: value})`` pairs, best-first — a KNN
        query against the ``vec0`` table for *collection*/*precision*,
        optionally restricted to *candidate_ids* (used by rerank stages).
        """
        table = f"vec_{collection}_{precision}"
        if not self._table_exists(table):
            return []

        value, expr = self._query_expr(precision, stored, float_vector)
        where = [f"embedding MATCH {expr}"]
        params: list = [value]

        if candidate_ids is not None:
            if not candidate_ids:
                return []
            rowid_map = self._rowids_for_ids(collection, candidate_ids)
            rowids = list(rowid_map.values())
            if not rowids:
                return []
            placeholders = ",".join("?" for _ in rowids)
            where.append(f"rowid IN ({placeholders})")
            params.extend(rowids)

        where.append("k = ?")
        params.append(limit)

        cur = self.conn.execute(
            f"SELECT rowid, distance FROM {table} WHERE {' AND '.join(where)} ORDER BY distance", params
        )
        rows = cur.fetchall()
        if not rows:
            return []
        field = _SCORE_FIELD[precision]
        rowid_to_id = self._ids_for_rowids(collection, [rowid for rowid, _ in rows])
        return [
            (rowid_to_id[rowid], {field: _score_from_distance(precision, dist)})
            for rowid, dist in rows
            if rowid in rowid_to_id
        ]

    def _rowids_for_ids(self, collection: str, ids: "list[str]") -> "dict[str, int]":
        ids = [str(i) for i in ids]
        placeholders = ",".join("?" for _ in ids)
        cur = self.conn.execute(f"SELECT id, rowid FROM {collection} WHERE id IN ({placeholders})", ids)
        return {row[0]: row[1] for row in cur.fetchall()}

    def _ids_for_rowids(self, collection: str, rowids: "list[int]") -> "dict[int, str]":
        rowids = list(dict.fromkeys(rowids))
        if not rowids:
            return {}
        placeholders = ",".join("?" for _ in rowids)
        cur = self.conn.execute(f"SELECT rowid, id FROM {collection} WHERE rowid IN ({placeholders})", rowids)
        return {row[0]: row[1] for row in cur.fetchall()}

    def _table_exists(self, name: str) -> bool:
        return self.conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
        ).fetchone() is not None

    # -- storage internals -----------------------------------------------

    def _create_tables(self) -> None:
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS documents (id TEXT PRIMARY KEY, payload TEXT NOT NULL)"
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
        self.conn.commit()

    def _ensure_vec_table(self, table: str, precision: str, dim: int) -> None:
        """Lazily create the ``vec0`` table for *table* (e.g.
        ``vec_documents_binary``) the first time a vector of that
        precision is indexed — the dimension isn't known until then. A
        no-op if the table already exists (its dimension was fixed by
        whichever call created it first)."""
        if self._table_exists(table):
            return
        coldef = _COLUMN_DEF[precision](dim)
        self.conn.execute(f"CREATE VIRTUAL TABLE {table} USING vec0({coldef})")
        self.conn.commit()

    def _load_vec_extension(self, ext_path: str) -> None:
        if ext_path == "auto":
            ext_path = sqlite_vec.loadable_path()

        try:
            self.conn.enable_load_extension(True)
        except AttributeError as exc:
            raise RuntimeError(
                "This Python build's sqlite3 module does not support loading extensions "
                "(enable_load_extension is missing). This usually means a macOS Python "
                "linked against Apple's SQLite, which disables extension loading -- the "
                "system Python and the CPython installed by actions/setup-python both are. "
                "Install Python via Homebrew (`brew install python`), uv (`uv python install "
                "<version>`), or pyenv with "
                "`PYTHON_CONFIGURE_OPTS='--enable-loadable-sqlite-extensions' pyenv install "
                "<version>`, which link against a SQLite that allows it."
            ) from exc
        try:
            self.conn.load_extension(ext_path)
        except sqlite3.OperationalError as exc:
            raise RuntimeError(
                f"Failed to load the sqlite-vec extension from {ext_path!r}. Make sure the "
                "'sqlite-vec' package is installed for your platform (`pip install "
                "sqlite-vec`), or pass an explicit vec_ext_path=... to Index.open()."
            ) from exc
        finally:
            self.conn.enable_load_extension(False)

    def _flush_payloads(self, table: str, payload_batch: list, text_batch: "list | None") -> "dict[str, int]":
        """Upsert *payload_batch* (``(id, payload_json)`` pairs) into
        *table* — an ``id`` that already exists keeps its ``rowid``
        (SQLite UPDATE-in-place semantics), which is what lets a
        re-indexed document/query's ``vec0`` rows stay in sync instead of
        going stale under a new rowid. Returns the resulting ``{id:
        rowid}`` map for the whole batch, used to key the embedding
        inserts. *text_batch*, if given, is (re)written into
        ``documents_fts`` (documents only)."""
        if not payload_batch:
            return {}
        with self.conn:
            self.conn.executemany(
                f"INSERT INTO {table} (id, payload) VALUES (?, ?) "
                "ON CONFLICT(id) DO UPDATE SET payload = excluded.payload",
                payload_batch,
            )
            if text_batch:
                # FTS5 has no INSERT OR REPLACE, so delete first.
                ids = [row[0] for row in text_batch]
                placeholders = ",".join("?" for _ in ids)
                self.conn.execute(f"DELETE FROM documents_fts WHERE id IN ({placeholders})", ids)
                self.conn.executemany("INSERT INTO documents_fts (id, text) VALUES (?, ?)", text_batch)
        return self._rowids_for_ids(table, [row[0] for row in payload_batch])

    def _flush_vec_embeddings(
        self, table: str, id_rowids: "dict[str, int]", emb_batch: list, precisions: "list[str]"
    ) -> None:
        """Write *emb_batch* (``(id, float32 ndarray)`` pairs) into the
        ``vec0`` table for each of *precisions*, creating it on first use.
        Existing rows for the same rowid are cleared first (``vec0`` has
        no ``INSERT OR REPLACE``), mirroring :meth:`_flush_payloads`'s FTS5
        handling."""
        if not emb_batch:
            return
        dim = len(emb_batch[0][1])
        for precision in precisions:
            vec_table = f"vec_{table}_{precision}"
            self._ensure_vec_table(vec_table, precision, dim)
            rows = [
                (id_rowids[item_id], sqlite3.Binary(vec.astype(np.float32).tobytes()))
                for item_id, vec in emb_batch
                if item_id in id_rowids
            ]
            if not rows:
                continue
            rowids = [row[0] for row in rows]
            placeholders = ",".join("?" for _ in rowids)
            expr = _QUANTIZE_EXPR[precision]
            with self.conn:
                self.conn.execute(f"DELETE FROM {vec_table} WHERE rowid IN ({placeholders})", rowids)
                self.conn.executemany(f"INSERT INTO {vec_table} (rowid, embedding) VALUES (?, {expr})", rows)

    def close(self) -> None:
        self.conn.close()
