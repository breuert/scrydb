# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`scrydb` is a Python library + CLI for lexical (FTS5/BM25), semantic (sqlite-vec), and hybrid (RRF) search where the raw documents, their embeddings, and the lexical index all live in **one SQLite file**. It targets Information Retrieval benchmarking (TREC run files) as much as interactive use.

## Commands

```bash
uv venv && uv pip install -e ".[all]"   # or: python -m venv .venv && pip install -e ".[all]"
uv run pytest                           # full suite
uv run pytest tests/test_search.py::test_hybrid_search_combines_lexical_and_semantic
uv run pytest -k "semantic and rerank"
uv run python -m scrydb.cli --help      # CLI without reinstalling

python -m build && twine check --strict dist/*   # what CI's build job runs
docker build -t scrydb .                          # add --build-arg EXTRAS=all for dense search
```

CI (`.github/workflows/ci.yml`) installs the **built package**, not the source tree (`pip install ".[eval,test]"`), across Python 3.9–3.13 on Linux/macOS/Windows. Two consequences: code must stay 3.9-compatible (`from __future__ import annotations` plus quoted `"str | None"` annotations everywhere — follow that convention), and anything the package needs at runtime must be declared in `pyproject.toml`/`MANIFEST.in`, not merely present in the repo.

Releases are tag-driven (`v*` → PyPI via trusted publishing); the workflow hard-fails if the tag doesn't match `project.version` in `pyproject.toml`, so bump the version in the same commit you tag.

## Architecture

Three files carry everything: `src/scrydb/core.py` (~1300 lines, all the logic), `src/scrydb/cli.py` (thin argparse wrapper), `src/scrydb/__init__.py` (re-exports `Index`, `Run`, `SearchResult`, `SentenceEmbedding`).

### The precision abstraction

The central design idea: an embedding is stored and searched at three precisions — `binary` (1 bit/dim, Hamming), `int8` (1 byte/dim, cosine), `float` (full, cosine) — each in its own `vec0` virtual table. Adding or changing a precision means touching the parallel module-level dicts at the top of `core.py`, all keyed by precision string:

- `_QUANTIZE_EXPR` — SQL to turn a raw float32 blob into stored form (int8 L2-normalizes first, so unnormalized dims don't saturate)
- `_CAST_EXPR` — SQL to tag an *already quantized* blob as a MATCH operand
- `_COLUMN_DEF` — the `vec0` column definition per dimension
- `_SCORE_FIELD` — which result field the score is reported under (`hamming_distance` / `int8_similarity` / `cosine_similarity`)
- `_NUMPY_DTYPE` — decoding stored blobs back to arrays

Keep them in sync; each is consumed by a different layer. All quantization happens **inside SQLite** via sqlite-vec, never in numpy.

`precision=`/`rerank=` both normalize through `_normalize_precision`, which accepts the v0.1.x aliases `"hamming"`→`binary` and `"cosine"`→`float`; `rerank=True` is a back-compat synonym for `"float"`. Tests pin these aliases — don't drop them.

### Schema

- `documents` / `queries` — `(id TEXT PRIMARY KEY, payload TEXT)`, the full row as JSON
- `documents_fts` — FTS5 virtual table, `tokenize='porter'`
- `vec_{documents,queries}_{binary,int8,float}` — created **lazily** by `_ensure_vec_table` on first vector of that precision, because the dimension isn't known until then. A table's dim is fixed by whichever call created it.

The `vec0` tables key on the payload table's **rowid**, joined back to `id`. This is why `_flush_payloads` uses `ON CONFLICT(id) DO UPDATE` rather than `INSERT OR REPLACE`: an upsert preserves the rowid, so re-indexing a document keeps its vectors in sync instead of orphaning them under a new rowid. Neither FTS5 nor `vec0` support `INSERT OR REPLACE`, so both flush paths DELETE-then-INSERT.

### Query pipeline

`search()` (one query string) and `batch_search()` (stored query ids → `Run`) are deliberate near-duplicates sharing the same private stages: `_lexical_rank` → `_semantic_rank` → `_rerank_vec` → `_reciprocal_rank_fusion` → `_materialize`. Changes to ranking behavior usually need editing **both** entry points.

Invariants worth preserving:

- Every ranker's score is **higher-is-better**. `_lexical_rank` negates FTS5's `bm25()` for this reason.
- Stages pass `(doc_id, extra_dict)` pairs; `_rerank_vec` merges the rerank stage's fields *on top of* the base stage's, so a BM25→rerank result carries both `score` and the rerank field. `Run._SCORE_FIELDS` encodes the resulting priority order for the TREC SCORE column — rerank fields must outrank the stale lexical `score`, or eval tools (which read only the SCORE column) silently ignore the rerank stage.
- `_query_material` prefers a *stored* query embedding over re-encoding the text, so a query searched at `precision="binary"` needs neither a float vector nor a model. This is what makes benchmark reproduction exact.
- Free-form query text goes through `_as_or_query`, an allowlist (`\w+` terms, quoted and OR-joined) rather than a blocklist of FTS5 metacharacters. `raw=True` opts out for callers writing FTS5 syntax themselves.

### Extension loading

`Index.open(..., vec_ext_path="auto")` loads `sqlite_vec.loadable_path()` from the pip package. `None` skips it — lexical search must keep working in that mode (`tests/test_vec_extension.py` enforces this). macOS system Python lacks `enable_load_extension`; `_load_vec_extension` catches that and raises a `RuntimeError` explaining the Homebrew/pyenv fix.

### Model plug-in point

`SentenceEmbedding` is just a `encode_documents`/`encode_queries` pair with a lazily-imported `sentence-transformers` model. `Index` never imports it directly — any object with that interface works via `add_model()`. `sentence-transformers` is an optional extra (`[dense]`); keep it out of the import path of anything the base install must run.

### CLI

`cli.py` mirrors the library API as `index` / `search` / `batch-search` / `auto`. Every option is settable via flag or `SCRYDB_*` env var (flag wins), because the Docker image is configured with `docker run -e`. `auto` is the image's default `CMD`: it indexes whatever JSONL files exist under `/data`, then batch-searches stored queries or answers `SCRYDB_QUERY`. New options should follow the flag-plus-env-var pattern and be reflected in the Docker section of `README.md` and `cli.py`'s module docstring.

## Conventions

- Public API surface is documented in `core.py`'s module docstring and mirrored in `README.md` — when the `mode`/`precision`/`rerank` vocabulary changes, update both.
- Comments in this codebase explain *why* (the rowid/upsert coupling, the score-field priority, int8 normalization). Match that: explain the constraint, not the syntax.
- `SearchResult` is a `Mapping` with attribute access; `Run` is a `dict` subclass. Both are intentionally thin so results drop straight into pandas/JSON.
