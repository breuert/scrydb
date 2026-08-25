<p align="center">
    <img src="./docs/images/logo-light.png" width=250/>
    <h1 align="center">scrydb</h1>
</p>

``scrydb``'s purpose is making lexical, dense, and hybrid search possible with [SQLite](https://sqlite.org/). Hardware requirements are kept low and everything is self-contained in a single file, i.e., the raw documents, their embeddings, and the lexical index are stored in a single SQLite file.

- Lexical search is made possible by the [FTS5 extension for SQLite](https://sqlite.org/fts5.html).
- Semantic search over the entire index is made possible by [`sqlite-vec`](https://github.com/asg017/sqlite-vec), a small, dependency-free vector search extension for SQLite. Every embedding can be searched at three precisions: **binary** (1 bit/dim, Hamming distance), **int8** (1 byte/dim, cosine), and **float** (full precision, cosine) — trading index size and speed against ranking quality, and combinable as a two-stage rerank (e.g. binary-first >> float-precision rerank).
- Hybrid search relies on [Reciprocal Rank Fusion](https://dl.acm.org/doi/10.1145/1571941.1572114) to fuse lexical and semantic search results.

The library is compatible with [Sentence Transformers](https://www.sbert.net/index.html). However, it is also possible to store precomputed embeddings for both queries and documents.

> [!NOTE]  
> The evaluation protocol, benchmark results, and examples are available at [`scrydb-eval`](https://github.com/breuert/scrydb-eval/), the corresponding data is shared on [Hugging Face](https://huggingface.co/datasets/breuert/scrydb-eval).

## Usage examples

``scrydb`` can be used interactively as follows:
```python
from scrydb import Index, SentenceEmbedding

with Index.open("idx.db") as index:
    index.add_model(SentenceEmbedding())
    index.index_documents("corpus.jsonl", id_field="docid", text_field="text")
    results = index.search("some query", mode="hybrid", rerank=True)
```

Batch search for Information Retrieval benchmarks with precomputed embeddings can be run as follows:
```python
import scrydb

idx = scrydb.Index.open("./path/to/index.db")

idx.index_documents(
    source="./path/to/corpus.jsonl",
    id_field="docid",
    text_field="text",
    embedding_field="embedding",
    store_int8_embeddings=True,  # opt in to int8 storage alongside binary/float
    )

idx.index_queries(
    source="./path/to/queries.jsonl",
    id_field="qid",
    text_field="text",
    embedding_field="embedding",
    store_int8_embeddings=True,
    )

idx.batch_search(mode="lexical").write_trec("./path/to/lexical/run")
idx.batch_search(mode="semantic", precision="binary").write_trec("./path/to/binary/run")
idx.batch_search(mode="semantic", precision="int8").write_trec("./path/to/int8/run")
idx.batch_search(mode="semantic", precision="float").write_trec("./path/to/float/run")
idx.batch_search(mode="semantic", precision="binary", rerank="float").write_trec("./path/to/binary-rerank-float/run")
idx.batch_search(mode="hybrid").write_trec("./path/to/hybrid/run")
```

### Search modes, precision, and rerank

`search()`/`batch_search()` take three orthogonal knobs:

- `mode` — `"lexical"` (BM25 over FTS5), `"semantic"` (vector search), or `"hybrid"` (Reciprocal Rank Fusion of both).
- `precision` — which vector representation `mode="semantic"`/the semantic side of `mode="hybrid"` ranks with: `"binary"` (default), `"int8"`, or `"float"`.
- `rerank` — `False` (default), or a second-stage rerank over the top candidates from `mode`, at `"binary"`, `"int8"`, or `"float"` precision (`True` is a synonym for `"float"`).

```python
idx.search("some query", mode="lexical")                                         # BM25
idx.search("some query", mode="lexical", rerank="float")                         # BM25 >> Float/Cosine
idx.search("some query", mode="semantic", precision="binary")                    # Binary/Hamming
idx.search("some query", mode="semantic", precision="int8")                      # Int8/Cosine
idx.search("some query", mode="semantic", precision="binary", rerank="float")    # Binary >> Float/Cosine
idx.search("some query", mode="hybrid", rerank=True)                             # Hybrid/RRF
```

## Installing

```bash
pip install scrydb
```

Or with [uv](https://docs.astral.sh/uv/) (faster, and manages the virtualenv for you):

```bash
uv venv && uv pip install scrydb
# or, inside a uv-managed project:
uv add scrydb
```

No C compiler or SQLite development headers required: vector search is
powered by [`sqlite-vec`](https://github.com/asg017/sqlite-vec), a pure
pip dependency that ships prebuilt binaries, so `pip install scrydb` is a
plain, fast, wheel-only install.

`Index.open()` loads the `sqlite-vec` extension automatically. This
requires a Python build whose `sqlite3` module supports
`enable_load_extension()` — true for Homebrew, pyenv, and
[uv](https://docs.astral.sh/uv/)-managed builds on macOS, virtually all
Linux distro packages, and the official Windows builds, but **not** for
macOS Pythons that link against Apple's SQLite, which disables extension
loading. That includes macOS's system Python and the CPython that
`actions/setup-python` installs on GitHub Actions runners. If you hit a
`RuntimeError` mentioning `enable_load_extension`, switch to Python from
Homebrew (`brew install python`), uv (`uv python install <version>`), or
pyenv (`PYTHON_CONFIGURE_OPTS='--enable-loadable-sqlite-extensions'
pyenv install <version>`).

If you'd rather not load the extension at all, you can still install and
use scrydb for lexical (BM25) search only — just disable it explicitly:

```python
Index.open("idx.db", vec_ext_path=None)
```

### Try the CLI without installing (uvx)

[`uvx`](https://docs.astral.sh/uv/guides/tools/) runs the `scrydb` command-line
tool (`index`/`search`/`batch-search`/`auto` — see the [Docker](#docker)
section below for the full reference) in a throwaway environment, no venv or
persistent install needed:

```bash
uvx scrydb index --documents corpus.jsonl --queries queries.jsonl --db idx.db
uvx scrydb search "some query" --db idx.db --mode hybrid --rerank float
uvx scrydb batch-search --db idx.db --mode hybrid --output run.trec
```

Since scrydb has no compiled artifacts of its own, `uvx` just downloads the
wheel and its dependencies (including `sqlite-vec`'s prebuilt binary) into
its cache — no build step at all.

## Docker

No local Python install needed: the [`Dockerfile`](./Dockerfile) builds a
Linux image with scrydb already installed, driven by a bundled `scrydb`
CLI. Everything it reads and writes -- input JSONL, the SQLite index, TREC
run files -- lives under `/data`, so bind-mount a host directory there.

```bash
docker build -t scrydb .
```

Drop `documents.jsonl` (and, optionally, `queries.jsonl`) into `./data` and
run the image with no arguments: it indexes whatever's present into
`./data/index.db`, then either batch-searches the stored queries into
`./data/run.trec` or -- if there's nothing to index and no stored queries --
tells you what it's waiting for.

```bash
docker run --rm -v "$PWD/data":/data scrydb
```

If `./data/index.db` already exists (say, you built it locally, or a
previous run produced it), the same command re-uses it: skips indexing
whatever source files aren't present and searches straight away. To query
an existing index ad hoc instead of running a full batch, override the
default command:

```bash
docker run --rm -v "$PWD/data":/data scrydb search "some query" --mode hybrid --rerank float
```

Every option is also settable as an `SCRYDB_*` environment variable (handy
for `docker run -e`), and the JSONL id-field names for documents/queries
default to `docid`/`qid`-style overrides when they differ from `id`:

```bash
docker run --rm -v "$PWD/data":/data \
  -e SCRYDB_DOC_ID_FIELD=docid -e SCRYDB_QUERY_ID_FIELD=qid \
  -e SCRYDB_MODEL=mixedbread-ai/mxbai-embed-large-v1 \
  -e SCRYDB_MODE=hybrid -e SCRYDB_RERANK=float \
  scrydb
```

Run `docker run --rm scrydb --help` (or `... <subcommand> --help`) for the
full `index`/`search`/`batch-search`/`auto` reference, or see the
module docstring in [`src/scrydb/cli.py`](./src/scrydb/cli.py).

Notes:

- **Dense/hybrid search** (`sentence-transformers`) isn't in the image by
  default -- build with `--build-arg EXTRAS=all` (or `dense`) to add it.
  This installs the CPU-only `torch` build so the image doesn't pull in
  multi-gigabyte CUDA packages it can't use.
- **Multi-platform**: build once for both Intel/AMD and Apple
  Silicon/ARM hosts (each running natively, no QEMU emulation) with
  `docker buildx build --platform linux/amd64,linux/arm64 -t scrydb .`
- **Plain Python access**: `docker run --rm -it -v "$PWD/data":/data --entrypoint python3 scrydb`
  drops into an interpreter with `scrydb` importable.

## How extension discovery works at runtime

`Index.open()`/`Index()` default to `vec_ext_path="auto"`, which loads
`sqlite_vec.loadable_path()` — the copy of the extension bundled inside
the installed `sqlite-vec` pip package, prebuilt for the current
platform. Pass an explicit path to load a different build (e.g. a newer
`vec0` release), or `None` to skip loading it entirely.

## Development / editable installs

```bash
git clone <repo>
cd scrydb
python -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"
pytest
```

Or with uv:

```bash
git clone <repo>
cd scrydb
uv venv
uv pip install -e ".[all]"
uv run pytest
```

`uv run` picks up `.venv` automatically, so there's no `source .venv/bin/activate`
step — any command after it (`uv run pytest`, `uv run python -m scrydb.cli --help`,
`uv run python your_script.py`) runs inside the project's venv.

## Optional extras

- `pip install "scrydb[dense]"` / `uv pip install "scrydb[dense]"` — dense/hybrid search via `sentence-transformers`
- `pip install "scrydb[eval]"` / `uv pip install "scrydb[eval]"` — `Run.to_dataframe()` via `pandas`
- `pip install "scrydb[all]"` / `uv pip install "scrydb[all]"` — both
