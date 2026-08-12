<p align="center">
    <img src="./docs/images/logo-light.png" width=250/>
    <h1 align="center">scrydb</h1>
</p>

``scrydb`` is built for one purpose: making lexical, dense, and hybrid search possible with [SQLite](https://sqlite.org/). It follows a minimalist's approach where hardware requirements are kept low and everything is self-contained in a single file, i.e., the raw documents, their embeddings, and the lexical index are stored in a single SQLite file.

Technically, lexical search is made possible by the [FTS5 extension for SQLite](https://sqlite.org/fts5.html). Semantic search for the entire index is made possible with binary embeddings and the [Hamming distance](https://en.wikipedia.org/wiki/Hamming_distance) that is implemented with the help of an efficient [custom SQLite extension](./src/scrydb/ext/hamming.c). Optionally, the retrieved results can be reranked with the full embeddings and cosine similarity or the full embeddings can be discarded entirely to keep the disk usage low. Hybrid search relies on [Reciprocal Rank Fusion](https://dl.acm.org/doi/10.1145/1571941.1572114) to fuse lexical and semantic search results. 

The library is compatible with [Sentence Transformers](https://www.sbert.net/index.html). However, it is also possible to store precomputed embeddings for both queries and documents.

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
    embedding_field="embedding"
    )

idx.index_queries(
    source="./path/to/queries.jsonl",
    id_field="qid",
    text_field="text",
    embedding_field="embedding"
    )

idx.batch_search(mode="lexical").write_trec("./path/to/lexical/run")
idx.batch_search(mode="semantic").write_trec("./path/to/semantic/run")
idx.batch_search(mode="hybrid").write_trec("./path/to/hybrid/run")
```

## Installing

> [!NOTE]  
> **Platform support:** Linux and macOS only (see "Why source-only" below for Windows).

```bash
pip install scrydb
```

This package includes a native SQLite loadable extension
(`hamming_distance()`, used for fast binary/hex vector search) written in
C ([`src/scrydb/ext/hamming.c`](./src/scrydb/ext/hamming.c)). **A C compiler and the SQLite development
headers must be available on your machine at install time** — pip builds
the extension for your exact platform as part of the install:

- **macOS**: install the Xcode Command Line Tools once, if you haven't
  already:

  ```bash
  xcode-select --install
  ```

  (macOS ships `sqlite3ext.h` alongside the system SQLite headers, so no
  separate SQLite package is required.)

- **Debian/Ubuntu**:

  ```bash
  sudo apt-get install build-essential libsqlite3-dev
  ```

- **Fedora/RHEL**:

  ```bash
  sudo dnf install gcc sqlite-devel
  ```

- **Arch**:

  ```bash
  sudo pacman -S base-devel sqlite
  ```

If the compiler or headers are missing, `pip install scrydb` fails with a
message explaining exactly what to install (see `setup.py`).

If you'd rather not compile anything, you can still install and use scrydb
for lexical (BM25) and cosine-rerank search — just disable the extension
explicitly:

```python
Index.open("idx.db", hamming_ext_path=None)
```

### Why source-only (no prebuilt wheels)

`hamming.so`/`hamming.dylib` is a native shared library, and its ABI
depends on the platform (and, in principle, the CPU architecture). Rather
than maintain a wheel build matrix, this package ships as an sdist and
compiles the extension for your exact machine during `pip install`
(`setup.py`'s custom `build_py` step runs `cc`/`gcc`/`clang` against
`src/scrydb/ext/hamming.c` and drops the result into `scrydb/ext/` as package
data). This is the same approach used by many source-only Python
packages that wrap native code without prebuilt wheels.

If you want prebuilt wheels for CI/distribution, the natural next step is
wiring this same build step into
[`cibuildwheel`](https://cibuildwheel.pypa.io/), which runs it inside
manylinux/macOS containers for each target platform and uploads the
resulting wheels — that's outside the scope of this initial release.

## How extension discovery works at runtime

`Index.open()`/`Index()` default to `hamming_ext_path="auto"`, which
looks for `scrydb/ext/hamming.so` (Linux) or `scrydb/ext/hamming.dylib`
(macOS) inside the installed package (`scrydb.core._bundled_hamming_path()`).
Pass an explicit path to load a different build, or `None` to skip
loading it entirely.

## Development / editable installs

```bash
git clone <repo>
cd scrydb
python -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"
pytest
```

## Optional extras

- `pip install "scrydb[dense]"` — dense/hybrid search via `sentence-transformers`
- `pip install "scrydb[eval]"` — `Run.to_dataframe()` via `pandas`
- `pip install "scrydb[all]"` — both
