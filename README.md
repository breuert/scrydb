# scrydb

Lexical, semantic, and hybrid search built on SQLite: lexical (BM25),
dense (embedding), and hybrid search, with a pluggable retrieval-model
interface.

```python
from scrydb import Index, SentenceEmbedding

with Index.open("idx.db") as index:
    index.add_model(SentenceEmbedding())
    index.index_documents("corpus.jsonl", id_field="docid", text_field="text")
    results = index.search("some query", mode="hybrid", rerank=True)
```

## Platform support

Linux and macOS only (see "Why source-only" below for Windows).

## Installing

```bash
pip install scrydb
```

This package includes a native SQLite loadable extension
(`hamming_distance()`, used for fast binary/hex vector search) written in
C (`src/scrydb/ext/hamming.c`). **A C compiler and the SQLite development
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
