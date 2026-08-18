# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------
# scrydb — containerized, install-free image
#
# Vector search is powered by the `sqlite-vec` PyPI package, which ships
# prebuilt wheels for Linux/macOS/Windows — no C compiler or SQLite dev
# headers needed at install time, so this is a plain single-stage `pip
# install` image.
#
# The image bundles the `scrydb` CLI (src/scrydb/cli.py) as its entrypoint,
# so no local Python install is needed to index or search:
#
#   # Index a corpus + queries and batch-search them in one shot, writing
#   # index.db and run.trec into ./data on the host:
#   docker run --rm -v "$PWD/data":/data scrydb
#
#   # Query an index.db that already exists in ./data:
#   docker run --rm -v "$PWD/data":/data scrydb search "some query" --mode hybrid
#
#   # Drop into a Python shell with scrydb importable instead:
#   docker run --rm -it -v "$PWD/data":/data --entrypoint python3 scrydb
#
# See the "Docker" section in README.md for the full walkthrough of the
# `auto`/`index`/`search`/`batch-search` subcommands and their SCRYDB_*
# environment variables.
#
# Build:  docker build -t scrydb .
# ---------------------------------------------------------------------------

FROM python:3.12-slim-bookworm

WORKDIR /build

# Copy only what's needed to resolve/build the package first, so dependency
# layers are cached even when application code below changes.
COPY pyproject.toml MANIFEST.in README.md LICENSE ./
COPY src ./src

# Build arg lets you opt into the extras (dense/eval/all) at image-build
# time, e.g. `docker build --build-arg EXTRAS=all -t scrydb .`. Defaults to
# "" (lexical/BM25 + vector search, matching the "pip install scrydb" base
# install) to keep the image small: `dense`/`all` pull in
# sentence-transformers -> torch, and PyPI's default torch wheel bundles a
# full CUDA/NVIDIA runtime (multiple GB) that a CPU-only container never
# uses. When dense/all is requested, install the CPU-only torch build first
# so the CUDA packages are never pulled in the first place.
ARG EXTRAS=""
RUN python -m pip install --no-cache-dir --upgrade pip \
    && case ",$EXTRAS," in \
         *,dense,*|*,all,*) \
           pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu ;; \
       esac \
    && if [ -n "$EXTRAS" ]; then \
         pip install --no-cache-dir ".[$EXTRAS]"; \
       else \
         pip install --no-cache-dir .; \
       fi

WORKDIR /

# Everything scrydb reads/writes at runtime — input JSONL, the SQLite
# index, TREC run files — lives under /data, meant to be bind-mounted from
# the host (`-v "$PWD/data":/data`) so it survives past `docker run --rm`.
WORKDIR /data

ENTRYPOINT ["scrydb"]

# With no arguments, `docker run scrydb` runs the environment-driven
# pipeline: index ./documents.jsonl / ./queries.jsonl if present (relative
# to WORKDIR, i.e. /data), then either batch-search whatever queries ended
# up stored or answer a single SCRYDB_QUERY — see `scrydb auto --help` and
# cli.py's module docstring. Override to run any other subcommand, e.g.
# `docker run --rm -v "$PWD/data":/data scrydb search "..." --mode hybrid`.
CMD ["auto"]
