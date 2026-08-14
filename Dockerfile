# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------
# scrydb — containerized, install-free image
#
# scrydb ships a native SQLite loadable extension (hamming.c) that setup.py
# compiles at *install* time with the local C compiler + SQLite dev headers
# (see setup.py / README "Why source-only"). Building inside a fixed Linux
# image sidesteps the "works on my machine" platform variance the README
# warns about — every image build compiles hamming.so the same way, on the
# same glibc/SQLite combination, regardless of what the host OS is. Building
# the image with `docker buildx build --platform linux/amd64,linux/arm64`
# (rather than plain `docker build`) additionally makes it run natively —
# no QEMU emulation — on both Intel/AMD and Apple Silicon/ARM hosts.
#
# The image bundles the `scrydb` CLI (src/scrydb/cli.py) as its entrypoint,
# so no local Python/compiler install is needed to index or search:
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

FROM python:3.12-slim-bookworm AS base

# build-essential -> cc/gcc for setup.py's compile_hamming_extension() step
# libsqlite3-dev   -> sqlite3ext.h, required by hamming.c's #include
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libsqlite3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy only what's needed to resolve/build the package first, so dependency
# layers are cached even when application code below changes.
COPY pyproject.toml setup.py MANIFEST.in README.md LICENSE ./
COPY src ./src

# Build arg lets you opt into the extras (dense/eval/all) at image-build
# time, e.g. `docker build --build-arg EXTRAS=all -t scrydb .`. Defaults to
# "" (lexical/BM25 only, matching the "pip install scrydb" base install) to
# keep the image small: `dense`/`all` pull in sentence-transformers -> torch,
# and PyPI's default torch wheel bundles a full CUDA/NVIDIA runtime (multiple
# GB) that a CPU-only container never uses. When dense/all is requested,
# install the CPU-only torch build first so the CUDA packages are never
# pulled in the first place.
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

# ---------------------------------------------------------------------------
# Runtime stage: keep the compiler toolchain out of the final image. The
# compiled hamming.so was already produced above and installed into
# site-packages/scrydb/ext/ — it doesn't need build-essential to *run*, only
# the SQLite runtime library (already present via python:slim's libsqlite3).
# ---------------------------------------------------------------------------
FROM python:3.12-slim-bookworm

# Copy the fully-built environment (installed scrydb + its deps + the
# `scrydb` console-script) from the build stage instead of re-installing.
COPY --from=base /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=base /usr/local/bin /usr/local/bin

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
