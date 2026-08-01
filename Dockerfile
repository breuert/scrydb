# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------
# scrydb — containerized build
#
# scrydb ships a native SQLite loadable extension (hamming.c) that setup.py
# compiles at *install* time with the local C compiler + SQLite dev headers
# (see setup.py / README "Why source-only"). Building inside a fixed Linux
# image sidesteps the "works on my machine" platform variance the README
# warns about — every image build compiles hamming.so the same way, on the
# same glibc/SQLite combination, regardless of what the host OS is.
#
# Build:   docker build -t scrydb .
# Run:     docker run --rm -it -v "$PWD":/data scrydb
#          docker run --rm -v "$PWD":/data scrydb python3 /data/my_script.py
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

# Build arg lets you opt into the extras (dense/eval/all) at image-build time,
# e.g. `docker build --build-arg EXTRAS=all -t scrydb .`
ARG EXTRAS=""
RUN python -m pip install --no-cache-dir --upgrade pip \
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

# Copy the fully-built environment (installed scrydb + its deps) from the
# build stage instead of re-installing.
COPY --from=base /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=base /usr/local/bin /usr/local/bin

WORKDIR /data

# No CLI entrypoint ships with scrydb (it's a library) — default to an
# interactive interpreter with scrydb importable; override the command to
# run your own script instead, e.g.:
#   docker run --rm -v "$PWD":/data scrydb python3 -c "import scrydb; print(scrydb.__version__)"
CMD ["python3"]
