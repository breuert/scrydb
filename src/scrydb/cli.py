"""
scrydb command-line interface.

Thin argparse wrapper around the ``Index``/``Run`` API (see
:mod:`scrydb.core`) so scrydb can be indexed, searched, and batch-searched
without writing Python. This is what the Docker image (see ``../../Dockerfile``
at the repo root) runs by default -- it lets a mounted SQLite index be built
and queried from a container alone, no local install required.

Every option can be set via flag or the matching ``SCRYDB_*`` environment
variable (flags win); the Docker image relies on the latter since ``docker
run -e`` is the natural way to configure a container.

Subcommands
-----------
    scrydb index          -- index documents and/or queries into a database
    scrydb search          -- run one ad-hoc query, print JSON results
    scrydb batch-search    -- run every stored query, write a TREC run file
    scrydb auto             -- environment-driven pipeline: index whatever
                                input files are present, then batch-search
                                or run a single ad-hoc query -- the Docker
                                image's default command
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .core import Index, SearchResult, SentenceEmbedding


def _env(name: str, default: "str | None" = None) -> "str | None":
    value = os.environ.get(name)
    return value if value else default


def _rerank_type(value: str):
    normalized = value.lower()
    if normalized in ("none", "false", ""):
        return False
    if normalized in ("binary", "int8", "float", "hamming", "cosine"):
        return normalized
    if normalized == "true":
        return True
    raise argparse.ArgumentTypeError(
        f"invalid rerank {value!r}; expected none, binary, int8, float "
        "(or the legacy aliases hamming/cosine)"
    )


def _result_to_dict(result: SearchResult) -> dict:
    return dict(result)


def _print_results(results) -> None:
    print(json.dumps([_result_to_dict(r) for r in results], ensure_ascii=False, indent=2, default=str))


def _open_index(db: str, model_name: "str | None") -> Index:
    index = Index.open(db)
    if model_name:
        index.add_model(SentenceEmbedding(model_name=model_name))
    return index


# ===========================================================================
# Subcommands
# ===========================================================================

def cmd_index(args: argparse.Namespace) -> int:
    if not args.documents and not args.queries:
        print("scrydb index: nothing to do -- pass --documents and/or --queries", file=sys.stderr)
        return 1
    with _open_index(args.db, args.model) as index:
        if args.documents:
            print(f"scrydb: indexing documents from {args.documents} -> {args.db}", file=sys.stderr)
            index.index_documents(
                args.documents,
                id_field=args.doc_id_field or args.id_field,
                text_field=args.text_field,
                embedding_field=args.embedding_field,
                store_int8_embeddings=args.store_int8,
            )
        if args.queries:
            print(f"scrydb: indexing queries from {args.queries} -> {args.db}", file=sys.stderr)
            index.index_queries(
                args.queries,
                id_field=args.query_id_field or args.id_field,
                text_field=args.text_field,
                embedding_field=args.embedding_field,
                store_int8_embeddings=args.store_int8,
            )
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    with _open_index(args.db, args.model) as index:
        results = index.search(
            args.query, mode=args.mode, top_k=args.top_k, rerank=args.rerank, precision=args.precision
        )
        _print_results(results)
    return 0


def cmd_batch_search(args: argparse.Namespace) -> int:
    with _open_index(args.db, args.model) as index:
        run = _run_batch_search(index, args)
    return 0 if run is not None else 1


def _run_batch_search(index: Index, args: argparse.Namespace):
    if len(index.queries) == 0:
        print(f"scrydb: index {args.db!r} has no stored queries -- nothing to batch-search", file=sys.stderr)
        return None
    print(
        f"scrydb: batch-searching {len(index.queries)} stored queries "
        f"(mode={args.mode!r}, precision={args.precision!r}, rerank={args.rerank!r})",
        file=sys.stderr,
    )
    run = index.batch_search(
        mode=args.mode, top_k=args.top_k, rerank=args.rerank, precision=args.precision
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    run.write_trec(out, tag=args.tag)
    n_hits = sum(len(hits) for hits in run.values())
    print(f"scrydb: wrote {n_hits} results over {len(run)} queries to {out}", file=sys.stderr)
    return run


def cmd_auto(args: argparse.Namespace) -> int:
    """Environment-driven pipeline for the Docker image's default command:
    index whatever input files are present, then either answer one ad-hoc
    query or batch-search whatever queries ended up stored -- so the same
    invocation handles "index + search fresh data" and "query an existing
    index" without the caller having to pick a subcommand."""
    documents = args.documents if args.documents and Path(args.documents).is_file() else None
    queries = args.queries if args.queries and Path(args.queries).is_file() else None

    with _open_index(args.db, args.model) as index:
        if documents:
            print(f"scrydb: indexing documents from {documents} -> {args.db}", file=sys.stderr)
            index.index_documents(
                documents,
                id_field=args.doc_id_field or args.id_field,
                text_field=args.text_field,
                embedding_field=args.embedding_field,
                store_int8_embeddings=args.store_int8,
            )
        if queries:
            print(f"scrydb: indexing queries from {queries} -> {args.db}", file=sys.stderr)
            index.index_queries(
                queries,
                id_field=args.query_id_field or args.id_field,
                text_field=args.text_field,
                embedding_field=args.embedding_field,
                store_int8_embeddings=args.store_int8,
            )

        if args.query:
            results = index.search(
                args.query, mode=args.mode, top_k=args.top_k, rerank=args.rerank, precision=args.precision
            )
            _print_results(results)
            return 0

        if len(index.queries) > 0:
            return 0 if _run_batch_search(index, args) is not None else 1

        print(
            f"scrydb: nothing to do -- {index!r}\n"
            "  Mount documents/queries JSONL at the paths given by --documents/--queries\n"
            "  (env: SCRYDB_DOCUMENTS/SCRYDB_QUERIES) to index them, set --query/SCRYDB_QUERY\n"
            "  for a one-off search, or store queries first (`scrydb index --queries ...`)\n"
            "  to enable a batch run. See `docker run --rm scrydb --help`.",
            file=sys.stderr,
        )
        return 0


# ===========================================================================
# argparse wiring
# ===========================================================================

def _add_db_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--db", default=_env("SCRYDB_DB", "index.db"),
        help="Path to the SQLite index file (env: SCRYDB_DB; default: %(default)s)",
    )


def _add_model_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model", default=_env("SCRYDB_MODEL"),
        help="sentence-transformers model name for on-the-fly embedding "
             "(env: SCRYDB_MODEL). Omit to rely on precomputed embedding fields, "
             "or for lexical-only search.",
    )


def _add_field_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--id-field", default=_env("SCRYDB_ID_FIELD", "id"),
        help="Id field used for both documents and queries, unless overridden by "
             "--doc-id-field/--query-id-field (env: SCRYDB_ID_FIELD)",
    )
    parser.add_argument(
        "--doc-id-field", default=_env("SCRYDB_DOC_ID_FIELD"),
        help="Id field for documents only, e.g. 'docid'; overrides --id-field (env: SCRYDB_DOC_ID_FIELD)",
    )
    parser.add_argument(
        "--query-id-field", default=_env("SCRYDB_QUERY_ID_FIELD"),
        help="Id field for queries only, e.g. 'qid'; overrides --id-field (env: SCRYDB_QUERY_ID_FIELD)",
    )
    parser.add_argument("--text-field", default=_env("SCRYDB_TEXT_FIELD", "text"), help="env: SCRYDB_TEXT_FIELD")
    parser.add_argument(
        "--embedding-field", default=_env("SCRYDB_EMBEDDING_FIELD", "emb"), help="env: SCRYDB_EMBEDDING_FIELD"
    )
    parser.add_argument(
        "--store-int8", action="store_true", default=_env("SCRYDB_STORE_INT8", "") not in ("", "0", "false", "False"),
        help="Also store int8-quantized embeddings, in addition to the always-stored binary "
             "embeddings and (by default) full-precision embeddings (env: SCRYDB_STORE_INT8)",
    )


def _add_search_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--mode", choices=["lexical", "semantic", "hybrid"], default=_env("SCRYDB_MODE", "lexical"),
        help="env: SCRYDB_MODE (default: %(default)s)",
    )
    parser.add_argument(
        "--precision", choices=["binary", "int8", "float"], default=_env("SCRYDB_PRECISION", "binary"),
        help="Vector precision for mode=semantic/hybrid's semantic side "
             "(env: SCRYDB_PRECISION, default: %(default)s)",
    )
    parser.add_argument(
        "--rerank", type=_rerank_type, default=_rerank_type(_env("SCRYDB_RERANK", "none")),
        help="none, binary, int8, or float (legacy aliases hamming/cosine also accepted) "
             "(env: SCRYDB_RERANK, default: none)",
    )
    parser.add_argument(
        "--top-k", type=int, default=int(_env("SCRYDB_TOP_K", "10")), help="env: SCRYDB_TOP_K (default: %(default)s)"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scrydb", description="Lexical, semantic, and hybrid search over a SQLite index."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_index = sub.add_parser("index", help="Index documents and/or queries into a database")
    _add_db_arg(p_index)
    _add_model_arg(p_index)
    _add_field_args(p_index)
    p_index.add_argument("--documents", default=_env("SCRYDB_DOCUMENTS"), help="JSONL path (env: SCRYDB_DOCUMENTS)")
    p_index.add_argument("--queries", default=_env("SCRYDB_QUERIES"), help="JSONL path (env: SCRYDB_QUERIES)")
    p_index.set_defaults(func=cmd_index)

    p_search = sub.add_parser("search", help="Run one ad-hoc query, print JSON results")
    _add_db_arg(p_search)
    _add_model_arg(p_search)
    _add_search_args(p_search)
    p_search.add_argument("query")
    p_search.set_defaults(func=cmd_search)

    p_batch = sub.add_parser("batch-search", help="Run every stored query, write a TREC run file")
    _add_db_arg(p_batch)
    _add_model_arg(p_batch)
    _add_search_args(p_batch)
    p_batch.add_argument("--output", default=_env("SCRYDB_OUTPUT", "run.trec"), help="env: SCRYDB_OUTPUT")
    p_batch.add_argument("--tag", default=_env("SCRYDB_TAG", "scrydb"), help="env: SCRYDB_TAG")
    p_batch.set_defaults(func=cmd_batch_search)

    p_auto = sub.add_parser(
        "auto",
        help="Environment-driven pipeline: index what's present, then batch-search or "
             "ad-hoc search (Docker image default)",
    )
    _add_db_arg(p_auto)
    _add_model_arg(p_auto)
    _add_field_args(p_auto)
    _add_search_args(p_auto)
    p_auto.add_argument(
        "--documents", default=_env("SCRYDB_DOCUMENTS", "documents.jsonl"), help="env: SCRYDB_DOCUMENTS"
    )
    p_auto.add_argument("--queries", default=_env("SCRYDB_QUERIES", "queries.jsonl"), help="env: SCRYDB_QUERIES")
    p_auto.add_argument("--query", default=_env("SCRYDB_QUERY"), help="One-off ad-hoc query text (env: SCRYDB_QUERY)")
    p_auto.add_argument("--output", default=_env("SCRYDB_OUTPUT", "run.trec"), help="env: SCRYDB_OUTPUT")
    p_auto.add_argument("--tag", default=_env("SCRYDB_TAG", "scrydb"), help="env: SCRYDB_TAG")
    p_auto.set_defaults(func=cmd_auto)

    return parser


def main(argv: "list[str] | None" = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
