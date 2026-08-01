"""scrydb — lexical, semantic, and hybrid search built on SQLite."""

from .core import Index, Run, SearchResult, SentenceEmbedding

__all__ = ["Index", "Run", "SearchResult", "SentenceEmbedding"]

try:
    from importlib.metadata import version as _version

    __version__ = _version("scrydb")
except Exception:  # pragma: no cover - package not installed
    __version__ = "0.0.0"
