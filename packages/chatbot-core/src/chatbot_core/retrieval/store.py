"""Vector store abstraction.

Two backends, one interface:

* **LanceDB** -- embedded, file-backed, nothing to run. Right for development
  and for projects whose corpus fits on one disk.
* **pgvector** -- Postgres. Right for production and for any project running
  more than one API replica, since they then share a single index.

Projects pick per environment. The interface is deliberately small; anything a
backend can't do uniformly (index tuning, replication) stays in that backend.
"""

from __future__ import annotations

import abc
import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from ..config import RetrievalConfig, VectorBackend
from .embeddings import Embedder


@dataclass(slots=True)
class Document:
    """A chunk ready to be indexed."""

    #: Stable across re-ingests. Derived from source URI + chunk index so that
    #: re-indexing an unchanged document is an upsert, not a duplicate.
    id: str
    text: str
    title: str
    uri: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SearchHit:
    id: str
    text: str
    title: str
    uri: str | None
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


def make_id(uri: str, chunk_index: int) -> str:
    """Deterministic chunk id.

    Hashed rather than raw so that long filesystem paths and URLs with query
    strings stay within the key-length limits of both backends.
    """
    digest = hashlib.sha256(uri.encode("utf-8")).hexdigest()[:20]
    return f"{digest}:{chunk_index}"


class VectorStore(abc.ABC):
    def __init__(self, config: RetrievalConfig, collection: str, embedder: Embedder) -> None:
        self.config = config
        self.collection = collection
        self.embedder = embedder

    @abc.abstractmethod
    async def ensure_ready(self) -> None:
        """Create the collection if absent; verify dimensions if present.

        A dimension mismatch means the index was built with a different
        embedding model. Every score against it would be meaningless, so this
        must fail loudly at startup rather than degrade silently at query time.
        """

    @abc.abstractmethod
    async def upsert(self, documents: Sequence[Document]) -> int:
        """Insert or replace by id. Returns the number written."""

    @abc.abstractmethod
    async def delete_by_uri(self, uri: str) -> int:
        """Remove every chunk from one source document.

        Re-ingesting calls this first: a document that shrank between runs
        would otherwise leave orphaned chunks that still match queries.
        """

    @abc.abstractmethod
    async def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        where: dict[str, Any] | None = None,
        audiences: frozenset[str] | None = None,
    ) -> list[SearchHit]:
        """Nearest neighbours above ``score_threshold``, best first.

        Args:
            audiences: Restrict results to documents tagged with one of these.
                ``None`` means unrestricted -- correct only for trusted internal
                callers such as an ingest verifier. Every request path passes
                the caller's audiences explicitly.

        Audience filtering happens **here**, in the query, not downstream. A
        chunk the caller may not see is never fetched, so it cannot reach the
        model's context and cannot be extracted from it. Filtering after
        retrieval, or instructing the model to ignore what it can see, is not
        access control.
        """

    @abc.abstractmethod
    async def count(self) -> int: ...

    async def aclose(self) -> None:  # noqa: B027 - optional hook, not abstract
        """Release connections. Safe to call more than once.

        Deliberately concrete and empty rather than abstract: an embedded store
        holds no connection pool, so only backends that own one override this.
        """


def build_store(
    config: RetrievalConfig, collection: str, embedder: Embedder
) -> VectorStore:
    match config.backend:
        case VectorBackend.LANCEDB:
            from .lancedb_store import LanceDBStore

            return LanceDBStore(config, collection, embedder)
        case VectorBackend.PGVECTOR:
            from .pgvector_store import PgVectorStore

            return PgVectorStore(config, collection, embedder)
    raise ValueError(f"unknown vector backend: {config.backend}")


__all__ = ["Document", "SearchHit", "VectorStore", "build_store", "make_id"]
