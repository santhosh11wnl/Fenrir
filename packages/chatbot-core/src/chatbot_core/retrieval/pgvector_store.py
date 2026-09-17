"""pgvector store -- Postgres-backed, shared across replicas.

The production backend. Use it once a project runs more than one API replica,
since they then query one index instead of each holding a private copy.

One table per collection rather than a shared table with a ``collection``
column: a pgvector column has a fixed dimension, and projects do not all use
the same embedding model. Separate tables also mean per-project index tuning
and a per-project ``DROP TABLE`` for a clean re-ingest.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Any

import structlog

from ..auth import DEFAULT_AUDIENCE
from ..config import RetrievalConfig
from .embeddings import Embedder
from .store import Document, SearchHit, VectorStore

log = structlog.get_logger(__name__)

_SAFE_NAME = re.compile(r"^[a-z][a-z0-9_]{0,54}$")


class PgVectorStore(VectorStore):
    def __init__(
        self,
        config: RetrievalConfig,
        collection: str,
        embedder: Embedder,
        *,
        dsn: str | None = None,
    ) -> None:
        super().__init__(config, collection, embedder)
        import os

        raw = dsn or os.environ.get("DATABASE_URL", "")
        if not raw:
            raise ValueError("pgvector backend requires DATABASE_URL")
        # asyncpg takes a bare postgres:// DSN; strip any SQLAlchemy driver tag.
        self._dsn = raw.replace("postgresql+asyncpg://", "postgresql://")
        self._table = _table_name(collection)
        self._pool: Any = None
        self._ready = False

    async def _get_pool(self) -> Any:
        if self._pool is None:
            try:
                import asyncpg
            except ImportError as exc:  # pragma: no cover
                raise ImportError(
                    "The pgvector backend needs: uv add 'chatbot-core[pgvector]'"
                ) from exc
            self._pool = await asyncpg.create_pool(
                self._dsn, min_size=1, max_size=8, command_timeout=30
            )
        return self._pool

    async def ensure_ready(self) -> None:
        if self._ready:
            return
        pool = await self._get_pool()
        dims = self.embedder.dimensions

        async with pool.acquire() as conn:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            # Table and index names are validated identifiers, not user input --
            # see _table_name. They cannot be bound as parameters in DDL.
            await conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._table} (
                    id          TEXT PRIMARY KEY,
                    embedding   vector({dims}) NOT NULL,
                    text        TEXT NOT NULL,
                    title       TEXT NOT NULL,
                    uri         TEXT,
                    metadata    JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )
            # HNSW: better recall/latency than IVFFlat and, unlike IVFFlat, it
            # needs no training pass over existing rows -- so it can be created
            # on an empty table at first boot.
            await conn.execute(
                f"CREATE INDEX IF NOT EXISTS {self._table}_embedding_idx "
                f"ON {self._table} USING hnsw (embedding vector_cosine_ops)"
            )
            # delete_by_uri runs on every re-ingest and would otherwise scan.
            await conn.execute(
                f"CREATE INDEX IF NOT EXISTS {self._table}_uri_idx ON {self._table} (uri)"
            )

            actual = await conn.fetchval(
                """
                SELECT a.atttypmod FROM pg_attribute a
                JOIN pg_class c ON c.oid = a.attrelid
                WHERE c.relname = $1 AND a.attname = 'embedding'
                """,
                self._table,
            )
            if actual is not None and actual > 0 and actual != dims:
                raise ValueError(
                    f"Table {self._table!r} stores {actual}-dimensional vectors "
                    f"but the configured embedding model produces {dims}. "
                    f"Re-ingest the corpus after dropping the table."
                )

        self._ready = True

    async def upsert(self, documents: Sequence[Document]) -> int:
        if not documents:
            return 0
        await self.ensure_ready()
        vectors = await self.embedder.embed_documents([d.text for d in documents])
        rows = [
            (
                doc.id,
                _to_pg_vector(vector),
                doc.text,
                doc.title,
                doc.uri,
                json.dumps(doc.metadata, default=str),
            )
            for doc, vector in zip(documents, vectors, strict=True)
        ]
        pool = await self._get_pool()
        async with pool.acquire() as conn, conn.transaction():
            await conn.executemany(
                f"""
                INSERT INTO {self._table} (id, embedding, text, title, uri, metadata)
                VALUES ($1, $2::vector, $3, $4, $5, $6::jsonb)
                ON CONFLICT (id) DO UPDATE SET
                    embedding  = EXCLUDED.embedding,
                    text       = EXCLUDED.text,
                    title      = EXCLUDED.title,
                    uri        = EXCLUDED.uri,
                    metadata   = EXCLUDED.metadata,
                    updated_at = now()
                """,
                rows,
            )
        return len(rows)

    async def delete_by_uri(self, uri: str) -> int:
        await self.ensure_ready()
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            status = await conn.execute(
                f"DELETE FROM {self._table} WHERE uri = $1", uri
            )
        return int(status.rsplit(" ", 1)[-1] or 0)

    async def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        where: dict[str, Any] | None = None,
        audiences: frozenset[str] | None = None,
    ) -> list[SearchHit]:
        await self.ensure_ready()
        vector = _to_pg_vector(await self.embedder.embed_query(query))
        limit = top_k or self.config.top_k

        # Filters go through bound JSONB parameters, so neither a
        # model-generated filter value nor an audience name reaches SQL text.
        params: list[Any] = [vector]
        clauses: list[str] = []

        if where:
            params.append(json.dumps(where, default=str))
            clauses.append(f"metadata @> ${len(params)}::jsonb")

        if audiences is not None:
            params.append(list(sorted(audiences)))
            audience_clause = f"metadata->>'audience' = ANY(${len(params)})"
            if DEFAULT_AUDIENCE in audiences:
                # An untagged document belongs to the default audience, so a
                # caller who can see that must also match rows with no key.
                audience_clause = (
                    f"({audience_clause} OR metadata->>'audience' IS NULL)"
                )
            clauses.append(audience_clause)

        predicate = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(limit)

        sql = f"""
            SELECT id, text, title, uri, metadata,
                   1 - (embedding <=> $1::vector) AS score
            FROM {self._table}
            {predicate}
            ORDER BY embedding <=> $1::vector
            LIMIT ${len(params)}
        """

        pool = await self._get_pool()
        async with pool.acquire() as conn:
            records = await conn.fetch(sql, *params)

        return [
            SearchHit(
                id=r["id"],
                text=r["text"],
                title=r["title"],
                uri=r["uri"],
                score=float(r["score"]),
                metadata=_load(r["metadata"]),
            )
            for r in records
            if float(r["score"]) >= self.config.score_threshold
        ]

    async def count(self) -> int:
        await self.ensure_ready()
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            return int(await conn.fetchval(f"SELECT count(*) FROM {self._table}"))

    async def aclose(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
            self._ready = False


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _table_name(collection: str) -> str:
    """Derive and validate a table identifier from a collection name.

    Identifiers can't be bound as query parameters, so this is the only place
    a collection name reaches SQL text -- it is validated against an allowlist
    rather than escaped.
    """
    candidate = f"chunks_{collection.replace('-', '_').lower()}"
    if not _SAFE_NAME.match(candidate):
        raise ValueError(
            f"collection {collection!r} does not produce a valid table name; "
            f"use lowercase letters, digits, and hyphens only"
        )
    return candidate


def _to_pg_vector(vector: Sequence[float]) -> str:
    """pgvector's text input format."""
    return "[" + ",".join(f"{x:.7g}" for x in vector) + "]"


def _load(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


__all__ = ["PgVectorStore"]
