"""LanceDB store -- embedded, file-backed, no server to run.

The default because it makes a fresh project useful in one command. It is a
real store, not a toy: it handles corpora into the millions of chunks. What it
doesn't do is share an index between processes, so any project scaling past one
API replica should move to pgvector.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import structlog

from ..auth import DEFAULT_AUDIENCE
from ..config import RetrievalConfig
from .embeddings import Embedder
from .store import Document, SearchHit, VectorStore

log = structlog.get_logger(__name__)


class LanceDBStore(VectorStore):
    def __init__(
        self,
        config: RetrievalConfig,
        collection: str,
        embedder: Embedder,
        *,
        path: str | Path | None = None,
    ) -> None:
        super().__init__(config, collection, embedder)
        import os

        self._path = Path(path or os.environ.get("LANCEDB_PATH", "./storage/lancedb"))
        self._db: Any = None
        self._table: Any = None
        # LanceDB's sync client does blocking disk I/O. Every call goes through
        # a worker thread, and this lock keeps concurrent writers from
        # interleaving table creation.
        self._lock = asyncio.Lock()

    async def ensure_ready(self) -> None:
        async with self._lock:
            if self._table is not None:
                return
            self._path.mkdir(parents=True, exist_ok=True)
            self._db = await asyncio.to_thread(_connect, self._path)
            names = await asyncio.to_thread(_list_tables, self._db)

            if self.collection in names:
                self._table = await asyncio.to_thread(self._db.open_table, self.collection)
                await self._verify_dimensions()
            else:
                log.info("creating_collection", collection=self.collection)
                # schema= is a keyword: the second positional parameter is
                # `data`, and passing an Arrow schema there is silently wrong.
                self._table = await asyncio.to_thread(
                    lambda: self._db.create_table(
                        self.collection,
                        schema=_empty_arrow(self.embedder.dimensions),
                    )
                )

    async def _verify_dimensions(self) -> None:
        schema = await asyncio.to_thread(lambda: self._table.schema)
        field = schema.field("vector")
        existing = getattr(field.type, "list_size", None)
        if existing and existing != self.embedder.dimensions:
            raise ValueError(
                f"Collection {self.collection!r} was built with "
                f"{existing}-dimensional vectors but the configured embedding "
                f"model produces {self.embedder.dimensions}. Re-ingest the "
                f"corpus, or point the project at a different collection."
            )

    async def upsert(self, documents: Sequence[Document]) -> int:
        if not documents:
            return 0
        await self.ensure_ready()
        vectors = await self.embedder.embed_documents([d.text for d in documents])
        rows = [
            {
                "id": doc.id,
                "vector": vector,
                "text": doc.text,
                "title": doc.title,
                "uri": doc.uri or "",
                "metadata": _dump(doc.metadata),
            }
            for doc, vector in zip(documents, vectors, strict=True)
        ]
        async with self._lock:
            # merge_insert is the upsert: re-ingesting a document replaces its
            # chunks rather than accumulating duplicates alongside them.
            await asyncio.to_thread(
                lambda: self._table.merge_insert("id")
                .when_matched_update_all()
                .when_not_matched_insert_all()
                .execute(rows)
            )
        return len(rows)

    async def delete_by_uri(self, uri: str) -> int:
        await self.ensure_ready()
        escaped = uri.replace("'", "''")
        async with self._lock:
            before = await asyncio.to_thread(self._table.count_rows)
            await asyncio.to_thread(self._table.delete, f"uri = '{escaped}'")
            after = await asyncio.to_thread(self._table.count_rows)
        return before - after

    async def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        where: dict[str, Any] | None = None,
        audiences: frozenset[str] | None = None,
    ) -> list[SearchHit]:
        await self.ensure_ready()
        if await self.count() == 0:
            return []

        vector = await self.embedder.embed_query(query)
        limit = top_k or self.config.top_k
        predicate = _where_clause(where, audiences)

        def run() -> list[dict[str, Any]]:
            q = self._table.search(vector).metric("cosine").limit(limit)
            if predicate:
                # prefilter=True applies the restriction *before* the nearest-
                # neighbour cut. Post-filtering would search the whole corpus
                # and then discard, silently returning fewer than top_k results
                # to a restricted caller.
                q = q.where(predicate, prefilter=True)
            return q.to_list()

        rows = await asyncio.to_thread(run)

        hits: list[SearchHit] = []
        for row in rows:
            # LanceDB reports cosine *distance*; similarity is 1 - distance.
            score = 1.0 - float(row.get("_distance", 1.0))
            if score < self.config.score_threshold:
                continue
            hits.append(
                SearchHit(
                    id=row["id"],
                    text=row["text"],
                    title=row["title"],
                    uri=row.get("uri") or None,
                    score=score,
                    metadata=_load(row.get("metadata")),
                )
            )
        return hits

    async def count(self) -> int:
        await self.ensure_ready()
        return int(await asyncio.to_thread(self._table.count_rows))


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _list_tables(db: Any) -> set[str]:
    """List collection names, tolerating both lancedb list APIs.

    ``table_names()`` is deprecated but returns a plain list. ``list_tables()``
    returns a paginated ``ListTablesResponse`` whose names live under
    ``.tables`` -- iterating the response object directly yields the wrong
    thing and silently reports every table as missing, which then fails as
    "table already exists" on create.
    """
    lister = getattr(db, "list_tables", None)
    if lister is None:
        return set(db.table_names())

    names: set[str] = set()
    token: str | None = None
    while True:
        response = lister(page_token=token) if token else lister()
        page = getattr(response, "tables", response)
        names.update(page)
        token = getattr(response, "page_token", None)
        if not token:
            return names


def _connect(path: Path) -> Any:
    try:
        import lancedb
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "The lancedb backend needs: uv add 'chatbot-core[lancedb]'"
        ) from exc
    return lancedb.connect(str(path))


def _empty_arrow(dimensions: int) -> Any:
    """Create the table from an explicit Arrow schema.

    Inferring it from a sample row would give ``vector`` a variable-size list
    type, which can't be indexed for ANN search.
    """
    import pyarrow as pa

    return pa.schema(
        [
            pa.field("id", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), dimensions)),
            pa.field("text", pa.string()),
            pa.field("title", pa.string()),
            pa.field("uri", pa.string()),
            pa.field("metadata", pa.string()),
        ]
    )


def _where_clause(
    where: dict[str, Any] | None, audiences: frozenset[str] | None = None
) -> str | None:
    """Build a SQL predicate over metadata.

    Values are restricted to scalars and quotes are escaped. Filter values can
    originate from a model-generated tool call, so they are never trusted.

    The audience clause matches the JSON-encoded metadata column, since chunk
    metadata is stored as a JSON string rather than typed columns.
    """
    parts: list[str] = []

    if audiences is not None and not audiences:
        # A caller granted no audiences at all. This must match nothing --
        # emitting no predicate would turn "sees nothing" into "sees
        # everything", which is the worst available failure. `1 = 0` rather
        # than `()`, which is not parseable and would surface as a query error
        # instead of an empty result.
        return "1 = 0"

    if audiences is not None:
        # An untagged document is DEFAULT_AUDIENCE, so a caller who can see
        # that audience must also match rows with no audience key at all.
        clauses = []
        for audience in sorted(audiences):
            if not audience.replace("_", "").replace("-", "").isalnum():
                raise ValueError(f"invalid audience: {audience!r}")
            clauses.append(f"metadata LIKE '%\"audience\": \"{audience}\"%'")
        if DEFAULT_AUDIENCE in audiences:
            clauses.append("metadata NOT LIKE '%\"audience\"%'")
        parts.append("(" + " OR ".join(clauses) + ")")

    if not where:
        return " AND ".join(parts) or None

    for key, value in where.items():
        if not key.replace("_", "").isalnum():
            raise ValueError(f"invalid filter key: {key!r}")
        if isinstance(value, bool):
            parts.append(f"{key} = {str(value).lower()}")
        elif isinstance(value, (int, float)):
            parts.append(f"{key} = {value}")
        elif isinstance(value, str):
            parts.append(f"{key} = '{value.replace(chr(39), chr(39) * 2)}'")
        else:
            raise ValueError(f"unsupported filter value for {key!r}: {type(value).__name__}")
    return " AND ".join(parts)


def _dump(metadata: dict[str, Any]) -> str:
    import json

    return json.dumps(metadata, default=str)


def _load(raw: Any) -> dict[str, Any]:
    import json

    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


__all__ = ["LanceDBStore"]
