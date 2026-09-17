"""Analytics event store -- Postgres, shared across projects and replicas.

Why this exists at all: ``api.admin.UsageStats`` holds scalar counters in
memory, reset on restart and private to each replica. That answers "is this
project healthy right now" and nothing else. A date filter, a trend line, or a
retention cohort all need one durable row per turn, with a timestamp on it.

Why one shared table rather than one per project, which is how the pgvector
store does it: a chunks table carries a fixed-dimension vector column, so
projects genuinely cannot share one. Turn rows are identical across projects,
and the dashboard's whole purpose is comparing seven sites side by side --
which against seven tables would be a seven-way UNION generated at runtime.
The ``project`` column is indexed and every query filters on it.

**Identity.** A cohort needs a stable subject with a first-seen date. The
widget is anonymous, so the subject is a visitor id the widget mints and keeps
in first-party storage. When that visitor later authenticates, their row gains
a ``user_id`` and every query re-resolves through it -- so stitching is
retroactive and needs no backfill. See ``_SUBJECT_SQL``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import structlog

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# identity resolution
# ---------------------------------------------------------------------------

#: How a turn row resolves to the person it belongs to.
#:
#: Applied at query time rather than stored on the turn, which is what makes
#: identity stitching retroactive: the moment a visitor logs in and their
#: ``analytics_visitors`` row gains a ``user_id``, every turn they ever sent --
#: including the anonymous ones from before the login -- resolves to the same
#: subject. Storing a resolved subject on each turn would instead require
#: rewriting history on every login.
#:
#: The ``anon:`` prefix keeps the two id spaces from colliding: a site whose
#: user ids happen to look like UUIDs must not merge a logged-in user with an
#: unrelated anonymous visitor that shares the string.
_SUBJECT_SQL = "COALESCE(v.user_id, 'anon:' || v.visitor_id)"


@dataclass(slots=True)
class TurnRecord:
    """One completed turn, as the API observed it."""

    project: str
    conversation_id: str
    visitor_id: str
    model: str
    duration_ms: int
    #: Time to first token. The number that actually governs perceived
    #: latency, tracked separately from total duration because a fast first
    #: token with a slow tail feels good and the reverse feels broken.
    ttft_ms: int | None = None
    user_id: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    tool_calls: int = 0
    failed: bool = False
    error_kind: str | None = None
    roles: list[str] | None = None


class AnalyticsStore:
    """Durable turn history, plus the queries the dashboard runs against it.

    Every write path here is failure-tolerant by design: analytics is
    observability, and losing a row is strictly better than failing the chat
    turn that produced it. Read paths are not -- a dashboard that silently
    renders zeros is worse than one that shows an error.
    """

    def __init__(self, dsn: str | None = None) -> None:
        raw = dsn or os.environ.get("ANALYTICS_DATABASE_URL") or os.environ.get(
            "DATABASE_URL", ""
        )
        if not raw:
            raise ValueError(
                "analytics requires ANALYTICS_DATABASE_URL (or DATABASE_URL)"
            )
        # asyncpg takes a bare postgres:// DSN; strip any SQLAlchemy driver tag.
        self._dsn = raw.replace("postgresql+asyncpg://", "postgresql://")
        self._pool: Any = None
        self._ready = False

    # -- lifecycle ----------------------------------------------------------

    async def _get_pool(self) -> Any:
        if self._pool is None:
            try:
                import asyncpg
            except ImportError as exc:  # pragma: no cover
                raise ImportError(
                    "The analytics store needs: uv add 'chatbot-core[pgvector]'"
                ) from exc
            # Smaller than the retrieval pool: writes are single-row and
            # queries are dashboard-rate, not request-rate.
            self._pool = await asyncpg.create_pool(
                self._dsn, min_size=1, max_size=4, command_timeout=30
            )
        return self._pool

    async def ensure_ready(self) -> None:
        if self._ready:
            return
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS analytics_visitors (
                    project     TEXT NOT NULL,
                    visitor_id  TEXT NOT NULL,
                    user_id     TEXT,
                    first_seen  TIMESTAMPTZ NOT NULL DEFAULT now(),
                    last_seen   TIMESTAMPTZ NOT NULL DEFAULT now(),
                    PRIMARY KEY (project, visitor_id)
                )
                """
            )
            # Cohort queries group by the *user's* earliest first_seen across
            # every device they have used, so user_id needs its own index.
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS analytics_visitors_user_idx "
                "ON analytics_visitors (project, user_id) "
                "WHERE user_id IS NOT NULL"
            )

            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS analytics_turns (
                    id                BIGSERIAL PRIMARY KEY,
                    project           TEXT NOT NULL,
                    conversation_id   TEXT NOT NULL,
                    visitor_id        TEXT NOT NULL,
                    occurred_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
                    model             TEXT NOT NULL,
                    duration_ms       INTEGER NOT NULL,
                    ttft_ms           INTEGER,
                    input_tokens      INTEGER NOT NULL DEFAULT 0,
                    output_tokens     INTEGER NOT NULL DEFAULT 0,
                    cache_read_tokens INTEGER NOT NULL DEFAULT 0,
                    tool_calls        INTEGER NOT NULL DEFAULT 0,
                    failed            BOOLEAN NOT NULL DEFAULT false,
                    error_kind        TEXT,
                    roles             TEXT[] NOT NULL DEFAULT '{}'
                )
                """
            )
            # Every dashboard query is (project, time range), so the composite
            # index is the one that matters. DESC because the default view is
            # always "most recent first".
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS analytics_turns_project_time_idx "
                "ON analytics_turns (project, occurred_at DESC)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS analytics_turns_visitor_idx "
                "ON analytics_turns (project, visitor_id)"
            )
        self._ready = True
        log.info("analytics_ready")

    async def aclose(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
            self._ready = False

    # -- writes -------------------------------------------------------------

    async def record_turn(self, record: TurnRecord) -> None:
        """Append one turn. Never raises.

        Called from the chat request path after the stream completes. An
        analytics outage must not surface to the person using the chatbot, so
        every failure here is logged and swallowed -- the turn already
        succeeded by the time this runs.
        """
        try:
            await self.ensure_ready()
            pool = await self._get_pool()
            async with pool.acquire() as conn, conn.transaction():
                # Upsert the visitor first so a turn always has a row to
                # resolve through; without it the analytic joins would drop
                # the very first turn of every new visitor.
                await conn.execute(
                    """
                    INSERT INTO analytics_visitors (project, visitor_id, user_id)
                    VALUES ($1, $2, $3)
                    ON CONFLICT (project, visitor_id) DO UPDATE SET
                        last_seen = now(),
                        -- Never clear a known identity: a logged-in visitor
                        -- who later browses anonymously keeps their stitch.
                        user_id = COALESCE(EXCLUDED.user_id, analytics_visitors.user_id)
                    """,
                    record.project,
                    record.visitor_id,
                    record.user_id,
                )
                await conn.execute(
                    """
                    INSERT INTO analytics_turns (
                        project, conversation_id, visitor_id, model,
                        duration_ms, ttft_ms, input_tokens, output_tokens,
                        cache_read_tokens, tool_calls, failed, error_kind, roles
                    )
                    VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13)
                    """,
                    record.project,
                    record.conversation_id,
                    record.visitor_id,
                    record.model,
                    record.duration_ms,
                    record.ttft_ms,
                    record.input_tokens,
                    record.output_tokens,
                    record.cache_read_tokens,
                    record.tool_calls,
                    record.failed,
                    record.error_kind,
                    record.roles or [],
                )
        except Exception as exc:  # noqa: BLE001 - see docstring
            log.warning(
                "analytics_write_failed",
                project=record.project,
                error=str(exc),
            )

    async def link_identity(self, project: str, visitor_id: str, user_id: str) -> None:
        """Bind an anonymous visitor to an authenticated user.

        This is the stitch. Because ``_SUBJECT_SQL`` resolves through this
        table at query time, calling it once rewrites the subject of every
        turn that visitor has ever sent -- no backfill, no history rewrite.
        """
        try:
            await self.ensure_ready()
            pool = await self._get_pool()
            async with pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO analytics_visitors (project, visitor_id, user_id)
                    VALUES ($1, $2, $3)
                    ON CONFLICT (project, visitor_id) DO UPDATE SET
                        user_id = EXCLUDED.user_id,
                        last_seen = now()
                    """,
                    project,
                    visitor_id,
                    user_id,
                )
        except Exception as exc:  # noqa: BLE001
            log.warning("analytics_link_failed", project=project, error=str(exc))

    # -- reads --------------------------------------------------------------

    async def summary(
        self,
        project: str,
        *,
        since: datetime,
        until: datetime,
        model: str | None = None,
        role: str | None = None,
    ) -> dict[str, Any]:
        """Headline totals for one project over a window.

        Percentiles rather than means for latency: a mean turn time is
        dominated by the slow tail and describes nobody's experience. p50 is
        what a typical visitor waits, p95 is what the unlucky ones do.
        """
        params, predicate = self._filters(project, since, until, model, role)
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                f"""
                SELECT
                    count(*)                                        AS turns,
                    count(DISTINCT t.conversation_id)               AS conversations,
                    count(DISTINCT {_SUBJECT_SQL})                  AS visitors,
                    coalesce(sum(t.input_tokens), 0)                AS input_tokens,
                    coalesce(sum(t.output_tokens), 0)               AS output_tokens,
                    coalesce(sum(t.cache_read_tokens), 0)           AS cache_read_tokens,
                    coalesce(sum(t.tool_calls), 0)                  AS tool_calls,
                    count(*) FILTER (WHERE t.failed)                AS errors,
                    percentile_disc(0.5) WITHIN GROUP (ORDER BY t.duration_ms) AS p50_ms,
                    percentile_disc(0.95) WITHIN GROUP (ORDER BY t.duration_ms) AS p95_ms,
                    percentile_disc(0.5) WITHIN GROUP (ORDER BY t.ttft_ms)      AS p50_ttft_ms,
                    percentile_disc(0.95) WITHIN GROUP (ORDER BY t.ttft_ms)     AS p95_ttft_ms
                FROM analytics_turns t
                JOIN analytics_visitors v
                  ON v.project = t.project AND v.visitor_id = t.visitor_id
                {predicate}
                """,
                *params,
            )
        result = dict(row) if row else {}
        turns = result.get("turns") or 0
        result["error_rate"] = (
            round((result.get("errors") or 0) / turns, 4) if turns else 0.0
        )
        return result

    async def timeseries(
        self,
        project: str,
        *,
        since: datetime,
        until: datetime,
        bucket: str = "day",
        model: str | None = None,
        role: str | None = None,
    ) -> list[dict[str, Any]]:
        """Bucketed activity, gap-filled.

        ``generate_series`` supplies the left side of the join so that a day
        with no traffic returns a zero row rather than vanishing. A chart that
        silently drops empty days draws a straight line through an outage,
        which is precisely the thing an operator needs to see.
        """
        if bucket not in _BUCKETS:
            raise ValueError(f"bucket must be one of {sorted(_BUCKETS)}")

        params, predicate = self._filters(project, since, until, model, role)
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                WITH buckets AS (
                    SELECT generate_series(
                        date_trunc('{bucket}', $2::timestamptz),
                        date_trunc('{bucket}', $3::timestamptz),
                        '1 {bucket}'::interval
                    ) AS bucket
                ),
                agg AS (
                    SELECT
                        date_trunc('{bucket}', t.occurred_at) AS bucket,
                        count(*)                              AS turns,
                        count(DISTINCT {_SUBJECT_SQL})        AS visitors,
                        count(*) FILTER (WHERE t.failed)      AS errors,
                        coalesce(sum(t.input_tokens + t.output_tokens), 0) AS tokens,
                        percentile_disc(0.5) WITHIN GROUP (ORDER BY t.duration_ms) AS p50_ms,
                        percentile_disc(0.95) WITHIN GROUP (ORDER BY t.duration_ms) AS p95_ms
                    FROM analytics_turns t
                    JOIN analytics_visitors v
                      ON v.project = t.project AND v.visitor_id = t.visitor_id
                    {predicate}
                    GROUP BY 1
                )
                SELECT
                    b.bucket,
                    coalesce(a.turns, 0)    AS turns,
                    coalesce(a.visitors, 0) AS visitors,
                    coalesce(a.errors, 0)   AS errors,
                    coalesce(a.tokens, 0)   AS tokens,
                    a.p50_ms, a.p95_ms
                FROM buckets b
                LEFT JOIN agg a ON a.bucket = b.bucket
                ORDER BY b.bucket
                """,
                *params,
            )
        return [dict(r) for r in rows]

    async def cohorts(
        self,
        project: str,
        *,
        since: datetime,
        until: datetime,
        period: str = "week",
        periods: int = 8,
    ) -> dict[str, Any]:
        """Retention cohorts: subjects grouped by when they first appeared.

        The cohort key is the subject's *earliest* first_seen across every
        visitor row that resolves to them. That matters for stitched users:
        someone who browsed anonymously in week 1 and logged in during week 3
        belongs to the week-1 cohort, not week 3. Taking first_seen from the
        individual visitor row would instead count them as a new acquisition
        on every new device, permanently inflating the top of the funnel.
        """
        if period not in _BUCKETS:
            raise ValueError(f"period must be one of {sorted(_BUCKETS)}")
        periods = max(1, min(periods, 52))

        pool = await self._get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                WITH subjects AS (
                    SELECT
                        {_SUBJECT_SQL}                               AS subject,
                        date_trunc('{period}', min(v.first_seen))    AS cohort
                    FROM analytics_visitors v
                    WHERE v.project = $1 AND v.first_seen >= $2 AND v.first_seen < $3
                    GROUP BY 1
                ),
                activity AS (
                    SELECT DISTINCT
                        {_SUBJECT_SQL}                               AS subject,
                        date_trunc('{period}', t.occurred_at)        AS active_at
                    FROM analytics_turns t
                    JOIN analytics_visitors v
                      ON v.project = t.project AND v.visitor_id = t.visitor_id
                    WHERE t.project = $1 AND t.occurred_at >= $2
                )
                SELECT
                    s.cohort,
                    -- Whole periods elapsed between joining and being active.
                    -- Period 0 is the cohort's own period, so it is always 100%.
                    (EXTRACT(EPOCH FROM (a.active_at - s.cohort))
                        / EXTRACT(EPOCH FROM '1 {period}'::interval))::int AS period_index,
                    count(DISTINCT s.subject) AS subjects
                FROM subjects s
                JOIN activity a ON a.subject = s.subject AND a.active_at >= s.cohort
                GROUP BY 1, 2
                HAVING (EXTRACT(EPOCH FROM (a.active_at - s.cohort))
                        / EXTRACT(EPOCH FROM '1 {period}'::interval))::int < $4
                ORDER BY 1, 2
                """,
                project,
                since,
                until,
                periods,
            )

        # Pivot into the triangular grid a cohort table is actually drawn as.
        grid: dict[str, dict[int, int]] = {}
        for row in rows:
            key = row["cohort"].date().isoformat()
            grid.setdefault(key, {})[row["period_index"]] = row["subjects"]

        cohorts = []
        for key in sorted(grid):
            counts = grid[key]
            size = counts.get(0, 0)
            cohorts.append(
                {
                    "cohort": key,
                    "size": size,
                    # Absolute counts and retained fractions both ship: the
                    # percentage is what gets read, the count is what makes a
                    # 100%-of-two-people cell obviously meaningless.
                    "counts": [counts.get(i, 0) for i in range(periods)],
                    "retention": [
                        round(counts.get(i, 0) / size, 4) if size else 0.0
                        for i in range(periods)
                    ],
                }
            )
        return {"period": period, "periods": periods, "cohorts": cohorts}

    async def breakdown(
        self,
        project: str,
        *,
        since: datetime,
        until: datetime,
        dimension: str = "model",
        model: str | None = None,
        role: str | None = None,
    ) -> list[dict[str, Any]]:
        """Turn counts grouped by one dimension, for the filter sidebars."""
        column = _DIMENSIONS.get(dimension)
        if column is None:
            raise ValueError(f"dimension must be one of {sorted(_DIMENSIONS)}")

        params, predicate = self._filters(project, since, until, model, role)
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT {column} AS value,
                       count(*) AS turns,
                       count(*) FILTER (WHERE t.failed) AS errors,
                       percentile_disc(0.5) WITHIN GROUP (ORDER BY t.duration_ms) AS p50_ms
                FROM analytics_turns t
                JOIN analytics_visitors v
                  ON v.project = t.project AND v.visitor_id = t.visitor_id
                {predicate}
                GROUP BY 1
                ORDER BY turns DESC
                LIMIT 50
                """,
                *params,
            )
        return [dict(r) for r in rows]

    async def turns_for_export(
        self,
        project: str,
        *,
        since: datetime,
        until: datetime,
        model: str | None = None,
        role: str | None = None,
        limit: int = 100_000,
    ) -> list[dict[str, Any]]:
        """Raw rows behind a report download.

        Deliberately excludes message text. An export is the easiest way for
        conversation content to leave the building, and nothing on the
        dashboard needs it -- counts, latencies and identity keys are enough
        to rebuild every figure shown.
        """
        params, predicate = self._filters(project, since, until, model, role)
        params.append(max(1, min(limit, 500_000)))
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT
                    t.occurred_at, t.project, t.conversation_id,
                    {_SUBJECT_SQL} AS subject,
                    v.first_seen, t.model, t.duration_ms, t.ttft_ms,
                    t.input_tokens, t.output_tokens, t.cache_read_tokens,
                    t.tool_calls, t.failed, t.error_kind, t.roles
                FROM analytics_turns t
                JOIN analytics_visitors v
                  ON v.project = t.project AND v.visitor_id = t.visitor_id
                {predicate}
                ORDER BY t.occurred_at DESC
                LIMIT ${len(params)}
                """,
                *params,
            )
        return [dict(r) for r in rows]

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _filters(
        project: str,
        since: datetime,
        until: datetime,
        model: str | None,
        role: str | None,
    ) -> tuple[list[Any], str]:
        """Build the shared WHERE clause.

        Every value is a bound parameter. The only interpolated fragments
        anywhere in this module are bucket/dimension names, both checked
        against fixed allowlists before they reach a query.
        """
        params: list[Any] = [project, since, until]
        clauses = ["t.project = $1", "t.occurred_at >= $2", "t.occurred_at < $3"]
        if model:
            params.append(model)
            clauses.append(f"t.model = ${len(params)}")
        if role:
            params.append(role)
            # Role is an array column: a turn matches if the caller held the
            # role at all, not only if it was their sole role.
            clauses.append(f"${len(params)} = ANY(t.roles)")
        return params, "WHERE " + " AND ".join(clauses)


#: Allowlisted time buckets. Interpolated into SQL, so this is a security
#: boundary and not merely a convenience.
_BUCKETS = frozenset({"hour", "day", "week", "month"})

#: Allowlisted group-by columns, same reasoning as _BUCKETS.
_DIMENSIONS = {
    "model": "t.model",
    "error_kind": "coalesce(t.error_kind, 'none')",
    "conversation": "t.conversation_id",
}


__all__ = ["AnalyticsStore", "TurnRecord"]
