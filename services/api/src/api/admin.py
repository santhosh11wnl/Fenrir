"""Admin routes.

Everything here requires this project's ``admin`` permission, checked against
the caller's roles. The permission is checked, never a role name -- which roles
carry `admin` differs per site, and an endpoint has no business knowing that
one of them calls its privileged role "support".

Scope note: these report on **one project**, because the process serves one
project. A dashboard covering all seven aggregates across seven deployments by
polling each one's ``/admin`` -- there is deliberately no endpoint here that
can see another project's data.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import PlainTextResponse, Response
from pydantic import BaseModel, ConfigDict

from chatbot_core.analytics import cohort_csv, summary_markdown, turns_csv
from chatbot_core.auth import Permission, User

from .security import require_permission

log = structlog.get_logger(__name__)

router = APIRouter(prefix="/admin", tags=["admin"])

# Built once at import rather than on every route decoration. Each returns a
# dependency enforcing one permission -- never a role name, since which roles
# carry a permission is a per-project decision.
RequireAdmin = Depends(require_permission(Permission.ADMIN))
RequireManageUsers = Depends(require_permission(Permission.MANAGE_USERS))
RequireHistory = Depends(require_permission(Permission.HISTORY))


# ---------------------------------------------------------------------------
# usage tracking
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class UsageStats:
    """Rolling counters for this process.

    In-memory and per-replica, like the conversation store: restarts reset
    them, and two replicas each hold their own. That is fine for "is this
    project healthy and roughly what is it costing", and wrong for billing --
    which should read the structured logs, not this.
    """

    started_at: float = field(default_factory=time.time)
    turns: int = 0
    errors: int = 0
    tool_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    #: Total wall-clock seconds spent generating, for a mean latency figure.
    duration_seconds: float = 0.0

    def record_turn(self, *, duration: float, usage: Any, tools: int, failed: bool) -> None:
        self.turns += 1
        self.duration_seconds += duration
        self.tool_calls += tools
        if failed:
            self.errors += 1
        if usage is not None:
            self.input_tokens += getattr(usage, "input_tokens", 0) or 0
            self.output_tokens += getattr(usage, "output_tokens", 0) or 0
            self.cache_read_tokens += getattr(usage, "cache_read_input_tokens", 0) or 0

    def snapshot(self) -> dict[str, Any]:
        return {
            "uptime_seconds": int(time.time() - self.started_at),
            "turns": self.turns,
            "errors": self.errors,
            "tool_calls": self.tool_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "mean_seconds_per_turn": (
                round(self.duration_seconds / self.turns, 2) if self.turns else 0.0
            ),
            "error_rate": round(self.errors / self.turns, 3) if self.turns else 0.0,
        }


# ---------------------------------------------------------------------------
# schemas
# ---------------------------------------------------------------------------


class RoleSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    permissions: list[str]
    user_count: int


class UserSummary(BaseModel):
    """Never includes credential material -- not even a hash prefix."""

    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    roles: list[str]
    permissions: list[str]
    disabled: bool
    created_at: str


class OverviewResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project: str
    name: str
    status: str
    model: str
    provider: str
    mcp_connected: bool
    tools: list[str]
    indexed_chunks: int
    retrieval_enabled: bool
    auth_enabled: bool
    usage: dict[str, Any]


class ConversationSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    turns: int
    created_at: float
    updated_at: float
    preview: str


# ---------------------------------------------------------------------------
# routes
# ---------------------------------------------------------------------------


def _state(request: Request):
    state = getattr(request.app.state, "app_state", None)
    if state is None:  # pragma: no cover
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Service starting."
        )
    return state


@router.get("/overview", response_model=OverviewResponse)
async def overview(
    request: Request,
    _: User | None = RequireAdmin,
) -> OverviewResponse:
    """Everything a dashboard tile needs for this project, in one call."""
    state = _state(request)
    detail = await state.engine.health()
    config = state.config
    degraded = config.mcp.enabled and not detail["mcp_connected"]

    return OverviewResponse(
        project=config.project.id,
        name=config.project.name,
        status="degraded" if degraded else "ok" if detail["ready"] else "starting",
        model=detail["model"],
        provider=detail["provider"],
        mcp_connected=detail["mcp_connected"],
        tools=detail["tools"],
        indexed_chunks=detail["indexed_chunks"],
        retrieval_enabled=config.retrieval.enabled,
        auth_enabled=config.auth.enabled,
        usage=state.usage.snapshot(),
    )


@router.get("/roles", response_model=list[RoleSummary])
async def roles(
    request: Request,
    _: User | None = RequireAdmin,
) -> list[RoleSummary]:
    """This project's roles and what each grants."""
    state = _state(request)
    return [
        RoleSummary(
            name=name,
            description=spec.description,
            permissions=sorted(p.value for p in spec.permissions),
            user_count=sum(1 for u in state.users.users if u.has_role(name)),
        )
        for name, spec in sorted(state.config.auth.roles.items())
    ]


@router.get("/users", response_model=list[UserSummary])
async def users(
    request: Request,
    _: User | None = RequireManageUsers,
) -> list[UserSummary]:
    """This project's users and their effective permissions.

    Behind `manage_users` rather than `admin`: seeing who has access is a
    higher bar than seeing whether the service is healthy.
    """
    state = _state(request)
    config = state.config
    return [
        UserSummary(
            id=user.id,
            name=user.name,
            roles=sorted(user.roles),
            permissions=sorted(
                p.value for p in config.auth.permissions_for(user.roles)
            ),
            disabled=user.disabled,
            created_at=user.created_at,
        )
        for user in sorted(state.users.users, key=lambda u: u.id)
    ]


@router.get("/conversations", response_model=list[ConversationSummary])
async def conversations(
    request: Request,
    limit: int = 50,
    _: User | None = RequireHistory,
) -> list[ConversationSummary]:
    """Recent conversations, newest first.

    Previews only. Reading a full conversation means reading what users typed,
    so the full transcript stays behind `GET /conversations/{id}`.
    """
    state = _state(request)
    recent = sorted(
        state.conversations.all_for(state.config.project.id),
        key=lambda c: c.updated_at,
        reverse=True,
    )[: max(1, min(limit, 200))]

    return [
        ConversationSummary(
            id=convo.id,
            turns=len(convo.messages),
            created_at=convo.created_at,
            updated_at=convo.updated_at,
            preview=(convo.messages[0].content[:120] if convo.messages else ""),
        )
        for convo in recent
    ]


# ---------------------------------------------------------------------------
# analytics
# ---------------------------------------------------------------------------

#: Shorthand windows the dashboard's range picker sends: "7d", "12w", "6h".
_RANGE = re.compile(r"^(\d{1,4})([hdwm])$")

_RANGE_UNITS = {
    "h": lambda n: timedelta(hours=n),
    "d": lambda n: timedelta(days=n),
    "w": lambda n: timedelta(weeks=n),
    "m": lambda n: timedelta(days=30 * n),
}


@dataclass(slots=True)
class Window:
    since: datetime
    until: datetime


def time_window(
    range_: str = Query("30d", alias="range", description="e.g. 24h, 7d, 12w, 6m"),
    since: datetime | None = Query(None, description="Overrides `range`."),
    until: datetime | None = Query(None),
) -> Window:
    """Resolve the range picker into an explicit half-open interval.

    Explicit `since`/`until` win over the shorthand, so a saved dashboard link
    keeps pointing at the window it was shared with rather than silently
    sliding forward as a relative range would.
    """
    end = until or datetime.now(UTC)
    if since is not None:
        start = since
    else:
        match = _RANGE.match(range_)
        if not match:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="range must look like 24h, 7d, 12w or 6m",
            )
        amount, unit = int(match.group(1)), match.group(2)
        start = end - _RANGE_UNITS[unit](amount)

    # Naive datetimes from a query string would compare against timestamptz
    # columns as if they were UTC on some drivers and local on others; pin
    # them here so the window means the same thing everywhere.
    if start.tzinfo is None:
        start = start.replace(tzinfo=UTC)
    if end.tzinfo is None:
        end = end.replace(tzinfo=UTC)

    if start >= end:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="`since` must precede `until`.",
        )
    return Window(since=start, until=end)


RequireWindow = Depends(time_window)


def _analytics(request: Request):
    """The turn store, or a 503 explaining that it is not configured.

    A dashboard that renders zeroes because no store exists is
    indistinguishable from one reporting a genuinely quiet week, so this
    refuses rather than returning empty results.
    """
    state = _state(request)
    if state.analytics is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Analytics is not configured for this project. "
                "Set ANALYTICS_DATABASE_URL to enable metrics and cohorts."
            ),
        )
    return state.analytics


@router.get("/metrics")
async def metrics(
    request: Request,
    window: Window = RequireWindow,
    model: str | None = None,
    role: str | None = None,
    bucket: str = Query("day", pattern="^(hour|day|week|month)$"),
    _: User | None = RequireAdmin,
) -> dict[str, Any]:
    """Headline totals plus the bucketed series, in one round trip.

    Bundled deliberately: the summary and the chart above it are always shown
    together, and two endpoints would let them disagree when a turn lands
    between the two requests.
    """
    store = _analytics(request)
    project = _state(request).config.project.id
    return {
        "project": project,
        "since": window.since.isoformat(),
        "until": window.until.isoformat(),
        "bucket": bucket,
        "summary": await store.summary(
            project, since=window.since, until=window.until, model=model, role=role
        ),
        "series": await store.timeseries(
            project,
            since=window.since,
            until=window.until,
            bucket=bucket,
            model=model,
            role=role,
        ),
    }


@router.get("/breakdown")
async def breakdown(
    request: Request,
    window: Window = RequireWindow,
    dimension: str = Query("model", pattern="^(model|error_kind|conversation)$"),
    model: str | None = None,
    role: str | None = None,
    _: User | None = RequireAdmin,
) -> list[dict[str, Any]]:
    """Turn counts grouped by one dimension, for the filter sidebar."""
    store = _analytics(request)
    return await store.breakdown(
        _state(request).config.project.id,
        since=window.since,
        until=window.until,
        dimension=dimension,
        model=model,
        role=role,
    )


@router.get("/cohorts")
async def cohorts(
    request: Request,
    window: Window = RequireWindow,
    period: str = Query("week", pattern="^(day|week|month)$"),
    periods: int = Query(8, ge=1, le=52),
    _: User | None = RequireHistory,
) -> dict[str, Any]:
    """Retention cohorts by first-seen period.

    Behind `history` rather than `admin`: a cohort grid is per-person
    behaviour over time, which is a closer read on individuals than the
    aggregate health numbers `admin` covers.
    """
    store = _analytics(request)
    return await store.cohorts(
        _state(request).config.project.id,
        since=window.since,
        until=window.until,
        period=period,
        periods=periods,
    )


@router.get("/export/turns.csv", response_class=Response)
async def export_turns(
    request: Request,
    window: Window = RequireWindow,
    model: str | None = None,
    role: str | None = None,
    limit: int = Query(100_000, ge=1, le=500_000),
    _: User | None = RequireHistory,
) -> Response:
    """Raw turn rows as CSV. Contains no message text -- see the store."""
    store = _analytics(request)
    project = _state(request).config.project.id
    rows = await store.turns_for_export(
        project,
        since=window.since,
        until=window.until,
        model=model,
        role=role,
        limit=limit,
    )
    return _attachment(
        turns_csv(rows), f"{project}-turns-{window.since:%Y%m%d}.csv", "text/csv"
    )


@router.get("/export/cohorts.csv", response_class=Response)
async def export_cohorts(
    request: Request,
    window: Window = RequireWindow,
    period: str = Query("week", pattern="^(day|week|month)$"),
    periods: int = Query(8, ge=1, le=52),
    _: User | None = RequireHistory,
) -> Response:
    """The retention grid as a wide CSV, one row per cohort."""
    store = _analytics(request)
    project = _state(request).config.project.id
    grid = await store.cohorts(
        project, since=window.since, until=window.until, period=period, periods=periods
    )
    return _attachment(
        cohort_csv(grid),
        f"{project}-cohorts-{window.since:%Y%m%d}.csv",
        "text/csv",
    )


@router.get("/export/report.md", response_class=PlainTextResponse)
async def export_report(
    request: Request,
    window: Window = RequireWindow,
    model: str | None = None,
    role: str | None = None,
    include_cohorts: bool = True,
    _: User | None = RequireAdmin,
) -> Response:
    """A written report for the window -- headline, per-model, retention."""
    store = _analytics(request)
    project = _state(request).config.project.id
    grid = None
    if include_cohorts:
        grid = await store.cohorts(
            project, since=window.since, until=window.until, period="week", periods=6
        )
    body = summary_markdown(
        project=project,
        since=window.since,
        until=window.until,
        summary=await store.summary(
            project, since=window.since, until=window.until, model=model, role=role
        ),
        breakdown=await store.breakdown(
            project, since=window.since, until=window.until, dimension="model"
        ),
        cohorts=grid,
    )
    return _attachment(
        body, f"{project}-report-{window.since:%Y%m%d}.md", "text/markdown"
    )


def _attachment(body: str, filename: str, media_type: str) -> Response:
    """Serve a generated file as a download.

    The filename is built from a validated project id and a formatted date,
    so it cannot carry a quote or newline into the header -- which is what
    turns a Content-Disposition into a header-injection vector.
    """
    return Response(
        content=body,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.post("/reload-users", status_code=status.HTTP_204_NO_CONTENT)
async def reload_users(
    request: Request,
    caller: User | None = RequireManageUsers,
) -> None:
    """Re-read users.yaml without restarting.

    The store is loaded once at startup, so a user added by the CLI is
    otherwise invisible until the next deploy. This closes that gap -- and
    matters most for *revocation*, where waiting for a restart is the wrong
    answer.
    """
    state = _state(request)
    before = len(state.users)
    state.users.reload()
    log.info(
        "user_store_reloaded",
        project=state.config.project.id,
        by=caller.id if caller else "unknown",
        before=before,
        after=len(state.users),
    )


__all__ = ["UsageStats", "Window", "router", "time_window"]
