"""FastAPI application.

One image serves every project; ``PROJECT`` selects which. The engine, MCP
session, and embedding model are built once during lifespan startup -- the
embedding model alone is hundreds of megabytes, so per-request construction
would be ruinous.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

import structlog
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse

from chatbot_core import ChatEngine, InMemoryConversationStore, ProjectConfig
from chatbot_core.analytics import AnalyticsStore, TurnRecord
from chatbot_core.auth import Permission, User, UserStore
from chatbot_core.storage import history_for

from .admin import UsageStats
from .admin import router as admin_router
from .ratelimit import SlidingWindowLimiter
from .schemas import (
    ChatRequest,
    ConversationResponse,
    ConversationTurn,
    HealthResponse,
    ThemeResponse,
    ToolInfo,
)
from .security import (
    assert_safe_to_start,
    current_user,
    load_user_store,
    rate_limit_key,
    require_permission,
)
from .settings import APISettings
from .sse import stream_events

log = structlog.get_logger(__name__)

#: A conversation transcript is the user's own data, not an admin surface, so
#: it stays reachable when a project runs with no auth at all -- but the moment
#: auth is on, reading or deleting one needs the `history` permission. Built at
#: module scope because FastAPI resolves dependencies once per route.
RequireHistoryWhenAuthed = require_permission(
    Permission.HISTORY, open_when_auth_disabled=True
)


@dataclass(slots=True)
class AppState:
    """Everything built once at startup and shared across requests."""

    settings: APISettings
    config: ProjectConfig
    engine: ChatEngine
    conversations: InMemoryConversationStore
    limiter: SlidingWindowLimiter
    #: This project's users only. Another project's users are never loaded
    #: into this process, so cross-project access is structurally impossible.
    users: UserStore
    #: Rolling counters for the admin surface. Per-process and per-replica.
    usage: UsageStats
    #: Durable turn log. ``None`` when no DSN is configured -- analytics is
    #: optional, and a project without it serves chat exactly as before.
    analytics: AnalyticsStore | None = None


def create_app(settings: APISettings | None = None) -> FastAPI:
    settings = settings or APISettings()
    config = ProjectConfig.load(settings.config_path)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        state = AppState(
            settings=settings,
            config=config,
            engine=ChatEngine(config),
            conversations=InMemoryConversationStore(
                max_conversations=settings.max_conversations,
                ttl_seconds=settings.conversation_ttl_seconds,
            ),
            limiter=SlidingWindowLimiter(config.limits.requests_per_minute),
            users=load_user_store(config, settings.project_dir),
            usage=UsageStats(),
            analytics=(
                AnalyticsStore(settings.analytics_database_url)
                if settings.analytics_database_url
                else None
            ),
        )
        app.state.app_state = state

        # Before anything binds a port: refuse to serve an unauthenticated
        # endpoint outside a local environment, or an auth-enabled project
        # with no users (which would reject every request while looking
        # merely broken).
        assert_safe_to_start(config, state.users)

        # Let a failed startup crash the process. A container that reports
        # healthy while answering every request with an error is far harder to
        # diagnose than one that refuses to start.
        await state.engine.startup()

        # Analytics is the deliberate exception to the crash-on-failure rule
        # above. Creating the tables here means a bad DSN is visible in the
        # boot log rather than as a warning buried in the first conversation
        # -- but an unreachable metrics database must not take the chatbot
        # down with it, so this logs and continues where the engine would not.
        if state.analytics is not None:
            try:
                await state.analytics.ensure_ready()
            except Exception as exc:  # noqa: BLE001
                log.error(
                    "analytics_unavailable",
                    project=config.project.id,
                    error=str(exc),
                )

        log.info("api_ready", project=config.project.id, port=settings.port)
        try:
            yield
        finally:
            await state.engine.shutdown()
            if state.analytics is not None:
                await state.analytics.aclose()
            log.info("api_stopped", project=config.project.id)

    app = FastAPI(
        title=f"{config.project.name} API",
        description=config.project.description or "Chat API",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Content-Type", "Authorization"],
    )

    _register_routes(app)
    # Admin routes live on their own router so the permission gate is declared
    # once, next to the endpoints it guards, rather than repeated inline.
    app.include_router(admin_router)
    return app


def get_state(request: Request) -> AppState:
    state: AppState | None = getattr(request.app.state, "app_state", None)
    if state is None:  # pragma: no cover - only reachable outside lifespan
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Service starting."
        )
    return state



def _register_routes(app: FastAPI) -> None:
    @app.get("/health", response_model=HealthResponse, tags=["ops"])
    async def health(state: AppState = Depends(get_state)) -> HealthResponse:
        """Liveness plus enough detail to populate an admin dashboard tile."""
        detail = await state.engine.health()
        degraded = state.config.mcp.enabled and not detail["mcp_connected"]
        return HealthResponse(
            status="degraded" if degraded else "ok" if detail["ready"] else "starting",
            **{k: detail[k] for k in HealthResponse.model_fields if k in detail},
        )

    @app.get("/theme", response_model=ThemeResponse, tags=["config"])
    async def theme(state: AppState = Depends(get_state)) -> ThemeResponse:
        """Branding for the web client, so one build serves every project."""
        cfg = state.config
        return ThemeResponse(
            project_id=cfg.project.id,
            name=cfg.project.name,
            description=cfg.project.description,
            primary=cfg.theme.primary,
            accent=cfg.theme.accent,
            logo_url=cfg.theme.logo_url,
            greeting=cfg.theme.greeting,
            placeholder=cfg.theme.placeholder,
            suggestions=cfg.theme.suggestions,
        )

    @app.get("/tools", response_model=list[ToolInfo], tags=["config"])
    async def tools(state: AppState = Depends(get_state)) -> list[ToolInfo]:
        """What this assistant can actually do, for display and for debugging."""
        return [
            ToolInfo(name=s.name, description=s.description)
            for s in state.engine.tools
        ]

    @app.post("/chat", tags=["chat"])
    async def chat(
        payload: ChatRequest,
        request: Request,
        state: AppState = Depends(get_state),
        user: User | None = Depends(current_user),
    ) -> EventSourceResponse:
        """Send a message and stream the reply as server-sent events.

        Frames are typed by ``chatbot_core.events``: ``text_delta`` for answer
        text, ``tool_call``/``tool_result`` around each tool, ``citations`` for
        retrieved sources, and exactly one terminal ``done``.
        """
        cfg = state.config
        request_id = uuid.uuid4().hex[:12]

        if len(payload.message) > cfg.limits.max_input_chars:
            raise HTTPException(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                detail=f"Message exceeds {cfg.limits.max_input_chars} characters.",
            )

        key = rate_limit_key(request, user)
        if not state.limiter.check(key):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Too many requests.",
                headers={"Retry-After": str(state.limiter.retry_after(key))},
            )
        state.limiter.prune()

        project = cfg.project.id
        conversation_id = (
            payload.conversation_id or (await state.conversations.create(project)).id
        )

        convo = await state.conversations.append(
            project, conversation_id, "user", payload.message
        )
        history = list(history_for(convo, cfg.limits.max_history_messages))

        log.info(
            "chat_request",
            request_id=request_id,
            project=project,
            conversation_id=conversation_id,
            turns=len(history),
            user=user.id if user else "anonymous",
        )

        async def generate() -> AsyncIterator[dict[str, str]]:
            reply: list[str] = []
            started = time.perf_counter()
            final_usage = None
            tool_count = 0
            failed = False
            first_token_at: float | None = None
            error_kind: str | None = None

            async def collected() -> AsyncIterator:
                async for event in state.engine.stream(
                    conversation_id=conversation_id,
                    messages=history,
                    # None for an anonymous visitor, which resolves to the
                    # project's anonymous_audiences -- public material only.
                    roles=user.roles if user else None,
                ):
                    nonlocal final_usage, tool_count, failed
                    nonlocal first_token_at, error_kind
                    if event.type == "text_delta":
                        # Stamped on the first delta only: this is the number
                        # the visitor actually experiences as "did it
                        # respond", and it moves independently of total
                        # duration once tool calls are in play.
                        if first_token_at is None:
                            first_token_at = time.perf_counter()
                        reply.append(event.text)
                    elif event.type == "tool_call":
                        tool_count += 1
                    elif event.type == "error":
                        failed = True
                        error_kind = getattr(event, "code", None) or "stream_error"
                    elif event.type == "done":
                        final_usage = event.usage
                    yield event

            async for frame in stream_events(collected(), request_id=request_id):
                yield frame

            # Persist after streaming so the next turn has this one in context.
            # A client that disconnects mid-stream still gets the partial saved,
            # which is what makes "continue where we left off" work.
            if reply:
                await state.conversations.append(
                    project, conversation_id, "assistant", "".join(reply)
                )

            duration = time.perf_counter() - started
            turn_failed = failed or not reply
            state.usage.record_turn(
                duration=duration,
                usage=final_usage,
                tools=tool_count,
                failed=turn_failed,
            )

            if state.analytics is not None:
                await state.analytics.record_turn(
                    TurnRecord(
                        project=project,
                        conversation_id=conversation_id,
                        # A client that sends no visitor id still produces a
                        # usable turn row; it just resolves to a subject that
                        # lasts one conversation, so it counts toward volume
                        # and latency but never toward retention.
                        visitor_id=payload.visitor_id or f"conv-{conversation_id}",
                        user_id=user.id if user else None,
                        model=cfg.model.id,
                        duration_ms=int(duration * 1000),
                        ttft_ms=(
                            int((first_token_at - started) * 1000)
                            if first_token_at is not None
                            else None
                        ),
                        input_tokens=getattr(final_usage, "input_tokens", 0) or 0,
                        output_tokens=getattr(final_usage, "output_tokens", 0) or 0,
                        cache_read_tokens=(
                            getattr(final_usage, "cache_read_input_tokens", 0) or 0
                        ),
                        tool_calls=tool_count,
                        failed=turn_failed,
                        error_kind=error_kind or ("empty_reply" if not reply else None),
                        roles=sorted(user.roles) if user else [],
                    )
                )

        return EventSourceResponse(generate())

    @app.get(
        "/conversations/{conversation_id}",
        response_model=ConversationResponse,
        tags=["chat"],
    )
    async def get_conversation(
        conversation_id: str,
        state: AppState = Depends(get_state),
        _: User | None = Depends(RequireHistoryWhenAuthed),
    ) -> ConversationResponse:
        project = state.config.project.id
        convo = await state.conversations.get(project, conversation_id)
        if convo is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found."
            )
        return ConversationResponse(
            id=convo.id,
            project=convo.project,
            messages=[
                ConversationTurn(role=m.role, content=m.content) for m in convo.messages
            ],
            created_at=convo.created_at,
            updated_at=convo.updated_at,
        )

    @app.delete(
        "/conversations/{conversation_id}",
        status_code=status.HTTP_204_NO_CONTENT,
        tags=["chat"],
    )
    async def delete_conversation(
        conversation_id: str,
        state: AppState = Depends(get_state),
        _: User | None = Depends(RequireHistoryWhenAuthed),
    ) -> None:
        if not await state.conversations.delete(state.config.project.id, conversation_id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found."
            )


__all__ = ["AppState", "create_app", "get_state"]
