"""API surface tests.

The engine is stubbed so these run with no Anthropic key, no MCP server, and no
embedding model. What's under test is the HTTP layer: SSE framing, conversation
lifetime, limits, and error shapes -- the engine has its own tests.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import pytest
from fastapi.testclient import TestClient

from api.admin import UsageStats
from api.admin import router as admin_router
from api.app import AppState
from api.ratelimit import SlidingWindowLimiter
from chatbot_core import InMemoryConversationStore, ProjectConfig
from chatbot_core.auth import UserStore
from chatbot_core.events import (
    Citations,
    Done,
    ErrorEvent,
    MessageStart,
    Source,
    TextDelta,
    ToolCall,
    ToolResult,
    Usage,
)

CONFIG = """
project:
  id: test-bot
  name: Test Bot
  description: Fixture project.
system_prompt: You are a test assistant.
retrieval:
  enabled: false
mcp:
  enabled: false
theme:
  primary: "#112233"
  greeting: Hi there
limits:
  max_input_chars: 200
  requests_per_minute: 5
"""


class StubEngine:
    """Replays a scripted event sequence. Records what it was asked."""

    def __init__(self, config, events=None):
        self.config = config
        self._events = events
        self.calls: list[list] = []
        self.roles: list[frozenset[str] | None] = []

    async def startup(self) -> None: ...

    async def shutdown(self) -> None: ...

    @property
    def tools(self):
        from chatbot_core.providers.base import ToolSpec

        return [ToolSpec(name="ping", description="Check the pipeline.", input_schema={})]

    async def health(self):
        return {
            "project": self.config.project.id,
            "model": self.config.model.id,
            "provider": self.config.model.provider.value,
            "ready": True,
            "mcp_connected": False,
            "tools": ["ping"],
            "indexed_chunks": 42,
        }

    async def stream(
        self, *, conversation_id: str, messages, roles: frozenset[str] | None = None
    ) -> AsyncIterator:
        self.calls.append(list(messages))
        # Recorded so a test can assert the caller's roles reach the engine --
        # that is what decides which document audiences the turn may retrieve.
        self.roles.append(roles)
        events = self._events or [
            MessageStart(
                conversation_id=conversation_id, message_id="msg_1", model="claude-opus-5"
            ),
            TextDelta(text="Hello "),
            TextDelta(text="world"),
            Done(stop_reason="end_turn", usage=Usage(input_tokens=10, output_tokens=2)),
        ]
        for event in events:
            yield event


@pytest.fixture
def config(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(CONFIG)
    return ProjectConfig.load(path)


def build_client(config, events=None) -> tuple[TestClient, StubEngine]:
    from api.settings import APISettings

    settings = APISettings(project="test-bot")
    engine = StubEngine(config, events)

    application = _app_with_stub(settings, config, engine)
    return TestClient(application), engine


def _app_with_stub(settings, config, engine):
    """Build the real app, then swap the engine during lifespan.

    Patching after construction keeps the routes, middleware, and dependency
    graph exactly as production builds them -- only the engine is fake.
    """
    from contextlib import asynccontextmanager

    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware

    from api.app import _register_routes

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.app_state = AppState(
            settings=settings,
            config=config,
            engine=engine,
            conversations=InMemoryConversationStore(),
            limiter=SlidingWindowLimiter(config.limits.requests_per_minute),
            # The fixture config leaves auth disabled, so an empty store is
            # correct here: every request is anonymous.
            users=UserStore([]),
            usage=UsageStats(),
        )
        await engine.startup()
        yield
        await engine.shutdown()

    app = FastAPI(lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware, allow_origins=settings.cors_origins, allow_credentials=True
    )
    _register_routes(app)
    return app


def parse_sse(text: str) -> list[tuple[str, dict]]:
    """Parse an SSE body into (event_name, payload) pairs."""
    frames: list[tuple[str, dict]] = []
    name: str | None = None
    for line in text.splitlines():
        if line.startswith("event:"):
            name = line[6:].strip()
        elif line.startswith("data:") and name:
            frames.append((name, json.loads(line[5:].strip())))
            name = None
    return frames


class TestHealth:
    def test_reports_ok_and_detail(self, config):
        client, _ = build_client(config)
        with client:
            body = client.get("/health").json()
        assert body["status"] == "ok"
        assert body["project"] == "test-bot"
        assert body["indexed_chunks"] == 42

    def test_theme_exposes_branding(self, config):
        client, _ = build_client(config)
        with client:
            body = client.get("/theme").json()
        assert body["primary"] == "#112233"
        assert body["greeting"] == "Hi there"

    def test_tools_lists_what_the_assistant_can_do(self, config):
        client, _ = build_client(config)
        with client:
            body = client.get("/tools").json()
        assert body == [{"name": "ping", "description": "Check the pipeline."}]


class TestChat:
    def test_streams_text_and_terminates(self, config):
        client, _ = build_client(config)
        with client:
            response = client.post("/chat", json={"message": "hi"})
        assert response.status_code == 200
        frames = parse_sse(response.text)
        kinds = [name for name, _ in frames]
        assert kinds[0] == "message_start"
        assert kinds[-1] == "done"
        text = "".join(p["text"] for n, p in frames if n == "text_delta")
        assert text == "Hello world"

    def test_returns_conversation_id_for_a_new_chat(self, config):
        client, _ = build_client(config)
        with client:
            frames = parse_sse(client.post("/chat", json={"message": "hi"}).text)
        start = next(p for n, p in frames if n == "message_start")
        assert start["conversation_id"]

    def test_history_accumulates_across_turns(self, config):
        """The second turn must see the first -- otherwise every message is
        answered with no memory of the conversation."""
        client, engine = build_client(config)
        with client:
            frames = parse_sse(client.post("/chat", json={"message": "first"}).text)
            cid = next(p for n, p in frames if n == "message_start")["conversation_id"]
            client.post("/chat", json={"message": "second", "conversation_id": cid})

        assert [m.content for m in engine.calls[0]] == ["first"]
        assert [m.content for m in engine.calls[1]] == ["first", "Hello world", "second"]

    def test_assistant_reply_is_persisted(self, config):
        client, _ = build_client(config)
        with client:
            frames = parse_sse(client.post("/chat", json={"message": "hi"}).text)
            cid = next(p for n, p in frames if n == "message_start")["conversation_id"]
            convo = client.get(f"/conversations/{cid}").json()
        assert [m["role"] for m in convo["messages"]] == ["user", "assistant"]
        assert convo["messages"][1]["content"] == "Hello world"

    def test_tool_and_citation_frames_are_forwarded(self, config):
        events = [
            MessageStart(conversation_id="c", message_id="m", model="claude-opus-5"),
            ToolCall(id="t1", name="ping", input={"message": "hi"}),
            ToolResult(id="t1", name="ping", ok=True, preview="pong", duration_ms=12),
            Citations(
                sources=[Source(id="s1", title="Doc", uri="a.md", score=0.9, excerpt="...")]
            ),
            TextDelta(text="done"),
            Done(stop_reason="end_turn"),
        ]
        client, _ = build_client(config, events)
        with client:
            frames = parse_sse(client.post("/chat", json={"message": "hi"}).text)
        kinds = [n for n, _ in frames]
        assert kinds == [
            "message_start",
            "tool_call",
            "tool_result",
            "citations",
            "text_delta",
            "done",
        ]

    def test_error_event_is_forwarded_with_done(self, config):
        """A client that never sees `done` spins forever."""
        events = [ErrorEvent(message="nope", retriable=False), Done(stop_reason="error")]
        client, _ = build_client(config, events)
        with client:
            frames = parse_sse(client.post("/chat", json={"message": "hi"}).text)
        assert [n for n, _ in frames] == ["error", "done"]

    def test_missing_done_is_synthesised(self, config):
        """A provider returning without a terminal event is a bug, but the
        client shouldn't be the one that pays for it."""
        client, _ = build_client(config, [TextDelta(text="truncated")])
        with client:
            frames = parse_sse(client.post("/chat", json={"message": "hi"}).text)
        assert frames[-1][0] == "done"
        assert frames[-1][1]["stop_reason"] == "incomplete"


class TestLimits:
    def test_rejects_oversized_message(self, config):
        client, _ = build_client(config)
        with client:
            response = client.post("/chat", json={"message": "x" * 500})
        assert response.status_code == 413

    def test_rejects_empty_message(self, config):
        client, _ = build_client(config)
        with client:
            assert client.post("/chat", json={"message": ""}).status_code == 422

    def test_rejects_unknown_body_field(self, config):
        client, _ = build_client(config)
        with client:
            response = client.post("/chat", json={"message": "hi", "model": "sneaky"})
        assert response.status_code == 422

    def test_rate_limits_with_retry_after(self, config):
        client, _ = build_client(config)
        with client:
            for _ in range(config.limits.requests_per_minute):
                assert client.post("/chat", json={"message": "hi"}).status_code == 200
            response = client.post("/chat", json={"message": "hi"})
        assert response.status_code == 429
        assert int(response.headers["retry-after"]) >= 1


class TestConversations:
    def test_unknown_conversation_is_404(self, config):
        client, _ = build_client(config)
        with client:
            assert client.get("/conversations/nope").status_code == 404

    def test_delete_removes_conversation(self, config):
        client, _ = build_client(config)
        with client:
            frames = parse_sse(client.post("/chat", json={"message": "hi"}).text)
            cid = next(p for n, p in frames if n == "message_start")["conversation_id"]
            assert client.delete(f"/conversations/{cid}").status_code == 204
            assert client.get(f"/conversations/{cid}").status_code == 404

    def test_delete_unknown_is_404(self, config):
        client, _ = build_client(config)
        with client:
            assert client.delete("/conversations/nope").status_code == 404


class TestRateLimiter:
    def test_allows_up_to_the_limit(self):
        limiter = SlidingWindowLimiter(3)
        assert [limiter.check("a") for _ in range(4)] == [True, True, True, False]

    def test_keys_are_independent(self):
        limiter = SlidingWindowLimiter(1)
        assert limiter.check("a") and limiter.check("b")
        assert not limiter.check("a")

    def test_prune_drops_idle_keys(self):
        """Without this the map grows one entry per client IP forever."""
        limiter = SlidingWindowLimiter(5, window_seconds=0.0)
        limiter.check("a")
        limiter.prune()
        assert limiter._hits == {}  # noqa: SLF001 - asserting the leak is closed


class TestPublicPaths:
    """`auth.public_paths` lets one project serve a public widget and a
    private dashboard from the same process.

    The risk it introduces is over-reach: a bypass that is wider than declared,
    or one that also waves through a bad key. Both are asserted against here.
    """

    @staticmethod
    def _authed_config(tmp_path, public_paths):
        """A config with auth ON, one admin user, and a given public path list.

        Built through YAML rather than by mutation: `AuthConfig` is frozen, and
        a test that reached around that would stop exercising the real loading
        path -- which is where a bad auth block should be caught.
        """
        from chatbot_core.auth import new_user

        paths = "\n".join(f"      - {p}" for p in public_paths)
        path = tmp_path / "config.yaml"
        path.write_text(
            CONFIG
            + f"""
auth:
  enabled: true
  mode: api_key
  default_role: admin
  public_paths:
{paths}
  anonymous_audiences: [public]
  roles:
    admin:
      description: Admin
      permissions: [chat, history, admin, ingest, manage_users]
      audiences: [public, internal]
"""
        )
        cfg = ProjectConfig.load(path)
        user, key = new_user("root", roles={"admin"})
        return cfg, UserStore([user]), key

    def _client(self, config, users):
        from api.settings import APISettings

        settings = APISettings(project="test-bot")
        engine = StubEngine(config)
        app = _app_with_stub_users(settings, config, engine, users)
        return TestClient(app), engine

    def test_chat_is_401_when_not_declared_public(self, tmp_path):
        cfg, users, _ = self._authed_config(tmp_path, ["/health"])
        client, _ = self._client(cfg, users)
        with client:
            assert client.post("/chat", json={"message": "hi"}).status_code == 401

    def test_chat_is_anonymous_when_declared_public(self, tmp_path):
        cfg, users, _ = self._authed_config(tmp_path, ["/health", "/chat"])
        client, engine = self._client(cfg, users)
        with client:
            assert client.post("/chat", json={"message": "hi"}).status_code == 200
        # Anonymous, so the turn gets `anonymous_audiences` -- not admin's.
        assert engine.roles == [None]

    def test_a_valid_key_still_authenticates_on_a_public_path(self, tmp_path):
        """A signed-in customer must keep their role on an open endpoint."""
        cfg, users, key = self._authed_config(tmp_path, ["/health", "/chat"])
        client, engine = self._client(cfg, users)
        with client:
            response = client.post(
                "/chat", json={"message": "hi"}, headers={"Authorization": f"Bearer {key}"}
            )
        assert response.status_code == 200
        assert engine.roles == [frozenset({"admin"})]

    def test_a_bad_key_is_rejected_even_on_a_public_path(self, tmp_path):
        """A revoked key must fail loudly, not silently drop to anonymous.

        Otherwise a user keeps working with quietly reduced access and never
        learns their credential died.
        """
        cfg, users, _ = self._authed_config(tmp_path, ["/health", "/chat"])
        client, _ = self._client(cfg, users)
        with client:
            response = client.post(
                "/chat",
                json={"message": "hi"},
                headers={"Authorization": "Bearer mcp_bogus"},
            )
        assert response.status_code == 401

    def test_public_paths_do_not_open_privileged_endpoints(self, tmp_path):
        """Listing /chat must not leak into admin routes."""
        cfg, users, _ = self._authed_config(tmp_path, ["/health", "/chat"])
        client, _ = self._client(cfg, users)
        with client:
            assert client.get("/admin/overview").status_code == 401

    def test_matching_is_exact_not_prefix(self, tmp_path):
        """`/chat` must not open a different route sharing its prefix."""
        cfg, users, _ = self._authed_config(tmp_path, ["/chat"])
        client, _ = self._client(cfg, users)
        with client:
            # Same prefix, different route: must still demand credentials.
            assert client.get("/conversations/abc").status_code == 401


def _app_with_stub_users(settings, config, engine, users):
    """As _app_with_stub, but with a populated user store."""
    from contextlib import asynccontextmanager

    from fastapi import FastAPI

    from api.app import _register_routes

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.app_state = AppState(
            settings=settings,
            config=config,
            engine=engine,
            conversations=InMemoryConversationStore(),
            limiter=SlidingWindowLimiter(config.limits.requests_per_minute),
            users=users,
            usage=UsageStats(),
        )
        await engine.startup()
        yield
        await engine.shutdown()

    app = FastAPI(lifespan=lifespan)
    _register_routes(app)
    # Admin routes are included separately by create_app, so a helper that only
    # calls _register_routes would 404 on them and quietly pass any test
    # asserting they are protected.
    app.include_router(admin_router)
    return app


class TestConversationsRequireHistory:
    """Transcripts are not public just because their id is hard to guess.

    Before this was enforced, `GET`/`DELETE /conversations/{id}` carried no
    auth dependency at all: with auth enabled, anyone holding an id could read
    a customer's full transcript, or delete it. Ids leak through logs, referrer
    headers and shared links, so obscurity was the only thing protecting them.
    """

    def test_reading_a_transcript_needs_credentials(self, tmp_path):
        cfg, users, _ = TestPublicPaths._authed_config(tmp_path, ["/chat"])
        client, _ = TestPublicPaths()._client(cfg, users)
        with client:
            assert client.get("/conversations/abc").status_code == 401

    def test_deleting_a_transcript_needs_credentials(self, tmp_path):
        cfg, users, _ = TestPublicPaths._authed_config(tmp_path, ["/chat"])
        client, _ = TestPublicPaths()._client(cfg, users)
        with client:
            assert client.delete("/conversations/abc").status_code == 401

    def test_a_permitted_user_can_read_their_transcript(self, tmp_path):
        """The gate must not lock out the role that is supposed to have it."""
        cfg, users, key = TestPublicPaths._authed_config(tmp_path, ["/chat"])
        client, _ = TestPublicPaths()._client(cfg, users)
        auth = {"Authorization": f"Bearer {key}"}
        with client:
            frames = parse_sse(
                client.post("/chat", json={"message": "hi"}, headers=auth).text
            )
            cid = next(p for n, p in frames if n == "message_start")["conversation_id"]
            assert client.get(f"/conversations/{cid}", headers=auth).status_code == 200

    def test_still_reachable_when_the_project_has_no_auth(self, config):
        """With auth off every caller is anonymous and the service is already
        open, so gating this would kill it in local dev while protecting
        nothing. `assert_safe_to_start` is what keeps that config off a server."""
        client, _ = build_client(config)
        with client:
            frames = parse_sse(client.post("/chat", json={"message": "hi"}).text)
            cid = next(p for n, p in frames if n == "message_start")["conversation_id"]
            assert client.get(f"/conversations/{cid}").status_code == 200
