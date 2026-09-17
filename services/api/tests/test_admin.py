"""Admin route tests.

The assertions that matter are the negative ones: that a `customer` cannot
reach an admin endpoint, that a `support` user cannot list users, and that with
auth disabled the admin surface is closed rather than open. A dashboard that
quietly serves anyone is worse than no dashboard.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.admin import UsageStats
from api.admin import router as admin_router
from api.app import AppState, _register_routes
from api.ratelimit import SlidingWindowLimiter
from api.settings import APISettings
from chatbot_core import InMemoryConversationStore, ProjectConfig
from chatbot_core.auth import UserStore, new_user

from .test_api import StubEngine

CONFIG_WITH_AUTH = """
project:
  id: site-bot
  name: Site Bot
system_prompt: You are a test assistant.
retrieval:
  enabled: false
mcp:
  enabled: false
auth:
  enabled: true
  roles:
    admin:
      description: Full access
      permissions: [chat, history, admin, ingest, manage_users]
    support:
      description: Support staff
      permissions: [chat, history]
    customer:
      description: End user
      permissions: [chat]
  default_role: customer
"""

CONFIG_NO_AUTH = """
project:
  id: site-bot
  name: Site Bot
system_prompt: You are a test assistant.
retrieval:
  enabled: false
mcp:
  enabled: false
"""


def build(tmp_path, config_text: str):
    """Build the real app with a stub engine and a known set of users."""
    path = tmp_path / "config.yaml"
    path.write_text(config_text)
    config = ProjectConfig.load(path)

    keys: dict[str, str] = {}
    people = []
    for user_id, roles in [
        ("alice", {"admin"}),
        ("bob", {"support"}),
        ("carol", {"customer"}),
    ]:
        user, key = new_user(user_id, roles=roles)
        keys[user_id] = key
        people.append(user)

    engine = StubEngine(config)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.app_state = AppState(
            settings=APISettings(project="site-bot"),
            config=config,
            engine=engine,
            conversations=InMemoryConversationStore(),
            limiter=SlidingWindowLimiter(config.limits.requests_per_minute),
            users=UserStore(people),
            usage=UsageStats(),
        )
        await engine.startup()
        yield
        await engine.shutdown()

    app = FastAPI(lifespan=lifespan)
    _register_routes(app)
    app.include_router(admin_router)
    return TestClient(app), keys


def auth(key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}"}


@pytest.fixture
def secured(tmp_path):
    return build(tmp_path, CONFIG_WITH_AUTH)


class TestOverview:
    def test_admin_can_read_it(self, secured):
        client, keys = secured
        with client:
            response = client.get("/admin/overview", headers=auth(keys["alice"]))
        assert response.status_code == 200
        body = response.json()
        assert body["project"] == "site-bot"
        assert body["auth_enabled"] is True
        assert "turns" in body["usage"]

    def test_support_is_refused(self, secured):
        """`support` has history but not admin."""
        client, keys = secured
        with client:
            response = client.get("/admin/overview", headers=auth(keys["bob"]))
        assert response.status_code == 403
        assert "admin" in response.json()["detail"]

    def test_customer_is_refused(self, secured):
        client, keys = secured
        with client:
            assert (
                client.get("/admin/overview", headers=auth(keys["carol"])).status_code
                == 403
            )

    def test_no_credentials_is_401(self, secured):
        client, _ = secured
        with client:
            response = client.get("/admin/overview")
        assert response.status_code == 401
        assert response.headers.get("www-authenticate") == "Bearer"

    def test_invalid_key_is_401(self, secured):
        client, _ = secured
        with client:
            assert client.get("/admin/overview", headers=auth("mcp_bogus")).status_code == 401


class TestUsers:
    def test_manage_users_permission_required(self, secured):
        """Seeing who has access is a higher bar than seeing service health --
        `support` can read history but must not enumerate users."""
        client, keys = secured
        with client:
            assert client.get("/admin/users", headers=auth(keys["bob"])).status_code == 403
            assert client.get("/admin/users", headers=auth(keys["alice"])).status_code == 200

    def test_never_exposes_credential_material(self, secured):
        client, keys = secured
        with client:
            body = client.get("/admin/users", headers=auth(keys["alice"])).text
        assert "key_hash" not in body
        assert "scrypt" not in body

    def test_reports_effective_permissions(self, secured):
        client, keys = secured
        with client:
            users = client.get("/admin/users", headers=auth(keys["alice"])).json()
        by_id = {u["id"]: u for u in users}
        assert by_id["carol"]["permissions"] == ["chat"]
        assert set(by_id["bob"]["permissions"]) == {"chat", "history"}


class TestRoles:
    def test_lists_project_roles_with_counts(self, secured):
        client, keys = secured
        with client:
            roles = client.get("/admin/roles", headers=auth(keys["alice"])).json()
        by_name = {r["name"]: r for r in roles}
        assert set(by_name) == {"admin", "support", "customer"}
        assert by_name["support"]["permissions"] == ["chat", "history"]
        assert by_name["admin"]["user_count"] == 1


class TestConversations:
    def test_history_permission_is_enough(self, secured):
        """Unlike /admin/users, support staff legitimately need this."""
        client, keys = secured
        with client:
            assert (
                client.get("/admin/conversations", headers=auth(keys["bob"])).status_code
                == 200
            )

    def test_customer_is_refused(self, secured):
        client, keys = secured
        with client:
            assert (
                client.get(
                    "/admin/conversations", headers=auth(keys["carol"])
                ).status_code
                == 403
            )


class TestReloadUsers:
    def test_requires_manage_users(self, secured):
        client, keys = secured
        with client:
            assert (
                client.post("/admin/reload-users", headers=auth(keys["bob"])).status_code
                == 403
            )
            assert (
                client.post(
                    "/admin/reload-users", headers=auth(keys["alice"])
                ).status_code
                == 204
            )


class TestAuthDisabled:
    """With auth off there is no way to identify a caller, so privileged
    endpoints must fail shut rather than serve everyone."""

    def test_admin_is_closed_not_open(self, tmp_path):
        client, _ = build(tmp_path, CONFIG_NO_AUTH)
        with client:
            response = client.get("/admin/overview")
        assert response.status_code == 403
        assert "authentication" in response.json()["detail"].lower()

    def test_chat_still_works_anonymously(self, tmp_path):
        """Disabling auth is a deliberate local-development choice; it should
        not break the thing the service exists to do."""
        client, _ = build(tmp_path, CONFIG_NO_AUTH)
        with client:
            assert client.post("/chat", json={"message": "hi"}).status_code == 200


class TestUsageStats:
    def test_counts_turns_and_tokens(self):
        class FakeUsage:
            input_tokens = 100
            output_tokens = 20
            cache_read_input_tokens = 50

        stats = UsageStats()
        stats.record_turn(duration=2.0, usage=FakeUsage(), tools=1, failed=False)
        stats.record_turn(duration=4.0, usage=FakeUsage(), tools=0, failed=True)

        snapshot = stats.snapshot()
        assert snapshot["turns"] == 2
        assert snapshot["errors"] == 1
        assert snapshot["tool_calls"] == 1
        assert snapshot["input_tokens"] == 200
        assert snapshot["mean_seconds_per_turn"] == 3.0
        assert snapshot["error_rate"] == 0.5

    def test_empty_snapshot_does_not_divide_by_zero(self):
        assert UsageStats().snapshot()["mean_seconds_per_turn"] == 0.0
