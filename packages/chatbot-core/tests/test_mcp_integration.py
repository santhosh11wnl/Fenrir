"""End-to-end tool chain: MCPClient -> MCPServer -> tool -> back.

Runs the real mcp 2.x client against a real in-process server, so it catches
the class of bug that pure mocking hides -- wrong field names, wrong result
shapes, renamed APIs between SDK versions. No subprocess and no bound port, so
it stays fast enough to run on every commit.
"""

from __future__ import annotations

import pytest

from chatbot_core.config import MCPConfig, MCPTransport, ToolFilter
from chatbot_core.engine import CompositeExecutor
from chatbot_core.mcp import MCPClient

pytest.importorskip("mcp.server.mcpserver", reason="requires mcp>=2")


@pytest.fixture
def server():
    from mcp.server.mcpserver import MCPServer

    srv = MCPServer("test-server")

    @srv.tool()
    def add(a: int, b: int) -> int:
        """Add two integers."""
        return a + b

    @srv.tool()
    def greet(name: str) -> str:
        """Greet someone by name."""
        return f"Hello, {name}!"

    @srv.tool()
    def explode() -> str:
        """Always raises, to exercise the failure path."""
        raise RuntimeError("tool blew up")

    @srv.tool()
    def no_args() -> str:
        """Takes no arguments."""
        return "ok"

    return srv


def config(**kwargs) -> MCPConfig:
    return MCPConfig(
        enabled=True, transport=MCPTransport.HTTP, url="http://unused.test/mcp", **kwargs
    )


async def connected(server, **kwargs) -> MCPClient:
    client = MCPClient(config(**kwargs), server=server)
    await client.connect()
    return client


async def test_discovers_tools(server):
    client = await connected(server)
    try:
        assert client.connected
        assert {s.name for s in client.specs} == {"add", "greet", "explode", "no_args"}
    finally:
        await client.aclose()


async def test_specs_carry_usable_schemas(server):
    """Both Anthropic and OpenAI-compatible tool APIs reject a schema without
    `type` and `properties`, which servers omit for zero-argument tools."""
    client = await connected(server)
    try:
        by_name = {s.name: s for s in client.specs}
        assert by_name["add"].description == "Add two integers."
        assert by_name["add"].input_schema["type"] == "object"
        assert set(by_name["add"].input_schema["properties"]) == {"a", "b"}
        assert by_name["no_args"].input_schema["type"] == "object"
        assert by_name["no_args"].input_schema["properties"] == {}
    finally:
        await client.aclose()


async def test_executes_tool(server):
    client = await connected(server)
    try:
        outcome = await client.execute("add", {"a": 2, "b": 3})
        assert not outcome.is_error
        assert "5" in outcome.content
    finally:
        await client.aclose()


async def test_tool_failure_is_an_outcome_not_an_exception(server):
    """A failing tool must come back as a result the model can react to --
    raising would end the turn and strand the user."""
    client = await connected(server)
    try:
        outcome = await client.execute("explode", {})
        assert outcome.is_error
        assert outcome.content
    finally:
        await client.aclose()


async def test_unknown_tool_is_an_outcome(server):
    client = await connected(server)
    try:
        assert (await client.execute("nonexistent", {})).is_error
    finally:
        await client.aclose()


class TestFiltering:
    async def test_deny_hides_tool_from_discovery(self, server):
        client = await connected(server, tools=ToolFilter(allow=["*"], deny=["explode"]))
        try:
            assert "explode" not in {s.name for s in client.specs}
        finally:
            await client.aclose()

    async def test_denied_tool_is_rejected_at_call_time(self, server):
        """Discovery-time filtering alone is bypassable if the server's tool
        list changes mid-session, so the check is repeated on execute."""
        client = await connected(server, tools=ToolFilter(allow=["*"], deny=["explode"]))
        try:
            outcome = await client.execute("explode", {})
            assert outcome.is_error
            assert "not available" in outcome.content
        finally:
            await client.aclose()

    async def test_allowlist_restricts_to_named_tools(self, server):
        client = await connected(server, tools=ToolFilter(allow=["add", "greet"]))
        try:
            assert {s.name for s in client.specs} == {"add", "greet"}
        finally:
            await client.aclose()


class TestDisconnectedBehaviour:
    async def test_failed_connect_degrades_instead_of_raising(self):
        """An assistant that still answers from its corpus beats one that
        won't start because a tool backend is down."""
        client = MCPClient(
            MCPConfig(
                enabled=True,
                transport=MCPTransport.HTTP,
                url="http://127.0.0.1:1/mcp",  # nothing listens on port 1
            )
        )
        await client.connect()
        assert not client.connected
        assert list(client.specs) == []
        outcome = await client.execute("anything", {})
        assert outcome.is_error

    async def test_disabled_client_never_connects(self, server):
        client = MCPClient(MCPConfig(enabled=False), server=server)
        await client.connect()
        assert not client.connected


class TestCompositeExecutor:
    async def test_merges_mcp_tools_with_a_local_tool(self, server):
        from chatbot_core.providers.base import ToolOutcome, ToolSpec

        class LocalTool:
            @property
            def specs(self):
                return (
                    ToolSpec(
                        name="local_echo",
                        description="Echo.",
                        input_schema={"type": "object", "properties": {}},
                    ),
                )

            async def execute(self, name, arguments):
                return ToolOutcome(content="echoed")

        client = await connected(server)
        try:
            composite = CompositeExecutor(client, LocalTool())
            names = {s.name for s in composite.specs}
            assert "add" in names and "local_echo" in names
            assert (await composite.execute("add", {"a": 1, "b": 1})).content
            assert (await composite.execute("local_echo", {})).content == "echoed"
        finally:
            await client.aclose()

    async def test_unknown_tool_lists_what_is_available(self):
        """The model can recover from a wrong name if told the right ones."""
        composite = CompositeExecutor()
        outcome = await composite.execute("ghost", {})
        assert outcome.is_error
        assert "Available tools" in outcome.content
