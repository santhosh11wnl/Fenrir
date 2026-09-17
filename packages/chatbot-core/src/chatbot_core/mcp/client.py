"""MCP client -- discovers and runs tools from an MCP server.

Tools are discovered at connect time rather than declared in config, so adding
a tool to the server makes it available to every project whose filter permits
it, with no redeploy of the chat service.

Written against **mcp 2.x**. The v2 ``Client`` takes the server target directly
-- a URL string for HTTP, ``StdioServerParameters`` for a subprocess, or an
``MCPServer`` instance for in-process testing -- so there is no transport
plumbing here. (v1's ``stdio_client`` / ``streamablehttp_client`` helpers and
the camelCase ``inputSchema`` / ``isError`` fields are gone; see
``pyproject.toml`` for the version floor that guarantees this shape.)
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

import structlog

from ..config import MCPConfig, MCPTransport
from ..providers.base import ToolOutcome, ToolSpec

log = structlog.get_logger(__name__)


class MCPClient:
    """A connected MCP session exposing the ToolExecutor interface.

    One instance belongs to one application lifespan and one event loop.
    Concurrent *calls* within that loop are fine -- the session multiplexes
    them -- but the instance is not shareable across loops.
    """

    def __init__(self, config: MCPConfig, server: Any | None = None) -> None:
        """
        Args:
            config: The project's MCP settings.
            server: Overrides the target built from ``config``. Injecting an
                in-process ``MCPServer`` here is what lets the tool pipeline be
                tested without spawning a subprocess or binding a port.
        """
        self._config = config
        self._server_override = server
        self._client: Any = None
        self._specs: list[ToolSpec] = []
        self._connect_lock = asyncio.Lock()

    @property
    def specs(self) -> Sequence[ToolSpec]:
        return self._specs

    @property
    def connected(self) -> bool:
        return self._client is not None

    # -- lifecycle ---------------------------------------------------------

    async def connect(self) -> None:
        """Open the session and discover tools. Idempotent.

        A failure here is logged and swallowed rather than raised: an assistant
        that can still answer from its corpus is far better than one that
        refuses to start because a tool backend is down.
        """
        async with self._connect_lock:
            if self._client is not None or not self._config.enabled:
                return
            try:
                from mcp import Client

                client = Client(self._server_target())
                await client.__aenter__()
                self._client = client
                await self.refresh_tools()
                log.info(
                    "mcp_connected",
                    transport=self._config.transport.value,
                    tools=[s.name for s in self._specs],
                )
            except Exception:  # noqa: BLE001 - degraded start beats no start
                log.exception(
                    "mcp_connect_failed", transport=self._config.transport.value
                )
                self._client = None
                self._specs = []

    def _server_target(self) -> Any:
        """Resolve what the v2 ``Client`` should connect to."""
        if self._server_override is not None:
            return self._server_override
        if self._config.transport is MCPTransport.STDIO:
            from mcp import StdioServerParameters

            return StdioServerParameters(
                command=self._config.command[0],
                args=list(self._config.command[1:]),
                env=dict(self._config.env) or None,
            )
        assert self._config.url  # guaranteed by MCPConfig validation
        return self._config.url

    async def aclose(self) -> None:
        if self._client is not None:
            try:
                await self._client.__aexit__(None, None, None)
            except Exception:  # noqa: BLE001 - shutdown must not raise
                log.exception("mcp_close_failed")
        self._client = None
        self._specs = []

    # -- discovery ---------------------------------------------------------

    async def refresh_tools(self) -> None:
        """Re-read the server's tool list and re-apply the project's filter."""
        if self._client is None:
            self._specs = []
            return

        listing = await self._client.list_tools()
        allowed: list[ToolSpec] = []
        skipped: list[str] = []

        for tool in listing.tools:
            if not self._config.tools.permits(tool.name):
                skipped.append(tool.name)
                continue
            allowed.append(
                ToolSpec(
                    name=tool.name,
                    description=tool.description or f"MCP tool: {tool.name}",
                    input_schema=_normalise_schema(tool.input_schema),
                )
            )

        self._specs = allowed
        if skipped:
            log.info("mcp_tools_filtered", skipped=skipped)

    # -- execution ---------------------------------------------------------

    async def execute(self, name: str, arguments: dict[str, Any]) -> ToolOutcome:
        if self._client is None:
            return ToolOutcome(
                content="The tool backend is unavailable right now.", is_error=True
            )
        # Re-check at call time. Filtering only at discovery would be
        # bypassable if the server's tool list changed mid-session.
        if not self._config.tools.permits(name):
            log.warning("mcp_tool_denied", tool=name)
            return ToolOutcome(content=f"Tool {name!r} is not available.", is_error=True)

        try:
            result = await self._client.call_tool(name, arguments)
        except Exception as exc:  # noqa: BLE001 - a tool must not kill the turn
            log.exception("mcp_tool_call_failed", tool=name)
            return ToolOutcome(content=f"Tool {name!r} failed: {exc}", is_error=True)

        return ToolOutcome(
            content=_render_content(result.content),
            is_error=bool(getattr(result, "is_error", False)),
        )


def _normalise_schema(schema: dict[str, Any] | None) -> dict[str, Any]:
    """Coerce an MCP input schema into the shape the chat APIs expect.

    Servers legitimately omit ``type`` or ``properties`` for zero-argument
    tools, but both Anthropic and OpenAI-compatible tool APIs reject a schema
    without them.
    """
    base = dict(schema or {})
    base.setdefault("type", "object")
    base.setdefault("properties", {})
    return base


def _render_content(blocks: Any) -> str:
    """Flatten MCP content blocks into text for the model.

    Non-text blocks are summarised rather than dropped: the model needs to know
    a tool returned an image, even though it can't be inlined here.
    """
    if not blocks:
        return "(the tool returned no content)"
    parts: list[str] = []
    for block in blocks:
        kind = getattr(block, "type", None)
        if kind == "text":
            parts.append(getattr(block, "text", ""))
        elif kind == "resource":
            resource = getattr(block, "resource", None)
            text = getattr(resource, "text", None)
            parts.append(text or f"(resource: {getattr(resource, 'uri', 'unknown')})")
        else:
            parts.append(f"({kind or 'unknown'} content omitted)")
    return "\n".join(p for p in parts if p) or "(the tool returned no content)"


__all__ = ["MCPClient"]
