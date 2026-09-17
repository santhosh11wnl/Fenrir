"""Server construction.

One factory builds the server for every transport. The transport is chosen at
the edge (``__main__``), not baked in here, so the same tool code serves a
local subprocess under Claude Code and a container in production -- which is
what makes "it works in my editor" evidence that it works deployed.
"""

from __future__ import annotations

from typing import Any

import structlog

from .registry import register_directory, register_package
from .settings import ServerSettings

log = structlog.get_logger(__name__)

#: Built-in tools, always registered. Project tools are layered on top.
BUILTIN_TOOL_PACKAGE = "mcp_server_core.tools"


def build_server(settings: ServerSettings | None = None) -> Any:
    """Build an ``MCPServer`` with built-in and project tools registered.

    Args:
        settings: Configuration. Read from the environment when omitted.

    Returns:
        A configured ``mcp.server.mcpserver.MCPServer``, not yet running.

    Raises:
        ToolRegistrationError: A tool module failed to load. Deliberately fatal
            -- a server that starts without the tools it advertises produces an
            assistant that fails mysteriously at request time instead.
    """
    from mcp.server.mcpserver import MCPServer

    settings = settings or ServerSettings()
    server = MCPServer(
        name=settings.name,
        instructions=(
            "Tools for retrieving and acting on this project's data. Prefer a "
            "tool over guessing whenever a question depends on live or "
            "project-specific information."
        ),
        log_level=settings.log_level,
    )

    builtin = register_package(server, BUILTIN_TOOL_PACKAGE)

    project_modules: list[str] = []
    tools_dir = settings.project_tools_dir
    if tools_dir is not None:
        project_modules = register_directory(server, tools_dir)
        if not project_modules:
            # Not fatal: a project can legitimately rely only on retrieval.
            # But silence here leads to "why can't it look anything up?".
            log.warning(
                "no_project_tools_found",
                project=settings.project,
                path=str(tools_dir),
            )

    log.info(
        "server_built",
        name=settings.name,
        project=settings.project or "(none)",
        builtin_modules=builtin,
        project_modules=project_modules,
    )
    return server


__all__ = ["BUILTIN_TOOL_PACKAGE", "build_server"]
