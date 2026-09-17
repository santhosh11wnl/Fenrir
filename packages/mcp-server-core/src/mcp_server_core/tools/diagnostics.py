"""Diagnostic tools.

Present in every project so there is always a way to prove the chain --
chat API -> MCP client -> server -> tool -> back -- without depending on a
project's own data being ingested and correct. When an assistant "can't use
tools", this is the first thing to call.
"""

from __future__ import annotations

import os
import platform
import time
from datetime import UTC, datetime
from typing import Any

_STARTED_AT = time.time()


def register(server: Any) -> None:
    @server.tool()
    def ping(message: str = "ping") -> str:
        """Verify the tool pipeline is working end to end.

        Echoes a message back with a server timestamp. Use this when tool calls
        appear to be failing, to establish whether the problem is the pipeline
        or an individual tool.

        Args:
            message: Text to echo back.
        """
        now = datetime.now(UTC).isoformat(timespec="seconds")
        return f"pong: {message} (server time {now})"

    @server.tool()
    def server_info() -> str:
        """Report which server, project, and runtime is handling tool calls.

        Useful when several projects share a deployment and an assistant seems
        to be reaching the wrong tool set.
        """
        uptime = int(time.time() - _STARTED_AT)
        lines = [
            f"server:  {os.environ.get('MCP_SERVER_NAME', 'mcp-platform')}",
            f"project: {os.environ.get('MCP_SERVER_PROJECT') or '(none -- built-in tools only)'}",
            f"python:  {platform.python_version()} on {platform.system()}",
            f"uptime:  {uptime}s",
        ]
        return "\n".join(lines)
