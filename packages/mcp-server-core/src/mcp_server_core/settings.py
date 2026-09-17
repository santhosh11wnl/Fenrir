"""Server settings.

Environment-driven, because the server runs in three different shapes -- a
subprocess under Claude Code, a container in compose, a service in production --
and only the environment differs between them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

Transport = Literal["stdio", "streamable-http", "sse"]


class ServerSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="MCP_SERVER_",
        env_file=".env",
        extra="ignore",
    )

    name: str = "mcp-platform"
    #: Project whose tool directory to load. Empty means built-in tools only,
    #: which is the right default for a shared server.
    project: str = ""
    #: Where project tool modules live. Resolved relative to the repo root.
    projects_dir: Path = Path("projects")

    host: str = "0.0.0.0"
    port: int = Field(default=8765, ge=1, le=65535)
    #: Path the streamable-HTTP endpoint is mounted at. Clients connect to
    #: ``http://host:port{mount_path}``.
    mount_path: str = "/mcp"

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    @property
    def project_tools_dir(self) -> Path | None:
        if not self.project:
            return None
        return self.projects_dir / self.project / "tools"


__all__ = ["ServerSettings", "Transport"]
