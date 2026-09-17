"""Service settings.

Deployment-shaped values only. Anything about *how the assistant behaves* lives
in the project's ``config.yaml``, not here -- that split is what lets one image
serve seven projects by changing a single environment variable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class APISettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    #: Which project this process serves. Must name a directory under
    #: ``projects_dir``. One container per project.
    project: str = "_template"
    projects_dir: Path = Path("projects")

    host: str = Field(default="0.0.0.0", alias="API_HOST")
    port: int = Field(default=8000, ge=1, le=65535, alias="API_PORT")

    #: Browser origins allowed to call the chat API. Comma-separated.
    #: Never "*" with credentials -- that combination is rejected by browsers
    #: and would be an open door if it weren't.
    #:
    #: ``NoDecode`` is load-bearing: without it pydantic-settings tries to
    #: JSON-parse any env value destined for a complex type *before* validators
    #: run, so a plain comma-separated list raises a JSONDecodeError at startup.
    #: Requiring operators to write a JSON array in an env var would be a poor
    #: trade, so decoding is disabled and `_split_csv` below does the work.
    cors_origins: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["http://localhost:5173"]
    )

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_format: Literal["console", "json"] = "console"

    #: Conversation retention for the in-memory store. See the note in
    #: chatbot_core.storage about why this is single-replica only.
    max_conversations: int = Field(default=1000, ge=1)
    conversation_ttl_seconds: float = Field(default=86_400, ge=0)

    #: Postgres DSN for the durable turn log behind the metrics, filter and
    #: cohort views. Unset disables analytics entirely: the API still serves
    #: chat normally and the dashboard's metrics panel reports that it has no
    #: store, which is the honest failure rather than a page of zeroes.
    analytics_database_url: str = ""

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_csv(cls, value: object) -> object:
        if isinstance(value, str):
            return [o.strip() for o in value.split(",") if o.strip()]
        return value

    @property
    def project_dir(self) -> Path:
        return self.projects_dir / self.project

    @property
    def config_path(self) -> Path:
        return self.project_dir / "config.yaml"


__all__ = ["APISettings"]
