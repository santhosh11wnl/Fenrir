"""Control service settings.

Deployment shape only, same split as the API's settings: what the fleet *is*
lives in ``registry.yaml``, not here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class ControlSettings(BaseSettings):
    # populate_by_name is load-bearing, not cosmetic. With a bare
    # `Field(alias="CONTROL_HOST")`, pydantic accepts *only* the alias, so
    # `ControlSettings(host=...)` from __main__ is silently dropped and
    # `--host` becomes a flag that does nothing. Allowing both means the env
    # var and the CLI override reach the same field.
    model_config = SettingsConfigDict(
        env_file=".env", extra="ignore", populate_by_name=True
    )

    projects_dir: Path = Path("projects")
    #: Defaults inside ``projects_dir`` so the fleet definition sits next to
    #: the projects it describes, and one volume mount covers both.
    registry_file: Path | None = None

    host: str = Field(default="0.0.0.0", alias="CONTROL_HOST")
    port: int = Field(default=8900, ge=1, le=65535, alias="CONTROL_PORT")

    cors_origins: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["http://localhost:5173"]
    )

    #: Shared secret the dashboard presents. This service writes YAML and can
    #: stamp new projects, so it is strictly more privileged than any single
    #: site's admin key -- it must never run open. `assert_safe_to_start`
    #: refuses to bind without one outside localhost.
    control_token: str = ""

    #: Where the project-stamping script lives, invoked to create a new site's
    #: directory. Configurable so a container can mount it elsewhere.
    new_project_script: Path = Path("scripts/new_project.py")

    #: How long to wait on a proxied admin call to a site. Short: the
    #: dashboard polls every 15s and a hung site must not stack up requests.
    site_timeout_seconds: float = Field(default=10.0, ge=1.0, le=120.0)

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_format: Literal["console", "json"] = "console"

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_csv(cls, value: object) -> object:
        if isinstance(value, str):
            return [o.strip() for o in value.split(",") if o.strip()]
        return value

    @property
    def registry_path(self) -> Path:
        return self.registry_file or (self.projects_dir / "registry.yaml")


__all__ = ["ControlSettings"]
