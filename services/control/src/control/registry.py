"""The site and model registry.

Files are the source of truth, not this process and not the dashboard's
browser tab. ``registry.yaml`` is what the control service reads on every
request and rewrites on every change, so the whole fleet's shape lives in one
reviewable, diffable, revertable file. A dashboard that held this in memory
would be infrastructure with no history and no rollback.

Two collections, because they answer different questions:

``models``
    How to reach a model server -- base url, model id, whether it supports
    tools. Defined once and referenced by name, so pointing seven sites at a
    new model server is one edit rather than seven.

``sites``
    One deployed assistant: which project directory it serves, where its API
    listens, and which named model it uses. Adding site number eight is an
    entry here plus a ``projects/<id>/`` directory -- no new code, no new
    repository, no new model.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import structlog
import yaml
from pydantic import BaseModel, ConfigDict, Field

log = structlog.get_logger(__name__)

#: Same shape ``ProjectMeta.id`` enforces in chatbot-core. Repeated rather than
#: imported because it also guards path construction here -- see `project_dir`.
PROJECT_ID = r"^[a-z][a-z0-9-]{1,48}[a-z0-9]$"

#: Model keys are referenced from sites and used as dict keys, never as paths.
MODEL_KEY = r"^[a-z][a-z0-9_-]{0,38}[a-z0-9]$"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModelEntry(_Strict):
    """Connection details for one model server."""

    label: str = ""
    #: OpenAI-compatible base url: vLLM, Ollama, TGI and llama.cpp all serve
    #: this shape, which is why a site only ever needs a url and an id.
    base_url: str
    id: str
    supports_tools: bool = False
    max_tokens: int = Field(default=4096, ge=256, le=128_000)
    #: Name of the environment variable holding the key, never the key. A
    #: registry file is meant to be committed; a secret in it would be too.
    api_key_env: str | None = None
    notes: str = ""


class SiteEntry(_Strict):
    """One deployed assistant."""

    id: str = Field(pattern=PROJECT_ID)
    label: str = ""
    #: Where this site's API answers, as the dashboard reaches it.
    base_url: str
    #: Which ``models`` key it uses. Validated on save -- a site pointing at a
    #: model that does not exist is the single most likely registry typo.
    model: str = Field(default="default", pattern=MODEL_KEY)
    #: Env var holding this site's admin key. Same reasoning as above: the
    #: dashboard resolves it server-side, so no key reaches the browser.
    admin_key_env: str | None = None
    #: Public origin the widget is embedded from, for the generated snippet.
    widget_origin: str = ""
    enabled: bool = True
    tags: list[str] = Field(default_factory=list)


class Registry(_Strict):
    models: dict[str, ModelEntry] = Field(default_factory=dict)
    sites: list[SiteEntry] = Field(default_factory=list)

    def site(self, site_id: str) -> SiteEntry | None:
        return next((s for s in self.sites if s.id == site_id), None)

    def validate_references(self) -> None:
        """Every site points at a model that exists, and ids are unique.

        Run before every write rather than only on load: the failure mode
        this prevents is a dashboard edit that saves cleanly and breaks a
        site's next deploy, which is far harder to trace back than a rejected
        save.
        """
        seen: set[str] = set()
        for entry in self.sites:
            if entry.id in seen:
                raise ValueError(f"duplicate site id: {entry.id!r}")
            seen.add(entry.id)
            if entry.model not in self.models:
                raise ValueError(
                    f"site {entry.id!r} references unknown model {entry.model!r}; "
                    f"known models: {sorted(self.models) or 'none'}"
                )


# ---------------------------------------------------------------------------
# persistence
# ---------------------------------------------------------------------------


def load(path: Path) -> Registry:
    """Read the registry, tolerating absence but not corruption.

    A missing file is an empty fleet, which is the correct reading on a fresh
    checkout. Malformed YAML raises -- silently serving an empty registry
    because of a stray tab would make every site vanish from the dashboard
    with no indication why.
    """
    if not path.exists():
        return Registry()
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a mapping at the top level")
    registry = Registry.model_validate(raw)
    registry.validate_references()
    return registry


def save(path: Path, registry: Registry) -> None:
    """Write the registry atomically.

    Temp file plus ``os.replace`` rather than a plain write: a crash midway
    through rewriting this file would otherwise leave truncated YAML, and the
    next load would take down the dashboard's view of every site at once.
    ``os.replace`` is atomic on POSIX and Windows alike.
    """
    registry.validate_references()
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = registry.model_dump(mode="json", exclude_defaults=False)
    body = yaml.safe_dump(payload, sort_keys=False, allow_unicode=True, width=100)

    # Same directory as the target, so the replace is a rename within one
    # filesystem rather than a cross-device copy (which is not atomic).
    handle, temp_path = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(_HEADER)
            stream.write(body)
            # fsync before replace: without it the rename can land before the
            # contents on a power loss, leaving an empty registry.
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, path)
    except BaseException:
        Path(temp_path).unlink(missing_ok=True)
        raise

    log.info("registry_saved", path=str(path), sites=len(registry.sites))


_HEADER = """\
# Site and model registry -- the source of truth for the fleet.
#
# Written by the control service when you edit the dashboard, and safe to edit
# by hand. Commit it: this file is the only place the shape of all seven sites
# is visible at once, and its history is your rollback.
#
# Secrets live in environment variables named here, never inline.
"""


def project_dir(projects_root: Path, site_id: str) -> Path:
    """Resolve a project directory, refusing anything outside the root.

    ``site_id`` reaches this from an HTTP path segment, so the pattern check
    is a security boundary rather than a formatting preference: it is what
    stops ``../../etc`` from becoming a config write. The realpath comparison
    afterwards is belt and braces, covering a symlinked project directory that
    points somewhere it should not.
    """
    import re

    if not re.match(PROJECT_ID, site_id):
        raise ValueError(f"invalid project id: {site_id!r}")

    root = projects_root.resolve()
    candidate = (root / site_id).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError(f"project path escapes {root}")
    return candidate


def read_project_config(projects_root: Path, site_id: str) -> dict[str, Any]:
    path = project_dir(projects_root, site_id) / "config.yaml"
    if not path.exists():
        raise FileNotFoundError(f"no config.yaml for project {site_id!r}")
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def write_project_config(
    projects_root: Path, site_id: str, config: dict[str, Any]
) -> Path:
    """Persist a project's config.yaml atomically.

    The caller is expected to have validated ``config`` against
    ``ProjectConfig`` first -- see ``app.update_config``. Writing an invalid
    config would take the site down on its next restart, at which point the
    dashboard that broke it is also the only tool that could show you why.
    """
    path = project_dir(projects_root, site_id) / "config.yaml"
    body = yaml.safe_dump(config, sort_keys=False, allow_unicode=True, width=100)

    handle, temp_path = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, path)
    except BaseException:
        Path(temp_path).unlink(missing_ok=True)
        raise

    log.info("project_config_saved", project=site_id, path=str(path))
    return path


__all__ = [
    "MODEL_KEY",
    "PROJECT_ID",
    "ModelEntry",
    "Registry",
    "SiteEntry",
    "load",
    "project_dir",
    "read_project_config",
    "save",
    "write_project_config",
]
