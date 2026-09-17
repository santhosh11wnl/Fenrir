"""Control plane: the registry API and the per-site admin proxy.

Two jobs, and the second is the reason this service exists at all.

**Registry CRUD.** Sites and models are read from and written to
``registry.yaml``. The dashboard edits through here, files stay the source of
truth.

**Admin proxy.** Every site's admin key is resolved from this process's
environment and attached server-side, so the dashboard holds no credentials.
That replaces the previous arrangement, where ``VITE_ADMIN_PROJECTS`` shipped
admin keys for all seven sites inside a JavaScript bundle -- anyone who could
load the page had them. Here the browser authenticates to the control service
once, and the control service authenticates to each site.

The proxy is also what makes cross-site rollups possible without granting any
site visibility into another: this process calls seven isolated APIs and
aggregates the answers, rather than any API gaining a way to read its
neighbours.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
import structlog
from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

from . import registry as reg
from .settings import ControlSettings

log = structlog.get_logger(__name__)

#: Admin paths the dashboard may reach through the proxy.
#:
#: An allowlist rather than a blanket forward: this service holds keys with
#: `manage_users` on every site, so an open proxy would let anyone who reached
#: the dashboard call any admin endpoint on any site with the highest
#: privileges in the fleet. Adding a route here is a deliberate act.
_PROXY_ALLOW = frozenset(
    {
        "overview",
        "roles",
        "users",
        "conversations",
        "metrics",
        "breakdown",
        "cohorts",
        "export/turns.csv",
        "export/cohorts.csv",
        "export/report.md",
    }
)


# ---------------------------------------------------------------------------
# request/response models
# ---------------------------------------------------------------------------


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SiteCreate(_Strict):
    id: str = Field(pattern=reg.PROJECT_ID)
    label: str = ""
    base_url: str
    model: str = Field(default="default", pattern=reg.MODEL_KEY)
    admin_key_env: str | None = None
    widget_origin: str = ""
    description: str = ""
    tags: list[str] = Field(default_factory=list)
    #: Stamp ``projects/<id>/`` from the template as part of adding the site.
    #: Off for a site whose directory already exists, which is how an existing
    #: deployment gets adopted into the registry without clobbering its config.
    scaffold: bool = True


class SiteUpdate(_Strict):
    label: str | None = None
    base_url: str | None = None
    model: str | None = Field(default=None, pattern=reg.MODEL_KEY)
    admin_key_env: str | None = None
    widget_origin: str | None = None
    enabled: bool | None = None
    tags: list[str] | None = None


class ModelUpsert(_Strict):
    label: str = ""
    base_url: str
    id: str
    supports_tools: bool = False
    max_tokens: int = Field(default=4096, ge=256, le=128_000)
    api_key_env: str | None = None
    notes: str = ""


class ConfigPatch(_Strict):
    """A partial project config, deep-merged over what is on disk."""

    config: dict[str, Any]


# ---------------------------------------------------------------------------
# app
# ---------------------------------------------------------------------------


def create_app(settings: ControlSettings | None = None) -> FastAPI:
    settings = settings or ControlSettings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        _assert_safe_to_start(settings)
        # One client for the process: a new connection pool per proxied
        # request would defeat keep-alive to every site and add a TLS
        # handshake to each of the dashboard's polls.
        app.state.http = httpx.AsyncClient(timeout=settings.site_timeout_seconds)
        app.state.settings = settings
        # Serialises registry writes. Two dashboard tabs saving at once would
        # otherwise read-modify-write the same file and silently drop one
        # edit -- the atomic replace in `registry.save` makes each write whole,
        # not each read-modify-write sequence.
        app.state.write_lock = asyncio.Lock()
        log.info(
            "control_ready", registry=str(settings.registry_path), port=settings.port
        )
        try:
            yield
        finally:
            await app.state.http.aclose()

    app = FastAPI(
        title="Control plane",
        description="Site registry, project config, and the per-site admin proxy.",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PATCH", "DELETE"],
        allow_headers=["Content-Type", "Authorization"],
    )
    _register_routes(app)
    return app


def _assert_safe_to_start(settings: ControlSettings) -> None:
    """Refuse to bind an unauthenticated control plane to a public interface.

    Mirrors the API's ``assert_safe_to_start``. This service is the more
    dangerous of the two -- it writes config and holds every site's admin key
    -- so the check is stricter: no token means localhost only, with no
    environment-variable escape hatch.
    """
    if settings.control_token:
        return
    if settings.host in {"127.0.0.1", "localhost", "::1"}:
        log.warning("control_token_unset", detail="bound to localhost only")
        return
    raise RuntimeError(
        f"CONTROL_TOKEN is unset and CONTROL_HOST is {settings.host!r}. "
        f"This service writes project config and holds every site's admin key; "
        f"it must not listen on a non-loopback address unauthenticated. "
        f"Set CONTROL_TOKEN, or bind to 127.0.0.1."
    )


def require_token(
    request: Request,
    authorization: str | None = Header(default=None),
) -> None:
    """Gate every route on the shared secret.

    Compared with ``secrets.compare_digest`` rather than ``==``: token checks
    are the textbook case for a timing side channel, and the fix costs
    nothing.
    """
    import secrets

    expected: str = request.app.state.settings.control_token
    if not expected:
        return  # localhost-only mode; see _assert_safe_to_start.

    supplied = ""
    if authorization and authorization.lower().startswith("bearer "):
        supplied = authorization[7:].strip()
    if not supplied or not secrets.compare_digest(supplied, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Control token missing or invalid.",
            headers={"WWW-Authenticate": "Bearer"},
        )


RequireToken = Depends(require_token)


def _settings(request: Request) -> ControlSettings:
    return request.app.state.settings  # type: ignore[no-any-return]


def _load(request: Request) -> reg.Registry:
    try:
        return reg.load(_settings(request).registry_path)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"registry.yaml could not be read: {exc}",
        ) from exc


def _register_routes(app: FastAPI) -> None:  # noqa: C901 - route table
    # -- registry ---------------------------------------------------------

    @app.get("/health", tags=["ops"])
    async def health(request: Request) -> dict[str, Any]:
        """Unauthenticated liveness. Reports shape, never contents."""
        path = _settings(request).registry_path
        try:
            registry = reg.load(path)
            return {
                "status": "ok",
                "registry": str(path),
                "sites": len(registry.sites),
                "models": len(registry.models),
            }
        except Exception as exc:  # noqa: BLE001
            return {"status": "degraded", "registry": str(path), "error": str(exc)}

    @app.get("/registry", tags=["registry"])
    async def get_registry(
        request: Request, _: None = RequireToken
    ) -> dict[str, Any]:
        """The whole fleet: sites, models, and which keys actually resolve.

        ``admin_key_present`` is computed rather than returning the key, so
        the dashboard can flag a site whose env var is missing without the
        secret ever crossing the wire.
        """
        registry = _load(request)
        return {
            "models": {k: v.model_dump() for k, v in registry.models.items()},
            "sites": [
                {
                    **site.model_dump(),
                    "admin_key_present": bool(
                        site.admin_key_env and os.environ.get(site.admin_key_env)
                    ),
                }
                for site in registry.sites
            ],
        }

    @app.post("/registry/sites", status_code=status.HTTP_201_CREATED, tags=["registry"])
    async def add_site(
        payload: SiteCreate, request: Request, _: None = RequireToken
    ) -> dict[str, Any]:
        """Register a site, optionally stamping its project directory."""
        settings = _settings(request)
        async with request.app.state.write_lock:
            registry = _load(request)
            if registry.site(payload.id) is not None:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=f"site {payload.id!r} already registered",
                )
            if payload.model not in registry.models:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f"unknown model {payload.model!r}",
                )

            if payload.scaffold:
                _scaffold(settings, payload)

            registry.sites.append(
                reg.SiteEntry(
                    id=payload.id,
                    label=payload.label or payload.id,
                    base_url=payload.base_url.rstrip("/"),
                    model=payload.model,
                    admin_key_env=payload.admin_key_env,
                    widget_origin=payload.widget_origin,
                    tags=payload.tags,
                )
            )
            reg.save(settings.registry_path, registry)

        log.info("site_added", site=payload.id, scaffolded=payload.scaffold)
        return {"id": payload.id, "scaffolded": payload.scaffold}

    @app.patch("/registry/sites/{site_id}", tags=["registry"])
    async def update_site(
        site_id: str, payload: SiteUpdate, request: Request, _: None = RequireToken
    ) -> dict[str, Any]:
        settings = _settings(request)
        async with request.app.state.write_lock:
            registry = _load(request)
            site = registry.site(site_id)
            if site is None:
                raise HTTPException(status.HTTP_404_NOT_FOUND, f"no site {site_id!r}")

            changes = payload.model_dump(exclude_none=True)
            if "base_url" in changes:
                changes["base_url"] = changes["base_url"].rstrip("/")
            for key, value in changes.items():
                setattr(site, key, value)

            # validate_references inside save catches a model rename that
            # would orphan this site, before anything reaches disk.
            reg.save(settings.registry_path, registry)
        return site.model_dump()

    @app.delete(
        "/registry/sites/{site_id}",
        status_code=status.HTTP_204_NO_CONTENT,
        tags=["registry"],
    )
    async def remove_site(
        site_id: str, request: Request, _: None = RequireToken
    ) -> None:
        """Deregister a site. Leaves ``projects/<id>/`` untouched.

        Deliberately not a delete: removing a site from the dashboard is a
        routine, reversible act, and deleting a corpus and its config from
        under a running container is not. Remove the directory by hand if
        that is genuinely what you want.
        """
        settings = _settings(request)
        async with request.app.state.write_lock:
            registry = _load(request)
            if registry.site(site_id) is None:
                raise HTTPException(status.HTTP_404_NOT_FOUND, f"no site {site_id!r}")
            registry.sites = [s for s in registry.sites if s.id != site_id]
            reg.save(settings.registry_path, registry)
        log.info("site_removed", site=site_id)

    @app.put("/registry/models/{key}", tags=["registry"])
    async def upsert_model(
        key: str, payload: ModelUpsert, request: Request, _: None = RequireToken
    ) -> dict[str, Any]:
        """Add or replace a named model."""
        import re

        if not re.match(reg.MODEL_KEY, key):
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_ENTITY, f"invalid model key {key!r}"
            )
        settings = _settings(request)
        async with request.app.state.write_lock:
            registry = _load(request)
            registry.models[key] = reg.ModelEntry(**payload.model_dump())
            reg.save(settings.registry_path, registry)
        return registry.models[key].model_dump()

    @app.delete(
        "/registry/models/{key}",
        status_code=status.HTTP_204_NO_CONTENT,
        tags=["registry"],
    )
    async def remove_model(key: str, request: Request, _: None = RequireToken) -> None:
        settings = _settings(request)
        async with request.app.state.write_lock:
            registry = _load(request)
            if key not in registry.models:
                raise HTTPException(status.HTTP_404_NOT_FOUND, f"no model {key!r}")
            users = [s.id for s in registry.sites if s.model == key]
            if users:
                # Refuse rather than cascade: silently repointing sites at a
                # different model would change what seven assistants run on,
                # from a delete button.
                raise HTTPException(
                    status.HTTP_409_CONFLICT,
                    f"model {key!r} is used by {', '.join(users)}; repoint them first",
                )
            del registry.models[key]
            reg.save(settings.registry_path, registry)

    # -- project config ---------------------------------------------------

    @app.get("/sites/{site_id}/config", tags=["config"])
    async def get_config(
        site_id: str, request: Request, _: None = RequireToken
    ) -> dict[str, Any]:
        settings = _settings(request)
        try:
            return reg.read_project_config(settings.projects_dir, site_id)
        except FileNotFoundError as exc:
            raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc

    @app.patch("/sites/{site_id}/config", tags=["config"])
    async def update_config(
        site_id: str, payload: ConfigPatch, request: Request, _: None = RequireToken
    ) -> dict[str, Any]:
        """Deep-merge a partial config into the site's ``config.yaml``.

        Validated against ``ProjectConfig`` before anything is written, so a
        bad edit is a 422 rather than a site that fails its next restart.
        Merge rather than replace, because the dashboard sends only the
        section it edited and a whole-document PUT would drop every key the
        UI does not yet render.
        """
        settings = _settings(request)
        try:
            current = reg.read_project_config(settings.projects_dir, site_id)
        except FileNotFoundError as exc:
            raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc

        merged = _deep_merge(current, payload.config)

        try:
            from chatbot_core import ProjectConfig

            ProjectConfig.model_validate(merged)
        except ImportError:  # pragma: no cover - chatbot-core always installed
            log.warning("config_validation_skipped", reason="chatbot-core missing")
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"resulting config is invalid: {exc}",
            ) from exc

        async with request.app.state.write_lock:
            reg.write_project_config(settings.projects_dir, site_id, merged)
        return merged

    # -- admin proxy ------------------------------------------------------

    @app.api_route(
        "/sites/{site_id}/admin/{path:path}",
        methods=["GET", "POST"],
        tags=["proxy"],
    )
    async def proxy_admin(
        site_id: str, path: str, request: Request, _: None = RequireToken
    ) -> Response:
        """Forward one admin call to a site, attaching its key server-side."""
        if path not in _PROXY_ALLOW:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"admin path {path!r} is not proxied",
            )

        registry = _load(request)
        site = registry.site(site_id)
        if site is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"no site {site_id!r}")

        headers = {}
        if site.admin_key_env:
            key = os.environ.get(site.admin_key_env, "")
            if not key:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=(
                        f"{site.admin_key_env} is not set in the control "
                        f"service's environment, so {site_id!r} cannot be queried."
                    ),
                )
            headers["Authorization"] = f"Bearer {key}"

        client: httpx.AsyncClient = request.app.state.http
        try:
            upstream = await client.request(
                request.method,
                f"{site.base_url}/admin/{path}",
                params=dict(request.query_params),
                headers=headers,
                content=await request.body() if request.method == "POST" else None,
            )
        except httpx.RequestError as exc:
            # 502, not 500: the control plane is fine, the site is not, and
            # the dashboard renders those two states differently.
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"{site_id} unreachable: {exc.__class__.__name__}",
            ) from exc

        # Content-Disposition is forwarded so a CSV proxied through here still
        # downloads as a file rather than rendering in the tab.
        passthrough = {
            k: v
            for k, v in upstream.headers.items()
            if k.lower() in {"content-type", "content-disposition"}
        }
        return Response(
            content=upstream.content,
            status_code=upstream.status_code,
            headers=passthrough,
        )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``patch`` into a copy of ``base``.

    Lists replace rather than concatenate. Appending would make it impossible
    to remove a suggestion or an audience through the dashboard, and a config
    edit that can only ever add is worse than one that replaces wholesale.
    """
    result = dict(base)
    for key, value in patch.items():
        existing = result.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            result[key] = _deep_merge(existing, value)
        else:
            result[key] = value
    return result


def _scaffold(settings: ControlSettings, payload: SiteCreate) -> None:
    """Stamp ``projects/<id>/`` by invoking the existing new_project script.

    A subprocess rather than an import: ``scripts/new_project.py`` is a CLI,
    not a library, and shelling out keeps one implementation of what a new
    project contains instead of a second copy that drifts.
    """
    target = reg.project_dir(settings.projects_dir, payload.id)
    if target.exists():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"{target} already exists. Register it with scaffold=false to "
                f"adopt it without overwriting."
            ),
        )
    script = settings.new_project_script
    if not script.exists():
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"scaffold script not found at {script}",
        )

    command = [
        sys.executable,
        str(script),
        payload.id,
        "--name",
        payload.label or payload.id,
        "--description",
        payload.description,
    ]
    try:
        # No shell, argument list, validated id: nothing here is
        # string-interpolated into a command line.
        result = subprocess.run(  # noqa: S603
            command, capture_output=True, text=True, timeout=120, check=False
        )
    except subprocess.TimeoutExpired as exc:
        raise HTTPException(
            status.HTTP_504_GATEWAY_TIMEOUT, "scaffold timed out"
        ) from exc

    if result.returncode != 0:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"scaffold failed: {result.stderr.strip() or result.stdout.strip()}",
        )


__all__ = ["create_app"]
