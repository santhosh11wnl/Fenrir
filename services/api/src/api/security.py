"""Request authentication.

Enforces the project's ``auth`` block. Two properties matter more than the
mechanics:

**Isolation.** The store loaded here belongs to one project, because the
process serves one project. There is no code path by which a key issued for
one website is checked against another website's users -- not because a check
forbids it, but because the other project's users are never loaded into this
process at all. Structural, not conditional.

**No accidental exposure.** Auth defaults to off so a freshly stamped project
runs locally with no setup. `assert_safe_to_start` is what keeps that default
from reaching a server: outside a local environment, an unauthenticated
project refuses to boot.
"""

from __future__ import annotations

import os
from pathlib import Path

import structlog
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from chatbot_core.auth import AuthMode, Permission, User, UserStore
from chatbot_core.config import ProjectConfig

log = structlog.get_logger(__name__)

#: Environments treated as local. Anything else is production for the purposes
#: of the start-up check below.
LOCAL_ENVIRONMENTS = frozenset({"local", "dev", "development", "test", "ci"})

# auto_error=False so a missing header reaches our handler, which can then
# distinguish "auth is off" from "credentials required".
_bearer = HTTPBearer(auto_error=False, description="Per-project API key")


def load_user_store(config: ProjectConfig, project_dir: Path) -> UserStore:
    """Load this project's users. Never another project's."""
    return UserStore.load(project_dir / config.auth.users_file)


def assert_safe_to_start(config: ProjectConfig, store: UserStore) -> None:
    """Refuse configurations that would expose a project unintentionally.

    Called during startup so the process dies loudly rather than serving an
    open endpoint. Both failures below are easy to create by accident and
    expensive to notice in production.
    """
    environment = os.environ.get("ENVIRONMENT", "local").lower()
    is_local = environment in LOCAL_ENVIRONMENTS

    if not config.auth.enabled:
        if config.auth.require_auth_in_production and not is_local:
            raise RuntimeError(
                f"auth is disabled but ENVIRONMENT={environment!r} is not local. "
                f"Refusing to start an unauthenticated chat endpoint.\n"
                f"  Enable auth:  set auth.enabled: true in the project config "
                f"and add users with scripts/manage_users.py\n"
                f"  Or override:  set auth.require_auth_in_production: false, "
                f"only if something in front of this enforces access."
            )
        log.warning(
            "auth_disabled",
            project=config.project.id,
            environment=environment,
            detail="every request is treated as anonymous",
        )
        return

    if len(store) == 0:
        # Auth on with nobody defined rejects every request. That is safe, but
        # it looks like a broken deployment, so say what it actually is.
        raise RuntimeError(
            f"auth is enabled for {config.project.id!r} but its user store is "
            f"empty -- every request would be rejected.\n"
            f"  Add a user:  uv run python scripts/manage_users.py add "
            f"{config.project.id} <user-id>"
        )

    log.info(
        "auth_enabled",
        project=config.project.id,
        mode=config.auth.mode.value,
        users=len(store),
        roles={
            name: sum(1 for u in store.users if u.has_role(name))
            for name in sorted(config.auth.roles)
        },
    )


def _unauthorized(detail: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        # Required by RFC 9110 on a 401, and it tells a client which scheme to use.
        headers={"WWW-Authenticate": "Bearer"},
    )


async def current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> User | None:
    """Resolve the caller, or raise 401.

    Returns None only when auth is disabled for this project, so downstream
    code can treat "no user" and "anonymous allowed" as the same case.
    """
    state = getattr(request.app.state, "app_state", None)
    if state is None:  # pragma: no cover - outside lifespan
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Service starting."
        )

    config: ProjectConfig = state.config
    if not config.auth.enabled or config.auth.mode is AuthMode.NONE:
        return None

    if credentials is None or not credentials.credentials:
        # A path the project has declared open may proceed anonymously. This is
        # what lets one project run an authenticated dashboard and a public
        # chat widget at the same time -- without it, `auth.enabled: true`
        # forces every visitor to hold a key, and `anonymous_audiences` could
        # never apply to anything.
        #
        # Anonymous here is a real restriction, not a bypass: the caller
        # resolves to None, which grants exactly `anonymous_audiences`.
        if request.url.path in config.auth.public_paths:
            return None
        raise _unauthorized("Missing credentials.")

    # Credentials were offered, so they are checked even on a public path. A
    # bad key must fail rather than quietly downgrade to anonymous, or a user
    # whose key was revoked would keep working with reduced access and never
    # find out.

    user = state.users.authenticate(credentials.credentials)
    if user is None:
        # Deliberately vague to the caller: distinguishing "no such key" from
        # "key belongs to a disabled user" would confirm a valid key exists.
        # The detail goes to the log instead.
        log.warning(
            "auth_failed",
            project=config.project.id,
            client=request.client.host if request.client else "unknown",
        )
        raise _unauthorized("Invalid credentials.")

    return user


async def require_user(user: User | None = Depends(current_user)) -> User | None:
    """Dependency for endpoints that need a caller when auth is on."""
    return user


def require_permission(permission: Permission, *, open_when_auth_disabled: bool = False):
    """Build a dependency that enforces one permission.

    Checks the *permission*, never a role name: which roles carry `admin`
    differs per project, and an endpoint has no business knowing that a given
    site calls its privileged role "support".

    With auth disabled there is no way to tell who is calling, so privileged
    endpoints stay closed rather than open to all -- failing shut is the only
    safe reading of "we cannot identify you".

    Args:
        open_when_auth_disabled: Serve the endpoint anyway when the project has
            no authentication at all. For endpoints that are *user*-scoped
            rather than privileged -- reading back your own conversation, say.
            With auth off every caller is anonymous and the whole service is
            already open, so gating these would only make them permanently
            dead in the default local setup while protecting nothing.
            `assert_safe_to_start` is what stops that configuration reaching a
            server. Leave it False for anything genuinely privileged.
    """

    async def dependency(
        request: Request, user: User | None = Depends(current_user)
    ) -> User | None:
        state = getattr(request.app.state, "app_state", None)
        if state is None:  # pragma: no cover
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Service starting.",
            )

        config: ProjectConfig = state.config
        if not config.auth.enabled:
            if open_when_auth_disabled:
                return None
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=(
                    f"The {permission.value!r} permission requires authentication "
                    f"to be enabled for this project."
                ),
            )

        assert user is not None  # auth is on, so current_user raised or returned
        granted = config.auth.permissions_for(user.roles)
        if permission not in granted:
            log.warning(
                "permission_denied",
                project=config.project.id,
                user=user.id,
                needed=permission.value,
                roles=sorted(user.roles),
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Requires the {permission.value!r} permission.",
            )
        return user

    return dependency


def rate_limit_key(request: Request, user: User | None) -> str:
    """Key for rate limiting.

    An authenticated subject is a real identity; an IP address is a guess that
    punishes everyone behind one NAT and is trivially rotated. Prefer the
    former whenever it exists.
    """
    if user is not None:
        return f"user:{user.id}"
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return f"ip:{forwarded.split(',')[0].strip()}"
    return f"ip:{request.client.host if request.client else 'unknown'}"


__all__ = [
    "LOCAL_ENVIRONMENTS",
    "assert_safe_to_start",
    "current_user",
    "load_user_store",
    "rate_limit_key",
    "require_permission",
    "require_user",
]
