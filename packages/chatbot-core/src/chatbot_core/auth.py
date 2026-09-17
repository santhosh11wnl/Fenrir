"""Per-project authentication and role-based access.

One base platform runs a separate assistant per website. Users belong to **one**
of those, not to the platform: a key issued for site A grants nothing on site
B, and no global user list exists that could accidentally span both.

Roles are defined by each project, permissions are fixed
--------------------------------------------------------
Sites need different vocabularies -- one wants ``admin``/``support``/
``customer``, another only ``staff``. So roles are whatever a project's config
declares. What roles *grant* comes from a fixed :class:`Permission` set,
because each permission corresponds to an actual check in the code; a
permission no endpoint enforces would read as protection that isn't there.

Where things live
-----------------
``projects/<id>/config.yaml``   the ``auth`` block: on/off, mode, policy
``projects/<id>/users.yaml``    the users -- **gitignored**, never committed

The split matters: config is checked in and reviewable, the user store holds
credential material and is not.

What is stored
--------------
Only a hash. An API key is shown exactly once, at creation, and cannot be
recovered afterwards -- if it is lost, issue a new one. Storing keys in
recoverable form would mean a read of the user file is a total compromise of
every project it covers.

Hashing uses scrypt from the standard library. It is memory-hard and available
everywhere, so there is no argon2/bcrypt build dependency to go wrong in a
slim container.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Self

import structlog
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

log = structlog.get_logger(__name__)

#: Keys carry a prefix so a leaked string is identifiable in a log or a repo
#: scan, and so a user can tell an API key from any other opaque token.
KEY_PREFIX = "mcp_"

#: 32 bytes of entropy. Long enough that guessing is not a threat model.
KEY_BYTES = 32

#: scrypt parameters. n=2**14 keeps verification near a millisecond on a
#: server while making offline cracking of a stolen file expensive.
_SCRYPT_N = 2**14
_SCRYPT_R = 8
_SCRYPT_P = 1
_SALT_BYTES = 16


class Permission(StrEnum):
    """What a caller may do. A fixed vocabulary, unlike roles.

    Permissions are fixed because each one corresponds to code that must check
    it -- inventing a permission no endpoint enforces would be worse than
    useless, since it reads as protection that isn't there. Roles, which group
    these, are defined per project.
    """

    #: Send messages to the assistant.
    CHAT = "chat"
    #: Read back conversations (including other users' on this project).
    HISTORY = "history"
    #: Read admin endpoints: usage, health detail, index status.
    ADMIN = "admin"
    #: Trigger a corpus re-ingest.
    INGEST = "ingest"
    #: Manage this project's users.
    MANAGE_USERS = "manage_users"


class RoleSpec(BaseModel):
    """A named bundle of permissions, defined by the project.

    Every website gets its own vocabulary -- one may need `admin`, `support`,
    and `customer`; another only `staff`. Hardcoding a role enum would force
    seven different access models into one shape.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    description: str = ""
    permissions: frozenset[Permission] = Field(default_factory=frozenset)

    #: Which document audiences this role may retrieve. Documents are tagged at
    #: ingest time (``metadata.audience``); a search runs filtered to these
    #: values, so a customer's query cannot return an internal document.
    #:
    #: This is enforced in the **query**, not in the prompt. Instructing a model
    #: not to mention something it can already see is not access control -- the
    #: text is in its context and can be extracted. Chunks the caller may not
    #: see are never fetched.
    audiences: frozenset[str] = Field(default_factory=lambda: frozenset({"public"}))

    #: Appended to the system prompt when this role is the caller. Use it to
    #: set expectations ("you are speaking to a support agent"), never to
    #: enforce access -- `audiences` does that.
    prompt: str = ""


#: Used when a project enables auth without defining roles. Deliberately
#: minimal: `admin` gets everything, `user` can only chat.
DEFAULT_ROLES: dict[str, RoleSpec] = {
    "admin": RoleSpec(
        description="Full access to this project.",
        permissions=frozenset(Permission),
        audiences=frozenset({"public", "staff", "admin"}),
    ),
    "user": RoleSpec(
        description="Can chat.",
        permissions=frozenset({Permission.CHAT}),
        audiences=frozenset({"public"}),
    ),
}

#: The audience an untagged document falls into. "public" so that forgetting to
#: tag a document makes it *visible*, not invisible -- a missing tag should be a
#: content mistake you notice, not a silent hole in the corpus.
#:
#: The inverse default would be safer for secrets and worse for everything else:
#: documents would vanish with no error and no clue why. Tag internal material
#: explicitly; see `ingest.yaml`.
DEFAULT_AUDIENCE = "public"


class AuthMode(StrEnum):
    #: No authentication. Only appropriate on a trusted network or localhost.
    NONE = "none"
    #: A per-user API key in `Authorization: Bearer <key>`.
    API_KEY = "api_key"


class AuthConfig(BaseModel):
    """The ``auth`` block of a project's config.yaml."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    #: Off by default so a freshly stamped project runs locally without setup.
    #: `require_auth_in_production` below is what stops that default reaching
    #: a server by accident.
    enabled: bool = False
    mode: AuthMode = AuthMode.API_KEY

    #: Where this project's users live, relative to the project directory.
    users_file: str = "users.yaml"

    #: This project's roles. Each website defines its own -- `admin` and
    #: `support` on one, something else on another.
    roles: dict[str, RoleSpec] = Field(default_factory=lambda: dict(DEFAULT_ROLES))

    #: Assigned to a user created without an explicit role.
    default_role: str = "user"

    #: Refuse to start with auth disabled when ENVIRONMENT is not a local one.
    #: The failure mode this prevents -- shipping a public, unauthenticated
    #: chat endpoint against a private corpus -- is bad enough to be worth a
    #: hard startup error.
    require_auth_in_production: bool = True

    #: Endpoints reachable without credentials, matched as exact paths. Add
    #: `/chat` to run a public website widget while keeping the dashboard and
    #: history endpoints authenticated -- the anonymous caller then retrieves
    #: exactly `anonymous_audiences`, so "open" still means "public documents
    #: only".
    #:
    #: A request that *does* carry a key is authenticated normally even here, so
    #: a signed-in customer keeps their role on a public path. Only privileged
    #: endpoints should be left off this list; anything on it is world-readable.
    public_paths: list[str] = Field(default_factory=lambda: ["/health"])

    #: What an unauthenticated visitor may retrieve. A website assistant is
    #: normally open to anyone browsing the site, so this is the anonymous
    #: customer's view -- and it must never include internal material.
    anonymous_audiences: frozenset[str] = Field(
        default_factory=lambda: frozenset({"public"})
    )

    def permissions_for(self, roles: frozenset[str]) -> frozenset[Permission]:
        """Union of the permissions granted by a user's roles.

        A role not defined by this project grants nothing rather than raising:
        a typo in users.yaml should reduce access, never widen it or take the
        service down.
        """
        granted: set[Permission] = set()
        for name in roles:
            spec = self.roles.get(name)
            if spec is None:
                log.warning("unknown_role", role=name, known=sorted(self.roles))
                continue
            granted |= spec.permissions
        return frozenset(granted)

    def audiences_for(self, roles: frozenset[str] | None) -> frozenset[str]:
        """Which document audiences a caller may retrieve.

        ``None`` means unauthenticated. Unknown roles contribute nothing, so a
        typo narrows access rather than widening it.
        """
        if roles is None:
            return self.anonymous_audiences

        allowed: set[str] = set()
        resolved = False
        for name in roles:
            spec = self.roles.get(name)
            if spec is None:
                log.warning("unknown_role", role=name, known=sorted(self.roles))
                continue
            resolved = True
            allowed |= spec.audiences

        if resolved:
            # At least one role was recognised, so the configuration has
            # actually spoken. Honour it exactly -- including an empty result.
            # A role deliberately given no audiences (a service account that
            # should use tools but never read the corpus) has to be
            # expressible, and silently topping it up to "public" would make
            # the config a lie.
            return frozenset(allowed)

        # Nothing resolved: every role this caller holds is unknown to the
        # project. Almost always a typo in users.yaml or a role deleted from
        # config while a user still carries it. Fall back to the anonymous
        # view rather than to nothing, since that is what the same person
        # would get by logging out -- it cannot grant more than a passer-by
        # already has, and a site assistant that answers nothing at all reads
        # as an outage rather than as a permissions problem.
        return self.anonymous_audiences

    def prompt_for(self, roles: frozenset[str] | None) -> str:
        """Role-specific guidance to append to the system prompt.

        Sets expectations about who is asking. It does not enforce anything --
        `audiences_for` does that, at query time.
        """
        if roles is None:
            return ""
        parts = [
            self.roles[name].prompt.strip()
            for name in sorted(roles)
            if name in self.roles and self.roles[name].prompt.strip()
        ]
        return "\n\n".join(parts)

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.enabled and self.mode is AuthMode.NONE:
            raise ValueError(
                "auth.enabled is true but auth.mode is 'none'; set a real mode "
                "or disable auth explicitly"
            )
        if not self.roles:
            raise ValueError("auth.roles cannot be empty; define at least one role")
        if self.default_role not in self.roles:
            raise ValueError(
                f"auth.default_role is {self.default_role!r} but the defined roles "
                f"are {sorted(self.roles)}"
            )
        if self.enabled and not any(
            Permission.ADMIN in spec.permissions for spec in self.roles.values()
        ):
            # Without this you can lock yourself out of your own project's
            # admin surface with no way back short of editing YAML by hand.
            raise ValueError(
                "auth is enabled but no role grants the 'admin' permission; "
                "at least one role must be able to administer this project"
            )
        return self


@dataclass(slots=True)
class User:
    """A user of ONE project.

    ``roles`` holds names defined by that project's config -- "admin",
    "support", whatever that website needs. Meaning is resolved against the
    project's `AuthConfig`, not assumed here, which is what lets seven projects
    run seven different access models over one User type.
    """

    id: str
    name: str
    roles: frozenset[str]
    key_hash: str
    disabled: bool = False
    created_at: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def has_role(self, role: str) -> bool:
        return role in self.roles


# ---------------------------------------------------------------------------
# hashing
# ---------------------------------------------------------------------------


def generate_key() -> str:
    """Mint a new API key. Shown once; only its hash is ever persisted."""
    return KEY_PREFIX + secrets.token_urlsafe(KEY_BYTES)


def hash_key(key: str) -> str:
    """Hash a key for storage: ``scrypt$n$r$p$salt$digest``, all base64."""
    salt = secrets.token_bytes(_SALT_BYTES)
    digest = hashlib.scrypt(
        key.encode(), salt=salt, n=_SCRYPT_N, r=_SCRYPT_R, p=_SCRYPT_P, dklen=32
    )
    return "$".join(
        [
            "scrypt",
            str(_SCRYPT_N),
            str(_SCRYPT_R),
            str(_SCRYPT_P),
            base64.b64encode(salt).decode(),
            base64.b64encode(digest).decode(),
        ]
    )


def verify_key(key: str, encoded: str) -> bool:
    """Check a key against a stored hash.

    Parameters come from the stored string rather than the constants above, so
    hashes written under older settings keep verifying after a tuning change.
    Comparison is constant-time: a byte-by-byte comparison leaks how much of a
    guess was correct, which is enough to reconstruct a secret given enough
    attempts.
    """
    try:
        scheme, n, r, p, salt_b64, digest_b64 = encoded.split("$")
        if scheme != "scrypt":
            return False
        expected = base64.b64decode(digest_b64)
        actual = hashlib.scrypt(
            key.encode(),
            salt=base64.b64decode(salt_b64),
            n=int(n),
            r=int(r),
            p=int(p),
            dklen=len(expected),
        )
    except (ValueError, TypeError):
        log.warning("malformed_key_hash")
        return False
    return hmac.compare_digest(actual, expected)


# ---------------------------------------------------------------------------
# store
# ---------------------------------------------------------------------------


class UserStore:
    """A project's users, loaded from its ``users.yaml``.

    Held in memory: the list is small, changes rarely, and a lookup sits on the
    request path. Call :meth:`reload` after editing the file.
    """

    def __init__(self, users: list[User], path: Path | None = None) -> None:
        self._users = {u.id: u for u in users}
        self._path = path

    # -- loading -----------------------------------------------------------

    @classmethod
    def load(cls, path: str | Path) -> UserStore:
        path = Path(path)
        if not path.is_file():
            # Not an error here. Startup decides whether an empty store is
            # acceptable, because that depends on whether auth is enabled.
            log.info("no_user_store", path=str(path))
            return cls([], path)

        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        entries = raw.get("users") or []
        users: list[User] = []

        for index, entry in enumerate(entries):
            try:
                users.append(_parse_user(entry))
            except (KeyError, ValueError) as exc:
                # Skip the bad entry rather than failing the whole file: one
                # malformed record should not lock every other user out.
                log.error("invalid_user_entry", index=index, error=str(exc))

        log.info("user_store_loaded", path=str(path), users=len(users))
        return cls(users, path)

    def reload(self) -> None:
        if self._path is not None:
            self._users = {u.id: u for u in UserStore.load(self._path)._users.values()}

    # -- queries -----------------------------------------------------------

    def __len__(self) -> int:
        return len(self._users)

    @property
    def users(self) -> list[User]:
        return list(self._users.values())

    def get(self, user_id: str) -> User | None:
        return self._users.get(user_id)

    def authenticate(self, key: str) -> User | None:
        """Find the user holding this key, or None.

        Every candidate is checked even after a match, so the time taken does
        not reveal *which* user matched or how far down the list they sit.
        The cost is bounded -- these lists are small by construction.
        """
        if not key:
            return None

        matched: User | None = None
        for user in self._users.values():
            if verify_key(key, user.key_hash) and not user.disabled:
                matched = matched or user
        return matched

    # -- mutation ----------------------------------------------------------

    def add(self, user: User) -> None:
        if user.id in self._users:
            raise ValueError(f"user {user.id!r} already exists")
        self._users[user.id] = user

    def remove(self, user_id: str) -> bool:
        return self._users.pop(user_id, None) is not None

    def save(self, path: str | Path | None = None) -> Path:
        """Write the store back to disk.

        Written via a temporary file and an atomic rename: a crash mid-write
        would otherwise leave a truncated user file, locking everyone out.
        """
        target = Path(path or self._path or "users.yaml")
        payload = {
            "users": [
                {
                    "id": u.id,
                    "name": u.name,
                    "roles": sorted(u.roles),
                    "key_hash": u.key_hash,
                    "disabled": u.disabled,
                    "created_at": u.created_at,
                    **({"metadata": u.metadata} if u.metadata else {}),
                }
                for u in self._users.values()
            ]
        }

        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        header = (
            "# Users for this project ONLY. Access here grants nothing on any\n"
            "# other project.\n"
            "#\n"
            "# Contains credential hashes: gitignored, and it must stay that way.\n"
            "# Keys cannot be recovered from these hashes -- issue a new one if\n"
            "# a user loses theirs.\n"
            "#\n"
            "# Manage with: uv run python scripts/manage_users.py --help\n\n"
        )
        temporary.write_text(
            header + yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
        )
        # 0600: this file is credential material.
        temporary.chmod(0o600)
        temporary.replace(target)
        return target


def _parse_user(entry: dict[str, Any]) -> User:
    user_id = str(entry["id"]).strip()
    if not user_id:
        raise ValueError("user id is empty")

    # Role names are validated against the project's config at permission
    # check time, not here -- this parser has no view of which project it is.
    roles = frozenset(str(r).strip() for r in (entry.get("roles") or []) if str(r).strip())

    key_hash = str(entry.get("key_hash") or "")
    if not key_hash:
        raise ValueError(f"user {user_id!r} has no key_hash")

    return User(
        id=user_id,
        name=str(entry.get("name") or user_id),
        roles=roles,
        key_hash=key_hash,
        disabled=bool(entry.get("disabled", False)),
        created_at=str(entry.get("created_at") or ""),
        metadata=dict(entry.get("metadata") or {}),
    )


def new_user(
    user_id: str, name: str = "", roles: set[str] | None = None
) -> tuple[User, str]:
    """Create a user and return them with their plaintext key.

    The key is returned exactly once. Show it to the operator, then let it go --
    nothing persists it, and it cannot be recovered from the stored hash.
    """
    key = generate_key()
    user = User(
        id=user_id,
        name=name or user_id,
        roles=frozenset(roles or {"user"}),
        key_hash=hash_key(key),
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )
    return user, key


__all__ = [
    "DEFAULT_AUDIENCE",
    "DEFAULT_ROLES",
    "KEY_PREFIX",
    "AuthConfig",
    "AuthMode",
    "Permission",
    "RoleSpec",
    "User",
    "UserStore",
    "generate_key",
    "hash_key",
    "new_user",
    "verify_key",
]
