"""Authentication and per-project access control.

The property these exist to protect: one base platform runs a separate
assistant per website, and a credential issued for one grants nothing on any
other. Everything else here is in service of that.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from chatbot_core.auth import (
    DEFAULT_ROLES,
    KEY_PREFIX,
    AuthConfig,
    AuthMode,
    Permission,
    RoleSpec,
    User,
    UserStore,
    generate_key,
    hash_key,
    new_user,
    verify_key,
)

SITE_ROLES = {
    "admin": RoleSpec(
        description="Full access",
        permissions={
            Permission.CHAT,
            Permission.HISTORY,
            Permission.ADMIN,
            Permission.INGEST,
            Permission.MANAGE_USERS,
        },
    ),
    "support": RoleSpec(
        description="Support staff", permissions={Permission.CHAT, Permission.HISTORY}
    ),
    "customer": RoleSpec(description="End user", permissions={Permission.CHAT}),
}


def site_config(**kwargs) -> AuthConfig:
    return AuthConfig(
        enabled=True, roles=SITE_ROLES, default_role="customer", **kwargs
    )


class TestKeys:
    def test_keys_are_prefixed_and_unique(self):
        keys = {generate_key() for _ in range(50)}
        assert len(keys) == 50
        assert all(k.startswith(KEY_PREFIX) for k in keys)

    def test_hash_verifies_the_right_key(self):
        key = generate_key()
        assert verify_key(key, hash_key(key))

    def test_hash_rejects_the_wrong_key(self):
        assert not verify_key(generate_key(), hash_key(generate_key()))

    def test_hashing_is_salted(self):
        """Identical keys must not produce identical hashes, or a stolen file
        reveals which users share a credential."""
        key = generate_key()
        assert hash_key(key) != hash_key(key)

    def test_hash_is_not_the_key(self):
        key = generate_key()
        assert key not in hash_key(key)

    @pytest.mark.parametrize(
        "bad", ["", "garbage", "scrypt$bad", "md5$1$1$1$a$b", "scrypt$a$b$c$d$e"]
    )
    def test_malformed_hash_fails_closed(self, bad):
        assert not verify_key("anything", bad)

    def test_stored_parameters_are_used_for_verification(self):
        """Hashes written under older cost settings must keep verifying, or a
        tuning change locks every existing user out."""
        key = generate_key()
        encoded = hash_key(key)
        scheme, n, r, p, salt, digest = encoded.split("$")
        assert scheme == "scrypt" and int(n) > 0
        assert verify_key(key, encoded)


class TestRoles:
    def test_roles_are_project_defined(self):
        """Each website names its own roles; the platform imposes none."""
        config = site_config()
        assert set(config.roles) == {"admin", "support", "customer"}

    def test_permissions_come_from_roles(self):
        config = site_config()
        assert config.permissions_for(frozenset({"customer"})) == {Permission.CHAT}
        assert config.permissions_for(frozenset({"support"})) == {
            Permission.CHAT,
            Permission.HISTORY,
        }

    def test_multiple_roles_union_their_permissions(self):
        config = site_config()
        granted = config.permissions_for(frozenset({"customer", "support"}))
        assert granted == {Permission.CHAT, Permission.HISTORY}

    def test_unknown_role_grants_nothing(self):
        """A typo in users.yaml must reduce access, never widen it or crash."""
        assert site_config().permissions_for(frozenset({"typo"})) == frozenset()

    def test_no_roles_grants_nothing(self):
        assert site_config().permissions_for(frozenset()) == frozenset()

    def test_two_projects_can_define_different_vocabularies(self):
        """The point of the whole design: one base, different access models."""
        other = AuthConfig(
            enabled=True,
            roles={
                "staff": RoleSpec(permissions={Permission.CHAT, Permission.ADMIN}),
            },
            default_role="staff",
        )
        assert "support" in site_config().roles
        assert "support" not in other.roles
        assert other.permissions_for(frozenset({"staff"})) == {
            Permission.CHAT,
            Permission.ADMIN,
        }

    def test_default_role_must_exist(self):
        with pytest.raises(ValidationError, match="default_role"):
            AuthConfig(roles=SITE_ROLES, default_role="nonexistent")

    def test_roles_cannot_be_empty(self):
        with pytest.raises(ValidationError, match="empty"):
            AuthConfig(roles={})

    def test_enabled_auth_needs_an_admin_capable_role(self):
        """Otherwise you can lock yourself out of your own project with no way
        back short of hand-editing YAML."""
        with pytest.raises(ValidationError, match="admin"):
            AuthConfig(
                enabled=True,
                roles={"user": RoleSpec(permissions={Permission.CHAT})},
                default_role="user",
            )

    def test_enabled_with_mode_none_is_rejected(self):
        with pytest.raises(ValidationError, match="mode"):
            AuthConfig(enabled=True, mode=AuthMode.NONE, roles=DEFAULT_ROLES)


class TestUserStore:
    def test_authenticates_the_holder_of_a_key(self):
        user, key = new_user("alice", roles={"admin"})
        store = UserStore([user])
        found = store.authenticate(key)
        assert found is not None and found.id == "alice"

    def test_rejects_an_unknown_key(self):
        user, _ = new_user("alice")
        assert UserStore([user]).authenticate(generate_key()) is None

    def test_rejects_an_empty_key(self):
        user, _ = new_user("alice")
        assert UserStore([user]).authenticate("") is None

    def test_disabled_user_cannot_authenticate(self):
        user, key = new_user("alice")
        user.disabled = True
        assert UserStore([user]).authenticate(key) is None

    def test_a_key_identifies_exactly_one_user(self):
        alice, alice_key = new_user("alice")
        bob, bob_key = new_user("bob")
        store = UserStore([alice, bob])
        assert store.authenticate(alice_key).id == "alice"
        assert store.authenticate(bob_key).id == "bob"

    def test_duplicate_id_is_rejected(self):
        alice, _ = new_user("alice")
        store = UserStore([alice])
        other, _ = new_user("alice")
        with pytest.raises(ValueError, match="already exists"):
            store.add(other)


class TestProjectIsolation:
    """The guarantee the whole design exists for."""

    def test_a_key_from_one_project_fails_on_another(self):
        alice, alice_key = new_user("alice", roles={"admin"})
        site_a = UserStore([alice])

        bob, _ = new_user("bob", roles={"admin"})
        site_b = UserStore([bob])

        assert site_a.authenticate(alice_key) is not None
        assert site_b.authenticate(alice_key) is None

    def test_same_user_id_on_two_sites_are_unrelated(self):
        """Two sites can both have an "admin" user; their keys are distinct."""
        a_admin, a_key = new_user("admin", roles={"admin"})
        b_admin, b_key = new_user("admin", roles={"admin"})

        assert UserStore([a_admin]).authenticate(b_key) is None
        assert UserStore([b_admin]).authenticate(a_key) is None

    def test_same_role_name_can_mean_different_things(self):
        """"support" grants history on one site and nothing extra on another."""
        strict = AuthConfig(
            enabled=True,
            roles={
                "admin": RoleSpec(permissions={Permission.ADMIN}),
                "support": RoleSpec(permissions={Permission.CHAT}),
            },
            default_role="support",
        )
        assert Permission.HISTORY in site_config().permissions_for(frozenset({"support"}))
        assert Permission.HISTORY not in strict.permissions_for(frozenset({"support"}))


class TestPersistence:
    def test_round_trips_through_yaml(self, tmp_path):
        alice, key = new_user("alice", name="Alice", roles={"admin", "support"})
        path = tmp_path / "users.yaml"
        UserStore([alice]).save(path)

        reloaded = UserStore.load(path)
        assert len(reloaded) == 1
        found = reloaded.authenticate(key)
        assert found is not None
        assert found.roles == frozenset({"admin", "support"})

    def test_saved_file_never_contains_the_key(self, tmp_path):
        alice, key = new_user("alice")
        path = tmp_path / "users.yaml"
        UserStore([alice]).save(path)
        assert key not in path.read_text()

    def test_saved_file_is_owner_only(self, tmp_path):
        """It holds credential material."""
        alice, _ = new_user("alice")
        path = tmp_path / "users.yaml"
        UserStore([alice]).save(path)
        assert path.stat().st_mode & 0o777 == 0o600

    def test_missing_file_is_an_empty_store(self, tmp_path):
        """Startup decides whether that is acceptable; loading does not."""
        assert len(UserStore.load(tmp_path / "absent.yaml")) == 0

    def test_one_bad_entry_does_not_lock_everyone_out(self, tmp_path):
        path = tmp_path / "users.yaml"
        alice, key = new_user("alice")
        path.write_text(
            "users:\n"
            "  - id: broken\n"          # no key_hash
            f"  - id: alice\n    key_hash: {alice.key_hash}\n    roles: [admin]\n"
        )
        store = UserStore.load(path)
        assert len(store) == 1
        assert store.authenticate(key) is not None

    def test_removal_revokes_immediately(self, tmp_path):
        alice, key = new_user("alice")
        store = UserStore([alice], tmp_path / "users.yaml")
        assert store.authenticate(key) is not None
        store.remove("alice")
        assert store.authenticate(key) is None


class TestNewUser:
    def test_returns_a_working_key_once(self):
        user, key = new_user("alice")
        assert verify_key(key, user.key_hash)

    def test_defaults_to_the_user_role(self):
        user, _ = new_user("alice")
        assert user.roles == frozenset({"user"})

    def test_records_creation_time(self):
        user, _ = new_user("alice")
        assert user.created_at

    def test_accepts_arbitrary_project_role_names(self):
        user, _ = new_user("alice", roles={"support", "escalation-lead"})
        assert user.has_role("escalation-lead")
        assert isinstance(user, User)
