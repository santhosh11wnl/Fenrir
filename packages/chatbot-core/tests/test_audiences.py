"""Audience-scoped retrieval.

The property under test: what a caller can retrieve is decided by their roles,
in the query, before any text reaches the model.

This is the distinction that matters. Instructing a model not to reveal
internal pricing is not access control -- the text is already in its context
window, and a sufficiently determined user will get it out. These tests assert
the restricted chunks are never *fetched*, so there is nothing to leak.

Covers the three layers that enforce it together:

* :class:`AuthConfig` resolving roles to an audience set,
* the store predicates that carry the set into the query,
* :class:`RetrievalTool` binding the set out of the model's reach.
"""

from __future__ import annotations

import pytest

from chatbot_core.auth import DEFAULT_AUDIENCE, AuthConfig, Permission, RoleSpec
from chatbot_core.retrieval.lancedb_store import _where_clause
from chatbot_core.retrieval.tool import TOOL_NAME, RetrievalTool

# A realistic shape: staff see their own material plus everything customers
# see; only an administrator reaches commercially sensitive documents.
SHOP_ROLES = {
    "super_admin": RoleSpec(
        description="Everything.",
        permissions=frozenset(Permission),
        audiences=frozenset({"public", "staff", "internal"}),
    ),
    "support": RoleSpec(
        description="Support desk.",
        permissions=frozenset({Permission.CHAT, Permission.HISTORY}),
        audiences=frozenset({"public", "staff"}),
        prompt="You are speaking to a support agent.",
    ),
    "customer": RoleSpec(
        description="Signed-in shopper.",
        permissions=frozenset({Permission.CHAT}),
        audiences=frozenset({"public"}),
        prompt="You are speaking to a customer.",
    ),
}


@pytest.fixture
def auth() -> AuthConfig:
    return AuthConfig(enabled=True, roles=SHOP_ROLES, default_role="customer")


class TestAudiencesForRoles:
    def test_a_role_gets_exactly_its_own_audiences(self, auth: AuthConfig) -> None:
        assert auth.audiences_for(frozenset({"customer"})) == frozenset({"public"})
        assert auth.audiences_for(frozenset({"support"})) == frozenset(
            {"public", "staff"}
        )

    def test_multiple_roles_union(self, auth: AuthConfig) -> None:
        """Holding two roles grants the union, not the intersection.

        A support agent who is also an administrator should not lose access by
        virtue of holding an extra role.
        """
        combined = auth.audiences_for(frozenset({"support", "super_admin"}))
        assert combined == frozenset({"public", "staff", "internal"})

    def test_anonymous_falls_back_to_the_configured_set(self, auth: AuthConfig) -> None:
        assert auth.audiences_for(None) == frozenset({DEFAULT_AUDIENCE})

    def test_anonymous_set_is_configurable_and_can_be_empty(self) -> None:
        """A project whose whole corpus is private can close the front door."""
        closed = AuthConfig(
            enabled=True,
            roles=SHOP_ROLES,
            default_role="customer",
            anonymous_audiences=frozenset(),
        )
        assert closed.audiences_for(None) == frozenset()

    def test_an_explicitly_empty_role_retrieves_nothing(self) -> None:
        """A role configured with no audiences must stay that way.

        The case: a service account that should call tools but never read the
        corpus. Quietly topping it up to the anonymous view would make the
        configuration a lie -- `audiences: []` has to mean what it says.
        """
        cfg = AuthConfig(
            enabled=True,
            roles={
                **SHOP_ROLES,
                "bot": RoleSpec(
                    description="Tools only.",
                    permissions=frozenset({Permission.CHAT}),
                    audiences=frozenset(),
                ),
            },
            default_role="customer",
        )
        assert cfg.audiences_for(frozenset({"bot"})) == frozenset()

    def test_wholly_unknown_roles_fall_back_to_the_anonymous_view(
        self, auth: AuthConfig
    ) -> None:
        """A typo degrades to the logged-out view, never above it.

        This is deliberately *not* an empty set: the same person gets the
        anonymous view by logging out, so falling back to it grants nothing a
        passer-by does not already have, and it fails as a permissions
        problem rather than as an apparent outage.
        """
        assert auth.audiences_for(frozenset({"deleted_role"})) == frozenset({"public"})

    def test_an_unknown_role_never_widens_a_known_one(self, auth: AuthConfig) -> None:
        """The fallback must not fire when any role did resolve."""
        assert auth.audiences_for(frozenset({"customer", "bogus"})) == frozenset(
            {"public"}
        )
        # And an unknown companion cannot drag internal material into scope.
        assert "internal" not in auth.audiences_for(frozenset({"support", "bogus"}))


class TestRolePrompt:
    def test_guidance_is_joined_for_multiple_roles(self, auth: AuthConfig) -> None:
        text = auth.prompt_for(frozenset({"customer", "support"}))
        assert "customer" in text and "support agent" in text

    def test_no_guidance_is_empty_not_none(self, auth: AuthConfig) -> None:
        assert auth.prompt_for(frozenset({"super_admin"})) == ""
        assert auth.prompt_for(None) == ""


class TestLanceDbPredicate:
    """The audience restriction has to survive into the SQL string."""

    def test_none_means_unrestricted(self) -> None:
        """Only for callers that have already been authorised elsewhere.

        ``None`` is distinct from an empty set: it is the ingest/admin path,
        not a caller with no audiences.
        """
        assert _where_clause(None, None) is None

    def test_empty_set_matches_nothing(self) -> None:
        """An empty audience set must not degrade into 'no filter'.

        The bug that would matter most: a caller granted nothing seeing
        everything. It also has to be *parseable* -- an unparseable predicate
        surfaces as a query error rather than as an empty result.
        """
        assert _where_clause(None, frozenset()) == "1 = 0"

    def test_empty_set_wins_over_any_metadata_filter(self) -> None:
        """Nothing can be ANDed onto 'no audiences' to make it match."""
        assert _where_clause({"lang": "en"}, frozenset()) == "1 = 0"

    def test_single_audience_restricts_to_that_tag(self) -> None:
        predicate = _where_clause(None, frozenset({"staff"}))
        assert predicate is not None
        assert '"audience": "staff"' in predicate
        assert "internal" not in predicate

    def test_untagged_rows_are_included_only_for_the_default_audience(self) -> None:
        """A document nobody tagged belongs to the default audience.

        Staff-only callers must not pick up untagged rows as a side effect.
        """
        public = _where_clause(None, frozenset({DEFAULT_AUDIENCE}))
        assert public is not None
        assert "NOT LIKE" in public

        staff_only = _where_clause(None, frozenset({"staff"}))
        assert staff_only is not None
        assert "NOT LIKE" not in staff_only

    def test_audience_names_are_validated_not_escaped(self) -> None:
        """Audience names reach SQL text, so they are allowlisted.

        They come from config rather than from a request, but a predicate
        built by string interpolation gets an allowlist regardless -- the
        distance between 'config' and 'user input' shrinks over time.
        """
        with pytest.raises(ValueError, match="invalid audience"):
            _where_clause(None, frozenset({"staff' OR '1'='1"}))

    def test_audience_and_metadata_filters_combine_with_and(self) -> None:
        predicate = _where_clause({"lang": "en"}, frozenset({"public"}))
        assert predicate is not None
        assert " AND " in predicate
        assert '"audience": "public"' in predicate


class TestRetrievalToolBinding:
    """The model must not be able to widen its own view."""

    def test_audience_is_not_a_tool_parameter(self) -> None:
        """If it were an argument, the model could simply ask for 'internal'."""
        spec = RetrievalTool(_NullStore(), audiences=frozenset({"public"})).specs[0]
        properties = spec.input_schema["properties"]
        assert "audience" not in properties
        assert "audiences" not in properties
        assert spec.input_schema["additionalProperties"] is False

    @pytest.mark.asyncio
    async def test_bound_audiences_are_passed_to_the_store(self) -> None:
        store = _NullStore()
        tool = RetrievalTool(store, audiences=frozenset({"public", "staff"}))
        await tool.execute(TOOL_NAME, {"query": "refund window"})
        assert store.seen == [frozenset({"public", "staff"})]

    @pytest.mark.asyncio
    async def test_an_injected_audience_argument_is_ignored(self) -> None:
        """Belt and braces: even if the schema were bypassed, it changes nothing.

        The tool reads audiences from the instance, never from `arguments`.
        """
        store = _NullStore()
        tool = RetrievalTool(store, audiences=frozenset({"public"}))
        await tool.execute(
            TOOL_NAME, {"query": "margins", "audiences": ["internal"], "audience": "internal"}
        )
        assert store.seen == [frozenset({"public"})]


class _NullStore:
    """Records the audiences it was queried with; returns no hits."""

    def __init__(self) -> None:
        self.seen: list[frozenset[str] | None] = []

    async def search(self, query, *, top_k=None, where=None, audiences=None):
        self.seen.append(audiences)
        return []
