"""URL policy tests.

The fetch tool's URL argument is model-generated, and a document retrieved from
the corpus can carry a prompt injection steering it. These assertions are the
guard against that turning into a server-side request forgery primitive.
"""

from __future__ import annotations

import pytest

from mcp_server_core.tools.web import allowed_hosts, check_url

PERMITTED = frozenset({"example.com", "docs.internal.test"})


def test_disabled_when_no_allowlist_configured():
    """Deny by default. An operator opts in per deployment."""
    reason = check_url("https://example.com/page", frozenset())
    assert reason is not None
    assert "disabled" in reason


def test_permits_exact_host():
    assert check_url("https://example.com/page", PERMITTED) is None


def test_permits_subdomain_of_allowed_host():
    assert check_url("https://api.example.com/v1", PERMITTED) is None


def test_rejects_lookalike_suffix():
    """Suffix matching alone would let evil-example.com through an
    example.com allowlist."""
    assert check_url("https://evil-example.com/", PERMITTED) is not None


def test_rejects_unlisted_host():
    reason = check_url("https://elsewhere.test/", PERMITTED)
    assert reason is not None and "allowlist" in reason


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "gopher://example.com/",
        "data:text/plain;base64,aGk=",
        "ftp://example.com/x",
    ],
)
def test_rejects_non_http_schemes(url):
    reason = check_url(url, PERMITTED)
    assert reason is not None
    assert "http" in reason.lower()


@pytest.mark.parametrize(
    "url",
    [
        "http://169.254.169.254/latest/meta-data/",  # cloud metadata endpoint
        "http://127.0.0.1:8000/admin",
        "http://10.0.0.5/internal",
        "http://[::1]:8000/",
    ],
)
def test_rejects_literal_ip_addresses(url):
    """Allowlisting is by hostname; a raw IP is the usual shape of a
    metadata-endpoint or internal-network probe."""
    assert check_url(url, PERMITTED) is not None


def test_rejects_url_without_host():
    assert check_url("https:///nohost", PERMITTED) is not None


def test_allowed_hosts_parses_env(monkeypatch):
    monkeypatch.setenv("MCP_HTTP_ALLOWED_HOSTS", " Example.com , docs.test ,, ")
    assert allowed_hosts() == frozenset({"example.com", "docs.test"})


def test_allowed_hosts_empty_by_default(monkeypatch):
    monkeypatch.delenv("MCP_HTTP_ALLOWED_HOSTS", raising=False)
    assert allowed_hosts() == frozenset()
