"""HTTP fetch tool.

A worked example of a tool that reaches outside the process, and the security
posture every such tool needs.

**The URL argument is model-generated and therefore untrusted.** A naive fetch
tool is a server-side request forgery primitive: the model can be steered --
by a prompt injection inside a document it just retrieved -- into fetching
``http://169.254.169.254/`` and returning cloud credentials as tool output.

Defences here, in order of importance:

1. **Allowlist.** Disabled unless ``MCP_HTTP_ALLOWED_HOSTS`` names hosts. Deny
   by default; an operator opts in per deployment.
2. **Scheme restriction.** ``http``/``https`` only -- no ``file://``,
   ``gopher://``, or ``data:``.
3. **No redirect following.** A permitted host must not be able to bounce the
   request to a denied one.
4. **Size and time caps.** A tool that streams forever is a denial of service
   against the chat request holding it open.

This is *defence for* a legitimate tool, not a bypass technique: the allowlist
is the security boundary, and everything else is depth behind it.
"""

from __future__ import annotations

import ipaddress
import os
from typing import Any
from urllib.parse import urlparse

import structlog

log = structlog.get_logger(__name__)

MAX_BYTES = 200_000
TIMEOUT_SECONDS = 15.0
ALLOWED_SCHEMES = frozenset({"http", "https"})


def allowed_hosts() -> frozenset[str]:
    """Hosts this deployment may fetch, from ``MCP_HTTP_ALLOWED_HOSTS``.

    Comma-separated. Empty (the default) disables the tool entirely.
    """
    raw = os.environ.get("MCP_HTTP_ALLOWED_HOSTS", "")
    return frozenset(h.strip().lower() for h in raw.split(",") if h.strip())


def check_url(url: str, permitted: frozenset[str]) -> str | None:
    """Validate a URL against the policy. Returns a reason string if rejected.

    Split out from the tool so the policy is directly testable -- security
    logic that can only be exercised through a live HTTP call tends not to be
    exercised at all.
    """
    if not permitted:
        return (
            "URL fetching is disabled on this server. An operator must set "
            "MCP_HTTP_ALLOWED_HOSTS to enable it."
        )

    try:
        parsed = urlparse(url)
    except ValueError:
        return f"{url!r} is not a valid URL."

    if parsed.scheme.lower() not in ALLOWED_SCHEMES:
        return f"Only http and https URLs are allowed, got {parsed.scheme!r}."

    host = (parsed.hostname or "").lower()
    if not host:
        return f"{url!r} has no host."

    # Reject literal IPs outright. Allowlisting is by hostname, and a raw IP is
    # the usual shape of a metadata-endpoint or internal-network probe.
    try:
        ipaddress.ip_address(host)
    except ValueError:
        pass
    else:
        return "Fetching IP addresses directly is not allowed; use a hostname."

    # Exact match or subdomain of a permitted host. Suffix matching alone would
    # let "evil-example.com" pass an "example.com" allowlist.
    if not any(host == p or host.endswith(f".{p}") for p in permitted):
        return f"Host {host!r} is not in this server's allowlist."

    return None


def register(server: Any) -> None:
    @server.tool()
    async def fetch_url(url: str) -> str:
        """Fetch the text content of a URL from an allowlisted host.

        Returns the response body as text, truncated if large. Only hosts this
        server has been configured to allow can be fetched.

        Args:
            url: Absolute http(s) URL to fetch.
        """
        import httpx

        permitted = allowed_hosts()
        if (reason := check_url(url, permitted)) is not None:
            log.warning("fetch_url_rejected", url=url, reason=reason)
            return f"Request rejected: {reason}"

        try:
            async with httpx.AsyncClient(
                timeout=TIMEOUT_SECONDS,
                # A permitted host must not be able to redirect us elsewhere.
                follow_redirects=False,
            ) as client:
                response = await client.get(url, headers={"User-Agent": "mcp-platform/0.1"})
        except httpx.TimeoutException:
            return f"Request to {url} timed out after {TIMEOUT_SECONDS:.0f}s."
        except httpx.HTTPError as exc:
            log.warning("fetch_url_failed", url=url, error=str(exc))
            return f"Could not fetch {url}: {exc}"

        if response.is_redirect:
            location = response.headers.get("location", "(none)")
            return (
                f"{url} returned a redirect to {location}, which is not followed. "
                f"Request the final URL directly if its host is allowed."
            )

        if response.status_code >= 400:
            return f"{url} returned HTTP {response.status_code}."

        body = response.text
        if len(body) > MAX_BYTES:
            body = body[:MAX_BYTES] + f"\n\n[truncated at {MAX_BYTES} characters]"

        log.info("fetch_url_ok", url=url, status=response.status_code, bytes=len(body))
        return body
