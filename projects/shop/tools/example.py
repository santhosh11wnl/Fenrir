"""Project-specific MCP tools.

Every ``*.py`` file in this directory that exposes ``register(server)`` is
loaded automatically by the MCP server. Files starting with ``_`` are treated
as helpers and skipped.

This is the extension point that keeps seven projects from forking the base:
you add tools here, never in ``packages/mcp-server-core``.

Writing a good tool
-------------------
* **The docstring is the prompt.** It's what the model reads to decide whether
  to call your tool. Say *when* to use it, not just what it does -- "Call this
  when the user asks about order status" beats "Gets order status".
* **Type hints become the schema.** Annotate every argument; the model sees
  those types and validates against them.
* **Return a string.** Write it for a reader, not a parser -- the model has to
  understand it to use it.
* **Treat arguments as untrusted.** They're model-generated, and the model can
  be steered by a prompt injection inside a document it just retrieved. Validate
  anything that reaches a filesystem, a database, or the network.
* **Fail with a message, don't raise.** A returned error explains itself to the
  model, which can then try something else. An exception just ends the turn.
"""

from __future__ import annotations

from typing import Any


def register(server: Any) -> None:
    """Register this module's tools. Called once at server startup."""

    @server.tool()
    def project_status(area: str = "all") -> str:
        """Report the current status of a project area.

        Call this when the user asks how something is going, whether a system
        is healthy, or what the current state of a given area is.

        Args:
            area: Which area to report on. Use "all" for a summary.
        """
        # Replace with a real lookup -- a database query, an internal API call,
        # a cache read. Keep credentials out of the sandbox: see
        # `chatbot_core` docs on keeping secrets host-side.
        known = {"all": "All systems nominal.", "ingest": "Last ingest succeeded."}
        if area not in known:
            # An informative failure lets the model correct itself; a raise
            # would just end the turn.
            return (
                f"Unknown area {area!r}. Known areas: {', '.join(sorted(known))}."
            )
        return known[area]
