"""Retrieval exposed as a tool.

Two ways to ground an answer in a corpus:

**As a tool** (``retrieval.as_tool: true``, the default) -- the model decides
when to search and what to search for. Turns that don't need the corpus don't
pay for it, and the model can re-query with better terms after a weak first
result. Costs one extra round trip when it does search.

**Prepended** (``as_tool: false``) -- every turn gets top-k chunks injected
before the model sees the question. One round trip, but it spends tokens on
every "thanks" and can't recover from a bad initial query.

Tool-based is right for assistants that mix corpus questions with tool work.
Prepend is right for a pure documentation Q&A bot where nearly every turn needs
the corpus anyway.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import structlog

from ..providers.base import ToolOutcome, ToolSpec
from .store import SearchHit, VectorStore

log = structlog.get_logger(__name__)

TOOL_NAME = "search_knowledge_base"


class RetrievalTool:
    """A :class:`~chatbot_core.providers.base.ToolExecutor` over one collection."""

    def __init__(
        self,
        store: VectorStore,
        *,
        description: str | None = None,
        audiences: frozenset[str] | None = None,
    ) -> None:
        self._store = store
        # Bound at construction, once per turn, from the authenticated caller.
        # Never a tool argument -- the model must not be able to widen its own
        # view by asking for a different audience.
        self._audiences = audiences
        self._description = description or (
            "Search the project's knowledge base for passages relevant to a "
            "question. Call this whenever the answer depends on project-specific "
            "documentation, policy, or reference material rather than general "
            "knowledge. Prefer a focused query over a broad one; if the first "
            "search returns nothing useful, try different terminology."
        )

    @property
    def specs(self) -> Sequence[ToolSpec]:
        return (
            ToolSpec(
                name=TOOL_NAME,
                description=self._description,
                input_schema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": (
                                "What to search for. Use the terminology you "
                                "expect in the source documents, not the user's "
                                "phrasing, when they differ."
                            ),
                        },
                        "top_k": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 20,
                            "description": "How many passages to return. Defaults to the project setting.",
                        },
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
            ),
        )

    async def execute(self, name: str, arguments: dict[str, Any]) -> ToolOutcome:
        if name != TOOL_NAME:
            return ToolOutcome(content=f"No such tool: {name}", is_error=True)

        query = (arguments.get("query") or "").strip()
        if not query:
            return ToolOutcome(
                content="The 'query' argument is required and must be non-empty.",
                is_error=True,
            )

        top_k = arguments.get("top_k")
        hits = await self._store.search(
            query,
            top_k=int(top_k) if isinstance(top_k, int) else None,
            audiences=self._audiences,
        )
        log.info("retrieval_search", query=query, hits=len(hits))

        if not hits:
            # Say so plainly. A model told "nothing found" will say it doesn't
            # know; a model handed weak matches will confabulate around them.
            return ToolOutcome(
                content=(
                    f"No passages in the knowledge base matched {query!r} above the "
                    f"relevance threshold. Do not guess an answer from this result -- "
                    f"either try a different query or tell the user the information "
                    f"isn't available."
                )
            )

        return ToolOutcome(content=format_hits(hits), sources=[_as_source(h) for h in hits])


def format_hits(hits: Sequence[SearchHit]) -> str:
    """Render passages for the model.

    Numbered and titled so the model can refer to "[2]" in its answer and the
    client can line that up with the citation list it received separately.
    """
    blocks: list[str] = []
    for i, hit in enumerate(hits, start=1):
        header = f"[{i}] {hit.title}"
        if hit.uri:
            header += f" ({hit.uri})"
        header += f" -- relevance {hit.score:.2f}"
        blocks.append(f"{header}\n{hit.text}")
    return (
        "Relevant passages from the knowledge base:\n\n"
        + "\n\n---\n\n".join(blocks)
        + "\n\nGround your answer in these passages and cite them as [1], [2], etc. "
        "If they don't contain the answer, say so rather than filling the gap."
    )


def _as_source(hit: SearchHit) -> dict[str, Any]:
    return {
        "id": hit.id,
        "title": hit.title,
        "uri": hit.uri,
        "score": round(hit.score, 4),
        "excerpt": hit.text[:300].strip(),
    }


__all__ = ["TOOL_NAME", "RetrievalTool", "format_hits"]
