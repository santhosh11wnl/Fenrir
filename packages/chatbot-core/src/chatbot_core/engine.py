"""The chat engine.

Owns the objects a project needs at runtime -- provider, MCP session, vector
store -- and hands the provider a single merged tool surface. Built once per
process at startup, not per request: the embedding model is hundreds of
megabytes and the MCP session is a live connection.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Any

import structlog

from .config import ProjectConfig
from .events import ChatEvent, Citations, Done, ErrorEvent, Source
from .mcp import MCPClient
from .providers import ChatProviderBase, Message, ToolOutcome, ToolSpec, build_provider
from .retrieval import RetrievalTool, build_embedder, build_store, format_hits
from .retrieval.store import VectorStore

log = structlog.get_logger(__name__)


class CompositeExecutor:
    """Presents MCP tools and the retrieval tool as one surface.

    Name collisions resolve in favour of whichever executor registered first,
    and are logged. Silently shadowing a tool would be a genuinely confusing
    bug to chase -- the model would call a name and get someone else's tool.
    """

    def __init__(self, *executors: Any) -> None:
        self._executors = [e for e in executors if e is not None]
        self._routes: dict[str, Any] = {}
        self._specs: list[ToolSpec] = []
        self.rebuild()

    def rebuild(self) -> None:
        self._routes.clear()
        self._specs.clear()
        for executor in self._executors:
            for spec in executor.specs:
                if spec.name in self._routes:
                    log.warning("duplicate_tool_name", tool=spec.name)
                    continue
                self._routes[spec.name] = executor
                self._specs.append(spec)

    @property
    def specs(self) -> Sequence[ToolSpec]:
        return self._specs

    async def execute(self, name: str, arguments: dict[str, Any]) -> ToolOutcome:
        executor = self._routes.get(name)
        if executor is None:
            available = ", ".join(sorted(self._routes)) or "none"
            return ToolOutcome(
                content=f"No such tool: {name!r}. Available tools: {available}.",
                is_error=True,
            )
        return await executor.execute(name, arguments)


#: Messages that are pure social nicety. Retrieval adds nothing to these, and
#: on a slow backend the extra prefill is most of the perceived wait.
#:
#: Matched whole-message only, after stripping punctuation. That is the
#: conservative direction: "hi" skips retrieval, "hi, where is my order?" does
#: not. A false skip means answering a real question with no corpus, which is
#: the failure this bot exists to avoid -- so anything not on this list
#: retrieves, including anything unrecognised.
_SMALL_TALK = frozenset(
    {
        "hi", "hii", "hey", "yo", "hello", "helo", "hiya", "howdy",
        "good morning", "good afternoon", "good evening", "morning", "evening",
        "thanks", "thank you", "thanks a lot", "thank you so much", "ty", "thx",
        "cheers", "nice", "great", "cool", "ok", "okay", "k", "got it", "sure",
        "bye", "goodbye", "see you", "see ya", "later", "good night", "night",
        "how are you", "how are you doing", "hows it going", "how is it going",
        "whats up", "sup", "you there", "are you there", "test", "testing",
        "who are you", "what are you", "what can you do", "help",
    }
)

#: Longer than this and it is a real question whatever it looks like, so the
#: cheap check is skipped rather than risking a clever false match.
_SMALL_TALK_MAX_CHARS = 32


def needs_retrieval(query: str) -> bool:
    """Whether a message is worth searching the knowledge base for.

    Fails towards retrieving: only an exact whole-message match against
    :data:`_SMALL_TALK` skips it.
    """
    text = query.strip().lower()
    if len(text) > _SMALL_TALK_MAX_CHARS:
        return True
    # Keep intra-word apostrophes out of the way ("what's" -> "whats") and drop
    # trailing punctuation, so "Hi!!" and "hi" are the same message.
    cleaned = "".join(c for c in text if c.isalnum() or c.isspace()).strip()
    cleaned = " ".join(cleaned.split())
    return cleaned not in _SMALL_TALK


class ChatEngine:
    def __init__(self, config: ProjectConfig) -> None:
        self.config = config
        self._provider: ChatProviderBase | None = None
        self._mcp: MCPClient | None = None
        self._store: VectorStore | None = None
        self._executor: CompositeExecutor | None = None
        self._started = False

    # -- lifecycle ---------------------------------------------------------

    async def startup(self) -> None:
        """Build everything the engine needs. Idempotent."""
        if self._started:
            return
        cfg = self.config
        log.info("engine_starting", project=cfg.project.id, model=cfg.model.id)

        self._provider = build_provider(cfg)

        retrieval_tool: RetrievalTool | None = None
        if cfg.retrieval.enabled:
            embedder = build_embedder(cfg.retrieval.embedding)
            self._store = build_store(cfg.retrieval, cfg.collection, embedder)
            await self._store.ensure_ready()
            count = await self._store.count()
            log.info("vector_store_ready", collection=cfg.collection, chunks=count)
            if count == 0:
                # Not fatal -- a project can be deployed before its corpus is
                # ingested -- but silence here leads to "why does it say it
                # doesn't know?" a week later.
                log.warning("vector_store_empty", collection=cfg.collection)
            if cfg.retrieval.as_tool:
                retrieval_tool = RetrievalTool(self._store)

        if cfg.mcp.enabled:
            self._mcp = MCPClient(cfg.mcp)
            await self._mcp.connect()

        self._executor = CompositeExecutor(self._mcp, retrieval_tool)
        log.info("engine_ready", tools=[s.name for s in self._executor.specs])
        self._started = True

    async def shutdown(self) -> None:
        for resource in (self._provider, self._mcp, self._store):
            if resource is None:
                continue
            try:
                await resource.aclose()
            except Exception:  # noqa: BLE001 - shutdown must not raise
                log.exception("shutdown_error", resource=type(resource).__name__)
        self._started = False

    # -- introspection -----------------------------------------------------

    @property
    def tools(self) -> Sequence[ToolSpec]:
        return self._executor.specs if self._executor else ()

    async def health(self) -> dict[str, Any]:
        return {
            "project": self.config.project.id,
            "model": self.config.model.id,
            "provider": self.config.model.provider.value,
            "ready": self._started,
            "mcp_connected": bool(self._mcp and self._mcp.connected),
            "tools": [s.name for s in self.tools],
            "indexed_chunks": (await self._store.count()) if self._store else 0,
        }

    # -- the turn ----------------------------------------------------------

    async def stream(
        self,
        *,
        conversation_id: str,
        messages: Sequence[Message],
        roles: frozenset[str] | None = None,
    ) -> AsyncIterator[ChatEvent]:
        """Run one assistant turn.

        Args:
            roles: The authenticated caller's roles, or ``None`` for an
                anonymous visitor. These decide which document audiences the
                turn may retrieve, and which role guidance is added to the
                system prompt.

        Never raises for a recoverable problem: the client always receives a
        well-formed stream ending in ``Done``, so a hung spinner is impossible.
        """
        if not self._started or self._provider is None or self._executor is None:
            yield ErrorEvent(message="The assistant is still starting up.", retriable=True)
            yield Done(stop_reason="not_ready")
            return

        if not messages:
            yield ErrorEvent(message="No message to respond to.", retriable=False)
            yield Done(stop_reason="empty_request")
            return

        # Resolved once per turn from the authenticated caller. The model is
        # never given a way to influence this -- it is not a tool argument and
        # not mentioned in the prompt as something negotiable.
        audiences = self.config.auth.audiences_for(roles)

        system = self.config.system_prompt
        if (role_guidance := self.config.auth.prompt_for(roles)):
            system = f"{system}\n\n{role_guidance}"

        history = list(messages)

        # Prepend mode: fetch context now, rather than letting the model decide
        # to search. Skipped for pure small talk -- "hi" cannot be answered any
        # better with three passages of returns policy in front of it, and on a
        # slow backend the extra prefill is most of the wait.
        if self._should_prepend() and needs_retrieval(history[-1].content):
            context, sources = await self._prefetch(history[-1].content, audiences)
            if sources:
                yield Citations(sources=sources)
            if context:
                # Deliberately attached to the user turn, NOT to the system
                # prompt. Retrieved context changes on every question, and the
                # system prompt is the very first thing in the prompt -- so
                # folding it in there changes the prefix and throws away the
                # backend's KV cache every single turn. Kept stable, the whole
                # system prompt stays cached and only the new context is
                # prefilled. That is the difference between re-processing ~600
                # tokens per turn and re-processing none.
                last = history[-1]
                history[-1] = Message(
                    role=last.role,
                    content=f"{context}\n\n---\n\nUsing the passages above where "
                    f"they are relevant, answer:\n\n{last.content}",
                )

        # Trim oldest turns first. The system prompt carries the cache
        # breakpoint, so trimming history leaves the cached prefix intact.
        limit = self.config.limits.max_history_messages
        if len(history) > limit:
            history = history[-limit:]

        # The retrieval tool is rebuilt per turn so it carries this caller's
        # audiences. MCP tools are stateless and shared.
        executor = self._executor
        if self._store is not None and self.config.retrieval.as_tool:
            executor = CompositeExecutor(
                self._mcp, RetrievalTool(self._store, audiences=audiences)
            )

        async for event in self._provider.stream(
            system=system,
            messages=history,
            executor=executor,
            conversation_id=conversation_id,
        ):
            yield event

    def _should_prepend(self) -> bool:
        return (
            self.config.retrieval.enabled
            and not self.config.retrieval.as_tool
            and self._store is not None
        )

    async def _prefetch(
        self, query: str, audiences: frozenset[str]
    ) -> tuple[str, list[Source]]:
        assert self._store is not None
        try:
            hits = await self._store.search(query, audiences=audiences)
        except Exception:  # noqa: BLE001 - degrade to un-grounded rather than fail
            log.exception("prefetch_failed", project=self.config.project.id)
            return "", []
        if not hits:
            return "", []
        sources = [
            Source(
                id=h.id,
                title=h.title,
                uri=h.uri,
                score=round(h.score, 4),
                excerpt=h.text[:300].strip(),
            )
            for h in hits
        ]
        return format_hits(hits), sources


__all__ = ["ChatEngine", "CompositeExecutor", "needs_retrieval"]
