"""Anthropic provider -- the default engine for every project.

Claude drives the agent loop because tool use is where it is furthest ahead,
and tool use is the whole point of an MCP-backed assistant.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Sequence
from typing import Any

import structlog
from anthropic import (
    APIConnectionError,
    APIStatusError,
    AsyncAnthropic,
    AuthenticationError,
    RateLimitError,
)

from ..config import ProjectConfig
from ..events import (
    ChatEvent,
    Citations,
    Done,
    ErrorEvent,
    MessageStart,
    Source,
    TextDelta,
    ThinkingDelta,
    ToolCall,
    ToolResult,
    Usage,
    truncate,
)
from .base import ChatProviderBase, Message, ToolExecutor, ToolOutcome

log = structlog.get_logger(__name__)


class AnthropicProvider(ChatProviderBase):
    def __init__(self, config: ProjectConfig, client: AsyncAnthropic | None = None) -> None:
        super().__init__(config)
        # Zero-arg construction resolves ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN,
        # or an `ant auth login` profile, in that order. Don't pass a key here.
        self._client = client or AsyncAnthropic(max_retries=3)
        self._owns_client = client is None

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.close()

    # -- request shaping ---------------------------------------------------

    def _request_kwargs(self, system: str, tools: list[dict[str, Any]]) -> dict[str, Any]:
        m = self.config.model
        kwargs: dict[str, Any] = {
            "model": m.id,
            "max_tokens": m.max_tokens,
            # A single cache breakpoint on the last system block covers tools +
            # system together, because the render order is tools -> system ->
            # messages. That is the large, stable prefix; the conversation after
            # it is what varies.
            "system": [
                {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
            ],
            "output_config": {"effort": m.effort.value},
        }
        if m.adaptive_thinking:
            thinking: dict[str, Any] = {"type": "adaptive"}
            if m.show_thinking:
                # Without this the thinking blocks stream with empty text, which
                # in a UI reads as a long unexplained pause before any output.
                thinking["display"] = "summarized"
            kwargs["thinking"] = thinking
        else:
            kwargs["thinking"] = {"type": "disabled"}
        if tools:
            kwargs["tools"] = tools
        return kwargs

    @staticmethod
    def _tool_payload(executor: ToolExecutor) -> list[dict[str, Any]]:
        return [
            {"name": s.name, "description": s.description, "input_schema": s.input_schema}
            for s in executor.specs
        ]

    # -- the loop ----------------------------------------------------------

    async def stream(
        self,
        *,
        system: str,
        messages: Sequence[Message],
        executor: ToolExecutor,
        conversation_id: str,
    ) -> AsyncIterator[ChatEvent]:
        tools = self._tool_payload(executor)
        base = self._request_kwargs(system, tools)
        api_messages: list[dict[str, Any]] = [
            {"role": m.role, "content": m.content} for m in messages
        ]

        usage = Usage()
        emitted_start = False
        seen_source_ids: set[str] = set()

        try:
            for _ in range(self.config.mcp.max_iterations):
                async with self._client.messages.stream(
                    **base, messages=api_messages
                ) as stream:
                    async for event in stream:
                        if event.type != "content_block_delta":
                            continue
                        if event.delta.type == "text_delta":
                            yield TextDelta(text=event.delta.text)
                        elif (
                            event.delta.type == "thinking_delta"
                            and self.config.model.show_thinking
                        ):
                            yield ThinkingDelta(text=event.delta.thinking)

                    final = await stream.get_final_message()

                if not emitted_start:
                    yield MessageStart(
                        conversation_id=conversation_id,
                        message_id=final.id,
                        model=final.model,
                    )
                    emitted_start = True

                _accumulate(usage, final.usage)

                # A refusal is a successful HTTP 200 with an empty or partial
                # body -- reading content[0] here would be the crash.
                if final.stop_reason == "refusal":
                    category = getattr(final.stop_details, "category", None)
                    log.warning("claude_refusal", category=category, project=self.config.project.id)
                    yield ErrorEvent(
                        message=(
                            "I can't help with that request. If you think this is a "
                            "mistake, try rephrasing it."
                        ),
                        retriable=False,
                    )
                    yield Done(stop_reason="refusal", usage=usage)
                    return

                # Append the assistant turn verbatim. Thinking blocks carry
                # signatures the API validates -- reconstructing or trimming
                # them is what causes mysterious 400s on the next request.
                api_messages.append({"role": "assistant", "content": final.content})

                # A server-side tool hit its iteration cap. Re-send as-is; the
                # server resumes. Do not inject a "continue" message.
                if final.stop_reason == "pause_turn":
                    continue

                if final.stop_reason != "tool_use":
                    yield Done(stop_reason=final.stop_reason, usage=usage)
                    return

                tool_uses = [b for b in final.content if b.type == "tool_use"]
                for block in tool_uses:
                    yield ToolCall(
                        id=block.id, name=block.name, input=dict(block.input or {})
                    )

                outcomes = await _run_tools(executor, tool_uses)

                results: list[dict[str, Any]] = []
                for block, (outcome, elapsed_ms) in zip(tool_uses, outcomes, strict=True):
                    yield ToolResult(
                        id=block.id,
                        name=block.name,
                        ok=not outcome.is_error,
                        preview=truncate(outcome.content),
                        duration_ms=elapsed_ms,
                    )
                    # Every tool_use needs a matching tool_result, including
                    # failures -- a missing one is a 400 on the next request.
                    results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": outcome.content,
                            "is_error": outcome.is_error,
                        }
                    )
                    fresh = [
                        Source(**s)
                        for s in outcome.sources
                        if s.get("id") not in seen_source_ids
                    ]
                    if fresh:
                        seen_source_ids.update(s.id for s in fresh)
                        yield Citations(sources=fresh)

                # All results in ONE user message. Splitting them across several
                # trains the model to stop making parallel tool calls.
                api_messages.append({"role": "user", "content": results})

            log.warning(
                "agent_loop_exhausted",
                project=self.config.project.id,
                max_iterations=self.config.mcp.max_iterations,
            )
            yield ErrorEvent(
                message=(
                    "I used up my tool budget for this turn without reaching an "
                    "answer. Try narrowing the question."
                ),
                retriable=True,
            )
            yield Done(stop_reason="max_iterations", usage=usage)

        except AuthenticationError:
            log.exception("anthropic_auth_failed", project=self.config.project.id)
            yield ErrorEvent(
                message="The assistant is misconfigured and can't reach its model.",
                retriable=False,
            )
            yield Done(stop_reason="error", usage=usage)
        except RateLimitError:
            log.warning("anthropic_rate_limited", project=self.config.project.id)
            yield ErrorEvent(message="Rate limited. Try again shortly.", retriable=True)
            yield Done(stop_reason="error", usage=usage)
        except APIConnectionError:
            log.exception("anthropic_connection_error", project=self.config.project.id)
            yield ErrorEvent(
                message="I couldn't reach the model. Try again.", retriable=True
            )
            yield Done(stop_reason="error", usage=usage)
        except APIStatusError as exc:
            log.exception("anthropic_api_error", status=exc.status_code)
            yield ErrorEvent(
                message="Something went wrong talking to the model.",
                retriable=exc.status_code >= 500,
            )
            yield Done(stop_reason="error", usage=usage)


async def _run_tools(
    executor: ToolExecutor, blocks: list[Any]
) -> list[tuple[ToolOutcome, int]]:
    """Run every requested tool concurrently and time each one."""

    async def one(block: Any) -> tuple[ToolOutcome, int]:
        started = time.perf_counter()
        try:
            outcome = await executor.execute(block.name, dict(block.input or {}))
        except Exception as exc:  # noqa: BLE001 - a tool must never kill the turn
            log.exception("tool_execution_failed", tool=block.name)
            outcome = ToolOutcome(
                content=f"Tool '{block.name}' failed: {exc}", is_error=True
            )
        return outcome, int((time.perf_counter() - started) * 1000)

    return list(await asyncio.gather(*(one(b) for b in blocks)))


def _accumulate(total: Usage, delta: Any) -> None:
    total.input_tokens += getattr(delta, "input_tokens", 0) or 0
    total.output_tokens += getattr(delta, "output_tokens", 0) or 0
    total.cache_read_input_tokens += getattr(delta, "cache_read_input_tokens", 0) or 0
    total.cache_creation_input_tokens += (
        getattr(delta, "cache_creation_input_tokens", 0) or 0
    )


__all__ = ["AnthropicProvider"]
