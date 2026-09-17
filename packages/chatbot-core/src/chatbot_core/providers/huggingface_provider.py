"""Hugging Face provider.

The escape hatch for a project that needs a self-hosted or fine-tuned model --
data residency, a domain-tuned voice, or cost at volume. It speaks the same
event protocol as the Anthropic provider, so swapping one project over is a
config change.

Caveats worth knowing before you pick this for a tool-heavy project:

* Tool calling depends entirely on the chosen model and on the serving stack
  advertising an OpenAI-compatible ``tools`` parameter. Many models on the
  Inference API do not, in which case the assistant degrades to text-only.
  ``supports_tools`` makes that choice explicit rather than silently lossy.
* Open-weight models are generally weaker at multi-step tool use than Claude.
  Prefer this for retrieval-grounded Q&A over agentic workflows.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator, Sequence
from typing import Any

import structlog

from ..config import ProjectConfig
from ..events import (
    ChatEvent,
    Citations,
    Done,
    ErrorEvent,
    MessageStart,
    Source,
    TextDelta,
    ToolCall,
    ToolResult,
    Usage,
    truncate,
)
from .base import ChatProviderBase, Message, ToolExecutor, ToolOutcome

log = structlog.get_logger(__name__)


class HuggingFaceProvider(ChatProviderBase):
    #: Flip to True only after confirming the target model reliably emits
    #: well-formed tool calls. A model that hallucinates tool syntax into its
    #: prose is worse than one with no tools at all.
    supports_tools: bool = False

    def __init__(self, config: ProjectConfig, client: Any | None = None) -> None:
        super().__init__(config)
        if client is not None:
            self._client = client
        else:
            try:
                from huggingface_hub import AsyncInferenceClient
            except ImportError as exc:  # pragma: no cover
                raise ImportError(
                    "The huggingface provider needs the 'huggingface' extra: "
                    "uv add 'chatbot-core[huggingface]'"
                ) from exc
            import os

            self._client = AsyncInferenceClient(
                model=config.model.hf_endpoint_url or config.model.id,
                token=os.environ.get("HF_TOKEN") or None,
            )

    async def aclose(self) -> None:
        close = getattr(self._client, "close", None)
        if close is not None:
            result = close()
            if asyncio.iscoroutine(result):
                await result

    async def stream(
        self,
        *,
        system: str,
        messages: Sequence[Message],
        executor: ToolExecutor,
        conversation_id: str,
    ) -> AsyncIterator[ChatEvent]:
        convo: list[dict[str, Any]] = [{"role": "system", "content": system}]
        convo += [{"role": m.role, "content": m.content} for m in messages]

        tools = _openai_tools(executor) if self.supports_tools else None
        usage = Usage()
        emitted_start = False
        seen_source_ids: set[str] = set()

        try:
            for _ in range(self.config.mcp.max_iterations):
                if not emitted_start:
                    yield MessageStart(
                        conversation_id=conversation_id,
                        message_id=f"hf-{int(time.time() * 1000)}",
                        model=self.config.model.id,
                    )
                    emitted_start = True

                kwargs: dict[str, Any] = {
                    "messages": convo,
                    "max_tokens": self.config.model.max_tokens,
                    "stream": True,
                }
                if self.config.model.temperature is not None:
                    kwargs["temperature"] = self.config.model.temperature
                if tools:
                    kwargs["tools"] = tools
                    kwargs["tool_choice"] = "auto"

                text_parts: list[str] = []
                # Tool-call fragments arrive spread across deltas and keyed by
                # index, so accumulate per index and only parse once complete.
                pending: dict[int, dict[str, Any]] = {}

                async for chunk in await self._client.chat_completion(**kwargs):
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta
                    if getattr(delta, "content", None):
                        text_parts.append(delta.content)
                        yield TextDelta(text=delta.content)
                    for call in getattr(delta, "tool_calls", None) or []:
                        slot = pending.setdefault(
                            call.index, {"id": "", "name": "", "args": ""}
                        )
                        if getattr(call, "id", None):
                            slot["id"] = call.id
                        fn = getattr(call, "function", None)
                        if fn is not None:
                            if getattr(fn, "name", None):
                                slot["name"] = fn.name
                            if getattr(fn, "arguments", None):
                                slot["args"] += fn.arguments
                    if getattr(chunk, "usage", None):
                        usage.input_tokens += chunk.usage.prompt_tokens or 0
                        usage.output_tokens += chunk.usage.completion_tokens or 0

                assistant_text = "".join(text_parts)

                if not pending:
                    convo.append({"role": "assistant", "content": assistant_text})
                    yield Done(stop_reason="end_turn", usage=usage)
                    return

                calls = [
                    {
                        "id": slot["id"] or f"call-{idx}",
                        "type": "function",
                        "function": {"name": slot["name"], "arguments": slot["args"]},
                    }
                    for idx, slot in sorted(pending.items())
                ]
                convo.append(
                    {
                        "role": "assistant",
                        "content": assistant_text or None,
                        "tool_calls": calls,
                    }
                )

                for call in calls:
                    yield ToolCall(
                        id=call["id"],
                        name=call["function"]["name"],
                        input=_safe_json(call["function"]["arguments"]),
                    )

                outcomes = await asyncio.gather(
                    *(
                        _invoke(
                            executor,
                            c["function"]["name"],
                            _safe_json(c["function"]["arguments"]),
                        )
                        for c in calls
                    )
                )

                for call, (outcome, elapsed_ms) in zip(calls, outcomes, strict=True):
                    yield ToolResult(
                        id=call["id"],
                        name=call["function"]["name"],
                        ok=not outcome.is_error,
                        preview=truncate(outcome.content),
                        duration_ms=elapsed_ms,
                    )
                    convo.append(
                        {
                            "role": "tool",
                            "tool_call_id": call["id"],
                            "content": outcome.content,
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

            yield ErrorEvent(
                message="I used up my tool budget for this turn.", retriable=True
            )
            yield Done(stop_reason="max_iterations", usage=usage)

        except Exception as exc:  # noqa: BLE001 - HF client raises a wide variety
            log.exception("huggingface_stream_failed", project=self.config.project.id)
            yield ErrorEvent(
                message="I couldn't reach the model. Try again.",
                retriable=not isinstance(exc, ValueError),
            )
            yield Done(stop_reason="error", usage=usage)


def _openai_tools(executor: ToolExecutor) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": s.name,
                "description": s.description,
                "parameters": s.input_schema,
            },
        }
        for s in executor.specs
    ]


def _safe_json(raw: str) -> dict[str, Any]:
    """Open-weight models emit malformed argument JSON often enough that this
    has to be non-fatal. An empty dict lets the tool report a clear validation
    error, which the model can then correct."""
    try:
        parsed = json.loads(raw or "{}")
    except json.JSONDecodeError:
        log.warning("tool_arguments_unparseable", raw=truncate(raw, 200))
        return {}
    return parsed if isinstance(parsed, dict) else {}


async def _invoke(
    executor: ToolExecutor, name: str, arguments: dict[str, Any]
) -> tuple[ToolOutcome, int]:
    started = time.perf_counter()
    try:
        outcome = await executor.execute(name, arguments)
    except Exception as exc:  # noqa: BLE001
        log.exception("tool_execution_failed", tool=name)
        outcome = ToolOutcome(content=f"Tool '{name}' failed: {exc}", is_error=True)
    return outcome, int((time.perf_counter() - started) * 1000)


__all__ = ["HuggingFaceProvider"]
