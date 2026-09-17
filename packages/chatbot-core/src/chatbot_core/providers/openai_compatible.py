"""Self-hosted chat models over the OpenAI-compatible protocol.

The default provider. It targets a protocol rather than a vendor, which means
one implementation covers every serving stack worth using:

===============  ========================================  ==================
Backend          ``base_url``                              Typical use
===============  ========================================  ==================
Ollama           ``http://ollama:11434/v1``                Laptop / small box
vLLM             ``http://vllm:8000/v1``                   GPU server, high QPS
llama.cpp        ``http://llama:8080/v1``                  CPU / edge
LM Studio        ``http://host.docker.internal:1234/v1``   Desktop GUI
HF TGI           ``http://tgi:80/v1``                      HF-native serving
===============  ========================================  ==================

Moving from a laptop to a GPU server is a ``base_url`` and a ``model.id`` in
``config.yaml``. No code changes, which is the whole point of putting the
protocol -- not the vendor -- behind the interface.

Tool calling
------------
Support varies by *model*, not by server. A model trained for function calling
(Qwen, Llama 3.1+, Mistral-Nemo, Hermes) emits structured ``tool_calls``;
others ignore the ``tools`` parameter and sometimes hallucinate tool syntax
into their prose, which is worse than having no tools at all. ``supports_tools``
in config makes that an explicit decision rather than a silent degradation.
"""

from __future__ import annotations

import asyncio
import json
import os
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


class OpenAICompatibleProvider(ChatProviderBase):
    def __init__(self, config: ProjectConfig, client: Any | None = None) -> None:
        super().__init__(config)
        if client is not None:
            self._client = client
            return

        try:
            from openai import AsyncOpenAI
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "The local provider needs the openai client (it speaks the "
                "OpenAI protocol to your own server): uv add openai"
            ) from exc

        model = config.model
        # Most self-hosted servers ignore the key entirely, but the client
        # refuses to construct without one, hence the placeholder.
        api_key = os.environ.get(model.api_key_env or "", "") or "not-needed"
        self._client = AsyncOpenAI(
            base_url=model.base_url,
            api_key=api_key,
            # A local model on a cold start loads weights before the first
            # token. The default 10 minutes is generous; the real protection
            # is that we stream, so a slow model shows progress meanwhile.
            timeout=model.request_timeout,
            max_retries=2,
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
        model = self.config.model
        convo: list[dict[str, Any]] = [{"role": "system", "content": system}]
        convo += [{"role": m.role, "content": m.content} for m in messages]

        tools = _as_openai_tools(executor) if model.supports_tools else None
        usage = Usage()
        emitted_start = False
        seen_source_ids: set[str] = set()

        try:
            for _ in range(self.config.mcp.max_iterations):
                if not emitted_start:
                    yield MessageStart(
                        conversation_id=conversation_id,
                        message_id=f"local-{int(time.time() * 1000)}",
                        model=model.id,
                    )
                    emitted_start = True

                kwargs: dict[str, Any] = {
                    "model": model.id,
                    "messages": convo,
                    "max_tokens": model.max_tokens,
                    "stream": True,
                    # Ask the server to report usage on the final chunk. Servers
                    # that don't understand this ignore it.
                    "stream_options": {"include_usage": True},
                }
                if model.temperature is not None:
                    kwargs["temperature"] = model.temperature
                if model.top_p is not None:
                    kwargs["top_p"] = model.top_p
                if tools:
                    kwargs["tools"] = tools
                    kwargs["tool_choice"] = "auto"

                text_parts: list[str] = []
                # Tool-call fragments arrive across many deltas, keyed by index.
                # Accumulate per index; only parse once the stream completes.
                pending: dict[int, dict[str, str]] = {}

                stream = await self._client.chat.completions.create(**kwargs)
                async for chunk in stream:
                    if getattr(chunk, "usage", None):
                        usage.input_tokens += chunk.usage.prompt_tokens or 0
                        usage.output_tokens += chunk.usage.completion_tokens or 0
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

                assistant_text = "".join(text_parts)

                if not pending:
                    convo.append({"role": "assistant", "content": assistant_text})
                    yield Done(stop_reason="end_turn", usage=usage)
                    return

                calls = [
                    {
                        "id": slot["id"] or f"call_{idx}",
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

            log.warning("agent_loop_exhausted", project=self.config.project.id)
            yield ErrorEvent(
                message="I used up my tool budget for this turn without reaching "
                "an answer. Try narrowing the question.",
                retriable=True,
            )
            yield Done(stop_reason="max_iterations", usage=usage)

        except Exception as exc:  # noqa: BLE001 - see _explain for the triage
            log.exception(
                "local_model_stream_failed",
                project=self.config.project.id,
                base_url=model.base_url,
                model=model.id,
            )
            message, retriable = _explain(exc, model.id)
            yield ErrorEvent(message=message, retriable=retriable)
            yield Done(stop_reason="error", usage=usage)


def _explain(exc: Exception, model_id: str) -> tuple[str, bool]:
    """Turn a client exception into something a user can act on.

    Self-hosted serving fails in a few very specific ways, and each has a
    different fix. A generic "something went wrong" would send someone hunting
    through logs for what is usually a one-line problem.
    """
    text = str(exc).lower()
    if "connection" in text or "connect" in text:
        return (
            "I can't reach the model server. Check that it's running and that "
            "base_url in config.yaml points at it.",
            True,
        )
    if "not found" in text or "404" in text:
        return (
            f"The model {model_id!r} isn't available on the server. Pull it "
            f"first (e.g. `ollama pull {model_id}`) or correct model.id.",
            False,
        )
    if "timeout" in text or "timed out" in text:
        return (
            "The model took too long to respond. It may still be loading "
            "weights -- try again, or raise request_timeout.",
            True,
        )
    if "context" in text and ("length" in text or "window" in text):
        return (
            "The conversation exceeded the model's context window. Start a new "
            "conversation, or lower max_history_messages or retrieval.top_k.",
            False,
        )
    return ("Something went wrong talking to the model.", True)


def _as_openai_tools(executor: ToolExecutor) -> list[dict[str, Any]]:
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
    """Parse tool arguments, tolerating malformed output.

    Open-weight models emit invalid argument JSON often enough that this must
    be non-fatal. An empty dict lets the tool return a clear validation error,
    which the model can then correct -- far better than ending the turn.
    """
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
    except Exception as exc:  # noqa: BLE001 - a tool must not kill the turn
        log.exception("tool_execution_failed", tool=name)
        outcome = ToolOutcome(content=f"Tool '{name}' failed: {exc}", is_error=True)
    return outcome, int((time.perf_counter() - started) * 1000)


__all__ = ["OpenAICompatibleProvider"]
