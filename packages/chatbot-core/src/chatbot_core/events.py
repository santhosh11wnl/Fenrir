"""The streaming event protocol.

One contract, three consumers: the engine emits these, the API serialises them
as SSE, the React client renders them. Changing a field here means changing all
three -- which is the point of having it in one place.

Every event carries a ``type`` discriminator so the client can switch on it
without guessing from shape.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class EventType(StrEnum):
    MESSAGE_START = "message_start"
    TEXT_DELTA = "text_delta"
    THINKING_DELTA = "thinking_delta"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    CITATIONS = "citations"
    ERROR = "error"
    DONE = "done"


class _Event(BaseModel):
    model_config = ConfigDict(extra="forbid")


class MessageStart(_Event):
    type: Literal[EventType.MESSAGE_START] = EventType.MESSAGE_START
    conversation_id: str
    message_id: str
    model: str


class TextDelta(_Event):
    """An incremental chunk of the visible answer. Append in arrival order."""

    type: Literal[EventType.TEXT_DELTA] = EventType.TEXT_DELTA
    text: str


class ThinkingDelta(_Event):
    """Summarised reasoning. Only emitted when ``model.show_thinking`` is on."""

    type: Literal[EventType.THINKING_DELTA] = EventType.THINKING_DELTA
    text: str


class ToolCall(_Event):
    """The model asked for a tool. Emitted *before* the tool runs, so the UI can
    show a pending row rather than a gap."""

    type: Literal[EventType.TOOL_CALL] = EventType.TOOL_CALL
    id: str
    name: str
    input: dict[str, Any]


class ToolResult(_Event):
    """A tool finished. ``preview`` is truncated for display -- the full result
    goes to the model, not to the browser, because tool output can be large and
    can contain data the end user shouldn't see."""

    type: Literal[EventType.TOOL_RESULT] = EventType.TOOL_RESULT
    id: str
    name: str
    ok: bool
    preview: str
    duration_ms: int | None = None


class Source(BaseModel):
    """One retrieved chunk, as shown to the user."""

    model_config = ConfigDict(extra="forbid")

    id: str
    title: str
    uri: str | None = None
    score: float
    excerpt: str


class Citations(_Event):
    """Retrieved context backing the answer. Emitted once, before the text that
    depends on it, so the client can render sources alongside the response."""

    type: Literal[EventType.CITATIONS] = EventType.CITATIONS
    sources: list[Source]


class ErrorEvent(_Event):
    """A failure the client should surface.

    ``message`` is user-facing and must never carry provider internals, stack
    traces, or credentials -- the full detail goes to the server log, keyed by
    ``request_id`` so support can correlate the two.
    """

    type: Literal[EventType.ERROR] = EventType.ERROR
    message: str
    retriable: bool = False
    request_id: str | None = None


class Usage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0


class Done(_Event):
    type: Literal[EventType.DONE] = EventType.DONE
    stop_reason: str | None = None
    usage: Usage = Field(default_factory=Usage)


ChatEvent = Annotated[
    MessageStart
    | TextDelta
    | ThinkingDelta
    | ToolCall
    | ToolResult
    | Citations
    | ErrorEvent
    | Done,
    Field(discriminator="type"),
]


def truncate(text: str, limit: int = 600) -> str:
    """Shorten tool output for display without lying about the length."""
    if len(text) <= limit:
        return text
    return f"{text[:limit]}... [{len(text) - limit} more characters]"


__all__ = [
    "ChatEvent",
    "Citations",
    "Done",
    "ErrorEvent",
    "EventType",
    "MessageStart",
    "Source",
    "TextDelta",
    "ThinkingDelta",
    "ToolCall",
    "ToolResult",
    "Usage",
    "truncate",
]
