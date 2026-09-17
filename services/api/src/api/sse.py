"""Server-sent event serialisation.

The bridge between ``chatbot_core.events`` and the wire. Each event becomes one
SSE frame whose ``event:`` field is the discriminator and whose ``data:`` is the
model's JSON -- so the browser can attach per-type listeners and the payload
matches the Pydantic model exactly.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import structlog

from chatbot_core.events import ChatEvent, Done, ErrorEvent

log = structlog.get_logger(__name__)


def to_frame(event: ChatEvent) -> dict[str, str]:
    """Render one event as an ``EventSourceResponse`` frame."""
    return {"event": event.type.value, "data": event.model_dump_json()}


async def stream_events(
    events: AsyncIterator[ChatEvent], *, request_id: str
) -> AsyncIterator[dict[str, str]]:
    """Serialise an event stream, guaranteeing a terminal frame.

    A client that never receives ``done`` shows a spinner forever, so an
    unexpected failure mid-stream is converted into ``error`` + ``done`` rather
    than being allowed to propagate and drop the connection.

    The user-facing message deliberately carries no exception detail; the full
    traceback goes to the log under ``request_id`` so support can correlate the
    two without leaking internals to the browser.
    """
    saw_done = False
    try:
        async for event in events:
            if isinstance(event, Done):
                saw_done = True
            yield to_frame(event)
    except Exception:  # noqa: BLE001 - a hung client is worse than a logged error
        log.exception("stream_failed", request_id=request_id)
        yield to_frame(
            ErrorEvent(
                message="Something went wrong generating a response.",
                retriable=True,
                request_id=request_id,
            )
        )
        yield to_frame(Done(stop_reason="error"))
        return

    if not saw_done:
        # A provider that returned without a terminal event is a bug, but the
        # client should not be the one that pays for it.
        log.warning("stream_ended_without_done", request_id=request_id)
        yield to_frame(Done(stop_reason="incomplete"))


__all__ = ["stream_events", "to_frame"]
