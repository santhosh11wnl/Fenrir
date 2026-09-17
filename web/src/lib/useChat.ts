/**
 * Chat state and the SSE stream.
 *
 * Why `fetch` and not `EventSource`: EventSource only issues GET requests and
 * cannot set headers, so it can carry neither a message body nor an
 * Authorization header. Reading the response stream manually costs a few lines
 * of parsing and gets both.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import type { ChatEvent, ChatMessage, Theme, ToolActivity } from "./types";

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "/api";

/** Per-user API key, when the project has auth enabled. */
const API_KEY = import.meta.env.VITE_API_KEY ?? "";

function authHeaders(): Record<string, string> {
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (API_KEY) headers.Authorization = `Bearer ${API_KEY}`;
  return headers;
}

let messageCounter = 0;
const nextId = () => `m${++messageCounter}`;

/**
 * Smooths streamed text onto the screen at a steady rate.
 *
 * A local model does not emit tokens evenly: measured here, text arrives in
 * bursts of ~4-5 characters every ~130ms, with gaps up to 240ms. Writing each
 * burst straight into React state renders exactly that -- a line that jerks
 * forward in visible clumps and stalls between them. The text is correct; it
 * just looks broken.
 *
 * So arrival and display are decoupled. Deltas land in a buffer, and a
 * rAF loop drains it a few characters per frame. Nothing is dropped and
 * nothing is invented -- the same characters appear, paced evenly.
 *
 * The drain rate adapts to how far behind the display is, which matters
 * because generation speed varies by an order of magnitude between a laptop
 * and a server. A fixed rate would either lag badly on a fast backend or
 * stutter on a slow one:
 *
 *   - a small backlog drains slowly, keeping a typewriter feel
 *   - a large backlog drains fast, so the UI never falls behind the stream
 *   - on `finish()` the remainder is flushed immediately, so the final text
 *     is never delayed by animation
 */
function createTextSmoother(onRender: (chunk: string) => void) {
  let pending = "";
  let frame: number | null = null;
  let finished = false;

  const step = () => {
    frame = null;
    if (pending.length === 0) {
      // Streaming ended and the buffer is drained: nothing left to schedule.
      if (finished) return;
      return;
    }

    // Aim to clear the backlog in roughly a dozen frames (~200ms), with a
    // floor so short bursts still animate and a ceiling so a huge backlog
    // does not arrive as one jump.
    const take = Math.max(1, Math.min(pending.length, Math.ceil(pending.length / 12)));
    onRender(pending.slice(0, take));
    pending = pending.slice(take);

    if (pending.length > 0) frame = requestAnimationFrame(step);
  };

  const schedule = () => {
    if (frame === null && pending.length > 0) frame = requestAnimationFrame(step);
  };

  return {
    push(text: string) {
      pending += text;
      schedule();
    },
    /** Flush everything still buffered, immediately. */
    finish() {
      finished = true;
      if (frame !== null) {
        cancelAnimationFrame(frame);
        frame = null;
      }
      if (pending.length > 0) {
        onRender(pending);
        pending = "";
      }
    },
  };
}

/**
 * Frame separator in an SSE stream: a blank line.
 *
 * Both endings must be handled. The spec permits CRLF, LF, or a bare CR, and
 * servers differ -- sse-starlette emits CRLF, so a parser that only looks for
 * "\n\n" finds nothing at all: "\r\n\r\n" contains no adjacent newlines. The
 * failure is silent and total. The stream runs to completion, zero events are
 * parsed, and the UI shows an empty reply with no error anywhere.
 */
const FRAME_SEPARATOR = /\r\n\r\n|\n\n|\r\r/;

/**
 * Parse an SSE byte stream into events.
 *
 * Frames can arrive split across chunks, so the tail of each chunk is held
 * back until its terminator shows up. Splitting on chunk boundaries instead
 * would corrupt any message larger than one packet.
 */
async function* readEvents(
  body: ReadableStream<Uint8Array>,
  signal: AbortSignal,
): AsyncGenerator<ChatEvent> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  try {
    while (!signal.aborted) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      for (;;) {
        const match = FRAME_SEPARATOR.exec(buffer);
        if (!match) break;

        const frame = buffer.slice(0, match.index);
        buffer = buffer.slice(match.index + match[0].length);

        // Field lines may end with CR, LF, or CRLF; split on any of them.
        const dataLine = frame
          .split(/\r\n|\n|\r/)
          .find((line) => line.startsWith("data:"));
        if (!dataLine) continue;

        try {
          yield JSON.parse(dataLine.slice(5).trim()) as ChatEvent;
        } catch {
          // A malformed frame is not worth killing the stream over; the
          // terminal `done` event still arrives and settles the UI.
          console.warn("skipping unparseable SSE frame");
        }
      }
    }
  } finally {
    reader.releaseLock();
  }
}

export function useChat() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [theme, setTheme] = useState<Theme | null>(null);
  const [streaming, setStreaming] = useState(false);
  const [connectionError, setConnectionError] = useState<string | null>(null);

  const conversationId = useRef<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    let cancelled = false;
    fetch(`${API_BASE}/theme`, { headers: authHeaders() })
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(String(r.status)))))
      .then((data: Theme) => !cancelled && setTheme(data))
      .catch(() => {
        // Non-fatal: the widget falls back to its default palette and copy.
        // Blocking the UI on branding would be a poor trade.
        if (!cancelled) setConnectionError("Could not reach the assistant.");
      });
    return () => {
      cancelled = true;
    };
  }, []);

  // Abort any in-flight stream when the widget unmounts, so a closed panel
  // does not leave a request running and writing into dead state.
  useEffect(() => () => abortRef.current?.abort(), []);

  const patchLast = useCallback(
    (update: (message: ChatMessage) => ChatMessage) => {
      setMessages((current) => {
        if (current.length === 0) return current;
        const copy = [...current];
        const last = copy[copy.length - 1];
        if (!last) return current;
        copy[copy.length - 1] = update(last);
        return copy;
      });
    },
    [],
  );

  const send = useCallback(
    async (text: string) => {
      const trimmed = text.trim();
      // Guard here as well as in the UI: a second submission mid-stream would
      // interleave two replies into one message.
      if (!trimmed || streaming) return;

      setConnectionError(null);
      const controller = new AbortController();
      abortRef.current = controller;

      setMessages((current) => [
        ...current,
        { id: nextId(), role: "user", text: trimmed, tools: [], sources: [], streaming: false },
        { id: nextId(), role: "assistant", text: "", tools: [], sources: [], streaming: true },
      ]);
      setStreaming(true);

      // Paces bursty deltas into an even stream. See createTextSmoother.
      const smoother = createTextSmoother((chunk) =>
        patchLast((m) => ({ ...m, text: m.text + chunk })),
      );

      try {
        const response = await fetch(`${API_BASE}/chat`, {
          method: "POST",
          headers: authHeaders(),
          body: JSON.stringify({
            message: trimmed,
            conversation_id: conversationId.current,
          }),
          signal: controller.signal,
        });

        if (!response.ok || !response.body) {
          const detail =
            response.status === 401
              ? "Not authorised. Check your API key."
              : response.status === 429
                ? "Too many messages — give it a moment."
                : `The assistant is unavailable (${response.status}).`;
          patchLast((m) => ({ ...m, streaming: false, error: detail }));
          return;
        }

        for await (const event of readEvents(response.body, controller.signal)) {
          switch (event.type) {
            case "message_start":
              conversationId.current = event.conversation_id;
              break;

            case "text_delta":
              smoother.push(event.text);
              break;

            case "tool_call":
              patchLast((m) => ({
                ...m,
                tools: [
                  ...m.tools,
                  { id: event.id, name: event.name, input: event.input, status: "running" },
                ],
              }));
              break;

            case "tool_result":
              patchLast((m) => ({
                ...m,
                tools: m.tools.map((tool): ToolActivity =>
                  tool.id === event.id
                    ? {
                        ...tool,
                        status: event.ok ? "ok" : "error",
                        preview: event.preview,
                        durationMs: event.duration_ms,
                      }
                    : tool,
                ),
              }));
              break;

            case "citations":
              patchLast((m) => {
                const seen = new Set(m.sources.map((s) => s.id));
                return {
                  ...m,
                  sources: [...m.sources, ...event.sources.filter((s) => !seen.has(s.id))],
                };
              });
              break;

            case "error":
              patchLast((m) => ({ ...m, error: event.message }));
              break;

            case "done":
              // Flush before clearing the flag, or the message would be marked
              // complete with the tail still sitting in the buffer.
              smoother.finish();
              patchLast((m) => ({ ...m, streaming: false }));
              break;
          }
        }
      } catch (error) {
        // An abort is a user action, not a failure worth reporting.
        if ((error as Error)?.name !== "AbortError") {
          patchLast((m) => ({
            ...m,
            error: "Lost connection to the assistant.",
          }));
        }
      } finally {
        // Flush unconditionally. On an abort or a dropped connection there is
        // still buffered text the user has already "received", and leaving it
        // in the buffer would silently truncate the visible answer. Idempotent
        // if `done` already flushed.
        smoother.finish();
        // Always clear the streaming flag. Leaving it set would disable the
        // composer permanently -- the worst possible failure for this widget.
        patchLast((m) => ({ ...m, streaming: false }));
        setStreaming(false);
        abortRef.current = null;
      }
    },
    [streaming, patchLast],
  );

  const stop = useCallback(() => abortRef.current?.abort(), []);

  const reset = useCallback(() => {
    abortRef.current?.abort();
    conversationId.current = null;
    setMessages([]);
    setConnectionError(null);
  }, []);

  return { messages, theme, streaming, connectionError, send, stop, reset };
}
