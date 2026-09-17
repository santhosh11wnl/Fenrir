/**
 * The message input.
 *
 * Locked while the assistant is answering. That is a deliberate product
 * decision, not a technical limitation: a second message mid-stream would
 * interleave two replies into one bubble, and the backend would bill for a
 * turn whose context is already stale.
 *
 * Disabling without explanation is hostile, so the lock is visible three ways
 * at once -- the field dims, the placeholder says what is happening, and the
 * send button becomes a stop button. A user who wants to interrupt can.
 */

import { useEffect, useLayoutEffect, useRef } from "react";

interface ComposerProps {
  value: string;
  placeholder: string;
  busy: boolean;
  onChange: (value: string) => void;
  onSubmit: () => void;
  onStop: () => void;
}

const MAX_ROWS = 5;

export function Composer({
  value,
  placeholder,
  busy,
  onChange,
  onSubmit,
  onStop,
}: ComposerProps) {
  const textarea = useRef<HTMLTextAreaElement>(null);

  // Grow with the content up to a ceiling, then scroll. Done in a layout
  // effect so the resize lands in the same frame as the text -- in a plain
  // effect the box visibly jumps a frame late.
  useLayoutEffect(() => {
    const node = textarea.current;
    if (!node) return;
    node.style.height = "auto";
    const lineHeight = parseInt(getComputedStyle(node).lineHeight || "20", 10);
    node.style.height = `${Math.min(node.scrollHeight, lineHeight * MAX_ROWS + 20)}px`;
  }, [value]);

  // Return focus when the assistant finishes, so a conversation can continue
  // without reaching for the mouse.
  useEffect(() => {
    if (!busy) textarea.current?.focus();
  }, [busy]);

  const handleKeyDown = (event: React.KeyboardEvent<HTMLTextAreaElement>) => {
    // Enter sends; Shift+Enter is a newline. The common case gets the
    // single keystroke.
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      if (!busy) onSubmit();
    }
  };

  const canSend = value.trim().length > 0 && !busy;

  return (
    <form
      className={`composer${busy ? " composer--busy" : ""}`}
      onSubmit={(event) => {
        event.preventDefault();
        if (canSend) onSubmit();
      }}
    >
      <textarea
        ref={textarea}
        className="composer__input"
        value={value}
        rows={1}
        disabled={busy}
        placeholder={busy ? "Answering..." : placeholder}
        onChange={(event) => onChange(event.target.value)}
        onKeyDown={handleKeyDown}
        aria-label="Message"
        // Tells assistive tech the field is temporarily unavailable rather
        // than permanently read-only.
        aria-busy={busy}
      />

      {busy ? (
        <button
          type="button"
          className="composer__action composer__action--stop"
          onClick={onStop}
          aria-label="Stop generating"
        >
          <svg viewBox="0 0 24 24" fill="none" aria-hidden="true">
            <rect x="7" y="7" width="10" height="10" rx="2" fill="currentColor" />
          </svg>
        </button>
      ) : (
        <button
          type="submit"
          className="composer__action"
          disabled={!canSend}
          aria-label="Send message"
        >
          <svg viewBox="0 0 24 24" fill="none" aria-hidden="true">
            <path
              d="M4 12h15M13 6l6 6-6 6"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
        </button>
      )}
    </form>
  );
}
