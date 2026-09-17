/**
 * The chat panel.
 *
 * Opens from the launcher and holds the conversation. Three behaviours worth
 * calling out:
 *
 * - **Escape closes it.** Expected of any overlay, and cheap to honour.
 * - **Autoscroll yields to the user.** It follows new tokens only while the
 *   view is already near the bottom; scrolling up to re-read something is not
 *   an invitation to be yanked back down.
 * - **Replies are announced politely.** `aria-live="polite"` means a screen
 *   reader reads the answer when it settles rather than interrupting on every
 *   streamed token.
 */

import { useEffect, useRef, useState } from "react";
import type { ChatMessage, Theme } from "../lib/types";
import { Composer } from "./Composer";
import { Message } from "./Message";

interface ChatPanelProps {
  open: boolean;
  theme: Theme | null;
  messages: ChatMessage[];
  streaming: boolean;
  connectionError: string | null;
  onSend: (text: string) => void;
  onStop: () => void;
  onReset: () => void;
  onClose: () => void;
}

/** How close to the bottom still counts as "following along", in pixels. */
const FOLLOW_THRESHOLD = 80;

export function ChatPanel({
  open,
  theme,
  messages,
  streaming,
  connectionError,
  onSend,
  onStop,
  onReset,
  onClose,
}: ChatPanelProps) {
  const [draft, setDraft] = useState("");
  const scroller = useRef<HTMLDivElement>(null);
  const following = useRef(true);

  useEffect(() => {
    if (!open) return;
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  // Track whether the user is still at the bottom. Read on scroll rather than
  // computed during render so it reflects intent, not layout churn.
  useEffect(() => {
    const node = scroller.current;
    if (!node) return;
    const onScroll = () => {
      const distance = node.scrollHeight - node.scrollTop - node.clientHeight;
      following.current = distance < FOLLOW_THRESHOLD;
    };
    node.addEventListener("scroll", onScroll, { passive: true });
    return () => node.removeEventListener("scroll", onScroll);
  }, [open]);

  useEffect(() => {
    if (!open || !following.current) return;
    const node = scroller.current;
    if (node) node.scrollTop = node.scrollHeight;
  }, [messages, open]);

  const submit = () => {
    const text = draft.trim();
    if (!text || streaming) return;
    setDraft("");
    following.current = true; // a new question always scrolls into view
    onSend(text);
  };

  const empty = messages.length === 0;
  const name = theme?.name ?? "Assistant";

  return (
    <section
      className={`panel${open ? " panel--open" : ""}`}
      role="dialog"
      aria-label={`${name} chat`}
      aria-modal="false"
      // Hidden from assistive tech and from Tab order while closed, so a
      // collapsed panel cannot swallow keyboard focus.
      {...(!open ? { inert: "" as unknown as boolean, "aria-hidden": true } : {})}
    >
      <header className="panel__head">
        <div className="panel__identity">
          {theme?.logo_url ? (
            <img className="panel__logo" src={theme.logo_url} alt="" />
          ) : (
            <span className="panel__mark" aria-hidden="true">
              {name.charAt(0).toUpperCase()}
            </span>
          )}
          <div className="panel__titles">
            <h2 className="panel__title">{name}</h2>
            <p className="panel__status">
              <span className={`dot${streaming ? " dot--busy" : ""}`} aria-hidden="true" />
              {streaming ? "Thinking..." : "Online"}
            </p>
          </div>
        </div>

        <div className="panel__actions">
          {messages.length > 0 && (
            <button
              type="button"
              className="icon-button"
              // Disabled while a reply streams. With the composer locked,
              // focus falls through to these controls -- and Space activates
              // a focused button, so an idle keypress could wipe the thread
              // mid-answer. Destructive controls stay unreachable until the
              // turn finishes.
              disabled={streaming}
              onClick={() => {
                onReset();
                setDraft("");
              }}
              aria-label="Start a new conversation"
              title={
                streaming
                  ? "Wait for the reply to finish"
                  : "New conversation"
              }
            >
              <svg viewBox="0 0 24 24" fill="none" aria-hidden="true">
                <path
                  d="M4 12a8 8 0 0 1 13.7-5.7L20 8M20 4v4h-4M20 12a8 8 0 0 1-13.7 5.7L4 16M4 20v-4h4"
                  stroke="currentColor"
                  strokeWidth="1.8"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              </svg>
            </button>
          )}
          <button
            type="button"
            className="icon-button"
            onClick={onClose}
            aria-label="Close chat"
          >
            <svg viewBox="0 0 24 24" fill="none" aria-hidden="true">
              <path d="M18 6 6 18M6 6l12 12" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
            </svg>
          </button>
        </div>
      </header>

      <div className="panel__body" ref={scroller}>
        {connectionError && (
          <div className="banner banner--error" role="alert">
            {connectionError}
          </div>
        )}

        {empty ? (
          <div className="welcome">
            <div className="welcome__mark" aria-hidden="true">
              <svg viewBox="0 0 24 24" fill="none">
                <path
                  d="M21 11.5a8.38 8.38 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.38 8.38 0 0 1-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.38 8.38 0 0 1 3.8-.9h.5a8.48 8.48 0 0 1 8 8v.5z"
                  stroke="currentColor"
                  strokeWidth="1.6"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              </svg>
            </div>
            <p className="welcome__greeting">{theme?.greeting ?? "How can I help?"}</p>
            {theme?.description && (
              <p className="welcome__description">{theme.description}</p>
            )}

            {theme?.suggestions && theme.suggestions.length > 0 && (
              <div className="welcome__suggestions">
                {theme.suggestions.map((suggestion) => (
                  <button
                    key={suggestion}
                    type="button"
                    className="suggestion"
                    onClick={() => !streaming && onSend(suggestion)}
                    disabled={streaming}
                  >
                    {suggestion}
                  </button>
                ))}
              </div>
            )}
          </div>
        ) : (
          <div className="thread" aria-live="polite" aria-busy={streaming}>
            {messages.map((message) => (
              <Message key={message.id} message={message} />
            ))}
          </div>
        )}
      </div>

      <footer className="panel__foot">
        <Composer
          value={draft}
          placeholder={theme?.placeholder ?? "Ask a question..."}
          busy={streaming}
          onChange={setDraft}
          onSubmit={submit}
          onStop={onStop}
        />
      </footer>
    </section>
  );
}
