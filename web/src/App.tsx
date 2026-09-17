/**
 * The widget root.
 *
 * Owns only what the launcher and panel must agree on: whether the panel is
 * open, and whether a reply arrived while it was closed. Everything about the
 * conversation itself lives in `useChat`.
 */

import { useEffect, useRef, useState } from "react";
import { ChatPanel } from "./components/ChatPanel";
import { Launcher } from "./components/Launcher";
import { useChat } from "./lib/useChat";

export function App() {
  const [open, setOpen] = useState(false);
  const [unread, setUnread] = useState(false);
  const { messages, theme, streaming, connectionError, send, stop, reset } = useChat();

  const lastSeen = useRef(0);

  // Flag a reply that landed while the panel was closed. Tracked by count
  // rather than by a boolean so re-opening and closing again cannot resurrect
  // a badge for something already read.
  useEffect(() => {
    if (open) {
      lastSeen.current = messages.length;
      setUnread(false);
    } else if (messages.length > lastSeen.current) {
      setUnread(true);
    }
  }, [messages.length, open]);

  // Apply the project's palette to the widget subtree only. Setting these on
  // :root would let one project's branding bleed into the host page.
  const style = theme
    ? ({ "--brand": theme.primary, "--accent": theme.accent } as React.CSSProperties)
    : undefined;

  return (
    <div className="mcp-widget" style={style}>
      <ChatPanel
        open={open}
        theme={theme}
        messages={messages}
        streaming={streaming}
        connectionError={connectionError}
        onSend={send}
        onStop={stop}
        onReset={reset}
        onClose={() => setOpen(false)}
      />
      <Launcher
        open={open}
        unread={unread}
        busy={streaming}
        onClick={() => setOpen((value) => !value)}
      />
    </div>
  );
}
