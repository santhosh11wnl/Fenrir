/**
 * One turn in the conversation, plus the tool activity behind it.
 *
 * Tool calls are shown rather than hidden. When an assistant pauses for two
 * seconds, "searching the knowledge base" is the difference between waiting
 * and assuming it has frozen -- and when an answer is wrong, seeing which tool
 * produced what is the whole debugging story.
 */

import { useState } from "react";
import type { ChatMessage, ToolActivity } from "../lib/types";

function ToolRow({ tool }: { tool: ToolActivity }) {
  const [expanded, setExpanded] = useState(false);
  const label = tool.name.replace(/_/g, " ");

  return (
    <div className={`tool tool--${tool.status}`}>
      <button
        type="button"
        className="tool__head"
        onClick={() => setExpanded((open) => !open)}
        aria-expanded={expanded}
      >
        <span className="tool__status" aria-hidden="true">
          {tool.status === "running" ? (
            <span className="tool__spinner" />
          ) : tool.status === "ok" ? (
            <svg viewBox="0 0 16 16" fill="none">
              <path
                d="m3.5 8.5 3 3 6-7"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
          ) : (
            <svg viewBox="0 0 16 16" fill="none">
              <path d="M4 4l8 8M12 4l-8 8" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
            </svg>
          )}
        </span>

        <span className="tool__name">{label}</span>

        {tool.durationMs != null && (
          <span className="tool__timing">{tool.durationMs}ms</span>
        )}
        <span className="tool__chevron" aria-hidden="true">
          <svg viewBox="0 0 16 16" fill="none">
            <path d="m4 6 4 4 4-4" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
          </svg>
        </span>
      </button>

      {expanded && (
        <div className="tool__body">
          {Object.keys(tool.input).length > 0 && (
            <>
              <div className="tool__label">Input</div>
              <pre className="tool__pre">{JSON.stringify(tool.input, null, 2)}</pre>
            </>
          )}
          {tool.preview && (
            <>
              <div className="tool__label">Result</div>
              <pre className="tool__pre">{tool.preview}</pre>
            </>
          )}
        </div>
      )}
    </div>
  );
}

export function Message({ message }: { message: ChatMessage }) {
  const isUser = message.role === "user";

  return (
    <div className={`msg msg--${message.role}`}>
      {!isUser && message.tools.length > 0 && (
        <div className="msg__tools">
          {message.tools.map((tool) => (
            <ToolRow key={tool.id} tool={tool} />
          ))}
        </div>
      )}

      {(message.text || message.streaming) && (
        <div className="msg__bubble">
          {/* Rendered as text, never as HTML. Model output is untrusted --
              a retrieved document could carry markup, and interpolating it
              would be a cross-site scripting hole. */}
          <span className="msg__text">{message.text}</span>

          {/* Only while genuinely empty: once tokens arrive the caret would
              compete with the text for attention. */}
          {message.streaming && !message.text && (
            <span className="msg__typing" aria-label="Assistant is typing">
              <i /><i /><i />
            </span>
          )}
        </div>
      )}

      {message.error && (
        <div className="msg__error" role="alert">
          {message.error}
        </div>
      )}

      {message.sources.length > 0 && (
        <div className="msg__sources">
          <div className="msg__sources-label">
            {message.sources.length === 1 ? "Source" : "Sources"}
          </div>
          {message.sources.map((source, index) => (
            <a
              key={source.id}
              className="source"
              href={source.uri ?? undefined}
              target="_blank"
              // noreferrer alongside noopener: the target must not be handed
              // a window reference or a referrer for the host page.
              rel="noopener noreferrer"
              title={source.excerpt}
            >
              <span className="source__index">{index + 1}</span>
              <span className="source__title">{source.title}</span>
              <span className="source__score">{(source.score * 100).toFixed(0)}%</span>
            </a>
          ))}
        </div>
      )}
    </div>
  );
}
