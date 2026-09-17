/**
 * The streaming event protocol.
 *
 * Mirrors `chatbot_core.events` on the Python side. One contract, three
 * consumers: the engine emits these, the API serialises them as SSE, this
 * client renders them. If a field changes there, it changes here.
 */

export type EventName =
  | "message_start"
  | "text_delta"
  | "thinking_delta"
  | "tool_call"
  | "tool_result"
  | "citations"
  | "error"
  | "done";

export interface MessageStartEvent {
  type: "message_start";
  conversation_id: string;
  message_id: string;
  model: string;
}

export interface TextDeltaEvent {
  type: "text_delta";
  text: string;
}

export interface ThinkingDeltaEvent {
  type: "thinking_delta";
  text: string;
}

export interface ToolCallEvent {
  type: "tool_call";
  id: string;
  name: string;
  input: Record<string, unknown>;
}

export interface ToolResultEvent {
  type: "tool_result";
  id: string;
  name: string;
  ok: boolean;
  preview: string;
  duration_ms: number | null;
}

export interface Source {
  id: string;
  title: string;
  uri: string | null;
  score: number;
  excerpt: string;
}

export interface CitationsEvent {
  type: "citations";
  sources: Source[];
}

export interface ErrorEvent {
  type: "error";
  message: string;
  retriable: boolean;
  request_id: string | null;
}

export interface Usage {
  input_tokens: number;
  output_tokens: number;
  cache_read_input_tokens: number;
  cache_creation_input_tokens: number;
}

export interface DoneEvent {
  type: "done";
  stop_reason: string | null;
  usage: Usage;
}

export type ChatEvent =
  | MessageStartEvent
  | TextDeltaEvent
  | ThinkingDeltaEvent
  | ToolCallEvent
  | ToolResultEvent
  | CitationsEvent
  | ErrorEvent
  | DoneEvent;

/** A tool invocation, pending until its result arrives. */
export interface ToolActivity {
  id: string;
  name: string;
  input: Record<string, unknown>;
  status: "running" | "ok" | "error";
  preview?: string;
  durationMs?: number | null;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  text: string;
  tools: ToolActivity[];
  sources: Source[];
  error?: string;
  /** True while this message is still being streamed. */
  streaming: boolean;
}

/** Project branding, from GET /theme. One build serves every project. */
export interface Theme {
  project_id: string;
  name: string;
  description: string;
  primary: string;
  accent: string;
  logo_url: string | null;
  greeting: string;
  placeholder: string;
  suggestions: string[];
}
