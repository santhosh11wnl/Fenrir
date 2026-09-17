/** Admin API shapes. Mirrors `api.admin` on the Python side. */

export interface Usage {
  uptime_seconds: number;
  turns: number;
  errors: number;
  tool_calls: number;
  input_tokens: number;
  output_tokens: number;
  cache_read_tokens: number;
  mean_seconds_per_turn: number;
  error_rate: number;
}

export interface Overview {
  project: string;
  name: string;
  status: "ok" | "degraded" | "starting";
  model: string;
  provider: string;
  mcp_connected: boolean;
  tools: string[];
  indexed_chunks: number;
  retrieval_enabled: boolean;
  auth_enabled: boolean;
  usage: Usage;
}

export interface Role {
  name: string;
  description: string;
  permissions: string[];
  user_count: number;
}

export interface AdminUser {
  id: string;
  name: string;
  roles: string[];
  permissions: string[];
  disabled: boolean;
  created_at: string;
}

/**
 * One deployed project the dashboard watches.
 *
 * Each project is a separate deployment with its own URL and its own key --
 * there is no shared credential, because there is no shared user list. Adding
 * a site to the dashboard means adding an entry here, not granting anything
 * new on the projects already listed.
 */
export interface ProjectEndpoint {
  id: string;
  label: string;
  baseUrl: string;
  apiKey: string;
}

/** What the dashboard knows about one project right now. */
export interface ProjectState {
  endpoint: ProjectEndpoint;
  overview?: Overview;
  /** Set when the project could not be reached or refused the key. */
  error?: string;
  loading: boolean;
  lastChecked?: number;
}
