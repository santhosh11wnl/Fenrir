/**
 * Which projects the dashboard watches.
 *
 * Configured at build time via `VITE_ADMIN_PROJECTS`, a JSON array:
 *
 *     VITE_ADMIN_PROJECTS='[
 *       {"id":"site-a","label":"Site A","baseUrl":"http://localhost:8000","apiKey":"mcp_..."},
 *       {"id":"site-b","label":"Site B","baseUrl":"http://localhost:8001","apiKey":"mcp_..."}
 *     ]'
 *
 * A note on those keys: they are **admin keys, embedded in a client bundle**.
 * Anyone who can load this page has them. That is acceptable for a dashboard
 * served on an internal network or behind a VPN, and unacceptable on the public
 * internet -- deploy it accordingly, or put a server-side proxy in front that
 * holds the keys and authenticates the operator itself.
 */

import type { ProjectEndpoint } from "./types";

const FALLBACK: ProjectEndpoint[] = [
  {
    id: "template-bot",
    label: "Template Assistant",
    baseUrl: import.meta.env.VITE_API_BASE_URL ?? "/api",
    apiKey: import.meta.env.VITE_API_KEY ?? "",
  },
];

export function loadEndpoints(): ProjectEndpoint[] {
  const raw = import.meta.env.VITE_ADMIN_PROJECTS;
  if (!raw) return FALLBACK;

  try {
    const parsed: unknown = JSON.parse(raw);
    if (!Array.isArray(parsed)) throw new Error("expected an array");

    return parsed.map((entry, index) => {
      const item = entry as Partial<ProjectEndpoint>;
      if (!item.baseUrl) {
        throw new Error(`entry ${index} has no baseUrl`);
      }
      return {
        id: item.id ?? `project-${index}`,
        label: item.label ?? item.id ?? `Project ${index + 1}`,
        // Trailing slashes would produce `//admin/overview`, which some
        // proxies reject outright.
        baseUrl: item.baseUrl.replace(/\/+$/, ""),
        apiKey: item.apiKey ?? "",
      };
    });
  } catch (error) {
    // Fall back rather than render nothing: a malformed env var should not
    // leave an operator staring at a blank page with no clue why.
    console.error("VITE_ADMIN_PROJECTS is not valid JSON:", error);
    return FALLBACK;
  }
}
