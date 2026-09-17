/**
 * Operations dashboard across every project.
 *
 * Each project is a separate deployment, so this polls each one's
 * `/admin/overview` independently. Failures are per-project: one site being
 * down must not blank the other six, which is exactly what a single
 * `Promise.all` would do.
 */

import { useCallback, useEffect, useState } from "react";
import { loadEndpoints } from "./endpoints";
import type { AdminUser, Overview, ProjectState, Role } from "./types";

const POLL_INTERVAL_MS = 15_000;

async function fetchJson<T>(
  baseUrl: string,
  path: string,
  apiKey: string,
  signal: AbortSignal,
): Promise<T> {
  const headers: Record<string, string> = {};
  if (apiKey) headers.Authorization = `Bearer ${apiKey}`;

  const response = await fetch(`${baseUrl}${path}`, { headers, signal });
  if (!response.ok) {
    // Status alone is not actionable; say what the operator should change.
    const reason =
      response.status === 401
        ? "key rejected"
        : response.status === 403
          ? "key lacks the admin permission"
          : `HTTP ${response.status}`;
    throw new Error(reason);
  }
  return (await response.json()) as T;
}

function relative(seconds: number): string {
  if (seconds < 60) return `${seconds}s`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m`;
  if (seconds < 86_400) return `${Math.floor(seconds / 3600)}h`;
  return `${Math.floor(seconds / 86_400)}d`;
}

function compact(value: number): string {
  if (value < 1000) return String(value);
  if (value < 1_000_000) return `${(value / 1000).toFixed(1)}k`;
  return `${(value / 1_000_000).toFixed(1)}M`;
}

function ProjectCard({
  state,
  selected,
  onSelect,
}: {
  state: ProjectState;
  selected: boolean;
  onSelect: () => void;
}) {
  const { endpoint, overview, error, loading } = state;

  // A project that cannot be reached is "down" -- distinct from one that
  // answers and reports itself degraded.
  const tone = error ? "down" : (overview?.status ?? "starting");

  return (
    <button
      type="button"
      className={`card card--${tone}${selected ? " card--selected" : ""}`}
      onClick={onSelect}
    >
      <div className="card__head">
        <span className={`pip pip--${tone}`} aria-hidden="true" />
        <span className="card__name">{overview?.name ?? endpoint.label}</span>
        {loading && <span className="card__spinner" aria-label="Refreshing" />}
      </div>

      {error ? (
        <p className="card__error">{error}</p>
      ) : overview ? (
        <>
          <dl className="card__stats">
            <div>
              <dt>Turns</dt>
              <dd>{compact(overview.usage.turns)}</dd>
            </div>
            <div>
              <dt>Chunks</dt>
              <dd>{compact(overview.indexed_chunks)}</dd>
            </div>
            <div>
              <dt>Mean</dt>
              <dd>{overview.usage.mean_seconds_per_turn}s</dd>
            </div>
            <div>
              <dt>Errors</dt>
              <dd className={overview.usage.error_rate > 0.1 ? "warn" : undefined}>
                {(overview.usage.error_rate * 100).toFixed(0)}%
              </dd>
            </div>
          </dl>

          <div className="card__flags">
            <span className="flag">{overview.model}</span>
            {/* Surfaced because an empty index is the single most common
                reason an assistant "doesn't know" things it should. */}
            {overview.retrieval_enabled && overview.indexed_chunks === 0 && (
              <span className="flag flag--warn">empty index</span>
            )}
            {!overview.mcp_connected && overview.tools.length === 0 && (
              <span className="flag flag--warn">no tools</span>
            )}
            {!overview.auth_enabled && <span className="flag flag--warn">no auth</span>}
          </div>
        </>
      ) : (
        <p className="card__error">Connecting...</p>
      )}
    </button>
  );
}

function Detail({ state }: { state: ProjectState }) {
  const [roles, setRoles] = useState<Role[] | null>(null);
  const [users, setUsers] = useState<AdminUser[] | null>(null);
  const [usersError, setUsersError] = useState<string | null>(null);

  const { endpoint, overview } = state;

  useEffect(() => {
    const controller = new AbortController();
    const { baseUrl, apiKey } = endpoint;

    fetchJson<Role[]>(baseUrl, "/admin/roles", apiKey, controller.signal)
      .then(setRoles)
      .catch(() => setRoles(null));

    // Users sit behind `manage_users`, a higher bar than `admin`. A key with
    // only `admin` legitimately gets 403 here, so it is reported inline rather
    // than treated as a failure of the whole panel.
    fetchJson<AdminUser[]>(baseUrl, "/admin/users", apiKey, controller.signal)
      .then((data) => {
        setUsers(data);
        setUsersError(null);
      })
      .catch((error: Error) => {
        if (error.name !== "AbortError") setUsersError(error.message);
      });

    return () => controller.abort();
  }, [endpoint]);

  if (!overview) {
    return (
      <aside className="detail">
        <p className="detail__empty">{state.error ?? "Not connected."}</p>
      </aside>
    );
  }

  return (
    <aside className="detail">
      <h2 className="detail__title">{overview.name}</h2>
      <p className="detail__sub">
        {overview.project} · {overview.provider} · {overview.model}
      </p>

      <section className="detail__section">
        <h3>Usage</h3>
        <dl className="detail__grid">
          <div><dt>Uptime</dt><dd>{relative(overview.usage.uptime_seconds)}</dd></div>
          <div><dt>Turns</dt><dd>{compact(overview.usage.turns)}</dd></div>
          <div><dt>Tool calls</dt><dd>{compact(overview.usage.tool_calls)}</dd></div>
          <div><dt>Errors</dt><dd>{overview.usage.errors}</dd></div>
          <div><dt>Input tokens</dt><dd>{compact(overview.usage.input_tokens)}</dd></div>
          <div><dt>Output tokens</dt><dd>{compact(overview.usage.output_tokens)}</dd></div>
        </dl>
      </section>

      <section className="detail__section">
        <h3>Tools <span className="count">{overview.tools.length}</span></h3>
        <div className="chips">
          {overview.tools.length > 0 ? (
            overview.tools.map((tool) => (
              <span key={tool} className="chip">{tool}</span>
            ))
          ) : (
            <span className="muted">none</span>
          )}
        </div>
      </section>

      <section className="detail__section">
        <h3>Roles <span className="count">{roles?.length ?? 0}</span></h3>
        {roles ? (
          roles.map((role) => (
            <div key={role.name} className="role">
              <div className="role__head">
                <strong>{role.name}</strong>
                <span className="muted">{role.user_count} user(s)</span>
              </div>
              {role.description && <p className="role__desc">{role.description}</p>}
              <div className="chips">
                {role.permissions.map((permission) => (
                  <span key={permission} className="chip chip--perm">{permission}</span>
                ))}
              </div>
            </div>
          ))
        ) : (
          <span className="muted">unavailable</span>
        )}
      </section>

      <section className="detail__section">
        <h3>Users <span className="count">{users?.length ?? 0}</span></h3>
        {usersError ? (
          <span className="muted">{usersError}</span>
        ) : users && users.length > 0 ? (
          users.map((user) => (
            <div key={user.id} className="user">
              <div className="user__head">
                <strong>{user.id}</strong>
                {user.disabled && <span className="flag flag--warn">disabled</span>}
              </div>
              <div className="chips">
                {user.roles.map((role) => (
                  <span key={role} className="chip">{role}</span>
                ))}
              </div>
            </div>
          ))
        ) : (
          <span className="muted">none</span>
        )}
      </section>
    </aside>
  );
}

export function Dashboard() {
  const [endpoints] = useState(loadEndpoints);
  const [projects, setProjects] = useState<ProjectState[]>(() =>
    endpoints.map((endpoint) => ({ endpoint, loading: true })),
  );
  const [selectedId, setSelectedId] = useState(endpoints[0]?.id ?? "");

  const refresh = useCallback(
    (signal: AbortSignal) => {
      // Deliberately not Promise.all: each project settles on its own, so one
      // unreachable site leaves the other six showing live data.
      endpoints.forEach((endpoint, index) => {
        setProjects((current) =>
          current.map((p, i) => (i === index ? { ...p, loading: true } : p)),
        );

        fetchJson<Overview>(endpoint.baseUrl, "/admin/overview", endpoint.apiKey, signal)
          .then((overview) =>
            setProjects((current) =>
              current.map((p, i) =>
                i === index
                  ? { ...p, overview, error: undefined, loading: false, lastChecked: Date.now() }
                  : p,
              ),
            ),
          )
          .catch((error: Error) => {
            if (error.name === "AbortError") return;
            setProjects((current) =>
              current.map((p, i) =>
                i === index
                  ? { ...p, error: error.message, loading: false, lastChecked: Date.now() }
                  : p,
              ),
            );
          });
      });
    },
    [endpoints],
  );

  useEffect(() => {
    const controller = new AbortController();
    refresh(controller.signal);
    const timer = setInterval(() => refresh(controller.signal), POLL_INTERVAL_MS);
    return () => {
      controller.abort();
      clearInterval(timer);
    };
  }, [refresh]);

  const selected = projects.find((p) => p.endpoint.id === selectedId) ?? projects[0];
  const healthy = projects.filter((p) => p.overview?.status === "ok").length;
  const down = projects.filter((p) => p.error).length;

  return (
    <div className="dash">
      <header className="dash__head">
        <div>
          <h1>Assistants</h1>
          <p className="dash__sub">
            {projects.length} project{projects.length === 1 ? "" : "s"} ·{" "}
            {healthy} healthy
            {down > 0 && <span className="warn"> · {down} unreachable</span>}
          </p>
        </div>
        <button
          type="button"
          className="refresh"
          onClick={() => refresh(new AbortController().signal)}
        >
          Refresh
        </button>
      </header>

      <div className="dash__body">
        <div className="grid">
          {projects.map((project) => (
            <ProjectCard
              key={project.endpoint.id}
              state={project}
              selected={project.endpoint.id === selectedId}
              onSelect={() => setSelectedId(project.endpoint.id)}
            />
          ))}
        </div>
        {selected && <Detail state={selected} />}
      </div>
    </div>
  );
}
