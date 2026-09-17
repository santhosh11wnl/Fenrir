# Operations

Running it, and the things that actually go wrong.

## Layout

| Component | Port | What it is |
|---|---|---|
| Model server | 11434 | Ollama or vLLM. **Not** managed by this repo on macOS. |
| `mcp-server` | 8765 | Tool server. One per project (or shared). |
| `api` | 8000 | Chat API. **One process serves one project.** |
| `web` | 5173 | Chat widget + dashboard (dev server). |

Seven sites means seven `api` containers on seven ports, each with `PROJECT`
set. They can share one model server and one MCP server.

## Day-to-day

```bash
source scripts/aliases.sh      # then `mcp` lists everything

mcpup                          # start
mcpps                          # what is running
mcpla                          # follow api logs
mcpreingest                    # re-index, then restart api
mcpask "a question"            # ask the running stack
mcphealth                      # health json
```

Without Docker: `./scripts/stack.sh start <project>`.

## Adding a site

```bash
uv run python scripts/new_project.py acme-docs --name "Acme Docs"
# edit projects/acme-docs/config.yaml   <- system_prompt matters most
# add documents to projects/acme-docs/data/
uv run python -m chatbot_core.ingest projects/acme-docs
PROJECT=acme-docs uv run chat-api --port 8001
```

For Docker, copy the `api` service in `docker-compose.yml`, change `PROJECT`
and the published port.

## The dashboard

```
http://localhost:5173/admin.html      # dashboard
http://localhost:5173/                # demo site with the widget
```

The dashboard needs an authenticated admin, so the project must have
`auth.enabled: true` and at least one user holding the `admin` permission.
With auth off, every `/admin/*` route returns **403 by design** — with no way
to identify the caller, failing shut is the only safe reading.

```bash
uv run python scripts/manage_users.py add shop root --role super_admin
```

Put the issued key in `web/.env.local` as `VITE_ADMIN_PROJECTS`, then restart
Vite (env is read at startup, not per request).

### Keeping the widget public while the dashboard is private

`VITE_API_KEY` and `VITE_ADMIN_PROJECTS` are read by different things — the
widget and the dashboard. Leave `VITE_API_KEY` **empty**: a visitor on a real
website is not logged in, and an admin key there would make the public widget
answer as an administrator, serving internal material on the demo page.

Enabling auth would normally also close `/chat`. `auth.public_paths` is what
keeps it open:

```yaml
auth:
  enabled: true
  public_paths: [/health, /chat]   # exact paths
  anonymous_audiences: [public]
```

Open here still means restricted: an anonymous caller gets
`anonymous_audiences`, so public documents only. A request that *does* carry a
key is authenticated normally even on a public path, so a signed-in customer
keeps their role — and a revoked key still fails rather than silently dropping
to anonymous. Never add `/admin` or `/conversations` to this list.

## Restricting documents by role

Tag each ingest source with an audience in `projects/<id>/ingest.yaml`:

```yaml
sources:
  - type: directory
    path: ./data              # customer-facing
    metadata: { audience: public }
  - type: directory
    path: ./data/staff        # support desk only
    metadata: { audience: staff }
  - type: directory
    path: ./data/internal     # administrators only
    metadata: { audience: internal }
```

Then grant audiences per role in `config.yaml` (`auth.roles.<role>.audiences`).
Untagged documents are `public`, so tag the restricted material, not the rest.

After every re-ingest, confirm the tags actually landed:

```bash
docker compose exec -T api python scripts/check_audiences.py shop
```

Non-zero exit means some role reached an audience it was not granted. Run it in
the container, not on the host — with the LanceDB backend the host has its own
empty store, which passes every check for the wrong reason.

The failure this catches is silent: a mistagged source produces a corpus that
behaves normally until someone is served material they should not see. Nothing
in the logs marks it. See `docs/architecture.md` for why the filter is applied
in the query rather than in the prompt.

## Things that actually break

### "Could not reach the assistant" / container crash-loops

Almost always a **stale image**. The production image bakes the source in, while
`config.yaml` is mounted — so a config change applies instantly and a code
change does not. New config against old code fails validation.

```bash
docker compose logs api --tail 30   # look for ValidationError
docker compose build api && docker compose up -d api
```

For development, mount the source so the two cannot drift:

```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d
```

### "It says it doesn't know" about documents you indexed

Check the index is actually populated:

```bash
mcphealth        # indexed_chunks should be non-zero
```

Zero means ingest never ran *against the store the API reads*. In Docker the
index lives in a named volume, so ingesting on the host does not populate it:

```bash
docker compose run --rm ingest && docker compose restart api
```

The restart matters — the API opens its store handle once at startup.

If chunks exist but answers are poor, check `score_threshold` (default `0.3`).
Raise it to reject weak matches, lower it if good passages are being filtered.

### Answers are slow

Measure before tuning:

```bash
uv run python scripts/ask.py --raw "your question" | head -20
```

Typical causes, in order of impact:

1. **Two model turns.** With `retrieval.as_tool: true` the model spends a whole
   turn deciding to search. Set it to `false` to prepend passages instead —
   roughly halves latency for a documentation bot.
2. **Too many tools.** Every schema is prompt re-read each turn. Deny what the
   site does not need via `mcp.tools.deny`.
3. **`top_k` too high.** Each passage is up to 512 tokens of context.
4. **Model not resident.** Ollama unloads after 5 minutes by default:
   ```bash
   OLLAMA_KEEP_ALIVE=24h ollama serve
   ollama ps        # should show 100% GPU and a long UNTIL
   ```

A config change makes the *next* request slow (~20s) because the prompt shape
changed and the server reprocesses it. That is expected; the one after is fast.

### Tools missing

```bash
mcptools                                    # what the API can see
docker compose logs mcp-server --tail 20    # what the server registered
```

`mcp_connected: false` with `status: degraded` means the API started without
the tool server — by design, so a dead tool backend does not take the assistant
down. Check the MCP server is up and `MCP_SERVER_URL` points at it.

If a tool registered on the server but the API cannot see it, check
`mcp.tools.deny` in the project config.

### Nobody can log in

```bash
uv run python scripts/manage_users.py list <project>
```

Keys cannot be recovered — rotate to issue a new one:

```bash
uv run python scripts/manage_users.py rotate <project> alice
```

After editing `users.yaml` by hand, either restart or:

```bash
curl -X POST -H "Authorization: Bearer <admin-key>" \
  localhost:8000/admin/reload-users
```

That matters most for **revocation**, where waiting for a restart is the wrong
answer.

## Before going to a server

- [ ] `auth.enabled: true` on every project, with users created
- [ ] `ENVIRONMENT=production` set — the service then refuses to start
      unauthenticated, which is the point
- [ ] `CORS_ORIGINS` set to the real site origins, never `*`
- [ ] `retrieval.backend: pgvector` if running more than one replica per project
      (LanceDB cannot share an index between processes)
- [ ] Conversation store moved off in-memory if running more than one replica —
      otherwise a user's second message can land on a replica that never saw
      their first
- [ ] `MCP_HTTP_ALLOWED_HOSTS` set if the `fetch_url` tool is enabled; it is
      disabled by default for a reason
- [ ] Model server sized for the model — a 72B needs far more than a laptop
- [ ] The dashboard is **not** on the public internet: admin keys are embedded
      in its bundle. Internal network, VPN, or a server-side proxy that holds
      the keys itself.

## Backups

| What | Where | Rebuildable? |
|---|---|---|
| Vector index | `vector-store` volume / `./storage` | Yes — re-run ingest |
| Fetched documents | `projects/*/.cache/` | Yes — re-downloads |
| Source documents | `projects/*/data/` | **No — back these up** |
| Users | `projects/*/users.yaml` | **No — hashes are unrecoverable** |
| Config | `projects/*/config.yaml` | In version control |

Only two things are irreplaceable: the source documents and the user files.
