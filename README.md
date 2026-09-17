# MCP chatbot platform

One base, many assistants. A self-hosted chat platform where each of your
websites gets its own assistant — its own documents, its own tools, its own
users and roles — built from shared code that is never forked.

Nothing leaves your infrastructure. The model, the embeddings, and the vector
index all run on hardware you control.

```
┌── packages/ ────────────────────────────────────────────┐
│  chatbot-core       providers · retrieval · MCP client  │  the base
│  mcp-server-core    tool registry · transports          │
└──────────────────────────────────────────────────────────┘
                              │
┌── projects/ ─────────────────┴───────────────────────────┐
│  <site-a>/  config.yaml · tools/ · data/ · users.yaml    │  per site
│  <site-b>/  config.yaml · tools/ · data/ · users.yaml    │
└──────────────────────────────────────────────────────────┘
                              │
┌── services/ + web/ ──────────┴───────────────────────────┐
│  api        FastAPI, SSE streaming, one project each     │
│  web        chat widget + operations dashboard           │
└──────────────────────────────────────────────────────────┘
```

A project differs from the base by **config, tools, and documents** — never by
forked Python. Fix a bug in `chatbot-core` and every site gets it.

## Quick start

```bash
# 1. Model server (natively on macOS, so it gets the GPU)
brew install ollama
OLLAMA_KEEP_ALIVE=24h ollama serve &
ollama pull qwen2.5:7b

# 2. Dependencies
brew install uv
uv sync --all-packages --extra lancedb --extra local-embeddings --extra documents --group dev

# 3. Index the demo corpus (public Python PEPs and one PDF)
uv run python -m chatbot_core.ingest projects/_template

# 4. Run it
./scripts/stack.sh start _template      # without Docker
docker compose up -d                    # with Docker

curl localhost:8000/health
uv run python scripts/ask.py "What is the maximum line length for Python code?"
```

Shell shortcuts: `source scripts/aliases.sh`, then `mcp` for the list.

## Creating a site

```bash
uv run python scripts/new_project.py support-bot --name "Support Assistant"
```

Then three things, in order of how much they matter:

1. **`projects/support-bot/config.yaml`** — the `system_prompt` above all. A
   vague prompt produces a vague assistant.
2. **`projects/support-bot/data/`** — the documents. Markdown, text, PDF, HTML,
   or DOCX. Or add `url` sources to `ingest.yaml` to pull from the web.
3. **Ingest and run:**
   ```bash
   uv run python -m chatbot_core.ingest projects/support-bot
   PROJECT=support-bot uv run chat-api
   ```

Until you ingest, it will correctly say it doesn't know things.

## Running several sites at once

One API process serves one project — that is what makes cross-site isolation
structural rather than a runtime check. So several sites means several
services, which `docker-compose.projects.yml` holds:

```bash
docker compose -f docker-compose.yml -f docker-compose.projects.yml up -d
# or:  mcpall
```

Adding the fifth, sixth, seventh is copying a block in that file and changing
three things: the service name, `PROJECT`, and the published port. Nothing else
differs, because the image is identical — only configuration separates one
assistant from another.

They share one model server and one MCP server. Give a site its own
`mcp-server` service if its tools must not be visible to the others.

**Each site needs its own key.** There is no shared credential, because there
is no shared user list:

```bash
uv run python scripts/manage_users.py add site-a ops --role admin   # key A
uv run python scripts/manage_users.py add site-b ops --role admin   # key B
```

Key A returns 401 on site B. That is the design working, not a
misconfiguration — and the admin dashboard's config carries one key per
project for the same reason.

## Users and roles

Users belong to **one site**. A key issued for site A grants nothing on site B —
not because a check forbids it, but because site B's process never loads site
A's users.

Roles are defined **per project**, because sites need different vocabularies:

```yaml
auth:
  enabled: true
  roles:
    admin:    { permissions: [chat, history, admin, ingest, manage_users] }
    support:  { permissions: [chat, history] }
    customer: { permissions: [chat] }
  default_role: customer
```

Permissions are a fixed vocabulary — each maps to a real check in the code.
Roles group them however a site needs.

```bash
uv run python scripts/manage_users.py roles support-bot
uv run python scripts/manage_users.py add support-bot alice --role admin
uv run python scripts/manage_users.py rotate support-bot alice
```

Keys are scrypt-hashed and shown once. A lost key is rotated, never recovered.

**Auth is off by default** so a new project runs locally with no setup — and the
service *refuses to start* with auth off unless `ENVIRONMENT` is local, so that
default cannot reach a server by accident.

## The web client

```bash
cd web && npm install && npm run dev
```

- `http://localhost:5173` — the chat widget on a demo host page
- `http://localhost:5173/admin.html` — operations dashboard across all sites

The widget is a floating launcher that expands into a panel. It mounts itself
into `document.body`, so embedding is one script tag. Branding comes from the
project's `/theme` endpoint, so one build serves every site.

The dashboard polls each project's `/admin/overview` independently — one site
being down leaves the others showing live data.

## Choosing a model

The provider targets the OpenAI-compatible protocol, not a vendor, so the
backend is a `base_url`:

| Where | `model.id` | `model.base_url` |
|---|---|---|
| Laptop | `qwen2.5:7b` | `http://localhost:11434/v1` (Ollama) |
| Server | `qwen2.5:72b` | `http://vllm:8000/v1` (vLLM) |

Both are `${MODEL_ID}` / `${MODEL_BASE_URL}` overrides. No code changes.

## Performance

A 7B model on an M4 answers a grounded question in roughly **7 seconds**, with
first text at 0.4s. Two settings dominate that:

- **`retrieval.as_tool`** — `false` prepends the retrieved passages and costs
  one model turn. `true` lets the model decide to search, which is smarter for
  mixed workloads and costs two turns. For a documentation bot where nearly
  every question needs the corpus, `false` is roughly twice as fast.
- **Tool count** — every tool schema is prompt the model re-reads each turn.
  Deny the tools a given site does not need.

The first request after any config change is slow (~20s) because the prompt
shape changed and the server reprocesses it. Subsequent ones are fast.

## Documentation

- [`docs/architecture.md`](docs/architecture.md) — how the pieces fit
- [`docs/decisions.md`](docs/decisions.md) — why, including what was rejected
- [`docs/operations.md`](docs/operations.md) — running it, and what breaks

## Development

```bash
uv run pytest -q                      # 208 tests
uv run ruff check packages services
cd web && npm run build
```

Docker note: the production image **bakes the source in**, while `config.yaml`
is mounted. Change both and the container runs new config against old code —
which shows up as a crash loop or a 404 on a route you just wrote. For
development, mount the source instead:

```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d
```
