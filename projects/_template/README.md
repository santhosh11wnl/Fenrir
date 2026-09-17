# Project template

Copy this directory to create a new assistant. A project differs from the base
platform by three things and nothing else:

| File | What it controls |
|---|---|
| `config.yaml` | Persona, model, retrieval settings, tool filter, theme, limits |
| `tools/*.py` | Project-specific MCP tools |
| `data/` + `ingest.yaml` | The knowledge base |

There is deliberately no Python to fork. If you find yourself wanting to change
engine behaviour, that change belongs in `packages/chatbot-core` where all
seven projects get it — or the project has genuinely diverged, and it should be
extracted into its own repository.

## Creating a project

```bash
uv run python scripts/new_project.py my-project --name "My Assistant"
```

Then:

1. Edit `projects/my-project/config.yaml` — at minimum the `system_prompt`.
2. Put documents in `projects/my-project/data/`.
3. Ingest them:
   ```bash
   uv run python -m chatbot_core.ingest projects/my-project
   ```
4. Run it:
   ```bash
   PROJECT=my-project uv run chat-api
   ```

## Knowing it works

```bash
curl localhost:8000/health
```

`indexed_chunks` should be non-zero once you've ingested, and `mcp_connected`
should be true if the MCP server is running. A zero chunk count is the usual
reason an assistant says it doesn't know things it should.

To prove the tool chain end to end, ask the assistant to call `ping`. That
exercises chat API → MCP client → server → tool → back without depending on
your corpus being correct.
