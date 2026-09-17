# Architecture

How the pieces fit. For *why* they fit that way, see
[`decisions.md`](decisions.md).

## The shape

```
            browser
               │  SSE
               ▼
        ┌─────────────┐        ┌──────────────┐
        │     api     │───────▶│  model server │  Ollama / vLLM
        │  (FastAPI)  │        └──────────────┘
        │             │
        │  ChatEngine │───────▶ vector store   LanceDB / pgvector
        │             │
        │             │───────▶ mcp-server     tools
        └─────────────┘
```

One `api` process serves **one** project. Seven sites means seven processes,
which is what makes cross-site isolation structural rather than conditional.

## Request flow

A single turn:

1. `POST /chat` authenticates the caller against **this project's** user store.
2. Rate limit, keyed on the authenticated subject when there is one.
3. History is loaded and trimmed to `limits.max_history_messages`.
4. `ChatEngine.stream()`:
   - In prepend mode, retrieve now and fold passages into the system prompt.
   - Hand the provider a merged tool surface (MCP tools + retrieval-as-a-tool).
5. The provider drives its own loop: call model → if tool calls, run them
   concurrently → feed all results back in one message → repeat.
6. Events stream out as SSE and are rendered by the widget.

Every turn ends with exactly one `done` event, even on failure. A client that
never sees `done` spins forever, so that is guaranteed at three layers.

## Access control: who can retrieve what

Each role is granted a set of **audiences**, and each document is tagged with
one. A caller only ever retrieves documents whose audience their roles grant.

```
roles ──► AuthConfig.audiences_for() ──► frozenset[str]
                                            │
          ┌─────────────────────────────────┴───────────────┐
          ▼                                                 ▼
   prepend: store.search(audiences=…)        as_tool: RetrievalTool(audiences=…)
          │                                                 │
          └──────────────► SQL/vector predicate ◄───────────┘
```

**The filter is applied in the query, never in the prompt.** This is the whole
point. Telling a model "do not reveal internal pricing to customers" is not
access control: the text is already in the context window, and it can be
argued, tricked, or role-played back out. A chunk that was never fetched
cannot leak.

Consequences that follow from that, and are tested in
`packages/chatbot-core/tests/test_audiences.py`:

| Rule | Why |
|---|---|
| Audiences are bound when `RetrievalTool` is constructed, once per turn | If it were a tool *argument*, the model could ask for `internal` itself |
| An untagged document is `public` (`DEFAULT_AUDIENCE`) | A forgotten tag should be a visible content mistake, not a silent hole |
| An empty audience set matches nothing (`1 = 0`), never "no filter" | The worst failure available is a caller granted nothing seeing everything |
| A role with `audiences: []` keeps exactly that | Lets a tools-only service account be expressed without the config lying |
| Roles unknown to the project fall back to the anonymous view | A typo in `users.yaml` degrades to the logged-out view, never above it |
| Multiple roles grant the **union** | Holding an extra role must not cost you access |

`RoleSpec.prompt` exists alongside this and only sets expectations about who is
asking. It is not a boundary and nothing should depend on it.

Some rules outrank every role. The shop prompt refuses to repeat card numbers,
CVVs, or PINs for *anyone* — customer, support, or administrator — because
"an administrator asked" is exactly the pretext an attacker uses.

The tests cover the logic; they cannot tell you whether *this* project's
documents were tagged correctly during ingest. For that, query the real index:

```
docker compose exec -T api python scripts/check_audiences.py shop
```

It exits non-zero if any role reaches an audience it was not granted, so it can
gate a deploy. It is checked against a deliberately sabotaged filter, so a pass
means something.

## The event protocol

One contract, three consumers — `chatbot_core/events.py`, the SSE serialiser,
and `web/src/lib/types.ts`. Change a field and all three change.

| Event | Meaning |
|---|---|
| `message_start` | Turn began; carries the conversation id |
| `text_delta` | A chunk of the visible answer |
| `thinking_delta` | Reasoning summary, when enabled |
| `tool_call` | A tool is about to run (emitted *before* execution) |
| `tool_result` | It finished, with duration |
| `citations` | Retrieved sources backing the answer |
| `error` | User-facing failure |
| `done` | Terminal. Exactly one per turn. |

`tool_call` before execution is why a UI can show a pending row rather than a
gap.

## Extension points

Four places to add behaviour, none requiring a change to existing code:

| To add | Do this |
|---|---|
| A model backend | New `ChatProviderBase` subclass + one `match` arm |
| A vector store | New `VectorStore` subclass + one `match` arm |
| A document type | New extractor in `retrieval/extractors.py` |
| A tool | Drop a module with `register(server)` into `projects/<id>/tools/` |

The tool case is the one used most: a project adds a file and the tool appears
in that project's assistant, with nothing in `packages/` touched.

## Key abstractions

```python
class ChatProviderBase(ABC):       # anthropic | openai-compatible | huggingface
    def stream(...) -> AsyncIterator[ChatEvent]

class VectorStore(ABC):            # lancedb | pgvector
    async def upsert / search / delete_by_uri

class Embedder(ABC):               # local sentence-transformers | HF API

class ToolExecutor(Protocol):      # MCP client | retrieval tool | composite
    specs; execute(name, args) -> ToolOutcome
```

`ChatEngine` depends on these abstractions and never on a concrete backend.
`ToolExecutor` is a two-member `Protocol` so the retrieval tool can satisfy it
without inheriting anything.

## Configuration

`ProjectConfig` is the contract between base and project. It is strict —
`extra="forbid"` — so a typo fails at startup rather than silently disabling a
setting you believed you set.

Two cross-field checks earn their keep:

- `chunk.size <= embedding.max_sequence_length` — otherwise the model discards
  the overflow with no error and retrieval quietly degrades.
- `adaptive_thinking=false` requires `effort <= high` on Anthropic models, which
  the API would otherwise reject at request time.

`${VAR}` and `${VAR:-default}` expand from the environment, which is how a
committed config references secrets it must never contain.

## Data at rest

| What | Where | Committed? |
|---|---|---|
| Config | `projects/*/config.yaml` | Yes |
| Tools | `projects/*/tools/*.py` | Yes |
| Documents | `projects/*/data/` | Gitignored (usually large) |
| Fetched pages | `projects/*/.cache/` | Gitignored (rebuildable) |
| Vector index | `storage/` or a volume | Gitignored (rebuildable) |
| Users | `projects/*/users.yaml` | **Gitignored — credential hashes** |

## Known limits

Honest about what is not production-ready at scale:

- **Conversations are in-memory.** Single replica only. Two replicas and a
  user's second message can land on one that never saw their first.
- **Rate limiting is per-process.** Three replicas means three times the limit.
- **Usage counters reset on restart.** Fine for "is this healthy"; wrong for
  billing — read the structured logs for that.
- **LanceDB is single-process.** Move to pgvector before scaling out.

Each has a defined path out, noted in the module that implements it.
