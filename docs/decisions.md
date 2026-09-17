# Design decisions

Why the platform is shaped this way, including the options rejected. Written so
that a future change can tell whether it is correcting a mistake or re-treading
ground.

---

## 1. Retrieval, not fine-tuning

**Decision.** Each site gets an indexed corpus that is retrieved at query time.
No per-site model training.

The original brief said each version would be "trained on" its own data, which
usually means one of two very different things. Fine-tuning teaches *style and
format*; it does not reliably teach *facts*, cannot cite a source, and requires
a retrain every time a document changes. Seven sites would mean seven training
pipelines and seven retrains per content update.

Retrieval gives each site its own corpus, citations back to the source
document, and same-day updates. It is also the right substrate to add
fine-tuning onto later if one site genuinely needs a custom voice — the two
compose; they are not alternatives.

**Rejected:** fine-tuning per site. Revisit only for voice and format, never
for knowledge.

---

## 2. One versioned base, not seven repositories

**Decision.** A monorepo with `packages/chatbot-core` and
`packages/mcp-server-core` as the shared base; each site is a directory of
config, tools, and documents.

Seven forks means a bug fixed seven times, and six sites quietly running the
version where it is not fixed. Seven repositories with a shared library means
real versioning but a release cycle for every change.

The monorepo gives one place to fix things while each project still pins what
it depends on. A project that genuinely diverges can be extracted — that is the
escape hatch, not the default.

**Consequence:** projects are deliberately *not* uv workspace members. A project
has no `pyproject.toml` because it is not a Python package. (Discovered the hard
way: a `projects/*` glob in the workspace broke `uv` the moment a project was
stamped.)

---

## 3. Self-hosted models, addressed by protocol

**Decision.** The default provider speaks the OpenAI-compatible protocol to a
server you run. Anthropic is an optional extra.

Targeting the *protocol* rather than a vendor means one implementation covers
Ollama, vLLM, llama.cpp, LM Studio, and HF TGI. Moving from a laptop to a GPU
server is a `base_url` and a `model.id` in config.

**Rejected:** an Ollama-specific client. It would have been marginally simpler
and would have made the laptop-to-server move a code change.

**Consequence:** on macOS the model runs *natively*, not in Docker — containers
get no Metal GPU access there, so a containerised model is CPU-only. On a Linux
server the reverse holds. Hence `ollama` is an opt-in compose profile rather
than a required service.

---

## 4. Roles per project, permissions fixed

**Decision.** Each project's config declares its own roles. What roles *grant*
comes from a fixed `Permission` enum.

Sites need different vocabularies — one wants `admin`/`support`/`customer`,
another only `staff`. A hardcoded role enum would force seven access models
into one shape.

Permissions are fixed for the opposite reason: each one corresponds to an actual
check in the code. A permission that no endpoint enforces reads as protection
that is not there, which is worse than having no permission at all.

**Endpoints check permissions, never role names.** Which role carries `admin`
is a per-site decision, and an endpoint has no business knowing that one site
calls its privileged role "support".

---

## 5. Isolation is structural, not conditional

**Decision.** One API process serves one project. Users live in that project's
`users.yaml` and nowhere else.

A key issued for site A fails on site B not because a check rejects it, but
because site B's process never loads site A's users. There is no code path that
could be made to cross that boundary by a bug in a conditional.

**Rejected:** one multi-tenant service with a `project_id` column. Cheaper to
run, and one missing `WHERE` clause from a cross-tenant data leak.

---

## 6. Own the agent loop

**Decision.** Each provider drives its own request/execute/repeat loop rather
than delegating to an SDK's tool runner.

Two reasons. `tool_call` is emitted *before* a tool runs and `tool_result`
after, with a measured duration — a runner that executes tools inside its own
iteration leaves a pending-state UI with nothing to render. And the same loop
shape has to work for a provider whose SDK has no runner at all.

**Cost:** we own termination and error handling. `max_iterations` bounds the
loop, and every tool failure is fed back as a tool result rather than raised, so
the model can recover instead of the turn dying.

---

## 7. LanceDB for development, pgvector for production

**Decision.** Both, behind one `VectorStore` interface, chosen per environment.

LanceDB is embedded and file-backed, so a fresh checkout is useful immediately.
It cannot share an index between processes, so any project running more than one
replica moves to pgvector — a config change, not a rewrite.

---

## 8. Refuse to index what cannot be read

**Decision.** A document that cannot be extracted honestly is skipped and
reported, never indexed.

This is the most consequential rule in the ingest pipeline, and it was learned
from a real failure. A PDF fetched from a URL with no file extension was cached
as `.html` and parsed by the HTML extractor, producing **718 chunks of
mojibake** that matched queries and answered nothing. Nothing errored.

Three defences now: content type comes from magic bytes rather than the URL
path; extracted text must be *readable* (not merely long enough); and a
document below either bar is skipped with a log line saying why.

**The general principle:** in a retrieval pipeline, silent partial success is
worse than loud failure. A skipped document is visible. A corrupt one looks
exactly like a working one until someone acts on its answer.

---

## 9. Chunks must fit the embedding window

**Decision.** `chunk.size` is validated against `embedding.max_sequence_length`
at config load, and that value is cross-checked against the model actually
loaded.

Another lesson from a real bug. The default was `all-MiniLM-L6-v2` (a **256**
token window) with 512-token chunks. The model silently truncated half of every
chunk — no error, no warning, just quietly worse retrieval. Measured: cosine
similarity of 0.9954 between a chunk and that same chunk with 60 extra
sentences appended, proving the tail was never embedded.

The default model is now `bge-small-en-v1.5` (512-token window, same 384
dimensions, better benchmarks), and the mismatch is now impossible to ship:
config validation catches a bad pairing, and the embedder verifies the declared
window against the real one.

Heading prefixes are deducted from the budget *before* packing, because adding
them afterwards pushed chunks back over the window.

---

## 10. Fail shut, and fail loudly

A few places where the safe default was chosen over the convenient one:

- **Auth off + non-local `ENVIRONMENT` → refuse to start.** Shipping a public
  unauthenticated endpoint against a private corpus is bad enough to be worth a
  hard boot failure.
- **Auth on + no users → refuse to start.** It would reject every request
  anyway; better to say so than to look merely broken.
- **Admin endpoints with auth disabled → 403, not open.** With no way to
  identify a caller, "we cannot tell who you are" can only safely mean no.
- **Unknown role → grants nothing.** A typo in `users.yaml` must reduce access,
  never widen it.
- **A startup failure crashes the process.** A container reporting healthy while
  erroring on every request is far harder to diagnose than one that will not
  start.

The exception: **MCP connection failure is survivable.** An assistant that can
still answer from its corpus is better than one that will not boot because a
tool backend is down. It reports `status: degraded` and carries on.
