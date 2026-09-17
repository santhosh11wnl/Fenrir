"""Project configuration schema.

This module is the contract between the base platform and the projects stamped
from it. A project differs from the base by its ``config.yaml``, its tools
module, and its corpus -- not by forked Python. Adding a field here is how you
give every project a new knob; adding a field to one project's YAML that isn't
here is an error, by design (``extra="forbid"``), so typos surface at startup
rather than as silently-ignored config.
"""

from __future__ import annotations

import os
import re
from enum import StrEnum
from pathlib import Path
from typing import Any, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .auth import AuthConfig

# ---------------------------------------------------------------------------
# enums
# ---------------------------------------------------------------------------


class ChatProvider(StrEnum):
    #: A model you host, reached over the OpenAI-compatible protocol. Covers
    #: Ollama, vLLM, llama.cpp, LM Studio, and HF TGI -- the backend is a
    #: ``base_url``, not a code path. The default: no external API, no
    #: per-token cost, and data never leaves your infrastructure.
    LOCAL = "local"
    #: Hugging Face hosted Inference API.
    HUGGINGFACE = "huggingface"
    #: Anthropic's API. Optional -- requires ANTHROPIC_API_KEY.
    ANTHROPIC = "anthropic"


class EmbeddingProvider(StrEnum):
    #: sentence-transformers, running in-process. No API cost, no network at
    #: query time. The default because re-indexing 7 corpora against a metered
    #: embedding API gets expensive fast.
    LOCAL = "local"
    #: Hugging Face hosted Inference API.
    HUGGINGFACE = "huggingface"


class VectorBackend(StrEnum):
    #: Embedded, file-based. Zero setup; right for dev and small corpora.
    LANCEDB = "lancedb"
    #: Postgres extension. Right for prod, and for replicas sharing one index.
    PGVECTOR = "pgvector"


class MCPTransport(StrEnum):
    #: MCP server launched as a subprocess. Local dev, Claude Desktop/Code.
    STDIO = "stdio"
    #: MCP server as its own service. What deployed environments use.
    HTTP = "http"


class Effort(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"
    MAX = "max"


# ---------------------------------------------------------------------------
# base
# ---------------------------------------------------------------------------


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# ---------------------------------------------------------------------------
# sections
# ---------------------------------------------------------------------------


class ProjectMeta(_Strict):
    """Identity. ``id`` is load-bearing: it namespaces the vector collection,
    the conversation store, and the container names."""

    id: str = Field(pattern=r"^[a-z][a-z0-9-]{1,48}[a-z0-9]$")
    name: str
    description: str = ""


class ModelConfig(_Strict):
    provider: ChatProvider = ChatProvider.LOCAL

    #: The model to run. For ``local`` this is whatever your server calls it
    #: (``qwen2.5:7b`` on Ollama, a repo id on vLLM). For ``huggingface``, a
    #: repo id. For ``anthropic``, a model id such as ``claude-opus-5``.
    id: str = "qwen2.5:7b"

    #: Where the model server listens. ``local`` only. Must include ``/v1``.
    #: Inside compose this is the service name, not localhost -- containers
    #: have their own loopback, so ``localhost`` would point at the API itself.
    base_url: str = "http://ollama:11434/v1"

    #: Environment variable holding the API key, when the server wants one.
    #: Most self-hosted servers don't; vLLM behind a gateway might.
    api_key_env: str | None = None

    #: Seconds to wait for a response. Generous by default: a self-hosted model
    #: loads weights on first request, which can take a minute on a cold start.
    request_timeout: float = Field(default=300.0, ge=5.0, le=3600.0)

    #: Whether this model reliably emits structured tool calls. Off by default
    #: because a model that hallucinates tool syntax into its prose is worse
    #: than one with no tools -- turn it on once you've confirmed the model
    #: handles function calling (Qwen 2.5+, Llama 3.1+, Mistral-Nemo, Hermes).
    supports_tools: bool = False

    max_tokens: int = Field(default=4_096, ge=256, le=128_000)

    #: Sampling. Supported by local and Hugging Face models; rejected by
    #: current Claude models, which is enforced below.
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)

    #: --- Anthropic only, ignored elsewhere ---
    effort: Effort = Effort.HIGH
    adaptive_thinking: bool = True
    show_thinking: bool = False

    #: Hugging Face only: a dedicated Inference Endpoint URL.
    hf_endpoint_url: str | None = None

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.provider is ChatProvider.ANTHROPIC:
            if self.temperature is not None or self.top_p is not None:
                raise ValueError(
                    "temperature and top_p are rejected by current Claude "
                    "models; steer with the system prompt instead"
                )
            if not self.adaptive_thinking and self.effort in (Effort.XHIGH, Effort.MAX):
                raise ValueError(
                    f"adaptive_thinking=false is only accepted at effort <= high, "
                    f"got {self.effort}. Either enable thinking or lower effort."
                )
        if self.provider is ChatProvider.LOCAL:
            if not self.base_url:
                raise ValueError("model.base_url is required when provider is 'local'")
            if not self.base_url.rstrip("/").endswith("/v1"):
                raise ValueError(
                    f"model.base_url should end with /v1 (the OpenAI-compatible "
                    f"path), got {self.base_url!r}"
                )
        return self


class EmbeddingConfig(_Strict):
    provider: EmbeddingProvider = EmbeddingProvider.LOCAL

    #: 384-dim, 512-token window, ~130MB. Chosen over the popular
    #: all-MiniLM-L6-v2 because MiniLM's window is only 256 tokens, which
    #: silently truncates any larger chunk (see `max_sequence_length`).
    model: str = "BAAI/bge-small-en-v1.5"

    #: Must match the model. Checked against the store's existing collection at
    #: startup -- a mismatch means the index was built with a different model
    #: and every similarity score would be meaningless.
    dimensions: int = Field(default=384, ge=64, le=4096)

    #: The model's input window, in tokens. **Text beyond this is discarded by
    #: the model without any error**, so a chunk larger than this is half-
    #: indexed and retrieval quietly degrades. `RetrievalConfig` enforces
    #: `chunk.size <= max_sequence_length`, and the local embedder cross-checks
    #: this value against the model it actually loaded.
    #:
    #: Common values: bge-small/base/large 512, e5-* 512, MiniLM-L6 256,
    #: mpnet-base 384.
    max_sequence_length: int = Field(default=512, ge=64, le=32_768)

    batch_size: int = Field(default=32, ge=1, le=512)
    #: cpu | cuda | mps. None = autodetect.
    device: str | None = None


class ChunkConfig(_Strict):
    #: Tokens, not characters. Sized so a handful of chunks fit comfortably
    #: alongside the system prompt without crowding the answer.
    size: int = Field(default=512, ge=64, le=4096)
    overlap: int = Field(default=64, ge=0, le=1024)

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.overlap >= self.size:
            raise ValueError(f"chunk overlap ({self.overlap}) must be < size ({self.size})")
        return self


class RetrievalConfig(_Strict):
    """Turn this off for projects that are pure tool-use with no corpus."""

    enabled: bool = True
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    chunk: ChunkConfig = Field(default_factory=ChunkConfig)
    backend: VectorBackend = VectorBackend.LANCEDB
    #: Defaults to the project id. Set explicitly only when two projects
    #: deliberately share one corpus.
    collection: str | None = None
    top_k: int = Field(default=6, ge=1, le=50)
    #: Cosine similarity floor. Chunks below this are dropped even if they're
    #: in the top k -- better to retrieve nothing than to ground an answer in
    #: noise. Tune per corpus; 0.3 is a conservative start.
    score_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    #: Expose retrieval to the model as a tool it chooses to call, rather than
    #: always prepending results. Better when most turns don't need the corpus.
    as_tool: bool = True

    @model_validator(mode="after")
    def _check_chunk_fits_embedding_window(self) -> Self:
        """Refuse a chunk size the embedding model cannot actually read.

        This is the highest-value check in the whole config. An oversized chunk
        produces no error at any layer: the model truncates the overflow, the
        vector is written, searches return results, and answers are grounded in
        half a passage. The only symptom is that quality is mysteriously poor.
        """
        window = self.embedding.max_sequence_length
        if self.chunk.size > window:
            raise ValueError(
                f"chunk.size ({self.chunk.size}) exceeds the embedding model's "
                f"input window ({window} tokens for {self.embedding.model!r}). "
                f"Everything past the window is silently discarded before "
                f"embedding. Lower chunk.size to {window} or below, or choose a "
                f"model with a larger window."
            )
        return self


class ToolFilter(_Strict):
    """Allow/deny over the MCP server's advertised tools.

    Deny wins. ``["*"]`` in allow means every tool the server exposes -- which
    is fine when the server is yours, and not fine when it is shared.
    """

    allow: list[str] = Field(default_factory=lambda: ["*"])
    deny: list[str] = Field(default_factory=list)

    def permits(self, tool_name: str) -> bool:
        if any(_glob(p, tool_name) for p in self.deny):
            return False
        return any(_glob(p, tool_name) for p in self.allow)


class MCPConfig(_Strict):
    #: Off by default so a project with no tool server is a valid config rather
    #: than a startup error. The stamping template turns it on, so projects
    #: created the normal way get tools without thinking about it.
    enabled: bool = False
    transport: MCPTransport = MCPTransport.HTTP
    #: http transport.
    url: str | None = None
    #: stdio transport. argv of the server process.
    command: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    tools: ToolFilter = Field(default_factory=ToolFilter)
    #: Ceiling on agent-loop iterations. Stops a misbehaving tool from
    #: spinning up an unbounded bill.
    max_iterations: int = Field(default=12, ge=1, le=100)

    @model_validator(mode="after")
    def _check(self) -> Self:
        if not self.enabled:
            return self
        if self.transport is MCPTransport.HTTP and not self.url:
            raise ValueError("mcp.url is required when transport is 'http'")
        if self.transport is MCPTransport.STDIO and not self.command:
            raise ValueError("mcp.command is required when transport is 'stdio'")
        return self


class ThemeConfig(_Strict):
    """Passed to the React app so one build serves every project."""

    primary: str = Field(default="#2563eb", pattern=r"^#[0-9a-fA-F]{6}$")
    accent: str = Field(default="#7c3aed", pattern=r"^#[0-9a-fA-F]{6}$")
    logo_url: str | None = None
    greeting: str = "How can I help?"
    placeholder: str = "Ask a question..."
    suggestions: list[str] = Field(default_factory=list, max_length=6)


class LimitsConfig(_Strict):
    max_history_messages: int = Field(default=40, ge=2, le=500)
    max_input_chars: int = Field(default=16_000, ge=100)
    #: Per-session request ceiling. Crude, but it's the difference between a
    #: runaway client and a surprise invoice.
    requests_per_minute: int = Field(default=20, ge=1, le=1000)


class ProjectConfig(_Strict):
    """The whole of a project's configuration."""

    project: ProjectMeta
    system_prompt: str = Field(min_length=1)
    model: ModelConfig = Field(default_factory=ModelConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    mcp: MCPConfig = Field(default_factory=MCPConfig)
    #: Users are per-project: access here grants nothing on any other project.
    #: Defined in this project's `users.yaml`, which is gitignored.
    auth: AuthConfig = Field(default_factory=AuthConfig)
    theme: ThemeConfig = Field(default_factory=ThemeConfig)
    limits: LimitsConfig = Field(default_factory=LimitsConfig)

    @property
    def collection(self) -> str:
        return self.retrieval.collection or self.project.id

    @classmethod
    def load(cls, path: str | Path) -> ProjectConfig:
        """Read a config.yaml, expanding ``${VAR}`` and ``${VAR:-default}``.

        Secrets belong in the environment, never in the YAML -- these files are
        committed. The expansion is what lets a committed config reference them.
        """
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"project config not found: {path}")
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"{path}: expected a YAML mapping at the top level")
        return cls.model_validate(_expand_env(raw))


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _expand_env(value: Any) -> Any:
    """Recursively expand ``${VAR}`` / ``${VAR:-default}`` in strings.

    An unset variable with no default raises rather than expanding to empty --
    an empty API key produces a 401 at the first request, which is a much worse
    place to discover the problem than startup.
    """
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    if not isinstance(value, str):
        return value

    def replace(m: re.Match[str]) -> str:
        name, default = m.group(1), m.group(2)
        found = os.environ.get(name)
        if found is not None:
            return found
        if default is not None:
            return default
        raise ValueError(
            f"config references ${{{name}}} but it is not set in the environment "
            f"(use ${{{name}:-fallback}} to make it optional)"
        )

    return _ENV_PATTERN.sub(replace, value)


def _glob(pattern: str, name: str) -> bool:
    """``*`` matches any run of characters. Deliberately simpler than fnmatch --
    tool names are flat identifiers, so ``?`` and ``[...]`` would be noise."""
    if pattern == "*":
        return True
    regex = "^" + ".*".join(re.escape(p) for p in pattern.split("*")) + "$"
    return re.match(regex, name) is not None


__all__ = [
    "ChatProvider",
    "ChunkConfig",
    "Effort",
    "EmbeddingConfig",
    "EmbeddingProvider",
    "LimitsConfig",
    "MCPConfig",
    "MCPTransport",
    "ModelConfig",
    "ProjectConfig",
    "ProjectMeta",
    "RetrievalConfig",
    "ThemeConfig",
    "ToolFilter",
    "VectorBackend",
]
