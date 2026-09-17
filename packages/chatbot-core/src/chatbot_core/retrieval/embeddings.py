"""Embedding backends.

Local sentence-transformers is the default. With seven projects re-indexing
independently, a metered embedding API becomes the dominant cost line for what
is a solved, commoditised operation -- and local embedding means no network hop
on the query path either.
"""

from __future__ import annotations

import abc
import asyncio
from collections.abc import Sequence
from functools import lru_cache
from typing import Any

import structlog

from ..config import EmbeddingConfig, EmbeddingProvider

log = structlog.get_logger(__name__)


class Embedder(abc.ABC):
    def __init__(self, config: EmbeddingConfig) -> None:
        self.config = config

    @property
    def dimensions(self) -> int:
        return self.config.dimensions

    @abc.abstractmethod
    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed corpus chunks at index time."""

    @abc.abstractmethod
    async def embed_query(self, text: str) -> list[float]:
        """Embed a single user query.

        Separate from ``embed_documents`` because asymmetric models (E5, BGE,
        GTE) require different prefixes for the two sides, and using the wrong
        one quietly degrades every score.
        """


class LocalEmbedder(Embedder):
    """sentence-transformers, in-process.

    The model loads once per process and is shared across calls. Encoding is
    CPU/GPU-bound and releases the GIL, so it runs in a worker thread rather
    than blocking the event loop -- without that, one index batch would stall
    every concurrent chat request in the same process.
    """

    def __init__(self, config: EmbeddingConfig) -> None:
        super().__init__(config)
        self._model = _load_model(config.model, config.device)

        actual = _model_dimensions(self._model)
        if actual != config.dimensions:
            raise ValueError(
                f"embedding.dimensions is {config.dimensions} but "
                f"{config.model!r} produces {actual}. Fix the config; a "
                f"mismatch corrupts every similarity score in the index."
            )

        # The config declares a window and RetrievalConfig sizes chunks against
        # it -- but a wrong declaration would defeat that check entirely. Verify
        # against the model actually loaded. Overflow is discarded silently by
        # the model, so this must fail loudly rather than warn.
        window = int(getattr(self._model, "max_seq_length", 0) or 0)
        if window and window < config.max_sequence_length:
            raise ValueError(
                f"embedding.max_sequence_length is {config.max_sequence_length} "
                f"but {config.model!r} only accepts {window} tokens. Text beyond "
                f"the window is dropped before embedding, so chunks sized "
                f"against the wrong value are half-indexed. Set "
                f"max_sequence_length to {window} and chunk.size to {window} or "
                f"below."
            )

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        return await asyncio.to_thread(self._encode, list(texts), False)

    async def embed_query(self, text: str) -> list[float]:
        vectors = await asyncio.to_thread(self._encode, [text], True)
        return vectors[0]

    def _encode(self, texts: list[str], is_query: bool) -> list[list[float]]:
        prefix = _asymmetric_prefix(self.config.model, is_query)
        payload = [prefix + t for t in texts] if prefix else texts
        vectors = self._model.encode(
            payload,
            batch_size=self.config.batch_size,
            # Normalised vectors make cosine similarity a plain dot product,
            # which is what both store backends are configured to use.
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return [v.tolist() for v in vectors]


class HuggingFaceEmbedder(Embedder):
    """Hosted HF Inference feature-extraction endpoint.

    Use when the deployment target can't spare the memory for local weights.
    Adds a network round trip to every query.
    """

    def __init__(self, config: EmbeddingConfig, client: object | None = None) -> None:
        super().__init__(config)
        if client is not None:
            self._client = client
        else:
            try:
                from huggingface_hub import AsyncInferenceClient
            except ImportError as exc:  # pragma: no cover
                raise ImportError(
                    "The huggingface embedder needs: uv add 'chatbot-core[huggingface]'"
                ) from exc
            import os

            self._client = AsyncInferenceClient(
                model=config.model, token=os.environ.get("HF_TOKEN") or None
            )

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        out: list[list[float]] = []
        # Chunked to stay under the endpoint's per-request payload ceiling.
        for start in range(0, len(texts), self.config.batch_size):
            batch = list(texts[start : start + self.config.batch_size])
            raw = await self._client.feature_extraction(batch)  # type: ignore[attr-defined]
            out.extend(_normalise(v) for v in _as_lists(raw))
        return out

    async def embed_query(self, text: str) -> list[float]:
        prefix = _asymmetric_prefix(self.config.model, True)
        raw = await self._client.feature_extraction([prefix + text])  # type: ignore[attr-defined]
        return _normalise(_as_lists(raw)[0])


def build_embedder(config: EmbeddingConfig) -> Embedder:
    match config.provider:
        case EmbeddingProvider.LOCAL:
            return LocalEmbedder(config)
        case EmbeddingProvider.HUGGINGFACE:
            return HuggingFaceEmbedder(config)
    raise ValueError(f"unknown embedding provider: {config.provider}")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


@lru_cache(maxsize=4)
def _load_model(name: str, device: str | None):  # type: ignore[no-untyped-def]
    """Load a sentence-transformers model, preferring the local cache.

    Loading straight from the hub re-validates every model file against
    huggingface.co on each start. That adds seconds to startup and -- worse --
    makes the service refuse to start when the hub is unreachable, which is an
    absurd failure mode for a model already sitting on disk.

    So: try the cache first, and only reach for the network when the model
    genuinely isn't there yet.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Local embeddings need: uv add 'chatbot-core[local-embeddings]'"
        ) from exc

    try:
        model = SentenceTransformer(name, device=device, local_files_only=True)
    except Exception:  # noqa: BLE001 - not cached, or cache incomplete
        log.info("downloading_embedding_model", model=name)
        model = SentenceTransformer(name, device=device)
    else:
        log.info("loaded_embedding_model_from_cache", model=name, device=device or "auto")
        return model

    log.info("loaded_embedding_model", model=name, device=device or "auto")
    return model


def _model_dimensions(model: Any) -> int:
    """Read a model's output width across sentence-transformers versions.

    ``get_sentence_embedding_dimension`` was renamed to
    ``get_embedding_dimension``; support both so the package isn't pinned to
    one minor version.
    """
    for attr in ("get_embedding_dimension", "get_sentence_embedding_dimension"):
        getter = getattr(model, attr, None)
        if callable(getter):
            return int(getter())
    raise AttributeError(
        "cannot determine embedding dimensions from "
        f"{type(model).__name__}; set embedding.dimensions explicitly"
    )


def _asymmetric_prefix(model: str, is_query: bool) -> str:
    """Instruction prefixes required by the common asymmetric model families.

    Omitting these is a silent quality regression -- retrieval still returns
    results, they're just measurably worse -- so it's worth handling centrally
    rather than hoping each project remembers.
    """
    lowered = model.lower()
    if "e5" in lowered:
        return "query: " if is_query else "passage: "
    if "bge" in lowered and is_query:
        return "Represent this sentence for searching relevant passages: "
    return ""


def _as_lists(raw: object) -> list[list[float]]:
    tolist = getattr(raw, "tolist", None)
    data = tolist() if callable(tolist) else raw
    if isinstance(data, list) and data and isinstance(data[0], (int, float)):
        return [list(data)]  # type: ignore[arg-type]
    return [list(v) for v in data]  # type: ignore[union-attr]


def _normalise(vector: list[float]) -> list[float]:
    total = sum(x * x for x in vector) ** 0.5
    return [x / total for x in vector] if total else vector


__all__ = ["Embedder", "HuggingFaceEmbedder", "LocalEmbedder", "build_embedder"]
