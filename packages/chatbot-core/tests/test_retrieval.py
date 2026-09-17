"""Retrieval round-trip against a real LanceDB store.

Uses a deterministic stub embedder rather than a real model: the behaviour
under test is upsert/search/delete semantics, not embedding quality, and a stub
keeps the suite free of a 2GB torch dependency.
"""

from __future__ import annotations

import math

import pytest

from chatbot_core.config import EmbeddingConfig, RetrievalConfig, VectorBackend
from chatbot_core.engine import needs_retrieval
from chatbot_core.retrieval.embeddings import Embedder
from chatbot_core.retrieval.lancedb_store import LanceDBStore
from chatbot_core.retrieval.store import Document, make_id
from chatbot_core.retrieval.tool import RetrievalTool

# The config floor is 64; no real embedding model is narrower. Kept at the
# floor so the stub stays cheap while exercising the same validation path.
DIMS = 64


class StubEmbedder(Embedder):
    """Maps text to a unit vector by hashing its words into fixed buckets.

    Documents sharing vocabulary land near each other, which is enough to make
    ranking assertions meaningful.
    """

    def __init__(self) -> None:
        super().__init__(EmbeddingConfig(dimensions=DIMS))

    def _vector(self, text: str) -> list[float]:
        buckets = [0.0] * DIMS
        for word in text.lower().split():
            buckets[hash(word) % DIMS] += 1.0
        norm = math.sqrt(sum(b * b for b in buckets)) or 1.0
        return [b / norm for b in buckets]

    async def embed_documents(self, texts):
        return [self._vector(t) for t in texts]

    async def embed_query(self, text):
        return self._vector(text)


@pytest.fixture
def store(tmp_path):
    config = RetrievalConfig(
        backend=VectorBackend.LANCEDB,
        embedding=EmbeddingConfig(dimensions=DIMS),
        top_k=5,
        score_threshold=0.0,
    )
    return LanceDBStore(config, "test_collection", StubEmbedder(), path=tmp_path / "db")


def doc(uri: str, index: int, text: str, title: str = "Doc") -> Document:
    return Document(id=make_id(uri, index), text=text, title=title, uri=uri)


async def test_empty_store_returns_no_hits(store):
    await store.ensure_ready()
    assert await store.count() == 0
    assert await store.search("anything") == []


async def test_upsert_then_search(store):
    await store.upsert(
        [
            doc("a.md", 0, "invoices are due within thirty days", "Billing"),
            doc("b.md", 0, "refunds are processed in five business days", "Refunds"),
        ]
    )
    assert await store.count() == 2
    hits = await store.search("invoices due thirty days")
    assert hits
    assert hits[0].title == "Billing"
    assert hits[0].uri == "a.md"


async def test_results_are_ordered_best_first(store):
    await store.upsert(
        [
            doc("a.md", 0, "alpha beta gamma delta"),
            doc("b.md", 0, "epsilon zeta eta theta"),
        ]
    )
    hits = await store.search("alpha beta gamma delta")
    assert len(hits) == 2
    assert hits[0].score >= hits[1].score


async def test_upsert_is_idempotent(store):
    """Re-ingesting unchanged content must not duplicate it -- ids are derived
    from source uri + chunk index precisely so this holds."""
    rows = [doc("a.md", 0, "stable content here")]
    await store.upsert(rows)
    await store.upsert(rows)
    assert await store.count() == 1


async def test_upsert_replaces_content_for_same_id(store):
    await store.upsert([doc("a.md", 0, "original text")])
    await store.upsert([doc("a.md", 0, "revised text")])
    assert await store.count() == 1
    hits = await store.search("revised text")
    assert hits[0].text == "revised text"


async def test_delete_by_uri_removes_every_chunk(store):
    """A document that shrank between ingests must not leave orphaned chunks
    that still match queries."""
    await store.upsert([doc("a.md", i, f"chunk {i} content") for i in range(4)])
    await store.upsert([doc("b.md", 0, "other document")])
    removed = await store.delete_by_uri("a.md")
    assert removed == 4
    assert await store.count() == 1


async def test_score_threshold_filters_weak_matches(tmp_path):
    config = RetrievalConfig(
        backend=VectorBackend.LANCEDB,
        embedding=EmbeddingConfig(dimensions=DIMS),
        score_threshold=0.99,
    )
    strict = LanceDBStore(config, "strict", StubEmbedder(), path=tmp_path / "db")
    await strict.upsert([doc("a.md", 0, "completely unrelated vocabulary")])
    assert await strict.search("entirely different words") == []


async def test_dimension_mismatch_is_rejected(tmp_path):
    """An index built with a different embedding model makes every similarity
    score meaningless -- it must fail loudly, not silently degrade."""
    base = RetrievalConfig(embedding=EmbeddingConfig(dimensions=DIMS))
    first = LanceDBStore(base, "c", StubEmbedder(), path=tmp_path / "db")
    await first.upsert([doc("a.md", 0, "some content")])

    class WiderEmbedder(StubEmbedder):
        @property
        def dimensions(self) -> int:
            return DIMS * 2

    second = LanceDBStore(base, "c", WiderEmbedder(), path=tmp_path / "db")
    with pytest.raises(ValueError, match="dimensional"):
        await second.ensure_ready()


class TestRetrievalTool:
    async def test_reports_no_match_without_inventing_one(self, store):
        """Handing the model weak matches invites confabulation; telling it
        plainly that nothing matched makes it say it doesn't know."""
        await store.upsert([doc("a.md", 0, "alpha beta gamma")])
        store.config = store.config.model_copy(update={"score_threshold": 0.99})
        outcome = await RetrievalTool(store).execute(
            "search_knowledge_base", {"query": "totally different"}
        )
        assert not outcome.is_error
        assert "No passages" in outcome.content
        assert outcome.sources == []

    async def test_returns_numbered_passages_and_sources(self, store):
        await store.upsert([doc("a.md", 0, "invoices are due within thirty days", "Billing")])
        outcome = await RetrievalTool(store).execute(
            "search_knowledge_base", {"query": "invoices due thirty"}
        )
        assert "[1] Billing" in outcome.content
        assert len(outcome.sources) == 1
        assert outcome.sources[0]["title"] == "Billing"

    async def test_rejects_empty_query(self, store):
        await store.ensure_ready()
        outcome = await RetrievalTool(store).execute("search_knowledge_base", {"query": "  "})
        assert outcome.is_error

    async def test_rejects_unknown_tool_name(self, store):
        await store.ensure_ready()
        outcome = await RetrievalTool(store).execute("not_a_tool", {})
        assert outcome.is_error


class TestNeedsRetrieval:
    """The small-talk gate must fail towards searching.

    Skipping retrieval on a real question means answering it with no corpus --
    exactly the confident-but-ungrounded reply this bot exists to prevent. A
    needless search only costs latency.
    """

    @pytest.mark.parametrize(
        "message",
        ["hi", "Hi!", "  HELLO  ", "thanks", "Thank you.", "good morning",
         "bye", "ok", "how are you?", "what can you do?", "sup"],
    )
    def test_pure_small_talk_skips_retrieval(self, message: str) -> None:
        assert needs_retrieval(message) is False

    @pytest.mark.parametrize(
        "message",
        [
            "hi, how long does delivery take?",   # greeting + real question
            "thanks, but what is the returns window?",
            "hello there, can I change my address",
            "delivery",
            "returns policy",
            "do you ship internationally?",
            "ok so when does my refund arrive",
        ],
    )
    def test_anything_carrying_a_question_retrieves(self, message: str) -> None:
        assert needs_retrieval(message) is True

    def test_unknown_short_input_retrieves(self) -> None:
        """Not on the list means search it. Silence is not a safe default."""
        assert needs_retrieval("vat receipt") is True
        assert needs_retrieval("???") is True

    def test_long_messages_always_retrieve(self) -> None:
        """Past the length cap the cheap check is not even attempted."""
        assert needs_retrieval("thanks " * 20) is True

    def test_empty_message_does_not_crash(self) -> None:
        assert needs_retrieval("   ") is True
