from __future__ import annotations

from chatbot_core.config import ChunkConfig
from chatbot_core.retrieval.chunking import chunk_document, default_token_counter


def words(text: str) -> int:
    """Deterministic stand-in for a real tokenizer, so these tests assert on
    chunking behaviour rather than on tokenizer quirks."""
    return max(1, len(text.split()))


def test_short_document_is_one_chunk():
    chunks = chunk_document("Hello world.", ChunkConfig(size=128, overlap=16))
    assert len(chunks) == 1
    assert chunks[0].text == "Hello world."


def test_respects_token_budget():
    body = " ".join(f"word{i}" for i in range(600))
    chunks = chunk_document(body, ChunkConfig(size=64, overlap=8), count_tokens=words)
    assert len(chunks) > 1
    # Modest overshoot is expected: a unit is only split when it alone exceeds
    # the budget, so a chunk can end slightly over after admitting a last unit.
    assert all(words(c.text) <= 96 for c in chunks)


def test_headings_prefixed_onto_chunks():
    """Without the heading, a chunk reading 'It must be renewed annually' is
    useless to both the model and a human reading the citation."""
    doc = (
        "# Billing Policy\n\nInvoices are due in 30 days.\n\n"
        "# Refunds\n\nRefunds take 5 days."
    )
    chunks = chunk_document(doc, ChunkConfig(size=256, overlap=0))
    assert any("Billing Policy" in c.text and "30 days" in c.text for c in chunks)
    assert any("Refunds" in c.text and "5 days" in c.text for c in chunks)


def test_section_recorded_in_metadata():
    chunks = chunk_document(
        "# Onboarding\n\nStep one is verification.", ChunkConfig(size=256, overlap=0)
    )
    assert chunks[0].metadata["section"] == "Onboarding"


def test_metadata_is_not_shared_between_chunks():
    """Each chunk needs its own dict -- a shared one means mutating one chunk's
    metadata silently mutates every other chunk's."""
    doc = "# A\n\n" + "\n\n".join(f"Paragraph {i} with several words." for i in range(60))
    chunks = chunk_document(
        doc, ChunkConfig(size=64, overlap=0), metadata={"k": "v"}, count_tokens=words
    )
    assert len(chunks) > 1
    chunks[0].metadata["mutated"] = True
    assert "mutated" not in chunks[1].metadata


def test_oversized_single_sentence_is_hard_split():
    """A minified blob or a newline-stripped PDF extraction has no sentence
    boundary to split on; it must still be chunked rather than dropped."""
    chunks = chunk_document("x" * 20_000, ChunkConfig(size=64, overlap=0))
    assert len(chunks) > 1
    assert all(default_token_counter(c.text) <= 64 for c in chunks)


def test_overlap_carries_context_forward():
    units = [f"Sentence number {i} carries some content." for i in range(60)]
    chunks = chunk_document(
        "\n\n".join(units), ChunkConfig(size=64, overlap=16), count_tokens=words
    )
    assert len(chunks) > 1
    tail = chunks[0].text.split("\n\n")[-1]
    assert tail in chunks[1].text


def test_zero_overlap_does_not_repeat():
    units = [f"Sentence number {i} carries some content." for i in range(60)]
    chunks = chunk_document(
        "\n\n".join(units), ChunkConfig(size=64, overlap=0), count_tokens=words
    )
    assert len(chunks) > 1
    tail = chunks[0].text.split("\n\n")[-1]
    assert tail not in chunks[1].text


def test_empty_document_yields_nothing():
    assert chunk_document("   \n\n  ", ChunkConfig()) == []


def test_indices_are_sequential():
    body = "\n\n".join(f"Paragraph {i} with several words in it." for i in range(60))
    chunks = chunk_document(body, ChunkConfig(size=64, overlap=0), count_tokens=words)
    assert [c.index for c in chunks] == list(range(len(chunks)))


class TestBudgetIncludesHeadingPrefix:
    """Every chunk in a section carries that section's heading.

    If the heading's tokens aren't deducted from the budget before packing,
    chunks land slightly over the embedding model's window -- where the
    overflow is discarded with no error. That is the exact silent failure the
    budget exists to prevent, so the guarantee must hold *after* prefixing.
    """

    def test_prefixed_chunks_stay_within_budget(self):
        heading = "# " + " ".join(f"Heading{i}" for i in range(20))
        body = "\n\n".join(f"Paragraph {i} with several words in it." for i in range(80))
        chunks = chunk_document(
            f"{heading}\n\n{body}", ChunkConfig(size=64, overlap=0), count_tokens=words
        )
        assert len(chunks) > 1
        assert all(words(c.text) <= 64 for c in chunks), [
            words(c.text) for c in chunks if words(c.text) > 64
        ]

    def test_pathological_heading_drops_the_prefix(self):
        """A heading longer than the whole budget must not produce chunks that
        are all heading and no content."""
        heading = "# " + " ".join(f"word{i}" for i in range(200))
        body = "\n\n".join(f"Paragraph {i} here." for i in range(20))
        chunks = chunk_document(
            f"{heading}\n\n{body}", ChunkConfig(size=64, overlap=0), count_tokens=words
        )
        assert chunks
        assert all(words(c.text) <= 64 for c in chunks)
        # Content still made it in, and the section is still recorded.
        assert any("Paragraph" in c.text for c in chunks)
        assert chunks[0].metadata["section"].startswith("word0")
