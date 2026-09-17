"""Document chunking.

Splits on structure first -- headings, then paragraphs, then sentences --
and only falls back to a hard character cut when a single sentence is longer
than the budget. Splitting mid-sentence is what produces the retrieved chunks
that look relevant and answer nothing.

The token budget is measured against the *embedding model's* tokenizer, not a
chat tokenizer, because the constraint being respected is the embedding model's
input window (typically 512). A chunk that overflows it is silently truncated
by the model, so its tail never gets indexed at all.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

from ..config import ChunkConfig

#: Rough fallback when no tokenizer is available. Deliberately pessimistic:
#: under-filling a chunk costs a little recall, overflowing it loses text.
_CHARS_PER_TOKEN = 3.6

_HEADING = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
_PARAGRAPH = re.compile(r"\n\s*\n")
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")

#: Floor on the per-chunk content budget after a heading prefix is deducted.
#: Below this a chunk would be mostly heading and almost no content.
_MIN_BUDGET = 32


@dataclass(slots=True)
class Chunk:
    text: str
    #: 0-based position within its source document. Lets a citation say
    #: "section 3 of X" and lets an update replace a document's chunks wholesale.
    index: int
    metadata: dict[str, Any] = field(default_factory=dict)


TokenCounter = Callable[[str], int]


def default_token_counter(text: str) -> int:
    return max(1, int(len(text) / _CHARS_PER_TOKEN))


def tokenizer_counter(model: Any) -> TokenCounter:
    """Build a counter from a loaded SentenceTransformer.

    Falls back to the estimate if the model doesn't expose a usable tokenizer,
    rather than failing the whole ingest over a counting detail.
    """
    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None:
        return default_token_counter

    def count(text: str) -> int:
        try:
            return len(tokenizer.encode(text, add_special_tokens=False))
        except Exception:  # noqa: BLE001
            return default_token_counter(text)

    return count


def chunk_document(
    text: str,
    config: ChunkConfig,
    *,
    metadata: dict[str, Any] | None = None,
    count_tokens: TokenCounter = default_token_counter,
) -> list[Chunk]:
    """Split one document into overlapping, structure-aligned chunks."""
    metadata = metadata or {}
    chunks: list[Chunk] = []

    for section_title, body in _sections(text):
        section_meta = dict(metadata)
        if section_title:
            section_meta["section"] = section_title

        # The heading is prefixed to every chunk in this section, so its cost
        # comes out of the budget *before* packing. Adding it afterwards would
        # push chunks past the embedding model's window, where the overflow is
        # discarded with no error -- the exact failure this budget exists to
        # prevent.
        prefix = f"{section_title}\n\n" if section_title else ""
        budget = config.size - count_tokens(prefix) if prefix else config.size
        if budget < _MIN_BUDGET:
            # A pathologically long heading. Drop the prefix rather than emit
            # chunks that are mostly heading and no content.
            prefix, budget = "", config.size

        units = _split_units(body, budget, count_tokens)
        for piece in _pack(units, budget, config.overlap, count_tokens):
            # Prefixing the heading keeps the chunk self-describing. Without it
            # a chunk reading "It must be renewed annually" is unusable out of
            # context, both to the model and to a human reading the citation.
            chunks.append(
                Chunk(
                    text=f"{prefix}{piece}".strip(),
                    index=len(chunks),
                    metadata=dict(section_meta),
                )
            )

    return [c for c in chunks if c.text]


def _sections(text: str) -> Iterator[tuple[str, str]]:
    """Split on markdown headings, yielding ``(heading, body)``.

    Documents without headings yield a single untitled section.
    """
    matches = list(_HEADING.finditer(text))
    if not matches:
        yield "", text
        return

    preamble = text[: matches[0].start()].strip()
    if preamble:
        yield "", preamble

    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        yield match.group(2).strip(), text[match.end() : end].strip()


def _split_units(body: str, max_tokens: int, count: TokenCounter) -> list[str]:
    """Break a section into the smallest units that each fit the budget."""
    units: list[str] = []
    for paragraph in (p.strip() for p in _PARAGRAPH.split(body)):
        if not paragraph:
            continue
        if count(paragraph) <= max_tokens:
            units.append(paragraph)
            continue
        for sentence in (s.strip() for s in _SENTENCE.split(paragraph)):
            if not sentence:
                continue
            if count(sentence) <= max_tokens:
                units.append(sentence)
            else:
                # A single oversized sentence -- a table row, a minified blob,
                # a stripped-newline PDF extraction. Hard-cut it.
                units.extend(_hard_split(sentence, max_tokens, count))
    return units


def _hard_split(text: str, max_tokens: int, count: TokenCounter) -> list[str]:
    """Cut an oversized run of text into pieces that each fit the budget.

    The character-per-token estimate is only a starting guess: it holds for
    prose but not for dense scripts, code, or the mojibake a bad PDF extraction
    produces, where a character can cost more than a token. So any piece that
    still overflows is *split again* rather than trimmed -- trimming would
    silently discard the remainder, which is how text goes missing from an
    index without anything appearing to fail.
    """
    span = max(1, int(max_tokens * _CHARS_PER_TOKEN))
    queue = [text[i : i + span] for i in range(0, len(text), span)]
    out: list[str] = []

    while queue:
        piece = queue.pop(0)
        if not piece:
            continue
        if count(piece) <= max_tokens or len(piece) == 1:
            out.append(piece)
            continue
        # Still too big: halve it and re-examine *both* halves.
        midpoint = len(piece) // 2
        queue.insert(0, piece[midpoint:])
        queue.insert(0, piece[:midpoint])

    return out


def _pack(
    units: list[str], budget: int, overlap: int, count: TokenCounter
) -> Iterator[str]:
    """Greedily fill chunks up to ``budget``, carrying overlap between them.

    ``budget`` is the configured size minus any per-chunk prefix, so what this
    yields can be prefixed without exceeding the embedding window.
    """
    buffer: list[str] = []
    buffer_tokens = 0

    for unit in units:
        unit_tokens = count(unit)
        if buffer and buffer_tokens + unit_tokens > budget:
            yield "\n\n".join(buffer)
            buffer, buffer_tokens = _carry_overlap(buffer, overlap, count)
        buffer.append(unit)
        buffer_tokens += unit_tokens

    if buffer:
        yield "\n\n".join(buffer)


def _carry_overlap(
    buffer: list[str], overlap: int, count: TokenCounter
) -> tuple[list[str], int]:
    """Keep trailing whole units as the next chunk's lead-in.

    Whole units, not a token slice, so the overlap is still readable prose --
    a half-sentence of overlap helps neither retrieval nor the reader.
    """
    if overlap <= 0:
        return [], 0
    carried: list[str] = []
    total = 0
    for unit in reversed(buffer):
        unit_tokens = count(unit)
        if total + unit_tokens > overlap:
            break
        carried.insert(0, unit)
        total += unit_tokens
    return carried, total


__all__ = [
    "Chunk",
    "TokenCounter",
    "chunk_document",
    "default_token_counter",
    "tokenizer_counter",
]
