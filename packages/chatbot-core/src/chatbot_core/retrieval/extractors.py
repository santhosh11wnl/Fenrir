"""Text extraction from documents.

A corpus is rarely markdown. This turns PDFs, HTML, and Word files into clean
text the chunker can work with.

The rule everywhere below: **extract nothing rather than extract garbage.** A
scanned PDF with no text layer yields a handful of ligature artefacts; indexing
those produces chunks that match queries and answer nothing, which is worse
than the document being absent, because it looks like it worked. Every
extractor returns ``None`` when it cannot do the job honestly, and the caller
reports it as skipped.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import structlog

log = structlog.get_logger(__name__)

#: Below this many characters, assume extraction failed rather than that the
#: document was genuinely near-empty. Scanned PDFs typically yield tens of
#: characters of noise; a real page yields hundreds.
MIN_USEFUL_CHARS = 200

#: Minimum fraction of characters that must look like ordinary prose.
#: A PDF with a broken font encoding extracts to output of the right *length*
#: but the wrong *content* -- mojibake like ``T-ɿI]gE91pR~ED`` sails past a
#: length check and lands in the index as chunks that match queries and answer
#: nothing. Real prose in any Latin-script language scores well above 0.8.
MIN_READABLE_RATIO = 0.75

#: Longest run of characters with no space. Real prose breaks often; a long
#: unbroken run means word boundaries were lost, which makes the text useless
#: for retrieval even when the characters themselves decode.
MAX_UNBROKEN_RUN = 120


@dataclass(slots=True)
class Extracted:
    text: str
    title: str | None = None
    #: Extractor-specific detail worth keeping: page count, source URL, etc.
    metadata: dict[str, str] | None = None


@runtime_checkable
class Extractor(Protocol):
    """Turns raw bytes into text. One implementation per document family."""

    suffixes: frozenset[str]

    def extract(self, data: bytes, name: str) -> Extracted | None: ...


# ---------------------------------------------------------------------------
# plain text
# ---------------------------------------------------------------------------


class TextExtractor:
    suffixes = frozenset({".md", ".markdown", ".txt", ".rst", ".csv", ".json", ".yaml", ".yml"})

    def extract(self, data: bytes, name: str) -> Extracted | None:
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            log.warning("not_utf8", name=name)
            return None
        return Extracted(text=text) if text.strip() else None


# ---------------------------------------------------------------------------
# pdf
# ---------------------------------------------------------------------------


class PDFExtractor:
    suffixes = frozenset({".pdf"})

    def extract(self, data: bytes, name: str) -> Extracted | None:
        try:
            from pypdf import PdfReader
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "PDF ingestion needs: uv add 'chatbot-core[documents]'"
            ) from exc

        import io

        try:
            reader = PdfReader(io.BytesIO(data))
        except Exception as exc:  # noqa: BLE001 - malformed PDFs are common
            log.warning("pdf_unreadable", name=name, error=str(exc))
            return None

        if getattr(reader, "is_encrypted", False):
            # Some PDFs are encrypted with an empty owner password and open
            # fine; the rest genuinely can't be read.
            try:
                reader.decrypt("")
            except Exception:  # noqa: BLE001
                log.warning("pdf_encrypted", name=name)
                return None

        pages: list[str] = []
        for number, page in enumerate(reader.pages, start=1):
            try:
                body = page.extract_text() or ""
            except Exception as exc:  # noqa: BLE001 - one bad page, not the file
                log.warning("pdf_page_failed", name=name, page=number, error=str(exc))
                continue
            body = _clean_pdf_text(body)
            if body.strip():
                # A page marker gives the chunker a structural boundary and
                # lets a citation say which page a claim came from.
                pages.append(f"## Page {number}\n\n{body}")

        text = "\n\n".join(pages)
        if len(text) < MIN_USEFUL_CHARS:
            log.warning(
                "pdf_no_text_layer",
                name=name,
                chars=len(text),
                hint="likely a scanned image; OCR it before ingesting",
            )
            return None

        # Length alone is not enough: a broken font encoding yields output of
        # the right size and the wrong content, which would otherwise land in
        # the index as chunks that match queries and answer nothing.
        if not looks_like_prose(text, name):
            return None

        meta = reader.metadata or {}
        title = (getattr(meta, "title", None) or "").strip() or None
        return Extracted(
            text=text,
            title=title,
            metadata={"pages": str(len(reader.pages)), "format": "pdf"},
        )


def readable_ratio(text: str) -> float:
    """Fraction of characters that look like ordinary prose.

    Counts letters, digits, whitespace, and common punctuation. Deliberately
    Unicode-aware via ``str.isalnum`` so non-English text scores correctly --
    the target is broken *encoding*, not non-Latin scripts.
    """
    if not text:
        return 0.0
    allowed = set(" \t\n\r.,;:!?'\"()[]{}<>/\\-_=+*&%$#@|~`^")
    good = sum(1 for c in text if c.isalnum() or c in allowed)
    return good / len(text)


def looks_like_prose(text: str, name: str) -> bool:
    """Whether extracted text is usable, or garbage that merely has length.

    Two failure modes this catches, both of which pass a length check:

    * **Broken font encoding.** A PDF whose glyphs don't map to Unicode
      extracts as mojibake -- right length, meaningless content.
    * **Lost word boundaries.** Some extractions run every word together.
      The characters decode fine but the text is useless for retrieval.
    """
    if "�" in text:  # U+FFFD REPLACEMENT CHARACTER: decoding already failed
        log.warning("extraction_has_replacement_chars", name=name)
        return False

    ratio = readable_ratio(text)
    if ratio < MIN_READABLE_RATIO:
        log.warning(
            "extraction_not_readable",
            name=name,
            readable_ratio=round(ratio, 3),
            hint="broken font encoding; the PDF likely needs OCR",
        )
        return False

    # Only meaningful for scripts that separate words with spaces. Chinese,
    # Japanese, and Thai legitimately run for hundreds of characters without
    # one, so applying this to them would reject perfectly good documents.
    if not _is_scriptio_continua(text):
        longest = max((len(run) for run in text.split()), default=0)
        if longest > MAX_UNBROKEN_RUN:
            log.warning(
                "extraction_lost_word_boundaries", name=name, longest_run=longest
            )
            return False

    return True


def _is_scriptio_continua(text: str) -> bool:
    """Whether the text is mostly in a script written without word spaces."""
    sample = text[:4000]
    if not sample:
        return False
    count = sum(
        1
        for c in sample
        if "぀" <= c <= "ヿ"  # hiragana, katakana
        or "㐀" <= c <= "䶿"  # CJK extension A
        or "一" <= c <= "鿿"  # CJK unified ideographs
        or "가" <= c <= "힯"  # hangul syllables
        or "฀" <= c <= "๿"  # thai
    )
    return count / len(sample) > 0.15


def _clean_pdf_text(text: str) -> str:
    """Repair the usual damage from PDF text extraction.

    PDFs store glyph positions, not paragraphs, so extractors hard-wrap lines
    and hyphenate across them. Left alone, the chunker sees one sentence per
    line and splits in the wrong places.
    """
    # Re-join words hyphenated across a line break.
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    # A single newline mid-sentence is a layout artefact; a blank line is a
    # real paragraph break and must survive.
    text = re.sub(r"(?<![\n.!?:])\n(?![\n])", " ", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


# ---------------------------------------------------------------------------
# html
# ---------------------------------------------------------------------------


class HTMLExtractor:
    suffixes = frozenset({".html", ".htm"})

    def extract(self, data: bytes, name: str) -> Extracted | None:
        try:
            from bs4 import BeautifulSoup
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "HTML ingestion needs: uv add 'chatbot-core[documents]'"
            ) from exc

        soup = BeautifulSoup(data, "lxml")

        # Strip chrome before extracting. Navigation and cookie banners appear
        # on every page of a site, so indexing them buries the actual content
        # under hundreds of near-identical chunks.
        for tag in soup(
            ["script", "style", "nav", "header", "footer", "aside", "noscript", "form"]
        ):
            tag.decompose()

        title = (soup.title.string or "").strip() if soup.title else None

        # Prefer a semantic content container when the page has one.
        root = soup.find("main") or soup.find("article") or soup.body or soup

        parts: list[str] = []
        for element in root.find_all(
            ["h1", "h2", "h3", "h4", "p", "li", "pre", "td", "blockquote"]
        ):
            content = element.get_text(" ", strip=True)
            if not content:
                continue
            if element.name.startswith("h"):
                # Preserve heading level as markdown so the chunker's
                # structure-aware splitting still has something to work with.
                parts.append(f"{'#' * int(element.name[1])} {content}")
            elif element.name == "li":
                parts.append(f"- {content}")
            else:
                parts.append(content)

        text = re.sub(r"\n{3,}", "\n\n", "\n\n".join(parts)).strip()
        if len(text) < MIN_USEFUL_CHARS:
            log.warning("html_too_little_text", name=name, chars=len(text))
            return None
        return Extracted(text=text, title=title, metadata={"format": "html"})


# ---------------------------------------------------------------------------
# docx
# ---------------------------------------------------------------------------


class DocxExtractor:
    suffixes = frozenset({".docx"})

    def extract(self, data: bytes, name: str) -> Extracted | None:
        try:
            import docx
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "DOCX ingestion needs: uv add 'chatbot-core[documents]'"
            ) from exc

        import io

        try:
            document = docx.Document(io.BytesIO(data))
        except Exception as exc:  # noqa: BLE001
            log.warning("docx_unreadable", name=name, error=str(exc))
            return None

        parts: list[str] = []
        for paragraph in document.paragraphs:
            content = paragraph.text.strip()
            if not content:
                continue
            style = (paragraph.style.name or "").lower()
            if style.startswith("heading"):
                level = "".join(c for c in style if c.isdigit()) or "2"
                parts.append(f"{'#' * min(int(level), 6)} {content}")
            else:
                parts.append(content)

        # Tables carry real content in policy and spec documents; flatten each
        # row to a pipe-delimited line so it survives chunking as a unit.
        for table in document.tables:
            for row in table.rows:
                cells = [c.text.strip() for c in row.cells if c.text.strip()]
                if cells:
                    parts.append(" | ".join(cells))

        text = "\n\n".join(parts).strip()
        if not text:
            return None
        return Extracted(text=text, metadata={"format": "docx"})


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------

_EXTRACTORS: tuple[Extractor, ...] = (
    TextExtractor(),
    PDFExtractor(),
    HTMLExtractor(),
    DocxExtractor(),
)

SUPPORTED_SUFFIXES: frozenset[str] = frozenset(
    suffix for extractor in _EXTRACTORS for suffix in extractor.suffixes
)


#: Leading bytes that identify a format regardless of what it is named.
#: Checked before any filename or Content-Type, because those lie: a URL like
#: ``arxiv.org/pdf/1706.03762`` has no extension at all, and handing a PDF to
#: the HTML parser yields binary parsed as markup -- mojibake that passes a
#: length check and poisons the index.
_MAGIC: tuple[tuple[bytes, str], ...] = (
    (b"%PDF-", ".pdf"),
    (b"PK\x03\x04", ".docx"),  # zip container; docx is the one we extract
)


def sniff_suffix(data: bytes, name: str = "", content_type: str = "") -> str:
    """Determine a document's real type, most reliable signal first.

    Order matters: magic bytes are a property of the file, ``Content-Type`` is
    a claim by the server, and the filename is a guess by whoever named it.
    """
    head = data[:8]
    for magic, suffix in _MAGIC:
        if head.startswith(magic):
            return suffix

    mapped = {
        "application/pdf": ".pdf",
        "text/html": ".html",
        "application/xhtml+xml": ".html",
        "text/markdown": ".md",
        "text/plain": ".txt",
        "application/json": ".json",
        "text/csv": ".csv",
    }.get(content_type.split(";")[0].strip().lower())
    if mapped:
        return mapped

    suffix = Path(name).suffix.lower()
    if suffix in SUPPORTED_SUFFIXES:
        return suffix

    # A bare URL with no other signal is usually a web page.
    return ".html"


def extractor_for(name: str) -> Extractor | None:
    suffix = Path(name).suffix.lower()
    for extractor in _EXTRACTORS:
        if suffix in extractor.suffixes:
            return extractor
    return None


def extract(data: bytes, name: str) -> Extracted | None:
    """Extract text from a document, or return None if it can't be done well."""
    extractor = extractor_for(name)
    if extractor is None:
        log.debug("unsupported_type", name=name, suffix=Path(name).suffix)
        return None
    return extractor.extract(data, name)


__all__ = [
    "MIN_USEFUL_CHARS",
    "SUPPORTED_SUFFIXES",
    "DocxExtractor",
    "Extracted",
    "Extractor",
    "HTMLExtractor",
    "PDFExtractor",
    "TextExtractor",
    "extract",
    "extractor_for",
    "looks_like_prose",
    "readable_ratio",
    "sniff_suffix",
]
