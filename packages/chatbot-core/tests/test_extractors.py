"""Document extraction tests.

The behaviour that matters most here is the *refusal* path. An extractor that
returns a few ligature artefacts from a scanned PDF produces chunks that match
queries and answer nothing -- and unlike a crash, it looks like success. These
assert that such documents are skipped rather than indexed.
"""

from __future__ import annotations

import io

import pytest

from chatbot_core.retrieval.extractors import (
    MIN_USEFUL_CHARS,
    SUPPORTED_SUFFIXES,
    HTMLExtractor,
    PDFExtractor,
    TextExtractor,
    _clean_pdf_text,  # noqa: PLC2701
    extract,
    extractor_for,
)

PROSE = (
    "Invoices are due within thirty days of issue. Late payment incurs a fee "
    "of one and a half percent per month. Refunds are processed within five "
    "business days of approval, and are returned to the original payment "
    "method wherever that remains possible. "
) * 3


class TestDispatch:
    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("a.md", TextExtractor),
            ("a.txt", TextExtractor),
            ("a.pdf", PDFExtractor),
            ("a.html", HTMLExtractor),
            ("A.PDF", PDFExtractor),
        ],
    )
    def test_selects_by_suffix(self, name, expected):
        assert isinstance(extractor_for(name), expected)

    def test_unknown_suffix_has_no_extractor(self):
        assert extractor_for("a.xyz") is None
        assert extract(b"data", "a.xyz") is None

    def test_supported_suffixes_are_lowercase_with_dot(self):
        assert all(s.startswith(".") and s.islower() for s in SUPPORTED_SUFFIXES)


class TestText:
    def test_extracts_utf8(self):
        result = extract(b"# Title\n\nBody text.", "a.md")
        assert result is not None
        assert "Body text." in result.text

    def test_rejects_non_utf8(self):
        assert extract(b"\xff\xfe\x00invalid", "a.txt") is None

    def test_rejects_whitespace_only(self):
        assert extract(b"   \n\n  ", "a.md") is None


class TestPDFCleanup:
    def test_rejoins_hyphenated_line_breaks(self):
        assert "international" in _clean_pdf_text("inter-\nnational")

    def test_unwraps_mid_sentence_newlines(self):
        """PDFs store glyph positions, not paragraphs, so extractors hard-wrap.
        Left alone the chunker sees one sentence per line and splits wrongly."""
        assert _clean_pdf_text("the quick brown\nfox jumps") == "the quick brown fox jumps"

    def test_preserves_paragraph_breaks(self):
        assert "\n\n" in _clean_pdf_text("First para.\n\nSecond para.")

    def test_collapses_runs_of_blank_lines(self):
        assert "\n\n\n" not in _clean_pdf_text("A.\n\n\n\n\nB.")


class TestPDF:
    @staticmethod
    def make_pdf(pages: list[str]) -> bytes:
        pytest.importorskip("pypdf")
        pytest.importorskip(
            "reportlab", reason="needs reportlab to synthesise a PDF"
        )
        from reportlab.pdfgen import canvas

        buffer = io.BytesIO()
        pdf = canvas.Canvas(buffer)
        for body in pages:
            text_object = pdf.beginText(50, 800)
            for line in body.split(". "):
                text_object.textLine(line.strip() + ".")
            pdf.drawText(text_object)
            pdf.showPage()
        pdf.save()
        return buffer.getvalue()

    def test_extracts_text_with_page_markers(self):
        data = self.make_pdf([PROSE, PROSE])
        result = extract(data, "doc.pdf")
        assert result is not None
        # Page markers give the chunker a boundary and let a citation name a page.
        assert "## Page 1" in result.text
        assert "## Page 2" in result.text
        assert result.metadata["pages"] == "2"

    def test_rejects_pdf_with_no_text_layer(self):
        """A scanned PDF yields a few artefacts. Indexing those is worse than
        skipping the file, because it looks like it worked."""
        data = self.make_pdf(["x"])
        assert extract(data, "scan.pdf") is None

    def test_rejects_corrupt_pdf(self):
        assert extract(b"not a pdf at all", "broken.pdf") is None


class TestHTML:
    def test_extracts_body_content(self):
        html = f"<html><head><title>Docs</title></head><body><main><h1>Billing</h1><p>{PROSE}</p></main></body></html>"
        result = extract(html.encode(), "page.html")
        assert result is not None
        assert result.title == "Docs"
        assert "# Billing" in result.text
        assert "thirty days" in result.text

    def test_strips_navigation_and_scripts(self):
        """Nav and cookie banners repeat on every page of a site; indexing them
        buries real content under near-identical chunks."""
        html = (
            "<html><body>"
            "<nav>Home About Contact</nav>"
            "<script>var tracking = 1;</script>"
            "<footer>Cookie notice</footer>"
            f"<main><p>{PROSE}</p></main>"
            "</body></html>"
        )
        result = extract(html.encode(), "page.html")
        assert result is not None
        for noise in ("Home About Contact", "tracking", "Cookie notice"):
            assert noise not in result.text

    def test_preserves_heading_levels_as_markdown(self):
        html = f"<html><body><main><h2>Sub</h2><p>{PROSE}</p></main></body></html>"
        result = extract(html.encode(), "page.html")
        assert result is not None
        assert "## Sub" in result.text

    def test_renders_list_items_as_bullets(self):
        html = f"<html><body><main><ul><li>First item</li><li>Second item</li></ul><p>{PROSE}</p></main></body></html>"
        result = extract(html.encode(), "page.html")
        assert result is not None
        assert "- First item" in result.text

    def test_rejects_page_with_too_little_text(self):
        html = "<html><body><main><p>Hi</p></main></body></html>"
        assert extract(html.encode(), "thin.html") is None

    def test_threshold_is_the_documented_constant(self):
        body = "word " * 5
        assert len(body) < MIN_USEFUL_CHARS
        assert extract(f"<html><body><p>{body}</p></body></html>".encode(), "a.html") is None


class TestGarbageDetection:
    """Length alone doesn't prove extraction worked.

    A PDF with a broken font encoding yields output of the right size and the
    wrong content. It passes a length check and lands in the index as chunks
    that match queries and answer nothing -- worse than a skipped file, because
    it looks like success.
    """

    def test_mojibake_is_rejected(self):
        from chatbot_core.retrieval.extractors import looks_like_prose

        garbage = "T-ɿI]�gE91p��R~��ED��O�I#ĩ��A�)����{n/E,<�2��Z��D�G��6��l" * 12
        assert not looks_like_prose(garbage, "broken.pdf")

    def test_replacement_character_is_rejected(self):
        from chatbot_core.retrieval.extractors import looks_like_prose

        assert not looks_like_prose(PROSE + "�", "partial.pdf")

    def test_lost_word_boundaries_are_rejected(self):
        from chatbot_core.retrieval.extractors import looks_like_prose

        assert not looks_like_prose("word" * 200, "runtogether.pdf")

    def test_ordinary_prose_passes(self):
        from chatbot_core.retrieval.extractors import looks_like_prose

        assert looks_like_prose(PROSE, "fine.pdf")

    def test_non_english_prose_passes(self):
        """The target is broken encoding, not non-Latin scripts."""
        from chatbot_core.retrieval.extractors import looks_like_prose

        text = "l'échéance des factures est de trente jours après émission. " * 8
        assert looks_like_prose(text, "fr.pdf")
        assert looks_like_prose("发票应在开具后三十天内支付。" * 20, "zh.pdf")

    def test_code_heavy_text_passes(self):
        """Documentation is full of punctuation; it must not be mistaken for
        broken encoding."""
        from chatbot_core.retrieval.extractors import looks_like_prose

        code = "def handler(x: int) -> dict[str, Any]: return {'k': x * 2}\n" * 12
        assert looks_like_prose(code, "doc.md")
