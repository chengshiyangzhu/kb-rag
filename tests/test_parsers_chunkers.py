"""Tests for the Stage 1 parsers, the cleaner, and the Stage 2 chunkers."""
from __future__ import annotations

from pathlib import Path

from app.chunkers import ChunkerConfig, RecursiveCharChunker, get_chunker
from app.ingest import clean_text, get_parser
from app.ingest.parsers.markdown_parser import MarkdownParser
from app.ingest.parsers.txt_parser import TxtParser
from app.ingest.parsers.xlsx_parser import XlsxParser
from app.models.document import Document


def test_txt_parser(tmp_path: Path) -> None:
    """TxtParser returns a single Document with the file contents."""
    path = tmp_path / "note.txt"
    path.write_text("Hello world\nThis is a test.", encoding="utf-8")

    docs = TxtParser().parse(path)

    assert len(docs) == 1
    assert "Hello" in docs[0].text
    assert docs[0].metadata.source == "note.txt"
    assert docs[0].metadata.doc_id  # non-empty


def test_markdown_parser(tmp_path: Path) -> None:
    """MarkdownParser returns a single Document preserving raw Markdown."""
    path = tmp_path / "note.md"
    path.write_text("# Title\n\nSome **markdown** content.", encoding="utf-8")

    docs = MarkdownParser().parse(path)

    assert len(docs) == 1
    assert "Title" in docs[0].text
    assert docs[0].metadata.page is None


def test_xlsx_parser(tmp_path: Path) -> None:
    """XlsxParser returns one Document per sheet with the correct sheet name."""
    from openpyxl import Workbook

    path = tmp_path / "book.xlsx"
    wb = Workbook()
    wb.active.title = "Alpha"
    wb.active["A1"] = "name"
    wb.active["B1"] = "value"
    wb.active["A2"] = "x"
    wb.active["B2"] = 1
    wb.create_sheet("Beta")
    wb["Beta"]["A1"] = "y"
    wb.save(str(path))

    docs = XlsxParser().parse(path)

    assert len(docs) == 2
    assert {d.metadata.sheet for d in docs} == {"Alpha", "Beta"}
    assert all(d.metadata.page is None for d in docs)
    assert all(d.metadata.doc_id == docs[0].metadata.doc_id for d in docs)


def test_recursive_chunker() -> None:
    """RecursiveCharChunker splits a long string into bounded chunks."""
    text = "a" * 2000
    doc = Document.from_text(text=text, source="test", doc_id="doc1")
    chunker = RecursiveCharChunker(ChunkerConfig(chunk_size=512, overlap=64))

    chunks = chunker.chunk(doc)

    assert len(chunks) >= 2
    for c in chunks:
        assert len(c.text) <= 512 + 64
        assert c.metadata.doc_id == "doc1"


def test_recursive_chunker_factory(mock_settings) -> None:
    """get_chunker returns a RecursiveCharChunker for the default settings."""
    chunker = get_chunker(mock_settings)
    assert isinstance(chunker, RecursiveCharChunker)


def test_cleaner() -> None:
    """clean_text collapses runs of spaces and excessive newlines."""
    result = clean_text("hello   world\n\n\nfoo")

    assert "   " not in result
    assert "hello world" in result
    assert "\n\n\n" not in result
    # Paragraph break is preserved.
    assert "world\n\nfoo" in result


def test_get_parser_dispatch(tmp_path: Path) -> None:
    """get_parser dispatches by file extension."""
    from app.ingest.parsers.pdf_parser import PdfParser
    from app.ingest.parsers.txt_parser import TxtParser

    assert isinstance(get_parser(tmp_path / "a.txt"), TxtParser)
    assert isinstance(get_parser(tmp_path / "a.pdf"), PdfParser)
