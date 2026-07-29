"""app.ingest.parsers package.

Re-exports the :class:`Parser` ABC, every concrete parser, and the
:func:`get_parser` factory for convenient ``from app.ingest.parsers import ...``
imports.
"""
from __future__ import annotations

from app.ingest.parsers.base import Parser, _make_doc_id, _now_utc
from app.ingest.parsers.docx_parser import DocxParser
from app.ingest.parsers.factory import get_parser
from app.ingest.parsers.html_parser import HtmlParser
from app.ingest.parsers.markdown_parser import MarkdownParser
from app.ingest.parsers.pdf_parser import PdfParser
from app.ingest.parsers.txt_parser import TxtParser
from app.ingest.parsers.xlsx_parser import XlsxParser

__all__ = [
    "Parser",
    "DocxParser",
    "HtmlParser",
    "MarkdownParser",
    "PdfParser",
    "TxtParser",
    "XlsxParser",
    "get_parser",
    "_make_doc_id",
    "_now_utc",
]
