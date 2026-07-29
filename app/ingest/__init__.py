"""app.ingest package.

Exposes the text cleaner and the parser factory as the public surface of the
ingestion stage.
"""
from __future__ import annotations

from app.ingest.cleaner import clean_document, clean_documents, clean_text
from app.ingest.parsers import get_parser

__all__ = [
    "clean_text",
    "clean_document",
    "clean_documents",
    "get_parser",
]
