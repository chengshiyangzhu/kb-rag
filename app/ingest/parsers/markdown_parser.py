"""Markdown parser."""
from __future__ import annotations

from pathlib import Path

from app.ingest.parsers.base import Parser, _make_doc_id
from app.models.document import Document, Metadata


class MarkdownParser(Parser):
    """Parse a ``.md`` file into a single :class:`Document`.

    The raw Markdown source is preserved verbatim so downstream structural
    chunkers can still recognise heading and table syntax.
    """

    def parse(self, file_path: Path) -> list[Document]:
        """Read the Markdown file as text and wrap it in a single Document.

        Args:
            file_path: Path to the ``.md`` file.

        Returns:
            A one-element list containing the parsed :class:`Document`.
        """
        text = file_path.read_text(encoding="utf-8", errors="ignore")
        doc_id = _make_doc_id(file_path, None)
        metadata = Metadata(source=file_path.name, doc_id=doc_id)
        return [Document(id=doc_id, text=text, metadata=metadata)]
