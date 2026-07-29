"""DOCX parser using python-docx.

Iterates the document body in order, preserving the relative position of
paragraphs and tables. Tables are serialised as Markdown tables so downstream
structural chunkers can keep them intact.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

from docx.document import Document as _DocxDocument
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import Table
from docx.text.paragraph import Paragraph

from app.ingest.parsers.base import Parser, _make_doc_id
from app.models.document import Document, Metadata


def _iter_block_items(doc: _DocxDocument) -> Iterator[Paragraph | Table]:
    """Yield paragraphs and tables in the order they appear in the body.

    Args:
        doc: A python-docx :class:`Document` instance.

    Yields:
        :class:`Paragraph` or :class:`Table` objects in document order.
    """
    parent_elm = doc.element.body
    for child in parent_elm.iterchildren():
        if isinstance(child, CT_P):
            yield Paragraph(child, doc)
        elif isinstance(child, CT_Tbl):
            yield Table(child, doc)


def _table_to_markdown(table: Table) -> str:
    """Convert a python-docx table into a Markdown table string.

    Args:
        table: A python-docx :class:`Table` instance.

    Returns:
        A Markdown table representation (header row + separator + body rows).
    """
    rows = list(table.rows)
    if not rows:
        return ""
    ncols = len(rows[0].cells)
    lines: list[str] = []
    lines.append("| " + " | ".join(cell.text.strip() for cell in rows[0].cells) + " |")
    lines.append("| " + " | ".join("---" for _ in range(ncols)) + " |")
    for row in rows[1:]:
        cells = list(row.cells)
        # Pad / trim to match header column count.
        if len(cells) < ncols:
            cells = cells + [cells[-1]] * (ncols - len(cells))
        lines.append("| " + " | ".join(cell.text.strip() for cell in cells[:ncols]) + " |")
    return "\n".join(lines)


class DocxParser(Parser):
    """Parse a ``.docx`` file into a single :class:`Document`.

    The text of all paragraphs and tables (rendered as Markdown) is concatenated
    in document order. ``metadata.page`` is ``None`` because DOCX is not
    paginated in a reliable way.
    """

    def parse(self, file_path: Path) -> list[Document]:
        """Extract paragraph and table text from a DOCX file.

        Args:
            file_path: Path to the ``.docx`` file.

        Returns:
            A one-element list containing the parsed :class:`Document`.
        """
        from docx import Document as _OpenDoc

        docx_doc = _OpenDoc(str(file_path))
        parts: list[str] = []
        for block in _iter_block_items(docx_doc):
            if isinstance(block, Paragraph):
                text = block.text.strip()
                if text:
                    parts.append(text)
            elif isinstance(block, Table):
                md = _table_to_markdown(block)
                if md:
                    parts.append(md)
        text = "\n\n".join(parts)
        doc_id = _make_doc_id(file_path, None)
        metadata = Metadata(source=file_path.name, page=None, doc_id=doc_id)
        return [Document(id=doc_id, text=text, metadata=metadata)]
