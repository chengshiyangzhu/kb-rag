"""Structural chunker that respects Markdown headings and tables.

The document is split into blocks delimited by Markdown headings
(``^#{1,6}\\s``). Consecutive Markdown table rows are kept as an atomic block
so tables are never split mid-row. Any block exceeding ``chunk_size * 2``
characters is recursively re-chunked with :class:`RecursiveCharChunker`.
"""
from __future__ import annotations

import re

from app.chunkers.base import Chunker, ChunkerConfig
from app.chunkers.recursive_char import RecursiveCharChunker
from app.models.document import Chunk, Document, Metadata

_HEADING_RE = re.compile(r"^#{1,6}\s", re.MULTILINE)
_TABLE_ROW_RE = re.compile(r"^\|.*\|\s*$")


class StructuralChunker(Chunker):
    """Chunk by Markdown structure (headings + atomic tables)."""

    def chunk(self, doc: Document) -> list[Chunk]:
        """Split ``doc`` into structural blocks.

        Args:
            doc: The :class:`Document` to chunk.

        Returns:
            A list of :class:`Chunk` instances. Oversized blocks are re-chunked
            with :class:`RecursiveCharChunker`.
        """
        text = doc.text or ""
        if not text.strip():
            return []
        blocks = self._split_blocks(text)
        chunks: list[Chunk] = []
        idx = 0
        offset = 0
        max_block = self.config.chunk_size * 2
        for block in blocks:
            block = block.strip()
            if not block:
                continue
            if len(block) > max_block:
                # Recursive fallback for oversized blocks.
                sub_doc = self._make_sub_doc(doc, block)
                sub_chunks = RecursiveCharChunker(self.config).chunk(sub_doc)
                for sc in sub_chunks:
                    sc.metadata.chunk_index = idx
                    chunks.append(sc)
                    idx += 1
                offset += len(block)
            else:
                chunks.append(self._build_chunk(doc, block, idx, offset))
                offset += len(block)
                idx += 1
        return chunks

    def _split_blocks(self, text: str) -> list[str]:
        """Split ``text`` into heading-delimited blocks.

        Markdown table rows immediately following a non-table block are kept
        attached to that block; consecutive table rows form their own block.

        Args:
            text: The full document text.

        Returns:
            A list of block strings.
        """
        # Split on lines that start with a Markdown heading marker.
        positions: list[int] = [m.start() for m in _HEADING_RE.finditer(text)]
        if not positions:
            return self._split_tables(text)
        positions.append(len(text))
        blocks: list[str] = []
        for i in range(len(positions) - 1):
            segment = text[positions[i] : positions[i + 1]]
            blocks.extend(self._split_tables(segment))
        return blocks

    def _split_tables(self, segment: str) -> list[str]:
        """Separate Markdown table runs from prose within a segment.

        Args:
            segment: A text segment (typically one heading section).

        Returns:
            A list of blocks where each table run is kept intact.
        """
        lines = segment.split("\n")
        blocks: list[str] = []
        buf: list[str] = []
        in_table = False
        for line in lines:
            is_table = bool(_TABLE_ROW_RE.match(line))
            if is_table:
                if not in_table and buf:
                    blocks.append("\n".join(buf).strip())
                    buf = []
                buf.append(line)
                in_table = True
            else:
                if in_table and buf:
                    blocks.append("\n".join(buf).strip())
                    buf = []
                buf.append(line)
                in_table = False
        if buf:
            blocks.append("\n".join(buf).strip())
        return [b for b in blocks if b]

    def _make_sub_doc(self, doc: Document, block: str) -> Document:
        """Build a temporary Document for recursive sub-chunking.

        Args:
            doc: The parent :class:`Document`.
            block: The block text to wrap.

        Returns:
            A new :class:`Document` carrying the parent's metadata.
        """
        meta = Metadata(
            source=doc.metadata.source,
            page=doc.metadata.page,
            sheet=doc.metadata.sheet,
            tag=list(doc.metadata.tag),
            doc_id=doc.metadata.doc_id,
        )
        return Document(id=doc.id, text=block, metadata=meta)
