"""Recursive character chunker (LangChain-style).

Splits text by a prioritised list of separators, falling back to the next one
when a split is still larger than ``chunk_size``. The resulting pieces are
merged into chunks of approximately ``chunk_size`` characters with ``overlap``
characters of overlap between adjacent chunks.

The implementation is self-contained: it does not depend on LangChain.
"""
from __future__ import annotations

from app.chunkers.base import Chunker, ChunkerConfig
from app.models.document import Chunk, Document


class RecursiveCharChunker(Chunker):
    """Recursive character chunker with configurable separators."""

    SEPARATORS: list[str] = ["\n\n", "\n", "。", ".", " ", ""]

    def chunk(self, doc: Document) -> list[Chunk]:
        """Split ``doc`` recursively and merge into overlapping chunks.

        Args:
            doc: The :class:`Document` to chunk.

        Returns:
            A list of :class:`Chunk` instances. Each chunk text is at most
            ``chunk_size + overlap`` characters long.
        """
        text = doc.text or ""
        if not text.strip():
            return []
        raw_splits = self._recursive_split(text, self.SEPARATORS)
        merged = self._merge_splits(raw_splits)
        chunks: list[Chunk] = []
        idx = 0
        offset = 0
        max_len = self.config.chunk_size + self.config.overlap
        for piece in merged:
            piece = piece.strip()
            if not piece:
                continue
            # Safety hard cut so we never exceed chunk_size + overlap.
            if len(piece) > max_len:
                piece = piece[:max_len]
            chunks.append(self._build_chunk(doc, piece, idx, offset))
            offset += len(piece)
            idx += 1
        return chunks

    def _recursive_split(self, text: str, separators: list[str]) -> list[str]:
        """Recursively split ``text`` using the separator priority list.

        Args:
            text: The text to split.
            separators: Ordered list of separators to try.

        Returns:
            A list of text fragments all no larger than ``chunk_size`` (the
            empty-string separator guarantees the recursion terminates by
            falling back to hard character slicing).
        """
        if not text:
            return []
        if len(text) <= self.config.chunk_size:
            return [text]
        if not separators:
            return [text]
        sep = separators[0]
        remaining = separators[1:]
        if sep == "":
            # Character-level fallback: hard slice into chunk_size pieces.
            size = self.config.chunk_size
            return [text[i : i + size] for i in range(0, len(text), size)]
        parts = text.split(sep)
        # Re-attach the separator to the start of every part except the first so
        # paragraph/sentence boundaries survive the merge step.
        rebuilt: list[str] = []
        for i, part in enumerate(parts):
            if i == 0:
                rebuilt.append(part)
            else:
                rebuilt.append(sep + part)
        result: list[str] = []
        for part in rebuilt:
            if len(part) > self.config.chunk_size:
                result.extend(self._recursive_split(part, remaining))
            elif part:
                result.append(part)
        return result

    def _merge_splits(self, splits: list[str]) -> list[str]:
        """Merge small splits into chunks of ~``chunk_size`` with overlap.

        Args:
            splits: Pre-split text fragments (may be smaller or larger than
                ``chunk_size``).

        Returns:
            A list of merged chunk strings.
        """
        chunk_size = self.config.chunk_size
        overlap = self.config.overlap
        merged: list[str] = []
        current = ""
        for piece in splits:
            if not piece:
                continue
            candidate = current + piece if current else piece
            if len(candidate) <= chunk_size:
                current = candidate
                continue
            # Candidate overflows: flush the current buffer.
            if current:
                merged.append(current)
            if overlap > 0 and len(current) >= overlap:
                current = current[-overlap:] + piece
            else:
                current = piece
            # If the new current is still too large, hard-cut it.
            if len(current) > chunk_size:
                merged.append(current[:chunk_size])
                current = current[chunk_size:]
        if current:
            merged.append(current)
        return merged
