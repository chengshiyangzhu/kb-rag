"""Fixed-size token chunker using tiktoken.

Splits document text into overlapping windows measured in tokens (encoding
``cl100k_base``). Windows shorter than ``min_chunk_size`` tokens are skipped so
the final chunk is not a tiny fragment.
"""
from __future__ import annotations

from app.chunkers.base import Chunker, ChunkerConfig
from app.models.document import Chunk, Document


class FixedTokenChunker(Chunker):
    """Chunk text by token count using the ``cl100k_base`` tiktoken encoding."""

    ENCODING_NAME = "cl100k_base"

    def __init__(self, config: ChunkerConfig | None = None) -> None:
        """Initialise the chunker and load the tiktoken encoding.

        Args:
            config: Chunker tuning parameters.
        """
        super().__init__(config)
        import tiktoken

        self._enc = tiktoken.get_encoding(self.ENCODING_NAME)

    def chunk(self, doc: Document) -> list[Chunk]:
        """Split ``doc`` into overlapping token windows.

        Args:
            doc: The :class:`Document` to chunk.

        Returns:
            A list of :class:`Chunk` instances. Empty or sub-``min_chunk_size``
            trailing windows are skipped.
        """
        text = doc.text or ""
        if not text.strip():
            return []
        tokens = self._enc.encode(text)
        size = self.config.chunk_size
        overlap = self.config.overlap
        min_size = self.config.min_chunk_size
        step = max(1, size - overlap)

        chunks: list[Chunk] = []
        idx = 0
        i = 0
        while i < len(tokens):
            window = tokens[i : i + size]
            if len(window) < min_size:
                # Trailing fragment too small to keep; advancing cannot grow it.
                break
            chunk_text = self._enc.decode(window)
            if chunk_text.strip():
                chunks.append(self._build_chunk(doc, chunk_text, idx, offset=i))
                idx += 1
            i += step
        return chunks
