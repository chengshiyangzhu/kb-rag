"""Semantic chunker based on sentence-embedding cosine similarity.

The document is split into sentences (Chinese- and English-aware), each
sentence is embedded with a lightweight sentence-transformers model, and
adjacent sentences whose cosine similarity drops below a threshold are placed
into separate chunks. Sentence accumulation also respects ``chunk_size``
characters so no chunk grows without bound.

The sentence-transformers model is loaded lazily on first use; if it is not
available the chunker degrades gracefully and returns the whole document as a
single chunk rather than raising.
"""
from __future__ import annotations

import re
from typing import Any

from app.chunkers.base import Chunker, ChunkerConfig
from app.models.document import Chunk, Document
from app.observability.logging import get_logger

logger = get_logger(__name__)

# Split after sentence-ending punctuation (Chinese and English). The regex
# keeps the terminator attached to the preceding sentence.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.。！？!?])\s*")


class SemanticChunker(Chunker):
    """Chunk text by semantic similarity between adjacent sentences."""

    DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    DEFAULT_THRESHOLD = 0.75

    def __init__(
        self,
        config: ChunkerConfig | None = None,
        model_name: str | None = None,
        threshold: float | None = None,
    ) -> None:
        """Initialise the semantic chunker.

        Args:
            config: Chunker tuning parameters.
            model_name: Override for the sentence-transformers model name.
            threshold: Override for the cosine similarity split threshold.
        """
        super().__init__(config)
        self.model_name = model_name or self.DEFAULT_MODEL
        self.threshold = threshold if threshold is not None else self.DEFAULT_THRESHOLD
        # ``None`` = uninitialised, ``False`` = unavailable, an instance = ready.
        self._model: Any = None

    def _ensure_model(self) -> Any:
        """Lazily load the sentence-transformers model.

        Returns:
            A ``SentenceTransformer`` instance, or ``None`` if the dependency is
            unavailable.
        """
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self.model_name)
            except Exception as exc:  # pragma: no cover - depends on env
                logger.warning(
                    "sentence-transformers unavailable; semantic chunker degrading to single chunk",
                    model=self.model_name,
                    error=str(exc),
                )
                self._model = False
        return self._model if self._model is not False else None

    def _split_sentences(self, text: str) -> list[str]:
        """Split ``text`` into sentences using bilingual punctuation.

        Args:
            text: The text to split.

        Returns:
            A list of non-empty sentence strings.
        """
        parts = _SENTENCE_SPLIT_RE.split(text)
        return [s.strip() for s in parts if s.strip()]

    def chunk(self, doc: Document) -> list[Chunk]:
        """Split ``doc`` into semantic chunks.

        Args:
            doc: The :class:`Document` to chunk.

        Returns:
            A list of :class:`Chunk` instances. When the embedding model is
            unavailable the whole document is returned as a single chunk.
        """
        text = doc.text or ""
        if not text.strip():
            return []
        model = self._ensure_model()
        if model is None:
            return [self._build_chunk(doc, text, 0, 0)]
        sentences = self._split_sentences(text)
        if not sentences:
            return []
        embeddings = model.encode(sentences, normalize_embeddings=True)
        chunks: list[Chunk] = []
        current: list[str] = []
        current_len = 0
        offset = 0
        idx = 0
        for i, sent in enumerate(sentences):
            would_exceed = (
                current
                and current_len + len(sent) > self.config.chunk_size
            )
            boundary = False
            if i + 1 < len(sentences):
                sim = float(embeddings[i] @ embeddings[i + 1])
                if sim < self.threshold and current_len >= self.config.min_chunk_size:
                    boundary = True
            if (would_exceed or boundary) and current:
                chunk_text = "".join(current)
                chunks.append(self._build_chunk(doc, chunk_text, idx, offset))
                offset += len(chunk_text)
                idx += 1
                current = [sent]
                current_len = len(sent)
            else:
                current.append(sent)
                current_len += len(sent)
        if current:
            chunk_text = "".join(current)
            chunks.append(self._build_chunk(doc, chunk_text, idx, offset))
        return chunks
