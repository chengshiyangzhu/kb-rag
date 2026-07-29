"""Chunker abstract base class and shared configuration.

A :class:`Chunker` turns a single :class:`~app.models.document.Document` into a
list of :class:`~app.models.document.Chunk` instances. The
:meth:`Chunker.chunk_documents` batch entry point guarantees a globally
continuous ``chunk_index`` across all chunks emitted for a batch.
"""
from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from app.models.document import Chunk, Document, Metadata


@dataclass
class ChunkerConfig:
    """Tuning parameters shared by all chunker implementations.

    Attributes:
        chunk_size: Target size of a chunk. Interpreted as tokens by
            token-based chunkers and as characters by character-based ones.
        overlap: Number of tokens/characters of overlap between adjacent chunks.
        min_chunk_size: Minimum size below which a chunk is discarded/merged.
    """

    chunk_size: int = 512
    overlap: int = 64
    min_chunk_size: int = 50


class Chunker(ABC):
    """Abstract base class for all chunkers."""

    def __init__(self, config: ChunkerConfig | None = None) -> None:
        """Initialise the chunker with an optional configuration.

        Args:
            config: Chunker tuning parameters. Defaults to a fresh
                :class:`ChunkerConfig` when omitted.
        """
        self.config = config or ChunkerConfig()

    @abstractmethod
    def chunk(self, doc: Document) -> list[Chunk]:
        """Split a single document into chunks.

        Args:
            doc: The :class:`Document` to chunk.

        Returns:
            A list of :class:`Chunk` instances derived from ``doc``.
        """
        raise NotImplementedError

    def chunk_documents(self, docs: list[Document]) -> list[Chunk]:
        """Chunk a batch of documents with globally continuous ``chunk_index``.

        Args:
            docs: List of :class:`Document` instances to chunk.

        Returns:
            A flat list of :class:`Chunk` instances whose ``metadata.chunk_index``
            values form a continuous 0-based sequence across all input documents.
        """
        chunks: list[Chunk] = []
        global_idx = 0
        for doc in docs:
            doc_chunks = self.chunk(doc)
            for ch in doc_chunks:
                ch.metadata.chunk_index = global_idx
                global_idx += 1
            chunks.extend(doc_chunks)
        return chunks

    def _make_chunk_id(self, doc_id: str, idx: int) -> str:
        """Build a deterministic 16-char hex chunk id from a doc id and index.

        Args:
            doc_id: The parent document id.
            idx: The local chunk index within the document.

        Returns:
            The first 16 characters of the SHA-1 hex digest of
            ``"{doc_id}::chunk::{idx}"``.
        """
        raw = f"{doc_id}::chunk::{idx}".encode("utf-8")
        return hashlib.sha1(raw).hexdigest()[:16]

    def _build_metadata(self, doc: Document, idx: int, offset: int | None) -> Metadata:
        """Build chunk metadata inheriting document-level fields.

        Args:
            doc: The parent :class:`Document`.
            idx: Local chunk index within the document.
            offset: Optional character offset of the chunk within the document
                text; recorded in ``bbox.char_offset`` for traceability.

        Returns:
            A :class:`Metadata` instance populated with inherited fields.
        """
        bbox: dict[str, Any] | None = None
        if offset is not None:
            bbox = {"char_offset": offset}
        return Metadata(
            source=doc.metadata.source,
            page=doc.metadata.page,
            sheet=doc.metadata.sheet,
            tag=list(doc.metadata.tag),
            doc_id=doc.metadata.doc_id,
            chunk_index=idx,
            bbox=bbox,
        )

    def _build_chunk(
        self, doc: Document, text: str, idx: int, offset: int | None = None
    ) -> Chunk:
        """Build a :class:`Chunk` with deterministic id and inherited metadata.

        Args:
            doc: The parent :class:`Document`.
            text: The chunk text.
            idx: Local chunk index within the document.
            offset: Optional character offset of the chunk within the document.

        Returns:
            A populated :class:`Chunk` instance.
        """
        return Chunk(
            id=self._make_chunk_id(doc.metadata.doc_id, idx),
            text=text,
            metadata=self._build_metadata(doc, idx, offset),
        )
