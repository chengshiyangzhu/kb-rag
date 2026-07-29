"""Ingestion pipeline orchestrating parse -> clean -> chunk -> embed -> store.

The :class:`IngestPipeline` ties together every upstream stage (parsing,
cleaning, chunking, embedding, vector storage and BM25 indexing) into a single
end-to-end flow for file ingestion.  Heavy components (chunker, embedder,
vector store, BM25 retriever) are lazily initialised on first use so that
constructing the pipeline is cheap and model loading is deferred until the
first ``ingest_file`` call.
"""
from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel, Field

from app.chunkers import get_chunker
from app.embedders import get_embedder
from app.ingest import clean_documents, get_parser
from app.observability.logging import bind_trace_id, get_logger
from app.observability.metrics import record_ingest
from app.observability.tracing import start_span
from app.retrieval import BM25Retriever
from app.stores import get_vector_store

logger = get_logger(__name__)


class IngestResult(BaseModel):
    """Outcome of ingesting a single file.

    Attributes:
        doc_id: Identifier of the ingested document.
        num_chunks: Number of chunks produced and stored.
        file_type: File extension (without dot) of the source file.
        trace_id: Trace identifier correlating logs/metrics across the request.
        errors: Non-fatal errors encountered during ingestion.
    """

    doc_id: str
    num_chunks: int
    file_type: str
    trace_id: str
    errors: list[str] = Field(default_factory=list)


class IngestPipeline:
    """End-to-end document ingestion pipeline.

    The pipeline parses a file into documents, cleans the text, chunks it,
    embeds the chunks, persists them to the vector store, and updates the BM25
    index.  All heavy components are lazily initialised on first use.

    Args:
        settings: Application :class:`Settings` instance.  Only stored in the
            constructor; components are built on demand.
    """

    def __init__(self, settings: object) -> None:
        """Store settings; components are initialised lazily.

        Args:
            settings: Application :class:`Settings` instance.
        """
        self.settings = settings
        self._chunker = None
        self._embedder = None
        self._vector_store = None
        self._bm25_retriever: BM25Retriever | None = None

    # ------------------------------------------------------------------
    # Lazy component accessors
    # ------------------------------------------------------------------

    @property
    def chunker(self):
        """Lazily build and return the chunker."""
        if self._chunker is None:
            self._chunker = get_chunker(self.settings)
        return self._chunker

    @property
    def embedder(self):
        """Lazily build and return the embedder."""
        if self._embedder is None:
            self._embedder = get_embedder(self.settings)
        return self._embedder

    @property
    def vector_store(self):
        """Lazily build and return the vector store."""
        if self._vector_store is None:
            self._vector_store = get_vector_store(self.settings)
        return self._vector_store

    @property
    def bm25_retriever(self) -> BM25Retriever:
        """Lazily build and return the BM25 retriever."""
        if self._bm25_retriever is None:
            self._bm25_retriever = BM25Retriever(self.settings.bm25_index_path)
        return self._bm25_retriever

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ingest_file(self, file_path: Path) -> IngestResult:
        """Ingest a single file end-to-end.

        The flow is: parse -> clean -> chunk -> embed (batch) -> store.upsert
        -> bm25.add -> bm25.persist.  A ``trace_id`` is generated and bound to
        the logging context so every log line within the request is
        correlatable.

        Args:
            file_path: Path to the file to ingest.

        Returns:
            An :class:`IngestResult` describing the outcome.  Non-fatal errors
            are recorded in ``result.errors`` rather than raised.
        """
        trace_id = uuid4().hex
        file_type = file_path.suffix.lower().lstrip(".") or "unknown"
        errors: list[str] = []
        doc_id = uuid4().hex
        num_chunks = 0

        with bind_trace_id(trace_id), start_span("ingest"):
            logger.info(
                "ingest.start",
                file=str(file_path),
                file_type=file_type,
            )

            # 1. Parse
            try:
                parser = get_parser(file_path)
                docs = parser.parse(file_path)
            except Exception as exc:  # noqa: BLE001 - record and continue
                logger.error(
                    "ingest.parse.failed",
                    file=str(file_path),
                    error=str(exc),
                )
                errors.append(f"parse: {exc}")
                return IngestResult(
                    doc_id=doc_id,
                    num_chunks=0,
                    file_type=file_type,
                    trace_id=trace_id,
                    errors=errors,
                )

            if not docs:
                errors.append("parse: no documents produced")
                return IngestResult(
                    doc_id=doc_id,
                    num_chunks=0,
                    file_type=file_type,
                    trace_id=trace_id,
                    errors=errors,
                )

            doc_id = docs[0].metadata.doc_id

            # 2. Clean
            try:
                docs = clean_documents(docs)
            except Exception as exc:  # noqa: BLE001
                logger.warning("ingest.clean.failed", error=str(exc))
                errors.append(f"clean: {exc}")

            # 3. Chunk
            try:
                chunks = self.chunker.chunk_documents(docs)
            except Exception as exc:  # noqa: BLE001
                logger.error("ingest.chunk.failed", error=str(exc))
                errors.append(f"chunk: {exc}")
                return IngestResult(
                    doc_id=doc_id,
                    num_chunks=0,
                    file_type=file_type,
                    trace_id=trace_id,
                    errors=errors,
                )

            if not chunks:
                errors.append("chunk: no chunks produced")
                return IngestResult(
                    doc_id=doc_id,
                    num_chunks=0,
                    file_type=file_type,
                    trace_id=trace_id,
                    errors=errors,
                )

            num_chunks = len(chunks)

            # 4. Embed (batch)
            try:
                vectors = self.embedder.embed_texts([c.text for c in chunks])
            except Exception as exc:  # noqa: BLE001
                logger.error("ingest.embed.failed", error=str(exc))
                errors.append(f"embed: {exc}")
                return IngestResult(
                    doc_id=doc_id,
                    num_chunks=0,
                    file_type=file_type,
                    trace_id=trace_id,
                    errors=errors,
                )

            # 5. Store
            try:
                self.vector_store.upsert(chunks, vectors)
            except Exception as exc:  # noqa: BLE001
                logger.error("ingest.store.failed", error=str(exc))
                errors.append(f"store: {exc}")

            # 6. BM25 index
            try:
                self.bm25_retriever.add(chunks)
                self.bm25_retriever.persist()
            except Exception as exc:  # noqa: BLE001
                logger.error("ingest.bm25.failed", error=str(exc))
                errors.append(f"bm25: {exc}")

            record_ingest(file_type)
            logger.info(
                "ingest.done",
                file=str(file_path),
                doc_id=doc_id,
                num_chunks=num_chunks,
                errors=len(errors),
            )

            return IngestResult(
                doc_id=doc_id,
                num_chunks=num_chunks,
                file_type=file_type,
                trace_id=trace_id,
                errors=errors,
            )

    def ingest_directory(
        self,
        dir_path: Path,
        pattern: str = "**/*",
    ) -> list[IngestResult]:
        """Ingest every file under ``dir_path`` matching ``pattern``.

        Each file is processed independently via :meth:`ingest_file`; a failure
        on one file does not abort the batch.

        Args:
            dir_path: Root directory to scan.
            pattern: Glob pattern (default ``"**/*"`` for recursive scan).

        Returns:
            A list of :class:`IngestResult`, one per file encountered.
        """
        results: list[IngestResult] = []
        for file_path in sorted(dir_path.glob(pattern)):
            if not file_path.is_file():
                continue
            try:
                result = self.ingest_file(file_path)
            except Exception as exc:  # noqa: BLE001 - never crash the batch
                logger.error(
                    "ingest.file.unexpected",
                    file=str(file_path),
                    error=str(exc),
                )
                result = IngestResult(
                    doc_id=uuid4().hex,
                    num_chunks=0,
                    file_type=file_path.suffix.lower().lstrip(".") or "unknown",
                    trace_id=uuid4().hex,
                    errors=[f"unexpected: {exc}"],
                )
            results.append(result)
        return results

    def delete_document(self, doc_id: str) -> None:
        """Delete a document and all its chunks from every store.

        Removes chunks from the vector store and the BM25 index, then
        persists the BM25 index.

        Args:
            doc_id: Identifier of the document to remove.
        """
        logger.info("ingest.delete", doc_id=doc_id)
        try:
            self.vector_store.delete_by_doc(doc_id)
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "ingest.delete.store.failed",
                doc_id=doc_id,
                error=str(exc),
            )
        try:
            self.bm25_retriever.remove_by_doc(doc_id)
            self.bm25_retriever.persist()
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "ingest.delete.bm25.failed",
                doc_id=doc_id,
                error=str(exc),
            )
