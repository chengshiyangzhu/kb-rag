"""Query pipeline orchestrating retrieve -> rerank -> guardrail -> generate.

The :class:`QueryPipeline` ties together the embedding, hybrid retrieval,
reranking, guardrail and generation stages into a single end-to-end RAG query
flow.  Heavy components (embedder, vector store, BM25 index, hybrid retriever,
reranker, generator) are lazily initialised on first use so that constructing
the pipeline is cheap and model loading is deferred until the first ``query``
call.
"""
from __future__ import annotations

import time
from uuid import uuid4

from pydantic import BaseModel, Field

from app.embedders import get_embedder
from app.generation import CitationParser, Guardrail, get_generator
from app.observability.logging import bind_trace_id, get_logger
from app.observability.metrics import (
    record_generation_latency,
    record_no_result,
    record_query,
    record_retrieval_latency,
)
from app.observability.tracing import start_span
from app.rerank import get_reranker
from app.retrieval import BM25Retriever, HybridRetriever, VectorRetriever
from app.stores import get_vector_store

logger = get_logger(__name__)


class Reference(BaseModel):
    """A citation reference mapped back to a source chunk.

    Attributes:
        chunk_id: Identifier of the source chunk.
        source: Original file path or URI of the chunk.
        page: 1-indexed page number (if applicable).
        score: Relevance score from the retrieval/rerank stage.
        snippet: Truncated preview of the chunk text.
    """

    chunk_id: str
    source: str
    page: int | None = None
    score: float | None = None
    snippet: str = ""


class QueryResult(BaseModel):
    """Result of a RAG query.

    Attributes:
        answer: Generated natural-language answer (or the no-result fallback).
        references: Citation references mapped to source chunks.
        trace_id: Trace identifier correlating logs/metrics across the request.
        no_result: ``True`` when retrieval yielded no confident result.
        retrieval_latency: Retrieval stage latency in seconds.
        generation_latency: Generation stage latency in seconds.
    """

    answer: str
    references: list[Reference] = Field(default_factory=list)
    trace_id: str
    no_result: bool = False
    retrieval_latency: float = 0.0
    generation_latency: float = 0.0


class QueryPipeline:
    """End-to-end RAG query pipeline.

    The pipeline embeds the query, retrieves candidates via hybrid retrieval,
    reranks them, checks the guardrail, and generates an answer with
    citations.  All heavy components are lazily initialised on first use.

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
        self._embedder = None
        self._vector_store = None
        self._bm25_retriever: BM25Retriever | None = None
        self._hybrid_retriever: HybridRetriever | None = None
        self._reranker = None
        self._generator = None
        self._guardrail: Guardrail | None = None
        self._citation_parser: CitationParser | None = None

    # ------------------------------------------------------------------
    # Lazy component accessors
    # ------------------------------------------------------------------

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

    @property
    def hybrid_retriever(self) -> HybridRetriever:
        """Lazily build and return the hybrid retriever.

        The hybrid retriever wraps a :class:`VectorRetriever` (which needs the
        embedder and vector store) and the :class:`BM25Retriever`.
        """
        if self._hybrid_retriever is None:
            vector_retriever = VectorRetriever(
                store=self.vector_store,
                embedder=self.embedder,
            )
            self._hybrid_retriever = HybridRetriever(
                vector_retriever=vector_retriever,
                bm25_retriever=self.bm25_retriever,
                rrf_k=getattr(self.settings, "rrf_k", 60),
            )
        return self._hybrid_retriever

    @property
    def reranker(self):
        """Lazily build and return the reranker."""
        if self._reranker is None:
            self._reranker = get_reranker(self.settings)
        return self._reranker

    @property
    def generator(self):
        """Lazily build and return the generator."""
        if self._generator is None:
            self._generator = get_generator(self.settings)
        return self._generator

    @property
    def guardrail(self) -> Guardrail:
        """Return the guardrail (created on first access)."""
        if self._guardrail is None:
            self._guardrail = Guardrail()
        return self._guardrail

    @property
    def citation_parser(self) -> CitationParser:
        """Return the citation parser (created on first access)."""
        if self._citation_parser is None:
            self._citation_parser = CitationParser()
        return self._citation_parser

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def query(
        self,
        question: str,
        filters: dict | None = None,
        top_n: int | None = None,
    ) -> QueryResult:
        """Run a RAG query end-to-end.

        Flow: embed_query -> hybrid retrieve -> rerank -> guardrail check ->
        generate -> parse citations -> map references.

        A ``trace_id`` is generated and bound to the logging context so every
        log line within the request is correlatable.

        Args:
            question: Natural-language question.
            filters: Optional metadata filters forwarded to the retriever.
            top_n: Override for the number of candidates to retrieve.  Falls
                back to ``settings.retrieve_top_n`` when ``None``.

        Returns:
            A :class:`QueryResult` with the answer, references and latencies.
        """
        trace_id = uuid4().hex
        effective_top_n = top_n or getattr(self.settings, "retrieve_top_n", 20)
        threshold = getattr(self.settings, "rerank_threshold", 0.3)
        rerank_top_k = getattr(self.settings, "rerank_top_k", 5)

        with bind_trace_id(trace_id), start_span("query"):
            logger.info("query.start", question=question[:80])

            # 1-2. Retrieve candidates via hybrid retrieval.
            retrieval_start = time.perf_counter()
            try:
                candidates = self.hybrid_retriever.retrieve(
                    question,
                    top_n=effective_top_n,
                    filters=filters,
                )
            except Exception as exc:  # noqa: BLE001 - treat as empty result
                logger.error("query.retrieve.failed", error=str(exc))
                candidates = []
            retrieval_latency = time.perf_counter() - retrieval_start
            record_retrieval_latency(retrieval_latency)

            # 3. No-result path: empty candidates.
            if not candidates:
                record_no_result()
                record_query("no_result")
                logger.info("query.no_result", reason="empty_candidates")
                return QueryResult(
                    answer=self.guardrail.build_no_result_answer(),
                    references=[],
                    trace_id=trace_id,
                    no_result=True,
                    retrieval_latency=retrieval_latency,
                    generation_latency=0.0,
                )

            # 4. Rerank.
            try:
                reranked = self.reranker.rerank(
                    question,
                    candidates,
                    top_k=rerank_top_k,
                )
            except Exception as exc:  # noqa: BLE001 - fall back to raw candidates
                logger.error("query.rerank.failed", error=str(exc))
                reranked = candidates[:rerank_top_k]

            top_score = reranked[0].score if reranked else 0.0

            # 5. Guardrail: reject low-confidence results.
            #    check_confidence() already calls record_no_result() on reject,
            #    so we must not call it again here to avoid double counting.
            if not self.guardrail.check_confidence(top_score, threshold=threshold):
                record_query("no_result")
                logger.info(
                    "query.no_result",
                    reason="low_confidence",
                    top_score=top_score,
                )
                return QueryResult(
                    answer=self.guardrail.build_no_result_answer(),
                    references=[],
                    trace_id=trace_id,
                    no_result=True,
                    retrieval_latency=retrieval_latency,
                    generation_latency=0.0,
                )

            # 6. Generate answer.
            generation_start = time.perf_counter()
            try:
                result = self.generator.generate(question, reranked)
            except Exception as exc:  # noqa: BLE001
                logger.error("query.generate.failed", error=str(exc))
                generation_latency = time.perf_counter() - generation_start
                record_generation_latency(generation_latency)
                record_query("no_result")
                return QueryResult(
                    answer=self.guardrail.build_no_result_answer(),
                    references=[],
                    trace_id=trace_id,
                    no_result=True,
                    retrieval_latency=retrieval_latency,
                    generation_latency=generation_latency,
                )
            generation_latency = time.perf_counter() - generation_start
            record_generation_latency(generation_latency)

            # 7. Parse citations and map back to source chunks.
            citations = self.citation_parser.parse(result.answer)
            referenced_chunks = self.citation_parser.map_to_references(
                citations,
                reranked,
            )
            references = [
                Reference(
                    chunk_id=chunk.id,
                    source=chunk.metadata.source,
                    page=chunk.metadata.page,
                    score=chunk.score,
                    snippet=chunk.snippet(),
                )
                for chunk in referenced_chunks
            ]

            record_query("ok")
            logger.info(
                "query.done",
                answer_len=len(result.answer),
                references=len(references),
                retrieval_ms=int(retrieval_latency * 1000),
                generation_ms=int(generation_latency * 1000),
            )

            return QueryResult(
                answer=result.answer,
                references=references,
                trace_id=trace_id,
                no_result=False,
                retrieval_latency=retrieval_latency,
                generation_latency=generation_latency,
            )
