"""``POST /api/v1/query`` endpoint for running RAG queries."""
from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from app.pipeline import QueryPipeline
from backend import get_query_pipeline_dep
from backend.schemas import QueryRequest, QueryResponse, ReferenceOut

router = APIRouter()


@router.post(
    "",
    response_model=QueryResponse,
    summary="Query the RAG knowledge base",
)
def query(
    request: QueryRequest,
    pipeline: Annotated[QueryPipeline, Depends(get_query_pipeline_dep)],
) -> QueryResponse:
    """Run a RAG query end-to-end.

    Retrieves relevant chunks via hybrid retrieval, reranks them, checks the
    guardrail, and generates a natural-language answer with citation
    references.
    """
    filters_dict: dict | None = None
    if request.filters is not None:
        filters_dict = {
            k: v
            for k, v in request.filters.model_dump().items()
            if v is not None
        }
        if not filters_dict:
            filters_dict = None

    result = pipeline.query(
        question=request.question,
        filters=filters_dict,
        top_n=request.top_n,
    )

    return QueryResponse(
        answer=result.answer,
        references=[
            ReferenceOut(
                chunk_id=ref.chunk_id,
                source=ref.source,
                page=ref.page,
                score=ref.score,
                snippet=ref.snippet,
            )
            for ref in result.references
        ],
        trace_id=result.trace_id,
        no_result=result.no_result,
        retrieval_latency=result.retrieval_latency,
        generation_latency=result.generation_latency,
    )
