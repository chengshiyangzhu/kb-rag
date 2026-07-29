"""``GET /health`` endpoint for liveness and readiness probing."""
from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from app.config import get_settings
from app.observability.logging import get_logger
from app.stores import VectorStore
from backend import get_vector_store_dep
from backend.schemas import HealthResponse

logger = get_logger(__name__)

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    tags=["health"],
    summary="Liveness and readiness probe",
)
def health(
    store: Annotated[VectorStore, Depends(get_vector_store_dep)],
) -> HealthResponse:
    """Check API liveness and vector store connectivity.

    Calls ``vector_store.count()`` to verify the store is reachable.  Returns
    HTTP 503 when the store is unavailable.
    """
    settings = get_settings()
    try:
        chunks = store.count()
    except Exception as exc:  # noqa: BLE001 - surface as 503
        logger.error("health.check.failed", error=str(exc))
        raise HTTPException(
            status_code=503,
            detail="vector store unavailable",
        )
    return HealthResponse(
        status="ok",
        vector_store=settings.vector_store,
        chunks=chunks,
        version="0.1.0",
    )
