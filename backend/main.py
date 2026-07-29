"""FastAPI application entry point for kb-rag (Stage 9).

Wires the ingest/query pipelines and vector store into a REST API exposed
under ``/api/v1`` plus ``/health`` and ``/metrics`` at the root level.
"""
from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator

from app.config import get_settings
from app.observability.logging import configure_logging, get_logger
from app.observability.tracing import init_tracer

# Dependency accessors live in the backend package (__init__.py) to avoid
# circular imports between backend.main and the route modules.  Re-exported
# here for backward compatibility (``backend.main.get_ingest_pipeline_dep``).
from backend import (  # noqa: F401
    get_ingest_pipeline_dep,
    get_query_pipeline_dep,
    get_vector_store_dep,
)

# Configure logging once on import so module-level log calls are structured.
_settings_initial = get_settings()
configure_logging(level=_settings_initial.log_level)
logger = get_logger(__name__)

# Import routers.  Safe to do at this point: route modules pull the dependency
# accessors from the already-initialized ``backend`` package.
from backend.api.v1 import router as v1_router  # noqa: E402
from backend.api.v1.health import router as health_router  # noqa: E402


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan hook.

    On startup: configure logging, initialize the OpenTelemetry tracer, and
    log the active environment.  Container initialization is lazy (the
    pipelines are built on first request), so no eager warm-up is performed.
    """
    s = get_settings()
    configure_logging(level=s.log_level)
    init_tracer(service_name="kb-rag-api")
    logger.info("kb-rag API starting", env=s.app_env)
    yield
    logger.info("kb-rag API shutting down")


def create_app() -> FastAPI:
    """Build and configure the FastAPI application."""
    s = get_settings()
    app = FastAPI(
        title="kb-rag API",
        version="0.1.0",
        description="Enterprise RAG knowledge base API.",
        lifespan=lifespan,
    )

    # CORS: allow the Streamlit UI origin (default localhost:8501); in dev mode
    # also allow "*" so a browser running on any origin can call the API.
    origins: list[str] = ["http://localhost:8501"]
    if s.app_env == "dev":
        origins.append("*")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Prometheus metrics: automatically exposes GET /metrics.
    Instrumentator(
        should_group_status_codes=True,
        should_ignore_untemplated=True,
        should_respect_env_var=False,
    ).instrument(app).expose(app, endpoint="/metrics")

    # Mount the v1 router under /api/v1 and the health router at root.
    app.include_router(v1_router, prefix="/api/v1")
    app.include_router(health_router, tags=["health"])

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        """Catch-all handler returning a 500 with a fresh trace_id."""
        trace_id = uuid.uuid4().hex
        logger.error(
            "unhandled.exception",
            error=str(exc),
            trace_id=trace_id,
            path=request.url.path,
        )
        return JSONResponse(
            status_code=500,
            content={"detail": "internal server error", "trace_id": trace_id},
        )

    logger.info("kb-rag API initialized", env=s.app_env)
    return app


app = create_app()


def run() -> None:
    """Entry point for the ``kb-rag-api`` console script."""
    import uvicorn

    s = get_settings()
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=s.api_port,
        reload=False,
    )


if __name__ == "__main__":
    run()
