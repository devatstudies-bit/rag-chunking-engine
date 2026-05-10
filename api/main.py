"""FastAPI application entry point."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

import structlog
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.routes import health_router, ingest_router, query_router
from chunking_engine import __version__
from chunking_engine.config import configure_logging, get_settings
from chunking_engine.models.base import ProviderFactory
from chunking_engine.pipeline import build_ingestion_graph, build_retrieval_graph
from chunking_engine.vectorstore import MilvusClientWrapper

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    settings = get_settings()
    configure_logging(settings.log_level)

    logger.info("startup_begin", provider=settings.llm_provider, version=__version__)

    provider = ProviderFactory.create(settings)
    embeddings = provider.get_embeddings()
    llm = provider.get_chat_model(temperature=0.0)

    milvus = MilvusClientWrapper(settings)
    milvus.ensure_collection()

    app.state.settings = settings
    app.state.ingestion_graph = build_ingestion_graph(embeddings=embeddings, milvus=milvus, llm=llm)
    app.state.retrieval_graph = build_retrieval_graph(embeddings=embeddings, milvus=milvus, llm=llm)

    logger.info("startup_complete")
    yield
    logger.info("shutdown")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="Chunking Engine API",
        description="Production-grade adaptive document chunking for RAG pipelines",
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── API key guard ──────────────────────────────────────────────────────────
    if settings.api_key:
        @app.middleware("http")
        async def _api_key_middleware(request: Request, call_next):
            if request.url.path in {"/health", "/docs", "/redoc", "/openapi.json"}:
                return await call_next(request)
            key = request.headers.get("X-API-Key", "")
            if key != settings.api_key:
                return JSONResponse(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    content={"detail": "Invalid or missing API key"},
                )
            return await call_next(request)

    app.include_router(health_router)
    app.include_router(ingest_router)
    app.include_router(query_router)

    return app


app = create_app()
