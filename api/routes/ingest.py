from __future__ import annotations

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request

from api.schemas import IngestRequest, IngestResponse, StrategyInfo
from chunking_engine.chunkers import ChunkingConfig
from chunking_engine.registry import StrategyRegistry

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/api/v1", tags=["ingestion"])


@router.post("/ingest", response_model=IngestResponse, summary="Ingest a document into the vector store")
async def ingest_document(request: Request, body: IngestRequest) -> IngestResponse:
    graph = request.app.state.ingestion_graph
    if graph is None:
        raise HTTPException(status_code=503, detail="Ingestion graph not initialised")

    # Build chunking config from request overrides
    config = ChunkingConfig(
        section_patterns=body.section_patterns or [],
        language=body.language or "generic",
    )
    settings = request.app.state.settings
    config.chunk_size = settings.default_chunk_size
    config.chunk_overlap = settings.default_chunk_overlap
    config.semantic_threshold = settings.semantic_breakpoint_threshold

    metadata: dict = {**body.metadata}
    if body.doc_type:
        metadata["doc_type"] = body.doc_type

    try:
        result = graph.run(
            document_text=body.content,
            document_id=body.document_id,
            metadata=metadata,
        )
    except Exception as exc:
        logger.error("ingest_error", doc_id=body.document_id, error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return IngestResponse(
        document_id=body.document_id,
        strategy_used=result.get("strategy", "unknown"),
        doc_type_detected=result.get("document_type", "unknown"),
        chunks_indexed=result.get("indexed_count", 0),
        status=result.get("status", "unknown"),
    )


@router.get("/strategies", response_model=list[StrategyInfo], summary="List all available strategies")
async def list_strategies() -> list[StrategyInfo]:
    return [StrategyInfo(**s) for s in StrategyRegistry.list_strategies()]
