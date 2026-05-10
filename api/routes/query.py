from __future__ import annotations

import structlog
from fastapi import APIRouter, HTTPException, Request

from api.schemas import QueryRequest, QueryResponse

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/api/v1", tags=["retrieval"])


@router.post("/query", response_model=QueryResponse, summary="Query the RAG pipeline")
async def query_documents(request: Request, body: QueryRequest) -> QueryResponse:
    graph = request.app.state.retrieval_graph
    if graph is None:
        raise HTTPException(status_code=503, detail="Retrieval graph not initialised")

    try:
        result = graph.run(
            query=body.query,
            top_k=body.top_k,
            filter_expr=body.filter_expr,
            doc_type_filter=body.doc_type_filter,
        )
    except Exception as exc:
        logger.error("query_error", query=body.query, error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return QueryResponse(
        answer=result.get("answer", ""),
        sources=result.get("sources", []),
        status=result.get("status", "unknown"),
    )
