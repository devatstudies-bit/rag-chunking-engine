from __future__ import annotations

from fastapi import APIRouter, Depends

from api.schemas import HealthResponse
from chunking_engine import __version__
from chunking_engine.config import get_settings

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse, summary="Liveness + readiness probe")
async def health_check() -> HealthResponse:
    settings = get_settings()
    milvus_ok = _ping_milvus(settings.milvus_host, settings.milvus_port)
    return HealthResponse(
        status="ok" if milvus_ok else "degraded",
        provider=settings.llm_provider,
        milvus="connected" if milvus_ok else "unreachable",
        version=__version__,
    )


def _ping_milvus(host: str, port: int) -> bool:
    try:
        import socket
        with socket.create_connection((host, port), timeout=1):
            return True
    except OSError:
        return False
