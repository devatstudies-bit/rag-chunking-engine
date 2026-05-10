"""LangGraph ingestion pipeline: classify → select strategy → chunk → embed → index."""

from __future__ import annotations

from typing import Any

import structlog
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, StateGraph

from chunking_engine.chunkers import (
    AgenticChunker,
    ChunkingConfig,
    ChunkingStrategy,
    DocumentType,
)
from chunking_engine.pipeline.state import IngestionState
from chunking_engine.registry import StrategyRegistry
from chunking_engine.vectorstore import DocumentIndexer, MilvusClientWrapper

logger = structlog.get_logger(__name__)


# ── Node implementations ──────────────────────────────────────────────────────

def _classify_node(state: IngestionState) -> IngestionState:
    """Detect the document type from content signals."""
    text = state.get("document_text", "")
    meta = state.get("raw_metadata", {})
    doc_type = meta.get("doc_type") or StrategyRegistry.detect_document_type(text, meta)
    logger.info("ingestion_classified", doc_type=doc_type, doc_id=state.get("document_id"))
    return {**state, "document_type": doc_type, "errors": state.get("errors", [])}


def _select_strategy_node(state: IngestionState) -> IngestionState:
    """Map detected document type to the optimal chunking strategy."""
    doc_type = DocumentType(state.get("document_type", DocumentType.UNKNOWN))
    strategy = StrategyRegistry.select_strategy(doc_type)
    logger.info("ingestion_strategy_selected", strategy=strategy, doc_type=doc_type)
    return {**state, "strategy": strategy.value}


def _make_chunk_node(
    embeddings: Embeddings,
    llm: BaseChatModel | None,
    config: ChunkingConfig,
) -> Any:
    def _chunk_node(state: IngestionState) -> IngestionState:
        text = state["document_text"]
        doc_id = state.get("document_id", "unknown")
        strategy = ChunkingStrategy(state["strategy"])
        meta = {
            "source": doc_id,
            **(state.get("raw_metadata") or {}),
        }

        try:
            chunker = StrategyRegistry.build_chunker(
                strategy=strategy,
                config=config,
                embeddings=embeddings,
                llm=llm,
            )
            chunks = chunker.chunk(text, meta)
        except Exception as exc:
            logger.error("ingestion_chunk_error", error=str(exc))
            return {**state, "chunks": [], "errors": [*state.get("errors", []), str(exc)]}

        logger.info("ingestion_chunked", strategy=strategy, total=len(chunks))
        return {**state, "chunks": chunks}

    return _chunk_node


def _make_index_node(indexer: DocumentIndexer) -> Any:
    def _index_node(state: IngestionState) -> IngestionState:
        chunks: list[Document] = state.get("chunks", [])
        if not chunks:
            return {**state, "indexed_count": 0, "status": "partial"}
        try:
            source = state.get("document_id", "unknown")
            count = indexer.index_with_source_refresh(chunks, source)
            logger.info("ingestion_indexed", count=count, source=source)
            return {**state, "indexed_count": count, "status": "success"}
        except Exception as exc:
            logger.error("ingestion_index_error", error=str(exc))
            return {
                **state,
                "indexed_count": 0,
                "status": "failed",
                "errors": [*state.get("errors", []), str(exc)],
            }

    return _index_node


def _error_node(state: IngestionState) -> IngestionState:
    logger.error("ingestion_failed", errors=state.get("errors", []))
    return {**state, "status": "failed"}


def _route_after_chunking(state: IngestionState) -> str:
    return "error" if state.get("errors") else "index"


# ── Graph factory ─────────────────────────────────────────────────────────────

class IngestionGraph:
    """Compiled LangGraph for end-to-end document ingestion."""

    def __init__(self, graph: Any) -> None:
        self._graph = graph

    def run(self, document_text: str, document_id: str, metadata: dict[str, Any] | None = None) -> IngestionState:
        initial: IngestionState = {
            "document_text": document_text,
            "document_id": document_id,
            "raw_metadata": metadata or {},
            "errors": [],
        }
        return self._graph.invoke(initial)


def build_ingestion_graph(
    embeddings: Embeddings,
    milvus: MilvusClientWrapper,
    llm: BaseChatModel | None = None,
    config: ChunkingConfig | None = None,
) -> IngestionGraph:
    """Wire and compile the ingestion StateGraph."""
    cfg = config or ChunkingConfig()
    indexer = DocumentIndexer(embeddings=embeddings, milvus=milvus)

    workflow = StateGraph(IngestionState)
    workflow.add_node("classify", _classify_node)
    workflow.add_node("select_strategy", _select_strategy_node)
    workflow.add_node("chunk", _make_chunk_node(embeddings, llm, cfg))
    workflow.add_node("index", _make_index_node(indexer))
    workflow.add_node("error", _error_node)

    workflow.set_entry_point("classify")
    workflow.add_edge("classify", "select_strategy")
    workflow.add_edge("select_strategy", "chunk")
    workflow.add_conditional_edges("chunk", _route_after_chunking, {"index": "index", "error": "error"})
    workflow.add_edge("index", END)
    workflow.add_edge("error", END)

    compiled = workflow.compile()
    return IngestionGraph(compiled)
