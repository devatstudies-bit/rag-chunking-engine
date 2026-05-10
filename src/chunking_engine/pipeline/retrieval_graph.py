"""LangGraph retrieval pipeline: embed query → search → rerank → generate answer."""

from __future__ import annotations

from typing import Any

import structlog
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from chunking_engine.pipeline.state import RetrievalState
from chunking_engine.vectorstore import MilvusClientWrapper

logger = structlog.get_logger(__name__)

_RAG_SYSTEM = """You are a precise question-answering assistant.

Answer the user's question using ONLY the provided context chunks.
If the context does not contain enough information, say so clearly.
Cite source document IDs when available."""


# ── Node implementations ──────────────────────────────────────────────────────

def _make_embed_query_node(embeddings: Embeddings) -> Any:
    def _node(state: RetrievalState) -> RetrievalState:
        query = state["query"]
        vec = embeddings.embed_query(query)
        logger.info("retrieval_query_embedded", query_len=len(query))
        return {**state, "query_embedding": vec, "errors": state.get("errors", [])}
    return _node


def _make_search_node(milvus: MilvusClientWrapper) -> Any:
    def _node(state: RetrievalState) -> RetrievalState:
        vec = state["query_embedding"]
        top_k = state.get("top_k", 5)
        filter_expr = state.get("filter_expr") or _build_filter(state)
        try:
            results = milvus.search(vec, top_k=top_k, filter_expr=filter_expr or None)
        except Exception as exc:
            logger.error("retrieval_search_error", error=str(exc))
            return {**state, "raw_results": [], "errors": [*state.get("errors", []), str(exc)]}
        logger.info("retrieval_results_found", count=len(results))
        return {**state, "raw_results": results}
    return _node


def _rerank_node(state: RetrievalState) -> RetrievalState:
    """Simple score-based re-rank (swap for a cross-encoder for production)."""
    results = sorted(
        state.get("raw_results", []),
        key=lambda r: float(r.get("score", 0.0)),
        reverse=True,
    )
    return {**state, "reranked_results": results}


def _make_generate_node(llm: BaseChatModel) -> Any:
    def _node(state: RetrievalState) -> RetrievalState:
        results = state.get("reranked_results", [])
        if not results:
            return {**state, "answer": "No relevant documents found.", "sources": [], "status": "no_results"}

        context = "\n\n---\n\n".join(
            f"[Source: {r['source']} | Section: {r.get('section', 'N/A')}]\n{r['content']}"
            for r in results
        )
        response = llm.invoke([
            SystemMessage(content=_RAG_SYSTEM),
            HumanMessage(content=f"Context:\n{context}\n\nQuestion: {state['query']}"),
        ])
        sources = [
            {"source": r["source"], "section": r.get("section", ""), "score": r.get("score")}
            for r in results
        ]
        logger.info("retrieval_answer_generated", sources=len(sources))
        return {**state, "answer": response.content, "sources": sources, "status": "success"}
    return _node


def _error_node(state: RetrievalState) -> RetrievalState:
    logger.error("retrieval_failed", errors=state.get("errors", []))
    return {**state, "answer": "Retrieval failed — see errors.", "status": "failed"}


def _route_after_search(state: RetrievalState) -> str:
    return "error" if state.get("errors") else "rerank"


def _build_filter(state: RetrievalState) -> str:
    if dt := state.get("doc_type_filter"):
        return f'doc_type == "{dt}"'
    return ""


# ── Graph factory ─────────────────────────────────────────────────────────────

class RetrievalGraph:
    """Compiled LangGraph for end-to-end RAG retrieval."""

    def __init__(self, graph: Any) -> None:
        self._graph = graph

    def run(
        self,
        query: str,
        top_k: int = 5,
        filter_expr: str | None = None,
        doc_type_filter: str | None = None,
    ) -> RetrievalState:
        initial: RetrievalState = {
            "query": query,
            "top_k": top_k,
            "filter_expr": filter_expr,
            "doc_type_filter": doc_type_filter,
            "errors": [],
        }
        return self._graph.invoke(initial)


def build_retrieval_graph(
    embeddings: Embeddings,
    milvus: MilvusClientWrapper,
    llm: BaseChatModel,
) -> RetrievalGraph:
    """Wire and compile the retrieval StateGraph."""
    workflow = StateGraph(RetrievalState)
    workflow.add_node("embed_query", _make_embed_query_node(embeddings))
    workflow.add_node("search", _make_search_node(milvus))
    workflow.add_node("rerank", _rerank_node)
    workflow.add_node("generate", _make_generate_node(llm))
    workflow.add_node("error", _error_node)

    workflow.set_entry_point("embed_query")
    workflow.add_edge("embed_query", "search")
    workflow.add_conditional_edges("search", _route_after_search, {"rerank": "rerank", "error": "error"})
    workflow.add_edge("rerank", "generate")
    workflow.add_edge("generate", END)
    workflow.add_edge("error", END)

    compiled = workflow.compile()
    return RetrievalGraph(compiled)
