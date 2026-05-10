"""Typed state definitions for the LangGraph ingestion and retrieval graphs."""

from __future__ import annotations

from typing import Any

from langchain_core.documents import Document
from typing_extensions import TypedDict


class IngestionState(TypedDict, total=False):
    """Mutable state flowing through the ingestion LangGraph."""

    # Input
    document_text: str
    document_id: str
    raw_metadata: dict[str, Any]

    # Inferred
    document_type: str           # DocumentType enum value
    strategy: str                # ChunkingStrategy enum value

    # Intermediate
    chunks: list[Document]

    # Output
    indexed_count: int
    errors: list[str]
    status: str                  # "success" | "partial" | "failed"


class RetrievalState(TypedDict, total=False):
    """Mutable state flowing through the retrieval LangGraph."""

    # Input
    query: str
    top_k: int
    filter_expr: str | None
    doc_type_filter: str | None

    # Intermediate
    query_embedding: list[float]
    raw_results: list[dict[str, Any]]
    reranked_results: list[dict[str, Any]]

    # Output
    answer: str
    sources: list[dict[str, Any]]
    errors: list[str]
    status: str
