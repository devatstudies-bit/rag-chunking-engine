from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    document_id: str = Field(..., description="Unique identifier for this document")
    content: str = Field(..., min_length=1, description="Raw document text")
    doc_type: str | None = Field(
        default=None,
        description="Override auto-detected type: structured_document | general_text | transcript | source_code | tabular_data | technical_doc | heterogeneous",
    )
    language: str | None = Field(
        default=None,
        description="Programming language hint for source_code documents (python, java, go, …)",
    )
    section_patterns: list[str] | None = Field(
        default=None,
        description="Custom section header names for document_aware strategy",
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="Arbitrary metadata stored alongside chunks")


class IngestResponse(BaseModel):
    document_id: str
    strategy_used: str
    doc_type_detected: str
    chunks_indexed: int
    status: str


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Natural-language question")
    top_k: int = Field(default=5, ge=1, le=50)
    doc_type_filter: str | None = Field(default=None, description="Restrict search to a specific doc_type")
    filter_expr: str | None = Field(default=None, description="Raw Milvus boolean filter expression")


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict[str, Any]]
    status: str


class HealthResponse(BaseModel):
    status: str
    provider: str
    milvus: str
    version: str


class StrategyInfo(BaseModel):
    strategy: str
    document_type: str
    description: str
