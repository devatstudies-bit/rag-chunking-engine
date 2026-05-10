"""Integration tests for the LangGraph ingestion pipeline (mock dependencies)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from chunking_engine.chunkers import ChunkingConfig
from chunking_engine.pipeline import build_ingestion_graph
from chunking_engine.vectorstore import MilvusClientWrapper


class _ConstantEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [[0.1] * 8 for _ in texts]

    def embed_query(self, text):
        return [0.1] * 8


@pytest.fixture
def mock_milvus():
    m = MagicMock(spec=MilvusClientWrapper)
    m.ensure_collection.return_value = None
    m.insert.return_value = [1, 2, 3]
    m.delete_by_source.return_value = 0
    return m


@pytest.fixture
def ingestion_graph(mock_milvus):
    embeddings = _ConstantEmbeddings()
    return build_ingestion_graph(
        embeddings=embeddings,
        milvus=mock_milvus,
        config=ChunkingConfig(chunk_size=200, chunk_overlap=20),
    )


def test_ingestion_general_text_succeeds(ingestion_graph, general_text):
    result = ingestion_graph.run(
        document_text=general_text,
        document_id="doc-001",
    )
    assert result["status"] in {"success", "partial"}
    assert result.get("indexed_count", 0) > 0


def test_ingestion_csv_detects_tabular(ingestion_graph, csv_data):
    result = ingestion_graph.run(
        document_text=csv_data,
        document_id="doc-csv-001",
    )
    assert result["document_type"] == "tabular_data"


def test_ingestion_code_detects_source(ingestion_graph, python_code):
    result = ingestion_graph.run(
        document_text=python_code,
        document_id="doc-code-001",
        metadata={"file_extension": ".py"},
    )
    assert result["document_type"] == "source_code"


def test_explicit_doc_type_override(ingestion_graph, general_text):
    result = ingestion_graph.run(
        document_text=general_text,
        document_id="doc-override",
        metadata={"doc_type": "structured_document"},
    )
    assert result["document_type"] == "structured_document"
    assert result["strategy"] == "document_aware"


def test_strategy_registry_test():
    from chunking_engine.registry import StrategyRegistry
    from chunking_engine.chunkers.base import DocumentType

    strategy = StrategyRegistry.select_strategy(DocumentType.TRANSCRIPT)
    assert strategy.value == "semantic"

    strategy = StrategyRegistry.select_strategy(DocumentType.TABULAR_DATA)
    assert strategy.value == "row_aware"

    strategy = StrategyRegistry.select_strategy(DocumentType.HETEROGENEOUS)
    assert strategy.value == "agentic"
