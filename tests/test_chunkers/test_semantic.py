"""Tests for SemanticChunker (uses a mock embeddings model)."""

from __future__ import annotations

import pytest
from langchain_core.embeddings import Embeddings

from chunking_engine.chunkers import ChunkingConfig, SemanticChunker
from chunking_engine.chunkers.base import ChunkingStrategy


class _MockEmbeddings(Embeddings):
    """Deterministic mock: each sentence gets a vector based on its word hash."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)

    @staticmethod
    def _vec(text: str) -> list[float]:
        import hashlib
        seed = int(hashlib.md5(text.encode()).hexdigest(), 16) % (10**8)
        import random
        rng = random.Random(seed)
        return [rng.gauss(0, 1) for _ in range(8)]


@pytest.fixture
def mock_embeddings():
    return _MockEmbeddings()


def test_produces_chunks(transcript_text, mock_embeddings):
    config = ChunkingConfig(semantic_threshold=50.0)
    chunker = SemanticChunker(embeddings=mock_embeddings, config=config)
    chunks = chunker.chunk(transcript_text)
    assert len(chunks) >= 1


def test_position_metadata_present(transcript_text, mock_embeddings):
    config = ChunkingConfig(semantic_threshold=50.0)
    chunker = SemanticChunker(embeddings=mock_embeddings, config=config)
    chunks = chunker.chunk(transcript_text)
    for chunk in chunks:
        assert "position" in chunk.metadata
        assert "chunk_id" in chunk.metadata


def test_strategy_label(transcript_text, mock_embeddings):
    config = ChunkingConfig(semantic_threshold=50.0)
    chunker = SemanticChunker(embeddings=mock_embeddings, config=config)
    chunks = chunker.chunk(transcript_text)
    assert all(c.metadata["strategy"] == ChunkingStrategy.SEMANTIC.value for c in chunks)
