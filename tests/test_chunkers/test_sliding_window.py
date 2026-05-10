"""Tests for SlidingWindowChunker."""

import pytest

from chunking_engine.chunkers import ChunkingConfig, SlidingWindowChunker
from chunking_engine.chunkers.base import ChunkingStrategy


def test_produces_multiple_chunks(general_text):
    config = ChunkingConfig(chunk_size=200, chunk_overlap=50)
    chunker = SlidingWindowChunker(config)
    chunks = chunker.chunk(general_text)
    assert len(chunks) >= 2


def test_overlap_flag_on_non_first_chunks(general_text):
    config = ChunkingConfig(chunk_size=200, chunk_overlap=50)
    chunker = SlidingWindowChunker(config)
    chunks = chunker.chunk(general_text)
    if len(chunks) > 1:
        # First chunk has no overlap; subsequent ones do
        assert chunks[0].metadata["has_overlap"] is False
        assert all(c.metadata["has_overlap"] is True for c in chunks[1:])


def test_deduplication_removes_near_duplicates():
    from langchain_core.documents import Document
    docs = [
        Document(page_content="The quick brown fox jumps over the lazy dog"),
        Document(page_content="The quick brown fox jumps over the lazy dog and runs away"),
        Document(page_content="A completely different piece of content about space exploration"),
    ]
    deduped = SlidingWindowChunker.deduplicate(docs, threshold=0.7)
    assert len(deduped) < len(docs)
    # The very different doc must survive
    assert any("space exploration" in d.page_content for d in deduped)


def test_strategy_label(general_text, default_config):
    chunker = SlidingWindowChunker(default_config)
    chunks = chunker.chunk(general_text)
    assert all(c.metadata["strategy"] == ChunkingStrategy.SLIDING_WINDOW.value for c in chunks)
