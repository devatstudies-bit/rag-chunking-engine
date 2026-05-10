"""Tests for FixedSizeChunker (baseline strategy)."""

import pytest

from chunking_engine.chunkers import ChunkingConfig, FixedSizeChunker
from chunking_engine.chunkers.base import ChunkingStrategy


def test_chunks_are_produced(general_text, default_config):
    chunker = FixedSizeChunker(default_config)
    chunks = chunker.chunk(general_text)
    assert len(chunks) > 0


def test_strategy_label(default_config):
    chunker = FixedSizeChunker(default_config)
    chunks = chunker.chunk("Hello world. " * 50)
    assert all(c.metadata["strategy"] == ChunkingStrategy.FIXED_SIZE.value for c in chunks)


def test_chunk_ids_are_sequential(general_text, default_config):
    chunker = FixedSizeChunker(default_config)
    chunks = chunker.chunk(general_text)
    ids = [c.metadata["chunk_id"] for c in chunks]
    assert ids == list(range(len(chunks)))


def test_metadata_propagated(general_text, default_config):
    chunker = FixedSizeChunker(default_config)
    chunks = chunker.chunk(general_text, metadata={"source": "test_doc", "author": "pytest"})
    assert all(c.metadata["source"] == "test_doc" for c in chunks)
    assert all(c.metadata["author"] == "pytest" for c in chunks)


def test_empty_text_returns_empty(default_config):
    chunker = FixedSizeChunker(default_config)
    assert chunker.chunk("") == []
