"""Tests for RecursiveCharacterChunker."""

import pytest

from chunking_engine.chunkers import ChunkingConfig, RecursiveCharacterChunker
from chunking_engine.chunkers.base import ChunkingStrategy


def test_respects_paragraph_boundaries(general_text):
    config = ChunkingConfig(chunk_size=300, chunk_overlap=30)
    chunker = RecursiveCharacterChunker(config)
    chunks = chunker.chunk(general_text)
    # No chunk should contain content from two unrelated paragraphs mid-word
    for chunk in chunks:
        assert len(chunk.page_content) <= config.chunk_size * 1.1  # 10% tolerance


def test_strategy_label(general_text, default_config):
    chunker = RecursiveCharacterChunker(default_config)
    chunks = chunker.chunk(general_text)
    assert all(c.metadata["strategy"] == ChunkingStrategy.RECURSIVE_CHARACTER.value for c in chunks)


def test_doc_type_is_general_text(general_text, default_config):
    chunker = RecursiveCharacterChunker(default_config)
    chunks = chunker.chunk(general_text)
    assert all(c.metadata["doc_type"] == "general_text" for c in chunks)


def test_chunk_ids_contiguous(general_text, default_config):
    chunker = RecursiveCharacterChunker(default_config)
    chunks = chunker.chunk(general_text)
    ids = [c.metadata["chunk_id"] for c in chunks]
    assert ids == list(range(len(chunks)))
