"""Tests for RowAwareChunker."""

import pytest

from chunking_engine.chunkers import ChunkingConfig, RowAwareChunker
from chunking_engine.chunkers.base import ChunkingStrategy


def test_one_chunk_per_row(csv_data):
    chunker = RowAwareChunker()
    chunks = chunker.chunk(csv_data)
    # 4 data rows in fixture
    assert len(chunks) == 4


def test_headers_in_every_chunk(csv_data):
    chunker = RowAwareChunker()
    chunks = chunker.chunk(csv_data)
    for chunk in chunks:
        # Each chunk should contain labelled key-value pairs
        assert ":" in chunk.page_content


def test_column_values_in_metadata(csv_data):
    chunker = RowAwareChunker()
    chunks = chunker.chunk(csv_data)
    # 'Priority' column → lowercased 'priority' key in metadata
    assert all("priority" in c.metadata for c in chunks)


def test_empty_csv_raises(default_config):
    chunker = RowAwareChunker(default_config)
    with pytest.raises(ValueError, match="no header row"):
        chunker.chunk("")


def test_strategy_label(csv_data):
    chunker = RowAwareChunker()
    chunks = chunker.chunk(csv_data)
    assert all(c.metadata["strategy"] == ChunkingStrategy.ROW_AWARE.value for c in chunks)
