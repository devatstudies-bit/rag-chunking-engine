"""Tests for CodeAwareChunker."""

import pytest

from chunking_engine.chunkers import ChunkingConfig, CodeAwareChunker
from chunking_engine.chunkers.base import ChunkingStrategy


def test_python_splits_on_function_boundaries(python_code):
    config = ChunkingConfig(chunk_size=500, chunk_overlap=0, language="python")
    chunker = CodeAwareChunker(config)
    chunks = chunker.chunk(python_code)
    # Should produce at least 2 chunks (two functions + one class)
    assert len(chunks) >= 2


def test_language_in_metadata(python_code):
    config = ChunkingConfig(chunk_size=500, chunk_overlap=0, language="python")
    chunker = CodeAwareChunker(config)
    chunks = chunker.chunk(python_code)
    assert all(c.metadata.get("language") == "python" for c in chunks)


def test_generic_language_fallback():
    config = ChunkingConfig(language="ruby")
    ruby_code = (
        "def greet(name)\n  puts 'Hello, #{name}'\nend\n\n"
        "def farewell(name)\n  puts 'Goodbye, #{name}'\nend\n"
    )
    chunker = CodeAwareChunker(config)
    chunks = chunker.chunk(ruby_code)
    assert len(chunks) >= 1


def test_strategy_label(python_code):
    config = ChunkingConfig(language="python")
    chunker = CodeAwareChunker(config)
    chunks = chunker.chunk(python_code)
    assert all(c.metadata["strategy"] == ChunkingStrategy.CODE_AWARE.value for c in chunks)


def test_doc_type_is_source_code(python_code):
    config = ChunkingConfig(language="python")
    chunker = CodeAwareChunker(config)
    chunks = chunker.chunk(python_code)
    assert all(c.metadata["doc_type"] == "source_code" for c in chunks)
