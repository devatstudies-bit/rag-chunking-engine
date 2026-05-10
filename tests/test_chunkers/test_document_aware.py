"""Tests for DocumentAwareChunker."""

import pytest

from chunking_engine.chunkers import ChunkingConfig, DocumentAwareChunker
from chunking_engine.chunkers.base import ChunkingStrategy


def test_sections_become_separate_chunks(structured_document):
    chunker = DocumentAwareChunker()
    chunks = chunker.chunk(structured_document)
    sections = [c.metadata["section"] for c in chunks]
    assert "Overview" in sections
    assert "Findings" in sections
    assert "Recommendations" in sections


def test_header_prepended_to_each_chunk(structured_document):
    chunker = DocumentAwareChunker()
    chunks = chunker.chunk(structured_document, metadata={"source": "INC-0042"})
    # Every chunk must contain the doc header (INC number)
    for chunk in chunks:
        assert "INC-2024-0042" in chunk.page_content


def test_section_in_metadata(structured_document):
    chunker = DocumentAwareChunker()
    chunks = chunker.chunk(structured_document)
    for chunk in chunks:
        assert "section" in chunk.metadata
        assert chunk.metadata["section"]


def test_fallback_for_unstructured_text(default_config):
    chunker = DocumentAwareChunker(default_config)
    plain = "This document has no recognised section headers."
    chunks = chunker.chunk(plain)
    assert len(chunks) == 1
    assert chunks[0].metadata["section"] == "full_document"


def test_custom_section_patterns():
    config = ChunkingConfig(section_patterns=["Diagnosis", "Treatment", "Prognosis"])
    text = (
        "Patient Report — P-2024-001\n\n"
        "Diagnosis:\nType 2 diabetes with mild neuropathy.\n\n"
        "Treatment:\nMetformin 500mg twice daily.\n\n"
        "Prognosis:\nGood with dietary compliance."
    )
    chunker = DocumentAwareChunker(config)
    chunks = chunker.chunk(text)
    sections = {c.metadata["section"] for c in chunks}
    assert "Diagnosis" in sections
    assert "Treatment" in sections
    assert "Prognosis" in sections


def test_strategy_label(structured_document):
    chunker = DocumentAwareChunker()
    chunks = chunker.chunk(structured_document)
    assert all(c.metadata["strategy"] == ChunkingStrategy.DOCUMENT_AWARE.value for c in chunks)
