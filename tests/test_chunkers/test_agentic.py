"""Tests for AgenticChunker (uses a mock LLM)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage

from chunking_engine.chunkers import AgenticChunker, ChunkingConfig
from chunking_engine.chunkers.base import ChunkingStrategy


def _make_mock_llm(proposal_json: str) -> MagicMock:
    """Return an LLM mock whose structured output returns *proposal_json*."""
    from chunking_engine.chunkers.agentic import _ChunkProposal
    import json

    data = json.loads(proposal_json)
    proposal = _ChunkProposal(**data)

    structured_llm = MagicMock()
    structured_llm.invoke.return_value = proposal

    llm = MagicMock()
    llm.with_structured_output.return_value = structured_llm
    return llm


PROPOSAL = """{
  "chunks": [
    {"start_index": 0, "end_index": 100, "section_name": "Introduction", "chunk_type": "section"},
    {"start_index": 100, "end_index": 200, "section_name": "Details", "chunk_type": "general"}
  ]
}"""


def test_uses_llm_proposed_boundaries():
    text = "A" * 200
    llm = _make_mock_llm(PROPOSAL)
    chunker = AgenticChunker(llm=llm)
    chunks = chunker.chunk(text)
    assert len(chunks) == 2


def test_section_names_in_metadata():
    text = "A" * 200
    llm = _make_mock_llm(PROPOSAL)
    chunker = AgenticChunker(llm=llm)
    chunks = chunker.chunk(text)
    sections = {c.metadata["section"] for c in chunks}
    assert "Introduction" in sections
    assert "Details" in sections


def test_strategy_label():
    text = "A" * 200
    llm = _make_mock_llm(PROPOSAL)
    chunker = AgenticChunker(llm=llm)
    chunks = chunker.chunk(text)
    assert all(c.metadata["strategy"] == ChunkingStrategy.AGENTIC.value for c in chunks)


def test_empty_proposal_returns_empty():
    text = "Some text"
    llm = _make_mock_llm('{"chunks": []}')
    chunker = AgenticChunker(llm=llm)
    chunks = chunker.chunk(text)
    assert chunks == []
