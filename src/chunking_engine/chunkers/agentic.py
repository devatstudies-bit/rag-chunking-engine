"""Agentic chunker — LLM decides optimal chunk boundaries with reasoning."""

from __future__ import annotations

from typing import Any

import structlog
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential

from .base import BaseChunker, ChunkingConfig, ChunkingStrategy

logger = structlog.get_logger(__name__)

_SYSTEM_PROMPT = """You are an expert document-chunking analyst.

Your task is to identify the optimal semantic chunk boundaries in the document
provided by the user.

Rules:
1. Each chunk must be semantically self-contained and meaningful in isolation.
2. Never split mid-sentence, mid-list, or mid-code-block.
3. Identify the most specific chunk_type for each section:
   "section", "table", "code_block", "list", "introduction", "conclusion", or "general".
4. Return ONLY the JSON structure requested — no commentary.
"""

_HUMAN_TEMPLATE = """Analyse the following document and return an optimal chunking plan.

Document (length: {char_count} characters):
---
{document_text}
---

Return a JSON object with this exact structure:
{{
  "chunks": [
    {{
      "start_index": <int>,
      "end_index": <int>,
      "section_name": "<descriptive name>",
      "chunk_type": "<section|table|code_block|list|introduction|conclusion|general>"
    }}
  ]
}}
"""


class _ChunkItem(BaseModel):
    start_index: int
    end_index: int
    section_name: str
    chunk_type: str = "general"


class _ChunkProposal(BaseModel):
    chunks: list[_ChunkItem] = Field(default_factory=list)


class AgenticChunker(BaseChunker):
    """Uses an LLM to determine semantically optimal chunk boundaries.

    The full document is submitted to the LLM in a single call.  The model
    returns start/end character offsets and a section name for each proposed
    chunk.  This handles heterogeneous documents (tables embedded in prose,
    code snippets inside reports) that no single rule-based strategy covers.

    Cost: one LLM call per document.  Use for high-value, complex documents.
    For large corpora, use agentic output as training data for a cheaper model.
    """

    strategy = ChunkingStrategy.AGENTIC

    def __init__(
        self,
        llm: BaseChatModel,
        config: ChunkingConfig | None = None,
    ) -> None:
        super().__init__(config)
        self._llm = llm

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def chunk(self, text: str, metadata: dict[str, Any] | None = None) -> list[Document]:
        proposal = self._propose_chunks(text)
        chunks: list[Document] = []

        for item in proposal.chunks:
            start = max(0, item.start_index)
            end = min(len(text), item.end_index)
            if end <= start:
                continue
            content = text[start:end].strip()
            if not content:
                continue

            meta = self._base_metadata({
                "doc_type": "heterogeneous",
                "section": item.section_name,
                "chunk_type": item.chunk_type,
                "chunk_id": len(chunks),
                "char_start": start,
                "char_end": end,
                **(metadata or {}),
            })
            chunks.append(Document(page_content=content, metadata=meta))

        logger.info("agentic_chunking_complete", total_chunks=len(chunks))
        return chunks

    def _propose_chunks(self, text: str) -> _ChunkProposal:
        structured_llm = self._llm.with_structured_output(_ChunkProposal)
        human_msg = _HUMAN_TEMPLATE.format(
            char_count=len(text),
            document_text=text[:8000],  # guard against context overflow
        )
        result = structured_llm.invoke([
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=human_msg),
        ])
        return result  # type: ignore[return-value]
