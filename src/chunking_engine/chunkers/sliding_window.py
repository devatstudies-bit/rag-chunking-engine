"""Sliding-window chunker — overlapping fixed-size chunks for dense technical docs."""

from __future__ import annotations

from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from .base import BaseChunker, ChunkingConfig, ChunkingStrategy

_SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", " ", ""]


class SlidingWindowChunker(BaseChunker):
    """Fixed-size chunks with configurable overlap between adjacent windows.

    Overlap ensures that concepts spanning a chunk boundary appear complete in
    at least one chunk.  A 20 % overlap (200 chars per 1000 char chunk) is the
    standard rule of thumb.

    Includes a Jaccard-similarity deduplication helper to remove near-duplicate
    chunks before sending them to the LLM.

    Ideal for: dense technical documentation, API references, manuals with
    heavy cross-references between sections.
    """

    strategy = ChunkingStrategy.SLIDING_WINDOW

    def chunk(self, text: str, metadata: dict[str, Any] | None = None) -> list[Document]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=_SEPARATORS,
        )
        meta = self._base_metadata({"doc_type": "technical_doc", **(metadata or {})})
        docs = splitter.create_documents([text], metadatas=[meta])

        for i, doc in enumerate(docs):
            doc.metadata["chunk_id"] = i
            doc.metadata["has_overlap"] = i > 0

        return docs

    @staticmethod
    def deduplicate(docs: list[Document], threshold: float = 0.85) -> list[Document]:
        """Remove near-duplicate overlapping chunks using Jaccard token similarity."""
        seen: list[Document] = []
        for doc in docs:
            tokens = set(doc.page_content.split())
            is_dup = any(
                len(tokens & set(s.page_content.split()))
                / max(len(tokens | set(s.page_content.split())), 1)
                > threshold
                for s in seen
            )
            if not is_dup:
                seen.append(doc)
        return seen
