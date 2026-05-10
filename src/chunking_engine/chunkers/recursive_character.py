"""Recursive-character chunker — respects paragraph and sentence boundaries."""

from __future__ import annotations

from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from .base import BaseChunker, ChunkingConfig, ChunkingStrategy

# Separator hierarchy: widest natural boundary tried first.
DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", " ", ""]


class RecursiveCharacterChunker(BaseChunker):
    """Hierarchical text splitter that preserves sentence and paragraph boundaries.

    Ideal for: general prose, mixed-content documentation, any text where
    sections are separated by blank lines but no machine-readable structure exists.
    """

    strategy = ChunkingStrategy.RECURSIVE_CHARACTER

    def chunk(self, text: str, metadata: dict[str, Any] | None = None) -> list[Document]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=DEFAULT_SEPARATORS,
        )
        meta = self._base_metadata({"doc_type": "general_text", **(metadata or {})})
        docs = splitter.create_documents([text], metadatas=[meta])
        for i, doc in enumerate(docs):
            doc.metadata["chunk_id"] = i
        return docs
