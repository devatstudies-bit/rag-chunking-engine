"""Fixed-size chunker — naive baseline, never use in production."""

from __future__ import annotations

from typing import Any

from langchain_text_splitters import CharacterTextSplitter
from langchain_core.documents import Document

from .base import BaseChunker, ChunkingConfig, ChunkingStrategy


class FixedSizeChunker(BaseChunker):
    """Splits text every N characters with no structural awareness.

    ONLY for benchmarking against better strategies.
    Destroys sentence and paragraph coherence.
    """

    strategy = ChunkingStrategy.FIXED_SIZE

    def chunk(self, text: str, metadata: dict[str, Any] | None = None) -> list[Document]:
        splitter = CharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separator="",
        )
        meta = self._base_metadata({"doc_type": "unknown", **(metadata or {})})
        docs = splitter.create_documents([text], metadatas=[meta])
        for i, doc in enumerate(docs):
            doc.metadata["chunk_id"] = i
        return docs
