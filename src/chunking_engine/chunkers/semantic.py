"""Semantic chunker — detects topic shifts via embedding similarity."""

from __future__ import annotations

from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from .base import BaseChunker, ChunkingConfig, ChunkingStrategy


class SemanticChunker(BaseChunker):
    """Splits text at topic-shift boundaries detected by cosine-similarity drops.

    Each consecutive sentence pair is compared in embedding space.
    When similarity drops below the configured percentile threshold a new chunk
    begins.  No fixed token budget — chunk length is determined entirely by
    topic coherence.

    Ideal for: transcripts, interviews, email threads, any conversational text
    without explicit section headers.
    """

    strategy = ChunkingStrategy.SEMANTIC

    def __init__(
        self,
        embeddings: Embeddings,
        config: ChunkingConfig | None = None,
    ) -> None:
        super().__init__(config)
        self._embeddings = embeddings

    def chunk(self, text: str, metadata: dict[str, Any] | None = None) -> list[Document]:
        from langchain_experimental.text_splitter import SemanticChunker as _LC_Semantic

        chunker = _LC_Semantic(
            embeddings=self._embeddings,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=self.config.semantic_threshold,
        )
        meta = self._base_metadata({"doc_type": "transcript", **(metadata or {})})
        docs = chunker.create_documents([text], metadatas=[meta])

        total = len(docs)
        for i, doc in enumerate(docs):
            doc.metadata["chunk_id"] = i
            doc.metadata["position"] = f"{i + 1} of {total}"

        return docs
