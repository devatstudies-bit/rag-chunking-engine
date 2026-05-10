"""DocumentIndexer — embeds chunked documents and writes them to Milvus."""

from __future__ import annotations

import json
from typing import Any

import structlog
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from chunking_engine.vectorstore.milvus_client import MilvusClientWrapper

logger = structlog.get_logger(__name__)

_BATCH_SIZE = 128


class DocumentIndexer:
    """Converts LangChain Documents → embeddings → Milvus records.

    Pipeline:
      1. Batch the incoming Documents.
      2. Call the embedding model on each batch.
      3. Build flat dicts compatible with the Milvus schema.
      4. Write to Milvus via MilvusClientWrapper.
    """

    def __init__(
        self,
        embeddings: Embeddings,
        milvus: MilvusClientWrapper,
        batch_size: int = _BATCH_SIZE,
    ) -> None:
        self._embeddings = embeddings
        self._milvus = milvus
        self._batch_size = batch_size
        self._milvus.ensure_collection()

    # ── Public interface ───────────────────────────────────────────────────────

    def index(self, documents: list[Document]) -> int:
        """Embed and index *documents*; return total records written."""
        total = 0
        for batch in self._batches(documents):
            records = self._build_records(batch)
            self._milvus.insert(records)
            total += len(records)
            logger.info("indexer_batch_written", batch_size=len(records), total=total)
        return total

    def index_with_source_refresh(
        self,
        documents: list[Document],
        source: str,
    ) -> int:
        """Delete existing chunks for *source* before indexing fresh ones."""
        deleted = self._milvus.delete_by_source(source)
        logger.info("indexer_source_refreshed", source=source, deleted=deleted)
        return self.index(documents)

    # ── Private helpers ────────────────────────────────────────────────────────

    def _build_records(self, docs: list[Document]) -> list[dict[str, Any]]:
        texts = [d.page_content for d in docs]
        vectors = self._embeddings.embed_documents(texts)
        records: list[dict[str, Any]] = []
        for doc, vec in zip(docs, vectors):
            m = doc.metadata
            # Serialise any remaining metadata fields not in the fixed schema.
            extra = {
                k: v for k, v in m.items()
                if k not in {"source", "doc_type", "strategy", "section", "chunk_id"}
            }
            records.append({
                "embedding":     vec,
                "content":       doc.page_content[:65_535],
                "source":        str(m.get("source", "")),
                "doc_type":      str(m.get("doc_type", "")),
                "strategy":      str(m.get("strategy", "")),
                "section":       str(m.get("section", "")),
                "chunk_id":      int(m.get("chunk_id", 0)),
                "metadata_json": json.dumps(extra)[:65_535],
            })
        return records

    @staticmethod
    def _batches(docs: list[Document]) -> list[list[Document]]:
        return [docs[i: i + _BATCH_SIZE] for i in range(0, len(docs), _BATCH_SIZE)]
