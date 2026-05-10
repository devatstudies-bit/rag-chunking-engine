"""Milvus connection wrapper with schema management and CRUD operations."""

from __future__ import annotations

import json
from typing import Any

import structlog
from pymilvus import DataType, MilvusClient

from chunking_engine.config import Settings, get_settings

logger = structlog.get_logger(__name__)

# VARCHAR fields must declare a max_length; 65 535 is the Milvus upper bound.
_MAX_VARCHAR = 65_535
_MAX_SHORT_VARCHAR = 512


class MilvusClientWrapper:
    """Thin wrapper around MilvusClient that owns schema lifecycle.

    Responsibilities:
    - Create / verify the target collection with HNSW index on first run.
    - Insert vectors + payload in batches.
    - Execute ANN (approximate nearest-neighbour) searches with optional
      metadata filters.
    - Delete documents by source identifier.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        s = settings or get_settings()
        self._collection = s.milvus_collection_name
        self._dim = s.milvus_embedding_dim
        uri = f"http://{s.milvus_host}:{s.milvus_port}"
        self._client = MilvusClient(uri=uri)
        logger.info("milvus_connected", uri=uri, collection=self._collection)

    # ── Schema bootstrap ───────────────────────────────────────────────────────

    def ensure_collection(self) -> None:
        """Create the collection if it does not already exist."""
        if self._client.has_collection(self._collection):
            logger.info("milvus_collection_exists", collection=self._collection)
            return

        schema = self._client.create_schema(
            auto_id=True,
            enable_dynamic_field=True,
        )
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=self._dim)
        schema.add_field("content", DataType.VARCHAR, max_length=_MAX_VARCHAR)
        schema.add_field("source", DataType.VARCHAR, max_length=_MAX_SHORT_VARCHAR)
        schema.add_field("doc_type", DataType.VARCHAR, max_length=64)
        schema.add_field("strategy", DataType.VARCHAR, max_length=64)
        schema.add_field("section", DataType.VARCHAR, max_length=256)
        schema.add_field("chunk_id", DataType.INT64)
        schema.add_field("metadata_json", DataType.VARCHAR, max_length=_MAX_VARCHAR)

        index_params = self._client.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            index_type="HNSW",
            metric_type="COSINE",
            params={"M": 16, "efConstruction": 200},
        )

        self._client.create_collection(
            collection_name=self._collection,
            schema=schema,
            index_params=index_params,
        )
        logger.info("milvus_collection_created", collection=self._collection)

    # ── Write ──────────────────────────────────────────────────────────────────

    def insert(self, records: list[dict[str, Any]]) -> list[int]:
        """Insert *records* into the collection, return assigned IDs."""
        if not records:
            return []
        result = self._client.insert(collection_name=self._collection, data=records)
        ids: list[int] = result.get("ids", [])
        logger.info("milvus_inserted", count=len(ids))
        return ids

    def delete_by_source(self, source: str) -> int:
        """Remove all chunks whose *source* field matches the given value."""
        expr = f'source == "{source}"'
        result = self._client.delete(collection_name=self._collection, filter=expr)
        deleted: int = result.get("delete_count", 0)
        logger.info("milvus_deleted", source=source, count=deleted)
        return deleted

    # ── Read ───────────────────────────────────────────────────────────────────

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        filter_expr: str | None = None,
        output_fields: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """ANN search; returns list of result dicts with distance and payload."""
        fields = output_fields or ["content", "source", "doc_type", "strategy",
                                   "section", "chunk_id", "metadata_json"]
        search_params = {"metric_type": "COSINE", "params": {"ef": 64}}
        results = self._client.search(
            collection_name=self._collection,
            data=[query_vector],
            limit=top_k,
            filter=filter_expr or "",
            output_fields=fields,
            search_params=search_params,
        )

        hits: list[dict[str, Any]] = []
        for hit in results[0]:
            entity = hit.get("entity", {})
            extra_meta: dict[str, Any] = {}
            if raw := entity.get("metadata_json"):
                try:
                    extra_meta = json.loads(raw)
                except json.JSONDecodeError:
                    pass
            hits.append({
                "id": hit.get("id"),
                "score": hit.get("distance"),
                "content": entity.get("content", ""),
                "source": entity.get("source", ""),
                "doc_type": entity.get("doc_type", ""),
                "strategy": entity.get("strategy", ""),
                "section": entity.get("section", ""),
                "chunk_id": entity.get("chunk_id", -1),
                **extra_meta,
            })
        return hits

    def collection_stats(self) -> dict[str, Any]:
        stats = self._client.get_collection_stats(self._collection)
        return dict(stats)
