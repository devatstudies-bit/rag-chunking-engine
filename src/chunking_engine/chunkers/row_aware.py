"""Row-aware chunker — one tabular row per chunk with headers in every chunk."""

from __future__ import annotations

import csv
import json
from io import StringIO
from typing import Any

from langchain_core.documents import Document

from .base import BaseChunker, ChunkingConfig, ChunkingStrategy


class RowAwareChunker(BaseChunker):
    """Converts a CSV / tabular dataset to one Document per row.

    Column headers are embedded in every chunk as labelled key-value pairs so
    that the LLM receiving the chunk always knows the meaning of each field.
    Without headers the model would see 'SE16N, HIGH, MANDATORY' with no
    context about what these values represent.

    All column values are also stored as individual metadata fields for
    precise Milvus metadata filtering at query time.
    """

    strategy = ChunkingStrategy.ROW_AWARE

    def chunk(self, text: str, metadata: dict[str, Any] | None = None) -> list[Document]:
        reader = csv.DictReader(StringIO(text))
        if reader.fieldnames is None:
            raise ValueError("CSV input has no header row")

        chunks: list[Document] = []
        for i, row in enumerate(reader):
            content_lines = [f"{col}: {val}" for col, val in row.items() if val is not None]
            chunk_text = "\n".join(content_lines)

            row_meta: dict[str, Any] = {col.lower().replace(" ", "_"): val for col, val in row.items()}
            meta = self._base_metadata({
                "doc_type": "tabular_data",
                "chunk_id": i,
                "row_index": i,
                "columns_json": json.dumps(list(row.keys())),
                **row_meta,
                **(metadata or {}),
            })
            chunks.append(Document(page_content=chunk_text, metadata=meta))

        return chunks
