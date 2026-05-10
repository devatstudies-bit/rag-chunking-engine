"""Chunking quality metrics for offline evaluation and monitoring."""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field

from langchain_core.documents import Document


@dataclass
class ChunkingMetrics:
    """Compute and report quality metrics for a list of chunked Documents."""

    documents: list[Document]
    _sizes: list[int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._sizes = [len(d.page_content) for d in self.documents]

    @property
    def total_chunks(self) -> int:
        return len(self.documents)

    @property
    def total_characters(self) -> int:
        return sum(self._sizes)

    @property
    def mean_chunk_size(self) -> float:
        return statistics.mean(self._sizes) if self._sizes else 0.0

    @property
    def median_chunk_size(self) -> float:
        return statistics.median(self._sizes) if self._sizes else 0.0

    @property
    def std_chunk_size(self) -> float:
        return statistics.stdev(self._sizes) if len(self._sizes) > 1 else 0.0

    @property
    def min_chunk_size(self) -> int:
        return min(self._sizes, default=0)

    @property
    def max_chunk_size(self) -> int:
        return max(self._sizes, default=0)

    @property
    def empty_chunk_count(self) -> int:
        return sum(1 for s in self._sizes if s == 0)

    @property
    def strategy_distribution(self) -> dict[str, int]:
        dist: dict[str, int] = {}
        for doc in self.documents:
            strat = str(doc.metadata.get("strategy", "unknown"))
            dist[strat] = dist.get(strat, 0) + 1
        return dist

    def report(self) -> dict[str, object]:
        return {
            "total_chunks": self.total_chunks,
            "total_characters": self.total_characters,
            "mean_chunk_size": round(self.mean_chunk_size, 1),
            "median_chunk_size": round(self.median_chunk_size, 1),
            "std_chunk_size": round(self.std_chunk_size, 1),
            "min_chunk_size": self.min_chunk_size,
            "max_chunk_size": self.max_chunk_size,
            "empty_chunks": self.empty_chunk_count,
            "strategy_distribution": self.strategy_distribution,
        }
