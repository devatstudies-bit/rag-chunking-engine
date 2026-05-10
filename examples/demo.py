"""
Chunking Engine — Interactive Demo
===================================
Demonstrates all 8 chunking strategies against representative sample documents.
Does NOT require a running Milvus instance — showcases chunking logic only.

Usage:
    python examples/demo.py
    python examples/demo.py --strategy document_aware
    python examples/demo.py --all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# Ensure src/ is on the path when running directly
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from chunking_engine.chunkers import (
    ChunkingConfig,
    CodeAwareChunker,
    DocumentAwareChunker,
    FixedSizeChunker,
    RecursiveCharacterChunker,
    RowAwareChunker,
    SlidingWindowChunker,
)
from chunking_engine.utils.metrics import ChunkingMetrics

console = Console()
SAMPLES_DIR = Path(__file__).parent / "sample_documents"

STRATEGY_COLORS = {
    "fixed_size":          "red",
    "recursive_character": "cyan",
    "document_aware":      "green",
    "semantic":            "magenta",
    "code_aware":          "yellow",
    "row_aware":           "blue",
    "sliding_window":      "orange3",
    "agentic":             "bright_white",
}


def _header(title: str, color: str) -> None:
    console.print()
    console.rule(f"[bold {color}]{title}[/bold {color}]")
    console.print()


def _chunk_table(chunks, strategy_name: str) -> Table:
    color = STRATEGY_COLORS.get(strategy_name, "white")
    table = Table(
        title=f"[bold {color}]{strategy_name}[/bold {color}] — {len(chunks)} chunks",
        show_lines=True,
        border_style=color,
        header_style=f"bold {color}",
    )
    table.add_column("#", style="dim", width=4)
    table.add_column("Content (first 120 chars)", min_width=50)
    table.add_column("Metadata", min_width=30)

    for i, chunk in enumerate(chunks):
        preview = chunk.page_content[:120].replace("\n", " ")
        if len(chunk.page_content) > 120:
            preview += "…"
        meta_lines = [
            f"section: {chunk.metadata.get('section', '-')}",
            f"doc_type: {chunk.metadata.get('doc_type', '-')}",
            f"strategy: {chunk.metadata.get('strategy', '-')}",
        ]
        table.add_row(str(i), preview, "\n".join(meta_lines))
    return table


def demo_fixed_size(text: str) -> None:
    _header("Strategy 1 — Fixed Size  ⚠  BENCHMARK ONLY", "red")
    console.print("[dim italic]Never use in production. Shown only to demonstrate why better strategies exist.[/dim italic]\n")
    config = ChunkingConfig(chunk_size=300, chunk_overlap=50)
    chunks = FixedSizeChunker(config).chunk(text, {"source": "demo"})
    console.print(_chunk_table(chunks, "fixed_size"))
    _print_metrics(chunks)


def demo_recursive(text: str) -> None:
    _header("Strategy 2 — Recursive Character", "cyan")
    config = ChunkingConfig(chunk_size=400, chunk_overlap=60)
    chunks = RecursiveCharacterChunker(config).chunk(text, {"source": "demo"})
    console.print(_chunk_table(chunks, "recursive_character"))
    _print_metrics(chunks)


def demo_document_aware(text: str) -> None:
    _header("Strategy 3 — Document-Aware", "green")
    chunks = DocumentAwareChunker().chunk(text, {"source": "advisory-2024-0187"})
    console.print(_chunk_table(chunks, "document_aware"))
    _print_metrics(chunks)


def demo_code_aware(text: str) -> None:
    _header("Strategy 5 — Code-Aware", "yellow")
    config = ChunkingConfig(chunk_size=1000, chunk_overlap=100, language="python")
    chunks = CodeAwareChunker(config).chunk(text, {"source": "inventory.py"})
    console.print(_chunk_table(chunks, "code_aware"))
    _print_metrics(chunks)


def demo_row_aware(text: str) -> None:
    _header("Strategy 6 — Row-Aware (Tabular)", "blue")
    chunks = RowAwareChunker().chunk(text, {"source": "backlog.csv"})
    console.print(_chunk_table(chunks, "row_aware"))
    _print_metrics(chunks)


def demo_sliding_window(text: str) -> None:
    _header("Strategy 7 — Sliding Window", "orange3")
    config = ChunkingConfig(chunk_size=400, chunk_overlap=100)
    chunks = SlidingWindowChunker(config).chunk(text, {"source": "demo"})
    deduped = SlidingWindowChunker.deduplicate(chunks, threshold=0.8)
    console.print(_chunk_table(chunks, "sliding_window"))
    console.print(f"[dim]After deduplication: {len(deduped)} chunks (was {len(chunks)})[/dim]")
    _print_metrics(chunks)


def _print_metrics(chunks) -> None:
    m = ChunkingMetrics(chunks)
    report = m.report()
    console.print(
        f"[dim]  Chunks: {report['total_chunks']} | "
        f"Mean size: {report['mean_chunk_size']} chars | "
        f"Min: {report['min_chunk_size']} | Max: {report['max_chunk_size']}[/dim]"
    )


def run_all() -> None:
    general_text = (SAMPLES_DIR / "structured_note.txt").read_text()
    code_text = (SAMPLES_DIR / "code_sample.py").read_text()
    csv_text = (SAMPLES_DIR / "data.csv").read_text()

    console.print(Panel.fit(
        "[bold bright_white]Chunking Engine — Strategy Showcase[/bold bright_white]\n"
        "[dim]8 strategies demonstrated on domain-appropriate documents[/dim]",
        border_style="bright_white",
    ))

    demo_fixed_size(general_text)
    demo_recursive(general_text)
    demo_document_aware(general_text)
    demo_code_aware(code_text)
    demo_row_aware(csv_text)
    demo_sliding_window(general_text)

    console.print()
    console.print(Panel(
        "[bold cyan]Strategies 4 (Semantic) and 8 (Agentic) require live LLM connections.[/bold cyan]\n"
        "Configure your .env file and run:\n"
        "  [yellow]python -c \"from chunking_engine.chunkers import SemanticChunker; ...\"[/yellow]",
        title="Note",
        border_style="cyan",
    ))


def main() -> None:
    parser = argparse.ArgumentParser(description="Chunking Engine demo")
    parser.add_argument("--strategy", choices=[
        "fixed_size", "recursive", "document_aware", "code_aware", "row_aware", "sliding_window",
    ], help="Run a single strategy demo")
    parser.add_argument("--all", action="store_true", default=True, help="Run all demos (default)")
    args = parser.parse_args()

    if args.strategy:
        text = (SAMPLES_DIR / "structured_note.txt").read_text()
        dispatch = {
            "fixed_size": lambda: demo_fixed_size(text),
            "recursive": lambda: demo_recursive(text),
            "document_aware": lambda: demo_document_aware(text),
            "code_aware": lambda: demo_code_aware((SAMPLES_DIR / "code_sample.py").read_text()),
            "row_aware": lambda: demo_row_aware((SAMPLES_DIR / "data.csv").read_text()),
            "sliding_window": lambda: demo_sliding_window(text),
        }
        dispatch[args.strategy]()
    else:
        run_all()


if __name__ == "__main__":
    main()
