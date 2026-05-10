# Strategy 8 — Agentic

> **Best for: complex heterogeneous documents where a single rule-based strategy cannot handle all content types simultaneously.**
> The highest-fidelity strategy. Most candidates at senior-level interviews do not know this exists.

---

## How It Works

The **full document** is submitted to an LLM in a single prompt. The model is asked to read the content and propose optimal chunk boundaries with:
- Start and end character positions for each chunk
- A descriptive section name
- A content type classification (`section`, `table`, `code_block`, `list`, `introduction`, `conclusion`, `general`)

The engine then extracts the proposed slices from the original document, attaches metadata, and returns LangChain `Document` objects.

```
Input: A 3,000-character document containing:
  - An executive summary paragraph
  - An embedded data table (markdown format)
  - A code snippet
  - A recommendations list
  - References

LLM proposal:
  [
    { start: 0,    end: 450,  section: "Executive Summary",   type: "introduction" },
    { start: 450,  end: 900,  section: "Performance Metrics", type: "table"        },
    { start: 900,  end: 1400, section: "Implementation",      type: "code_block"   },
    { start: 1400, end: 1900, section: "Recommendations",     type: "list"         },
    { start: 1900, end: 2200, section: "References",          type: "general"      }
  ]
```

---

## The Domain Expert Analogy

Instead of giving scissors to a machine (fixed-size) or a linguist (semantic), you hand the document to a domain expert who reads it fully and decides where the natural semantic breaks are. The expert considers content type, logical flow, and semantic completeness — not token counts.

---

## Why No Other Strategy Can Handle Heterogeneous Documents

```
Document structure:
  ┌──────────────────────────────────────────────┐
  │ Executive Summary (prose)                     │ → needs Recursive or Document-Aware
  ├──────────────────────────────────────────────┤
  │ | Metric | Q1    | Q2    | Q3    |            │ → needs Row-Aware
  │ |--------|-------|-------|-------|            │
  │ | Revenue| $1.2M | $1.5M | $1.8M |           │
  ├──────────────────────────────────────────────┤
  │ def calculate_growth(q1, q2):                 │ → needs Code-Aware
  │     return (q2 - q1) / q1                    │
  ├──────────────────────────────────────────────┤
  │ 1. Expand to APAC markets                     │ → needs Recursive
  │ 2. Hire 3 additional engineers                │
  └──────────────────────────────────────────────┘

No single rule-based strategy handles all four sections correctly.
Agentic chunking adapts to each section individually.
```

---

## Cost Model

| Stage | Cost |
|---|---|
| Ingestion | **1 LLM call per document** |
| Retrieval | No additional cost — chunks are stored like any other strategy |

**Is it worth it?**

| Corpus size | Recommendation |
|---|---|
| < 500 complex documents | ✅ Use agentic directly — cost is negligible |
| 500–5,000 documents | ✅ Use agentic + cache results; re-chunk only on document update |
| > 5,000 documents | 🔄 Use agentic on a representative sample as training data; fine-tune a cheaper boundary classifier |

---

## Retry Logic

The `AgenticChunker` uses `tenacity` with exponential back-off (3 attempts, 1–10 second waits). This handles transient LLM API errors without failing the entire ingestion pipeline.

---

## Document Length Guard

The agentic prompt is capped at 8,000 characters of document text to prevent context overflow. For longer documents, pre-split the document with Recursive Character chunking into 8,000-character segments, then apply agentic chunking to each segment independently.

```python
from chunking_engine.chunkers import AgenticChunker, RecursiveCharacterChunker, ChunkingConfig

# Pre-split very long documents before agentic processing
presplit_config = ChunkingConfig(chunk_size=7500, chunk_overlap=0)
segments = RecursiveCharacterChunker(presplit_config).chunk(very_long_doc)

# Apply agentic chunking to each segment
agentic = AgenticChunker(llm=llm)
all_chunks = []
for seg in segments:
    all_chunks.extend(agentic.chunk(seg.page_content, seg.metadata))
```

---

## Using Agentic Output as Training Data

For large corpora, run agentic chunking on 1,000 representative documents and use the proposed boundaries to fine-tune a cheap text classifier. The classifier learns to mimic the LLM's boundary decisions at 1/100th the cost per document.

---

## Code Reference

```python
from chunking_engine.chunkers import AgenticChunker, ChunkingConfig
from chunking_engine.models.base import ProviderFactory

provider = ProviderFactory.create()
llm = provider.get_chat_model(temperature=0.0)

chunker = AgenticChunker(llm=llm)
chunks = chunker.chunk(complex_document, {"source": "report-2024-Q3"})

# Each chunk has:
# chunk.metadata["section"]    → "Performance Metrics"
# chunk.metadata["chunk_type"] → "table"
# chunk.metadata["char_start"] → 450
# chunk.metadata["char_end"]   → 900
```

---

## Metadata Fields

| Field | Value | Notes |
|---|---|---|
| `strategy` | `agentic` | |
| `doc_type` | `heterogeneous` | |
| `section` | LLM-assigned section name | |
| `chunk_type` | `section` / `table` / `code_block` / `list` / `introduction` / `conclusion` / `general` | |
| `char_start` | Integer | Start position in original text |
| `char_end` | Integer | End position in original text |
| `chunk_id` | Sequential integer | |

---

## Chunk Type Filtering

```python
# Retrieve only table chunks (for quantitative queries)
results = milvus.search(query_vec, filter_expr='chunk_type == "table"')

# Retrieve only code blocks (for implementation queries)
results = milvus.search(query_vec, filter_expr='chunk_type == "code_block"')
```

---

## Interview Line

> *"Agentic chunking is the highest-fidelity approach — I submit the full document to GPT-4o or Claude and ask it to propose chunk boundaries with reasoning. It handles complex heterogeneous documents that contain structured sections, embedded tables, and code snippets all mixed together. No single rule-based strategy handles all three simultaneously. The cost is one LLM call per document at ingestion time, which is justified for a small corpus of high-value documents. For a large corpus I would use agentic chunking to generate training examples for a cheaper, fine-tuned boundary classifier."*
