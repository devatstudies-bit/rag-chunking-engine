# Strategy 6 — Row-Aware (Tabular)

> **Best for: CSV files, spreadsheets, database exports, backlog tables, any structured tabular data.**

---

## How It Works

**One row = one chunk.** Column headers are included in every chunk as labelled key-value pairs. All column values are also stored as individual metadata fields for server-side Milvus filtering.

```
Input CSV:
  ID,Name,Category,Priority,Status
  ITEM-001,Legacy Auth Removal,Security,Critical,Mandatory
  ITEM-002,Database Migration,Infrastructure,High,Mandatory
  ITEM-003,Logging Upgrade,Observability,Medium,Recommended

Output chunks:

Chunk 0 (page_content):
  ID: ITEM-001
  Name: Legacy Auth Removal
  Category: Security
  Priority: Critical
  Status: Mandatory

Chunk 1 (page_content):
  ID: ITEM-002
  Name: Database Migration
  Category: Infrastructure
  Priority: High
  Status: Mandatory

Chunk 2 (page_content):
  ID: ITEM-003
  Name: Logging Upgrade
  Category: Observability
  Priority: Medium
  Status: Recommended
```

---

## Why Headers Must Be in Every Chunk

Without headers, a retrieved chunk reads:

```
ITEM-001, Security, Critical, Mandatory
```

The LLM has no idea whether "Critical" is a priority, a severity, a status, or something else entirely. With headers:

```
ID: ITEM-001
Category: Security
Priority: Critical
Status: Mandatory
```

The LLM immediately understands the meaning of each value without requiring the original schema to be in the prompt.

---

## The Spreadsheet Analogy

Every row in a spreadsheet is a self-contained record. You would never hand someone half a row and expect them to understand it — they need all the columns. Row-aware chunking applies this principle: each chunk is one complete, labelled record.

---

## Metadata Filtering Power

All column values are stored in chunk metadata (lowercased, spaces replaced with underscores). This enables extremely precise server-side filtering:

```python
# Find all Critical items
results = milvus.search(query_vec, filter_expr='priority == "Critical"')

# Find all Mandatory items in Security category
results = milvus.search(
    query_vec,
    filter_expr='status == "Mandatory" and category == "Security"'
)

# Find by specific ID
results = milvus.search(query_vec, filter_expr='id == "ITEM-001"')
```

This turns vector search into a hybrid semantic + structured query — far more powerful than text similarity alone.

---

## Handling Large Tables

For CSV files with thousands of rows, the indexer's batch embedding (128 rows per batch) keeps memory usage bounded. Each row is embedded independently — no cross-row context is needed.

---

## Code Reference

```python
from chunking_engine.chunkers import RowAwareChunker

chunker = RowAwareChunker()
chunks = chunker.chunk(csv_text, {"source": "backlog.csv"})

# Each chunk has:
# chunk.page_content → "ID: ITEM-001\nCategory: Security\nPriority: Critical\n…"
# chunk.metadata["priority"]   → "Critical"
# chunk.metadata["category"]   → "Security"
# chunk.metadata["status"]     → "Mandatory"
# chunk.metadata["row_index"]  → 0
```

---

## Input Requirements

- Valid CSV with a header row as the first line
- Any delimiter supported by Python's `csv.DictReader` (default: comma)
- Column names become metadata keys (lowercased, spaces → underscores)

---

## Metadata Fields

| Field | Value | Notes |
|---|---|---|
| `strategy` | `row_aware` | |
| `doc_type` | `tabular_data` | |
| `chunk_id` | Sequential integer | |
| `row_index` | Row number (0-based) | |
| `columns_json` | JSON list of column names | |
| `{col_name}` | Column value | One entry per column, lowercased |

---

## Interview Line

> *"Tabular data is chunked one row per document, always including column headers as labelled key-value pairs. Without headers the LLM would see 'ITEM-001, Critical, Mandatory' with no idea what these values mean. The column values stored in metadata enable powerful server-side filtering at retrieval time — for example, retrieving only mandatory critical items affecting a specific category, without scanning the entire collection."*
