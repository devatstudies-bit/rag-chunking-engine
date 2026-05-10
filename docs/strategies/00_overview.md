# Chunking Strategies — Quick Reference

> **The Golden Rule: There is NO single best strategy. The right strategy depends entirely on the document type.**

---

## Decision Guide

| Document Type | Strategy | Why |
|---|---|---|
| Structured doc (known sections) | [Document-Aware](03_document_aware.md) | Section headers are explicit boundaries. Header prepended to every chunk. |
| General prose / mixed content | [Recursive Character](02_recursive_character.md) | Hierarchy: paragraph → sentence → word. No structural knowledge needed. |
| Transcript / chat / interview | [Semantic](04_semantic.md) | No headers. Topic shifts mid-conversation. Embeddings detect the shifts. |
| Source code (any language) | [Code-Aware](05_code_aware.md) | Functions/classes are complete logical units. Never split mid-function. |
| CSV / table / spreadsheet | [Row-Aware](06_row_aware.md) | One row = one complete record. Headers label every value. |
| Dense technical docs / APIs | [Sliding Window](07_sliding_window.md) | Cross-references span boundaries. Overlap ensures completeness. |
| Mixed: prose + tables + code | [Agentic](08_agentic.md) | No rule-based strategy handles all types. LLM proposes boundaries. |
| Prototype / benchmark only | [Fixed Size](01_fixed_size.md) | Never in production. Split mid-sentence. Baseline only. |

---

## Strategy Comparison at a Glance

```
Structural Awareness ────────────────────────────────────► HIGH
LOW
│  Fixed Size        ← no awareness
│  Recursive Char    ← paragraph/sentence awareness
│  Sliding Window    ← same as recursive + overlap
│  Semantic          ← topic coherence awareness
│  Code-Aware        ← language syntax awareness
│  Document-Aware    ← document schema awareness
│  Row-Aware         ← tabular schema awareness
│  Agentic           ← full semantic understanding via LLM
▼ HIGH
Compute Cost
```

---

## The Interview Answer That Wins

> *"I do not pick one chunking strategy and apply it everywhere. In a real project I use multiple strategies simultaneously, selecting based on document type:*
> - *Document-Aware for structured reports with known section schemas*
> - *Semantic for transcripts and conversational data*
> - *Code-Aware for source files (function/class boundaries)*
> - *Row-Aware for CSV and tabular exports*
> - *Recursive Character as the general-purpose fallback*
>
> *Where structure is known and fixed, I exploit it directly. Where structure is absent, I let embeddings detect topic boundaries. Where the document is code, I respect language syntax. The goal is always the same: each chunk must be self-contained, meaningful, and retrievable in isolation."*

---

## Strategy Documents

1. [01 — Fixed Size](01_fixed_size.md) — baseline benchmark, never production
2. [02 — Recursive Character](02_recursive_character.md) — general-purpose prose
3. [03 — Document-Aware](03_document_aware.md) — structured documents with known sections
4. [04 — Semantic](04_semantic.md) — transcripts and conversational text
5. [05 — Code-Aware](05_code_aware.md) — source code, any programming language
6. [06 — Row-Aware](06_row_aware.md) — CSV files and tabular data
7. [07 — Sliding Window](07_sliding_window.md) — dense technical documentation
8. [08 — Agentic](08_agentic.md) — complex heterogeneous documents
