# Architecture Deep Dive

## Design Principles

### 1. Strategy Pattern — One Interface, Eight Implementations

Every chunker inherits from `BaseChunker` and implements a single `chunk(text, metadata) → list[Document]` method. The caller never needs to know which strategy is active.

```
BaseChunker
├── FixedSizeChunker           ← never use in production
├── RecursiveCharacterChunker  ← general prose fallback
├── DocumentAwareChunker       ← structured documents
├── SemanticChunker            ← transcripts, conversations
├── CodeAwareChunker           ← source code (any language)
├── RowAwareChunker            ← CSV / tabular data
├── SlidingWindowChunker       ← dense technical docs
└── AgenticChunker             ← complex heterogeneous docs
```

### 2. Provider Abstraction — Zero Code Changes to Switch LLMs

`LLMProvider` is an abstract class. `ProviderFactory.create()` reads `LLM_PROVIDER` from settings and returns the right implementation. Adding a new provider (e.g. Google Vertex AI) requires only a new class, not changes to any pipeline code.

```python
# Switching providers is a single .env change:
LLM_PROVIDER=bedrock   # or azure_openai
```

### 3. LangGraph Pipelines — Observable, Resumable, Extensible

Both the ingestion and retrieval pipelines are `StateGraph` instances. Every node is a pure function: `state_in → state_out`. This makes each step independently testable, loggable, and replaceable.

**Ingestion nodes:**
```
classify → select_strategy → chunk → index → END
                                   ↘ error → END
```

**Retrieval nodes:**
```
embed_query → search → rerank → generate → END
                     ↘ error → END
```

### 4. Auto-Detection Without Magic

`StrategyRegistry.detect_document_type()` scores each document against a set of heuristics (regex patterns, file extension, CSV comma density). The highest-scoring type wins. Every heuristic is a pure function — easy to test, easy to extend.

The caller can always override with an explicit `doc_type` in the metadata, which always takes precedence.

### 5. Metadata is a First-Class Citizen

Every chunk carries a `metadata` dict that travels from the chunker through the indexer into Milvus. At retrieval time, these fields enable server-side filtering — e.g. "only return chunks from structured documents in the Findings section." This is far more precise than pure vector similarity.

Key metadata fields stored in every chunk:
- `strategy` — which chunker produced this chunk
- `doc_type` — detected or provided document type
- `source` — document identifier
- `section` — section name (document-aware strategy)
- `chunk_id` — ordinal position within the document
- `metadata_json` — serialised remainder (for domain-specific fields)

---

## Component Interactions

```
Client
  │
  ▼
FastAPI (api/main.py)
  │  ┌──────────────────────────────┐
  │  │  app.state.ingestion_graph   │ ← built at startup
  │  │  app.state.retrieval_graph   │
  │  └──────────────────────────────┘
  │
  ├─ POST /api/v1/ingest
  │    IngestionGraph.run(text, id, meta)
  │      → classify_node         (StrategyRegistry.detect_document_type)
  │      → select_strategy_node  (StrategyRegistry.select_strategy)
  │      → chunk_node            (StrategyRegistry.build_chunker → chunker.chunk)
  │      → index_node            (DocumentIndexer.index_with_source_refresh)
  │           → embeddings.embed_documents  (provider)
  │           → MilvusClientWrapper.insert
  │
  └─ POST /api/v1/query
       RetrievalGraph.run(query, top_k, filter)
         → embed_query_node      (embeddings.embed_query)
         → search_node           (MilvusClientWrapper.search)
         → rerank_node           (score sort)
         → generate_node         (llm.invoke with context)
```

---

## Milvus Schema

| Field | Type | Notes |
|---|---|---|
| `id` | INT64 (PK, auto) | Assigned by Milvus |
| `embedding` | FLOAT_VECTOR(1536) | HNSW index, cosine metric |
| `content` | VARCHAR(65535) | Full chunk text |
| `source` | VARCHAR(512) | Document identifier |
| `doc_type` | VARCHAR(64) | Filterable |
| `strategy` | VARCHAR(64) | Filterable |
| `section` | VARCHAR(256) | Filterable |
| `chunk_id` | INT64 | Chunk ordinal |
| `metadata_json` | VARCHAR(65535) | Serialised extra fields |

**HNSW index parameters:**
- `M = 16` — graph connectivity (higher = better recall, more memory)
- `efConstruction = 200` — build quality (higher = better, slower build)
- `ef = 64` — search quality at query time

---

## Extending the Engine

### Adding a new chunking strategy

1. Create `src/chunking_engine/chunkers/my_strategy.py` inheriting `BaseChunker`
2. Add `MY_STRATEGY = "my_strategy"` to `ChunkingStrategy` enum in `base.py`
3. Add a new `DocumentType` entry if needed
4. Add the mapping to `_TYPE_TO_STRATEGY` in `strategy_registry.py`
5. Add a `case ChunkingStrategy.MY_STRATEGY` branch in `StrategyRegistry.build_chunker`
6. Export from `chunkers/__init__.py`
7. Write tests in `tests/test_chunkers/`

### Adding a new LLM provider

1. Create `src/chunking_engine/models/my_provider.py` inheriting `LLMProvider`
2. Implement `get_chat_model()` and `get_embeddings()`
3. Add the provider name to `Settings.llm_provider` type hint
4. Add a branch in `ProviderFactory.create()`
5. Add provider credentials to `.env.example`

### Adding a new document type heuristic

Add a new tuple to `_HEURISTICS` in `strategy_registry.py`:

```python
(
    DocumentType.MY_NEW_TYPE,
    lambda text, meta: _my_detection_fn(text, meta),
),
```

---

## Performance Notes

- **Batch embedding**: The indexer calls `embed_documents` in batches of 128, not one-by-one. This is 10–50× faster depending on the provider.
- **HNSW vs IVF_FLAT**: HNSW (chosen here) has O(log n) search time with better recall at low ef values. Switch to IVF_FLAT for datasets > 100M vectors.
- **Semantic chunking is expensive at ingestion**: One embedding call per sentence. Use only for transcripts/chat. For millions of documents, pre-classify and route only transcripts through the semantic path.
- **Agentic chunking is expensive**: One LLM call per document. Reserve for a small corpus of complex, high-value documents. Alternatively, use agentic output to fine-tune a cheaper classification model.
- **Deduplication**: `SlidingWindowChunker.deduplicate()` uses Jaccard similarity — O(n²) in the worst case. For large result sets, use MinHash LSH for approximate deduplication.
