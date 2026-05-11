# Chunking Engine

> **Production-grade adaptive document chunking for RAG pipelines**
> Built with LangChain · LangGraph · Milvus · AWS Bedrock · Azure OpenAI

---

## The Golden Rule

> **There is NO single best chunking strategy. The right strategy depends entirely on the document type.**

This engine automatically selects the optimal strategy for each document and orchestrates an end-to-end RAG pipeline with observable, testable, provider-agnostic components.

---

## Architecture Overview

```mermaid
graph TB
    classDef input     fill:#1565C0,stroke:#90CAF9,stroke-width:2px,color:#fff
    classDef ingestion fill:#1B5E20,stroke:#A5D6A7,stroke-width:2px,color:#fff
    classDef chunker   fill:#4A148C,stroke:#CE93D8,stroke-width:2px,color:#fff
    classDef provider  fill:#E65100,stroke:#FFCC02,stroke-width:2px,color:#fff
    classDef vector    fill:#006064,stroke:#80DEEA,stroke-width:2px,color:#fff
    classDef retrieval fill:#880E4F,stroke:#F48FB1,stroke-width:2px,color:#fff
    classDef api       fill:#33691E,stroke:#DCEDC8,stroke-width:2px,color:#fff

    subgraph INPUT["📄 Document Input Layer"]
        D1[Structured Document]:::input
        D2[General Text]:::input
        D3[Transcript]:::input
        D4[Source Code]:::input
        D5[Tabular Data]:::input
        D6[Technical Doc]:::input
        D7[Heterogeneous]:::input
    end

    subgraph INGESTION["⚙️ LangGraph Ingestion Pipeline"]
        N1["🔍 Classify Document"]:::ingestion
        N2["📋 Select Strategy"]:::ingestion
        N3["✂️ Execute Chunker"]:::ingestion
        N4["🔢 Generate Embeddings"]:::ingestion
        N5["💾 Index to Milvus"]:::ingestion
        N1 --> N2 --> N3 --> N4 --> N5
    end

    subgraph CHUNKERS["🧠 Chunking Strategy Engine"]
        C1["1️⃣ Fixed Size"]:::chunker
        C2["2️⃣ Recursive Character"]:::chunker
        C3["3️⃣ Document-Aware"]:::chunker
        C4["4️⃣ Semantic"]:::chunker
        C5["5️⃣ Code-Aware"]:::chunker
        C6["6️⃣ Row-Aware"]:::chunker
        C7["7️⃣ Sliding Window"]:::chunker
        C8["8️⃣ Agentic"]:::chunker
    end

    subgraph PROVIDERS["☁️ LLM Providers"]
        P1["🟠 AWS Bedrock"]:::provider
        P2["🔵 Azure OpenAI"]:::provider
    end

    subgraph VECTOR["🗄️ Vector Database"]
        M1["⚡ Milvus — HNSW / Cosine"]:::vector
    end

    subgraph RETRIEVAL["🔎 LangGraph Retrieval Pipeline"]
        R1["📝 Preprocess Query"]:::retrieval
        R2["🔢 Embed Query"]:::retrieval
        R3["🔍 ANN Search"]:::retrieval
        R4["📊 Rerank Results"]:::retrieval
        R5["💬 Generate Answer"]:::retrieval
        R1 --> R2 --> R3 --> R4 --> R5
    end

    subgraph API["🌐 REST API"]
        A1["POST /api/v1/ingest"]:::api
        A2["POST /api/v1/query"]:::api
        A3["GET  /health"]:::api
    end

    D1 & D2 & D3 & D4 & D5 & D6 & D7 --> N1
    N3 --> C1 & C2 & C3 & C4 & C5 & C6 & C7 & C8
    C1 & C2 & C3 & C4 & C5 & C6 & C7 & C8 --> N4
    N4 --> P1 & P2
    N5 --> M1
    R3 --> M1
    R2 --> P1 & P2
    R5 --> P1 & P2
    A1 --> N1
    A2 --> R1
```

---

## Strategy Selection Decision Tree

```mermaid
flowchart TD
    classDef start    fill:#1565C0,stroke:#1565C0,color:#fff
    classDef decision fill:#F57F17,stroke:#E65100,color:#fff
    classDef strategy fill:#1B5E20,stroke:#2E7D32,color:#fff
    classDef warning  fill:#B71C1C,stroke:#C62828,color:#fff

    START([🚀 Document Arrives]):::start
    Q1{Is it CSV or tabular?}:::decision
    Q2{Is it source code?}:::decision
    Q3{Has known section headers?}:::decision
    Q4{Conversational or transcript?}:::decision
    Q5{Dense technical prose?}:::decision
    Q6{Mixed prose + tables + code?}:::decision
    Q7{Benchmark or prototype only?}:::decision

    S1["6️⃣ Row-Aware Chunker<br/>1 row = 1 chunk<br/>Headers in every chunk<br/>Column values in metadata"]:::strategy
    S2["5️⃣ Code-Aware Chunker<br/>Splits at function/class boundaries<br/>Never cuts mid-function<br/>Multi-language support"]:::strategy
    S3["3️⃣ Document-Aware Chunker<br/>1 section = 1 chunk<br/>Header prepended to every chunk<br/>Section name in metadata"]:::strategy
    S4["4️⃣ Semantic Chunker<br/>Embeds every sentence<br/>Splits at similarity drops<br/>Topic-coherent chunks"]:::strategy
    S5["7️⃣ Sliding Window Chunker<br/>Fixed size + configurable overlap<br/>Boundary concepts preserved<br/>20% overlap rule of thumb"]:::strategy
    S6["8️⃣ Agentic Chunker<br/>LLM reads full document<br/>Proposes optimal boundaries<br/>1 LLM call per document"]:::strategy
    S7["2️⃣ Recursive Character Chunker<br/>Hierarchy: para → line → sentence<br/>Sentence/paragraph integrity<br/>Best general-purpose fallback"]:::strategy
    S8["1️⃣ Fixed Size Chunker<br/>⚠️ NEVER in production<br/>No structural awareness<br/>Benchmark baseline only"]:::warning

    START --> Q1
    Q1 -->|Yes| S1
    Q1 -->|No| Q2
    Q2 -->|Yes| S2
    Q2 -->|No| Q3
    Q3 -->|Yes| S3
    Q3 -->|No| Q4
    Q4 -->|Yes| S4
    Q4 -->|No| Q5
    Q5 -->|Yes| S5
    Q5 -->|No| Q6
    Q6 -->|Yes| S6
    Q6 -->|No| Q7
    Q7 -->|Yes| S8
    Q7 -->|No| S7
```

---

## Ingestion Pipeline (LangGraph)

```mermaid
flowchart LR
    classDef step  fill:#1565C0,stroke:#90CAF9,stroke-width:2px,color:#fff
    classDef ok    fill:#1B5E20,stroke:#A5D6A7,stroke-width:2px,color:#fff
    classDef error fill:#B71C1C,stroke:#EF9A9A,stroke-width:2px,color:#fff

    IN([📄 Raw Document]):::ok
    C["🔍 Classify<br/>Heuristic detection:<br/>CSV → tabular_data<br/>Code → source_code<br/>Headers → structured_document<br/>Speakers → transcript"]:::step
    S["📋 Select Strategy<br/>Registry lookup:<br/>doc_type → ChunkingStrategy<br/>Instantiate chunker"]:::step
    CH["✂️ Execute Chunker<br/>Runs selected strategy<br/>Returns list of Documents<br/>with strategy metadata"]:::step
    I["💾 Index to Milvus<br/>Batch embed 128 docs<br/>Delete prior version<br/>Insert with HNSW index"]:::step
    OK([✅ indexed_count]):::ok
    ERR([❌ error list]):::error

    IN --> C --> S --> CH
    CH -->|chunks ready| I
    CH -->|exception| ERR
    I --> OK
```

---

## Retrieval Pipeline (LangGraph)

```mermaid
flowchart LR
    classDef step  fill:#880E4F,stroke:#F48FB1,stroke-width:2px,color:#fff
    classDef ok    fill:#1B5E20,stroke:#A5D6A7,stroke-width:2px,color:#fff
    classDef error fill:#B71C1C,stroke:#EF9A9A,stroke-width:2px,color:#fff

    IN([🔎 User Query]):::ok
    E["🔢 Embed Query<br/>Provider: Bedrock or Azure<br/>float[1536] vector"]:::step
    SE["🔍 ANN Search<br/>Cosine similarity HNSW<br/>Optional metadata filter<br/>Returns top-k hits"]:::step
    R["📊 Rerank Results<br/>Score-based sort<br/>Plug in cross-encoder<br/>for production"]:::step
    G["💬 Generate Answer<br/>Context + question → LLM<br/>Bedrock or Azure<br/>Answer + cited sources"]:::step
    OK([✅ Answer + Sources]):::ok
    ERR([❌ Error]):::error

    IN --> E --> SE
    SE -->|results found| R
    SE -->|Milvus unreachable| ERR
    R --> G --> OK
```

---

## Strategy Comparison Matrix

```mermaid
quadrantChart
    title Chunking Strategy Trade-offs
    x-axis Low Structural Awareness --> High Structural Awareness
    y-axis Low Compute Cost --> High Compute Cost
    quadrant-1 Powerful but expensive
    quadrant-2 High-fidelity, affordable
    quadrant-3 Fast but imprecise
    quadrant-4 Structured and cheap
    Fixed Size: [0.05, 0.05]
    Recursive Character: [0.25, 0.10]
    Sliding Window: [0.20, 0.15]
    Row-Aware: [0.75, 0.08]
    Code-Aware: [0.80, 0.12]
    Document-Aware: [0.85, 0.10]
    Semantic: [0.45, 0.55]
    Agentic: [0.95, 0.90]
```

---

## Quick Start

### 1. Clone and setup

```bash
git clone https://github.com/devatstudies-bit/rag-chunking-engine
cd rag-chunking-engine
bash setup.sh
source .venv/bin/activate
```

### 2. Configure credentials

```bash
cp .env.example .env
# Fill in your Azure OpenAI or AWS Bedrock credentials
```

### 3. Start Milvus (Docker)

```bash
docker compose -f docker/docker-compose.yml up -d
```

### 4. Run the demo (no Milvus needed)

```bash
python examples/demo.py
```

### 5. Start the API server

```bash
uvicorn api.main:app --reload
# API docs: http://localhost:8000/docs
```

### 6. Run tests

```bash
pytest
pytest --cov=src/chunking_engine --cov-report=html
```

---

## API Reference

### Ingest a document

```bash
curl -X POST http://localhost:8000/api/v1/ingest \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-secret-key" \
  -d '{
    "document_id": "report-2024-0042",
    "content": "Overview:\nThis report covers...\n\nFindings:\nWe found...",
    "doc_type": "structured_document"
  }'
```

Response:
```json
{
  "document_id": "report-2024-0042",
  "strategy_used": "document_aware",
  "doc_type_detected": "structured_document",
  "chunks_indexed": 4,
  "status": "success"
}
```

### Query the RAG pipeline

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-secret-key" \
  -d '{
    "query": "What are the main findings from the 2024 report?",
    "top_k": 5,
    "doc_type_filter": "structured_document"
  }'
```

---

## Provider Switching

Switch between AWS Bedrock and Azure OpenAI with a single environment variable — no code changes required:

```bash
LLM_PROVIDER=azure_openai   # default
LLM_PROVIDER=bedrock
```

| Capability | Azure OpenAI | AWS Bedrock |
|---|---|---|
| Chat model | `gpt-4o` | `claude-3-5-sonnet-20241022-v2:0` |
| Embeddings | `text-embedding-3-large` (1536-d) | `titan-embed-text-v2:0` (1536-d) |
| Auth | API key | IAM / access key |

---

## Strategy Reference

| # | Strategy | Best For | Key Property |
|---|---|---|---|
| 1 | Fixed Size | Benchmarks only | Splits anywhere — no intelligence |
| 2 | Recursive Character | General prose | Hierarchy: `\n\n` → `\n` → `.` → ` ` |
| 3 | Document-Aware | Structured reports | 1 section = 1 chunk, header in every chunk |
| 4 | Semantic | Transcripts / chat | Embedding similarity drives boundaries |
| 5 | Code-Aware | Source code | Function/class boundaries respected |
| 6 | Row-Aware | CSV / tables | 1 row = 1 chunk, headers always present |
| 7 | Sliding Window | Dense technical docs | Configurable overlap prevents boundary loss |
| 8 | Agentic | Mixed / complex docs | LLM proposes boundaries with reasoning |

---

## Project Structure

```
rag-chunking-engine/
├── src/chunking_engine/
│   ├── config/           # Pydantic settings, structured logging
│   ├── models/           # LLM provider abstraction (Bedrock + Azure OpenAI)
│   ├── chunkers/         # All 8 chunking strategies
│   ├── vectorstore/      # Milvus client + document indexer
│   ├── pipeline/         # LangGraph ingestion + retrieval graphs
│   ├── registry/         # Strategy auto-detection + factory
│   └── utils/            # Deduplication, metrics
├── api/                  # FastAPI REST API
├── tests/                # Pytest test suite (all chunkers + pipeline)
├── examples/             # Sample documents + interactive demo
├── docker/               # Dockerfile + docker-compose (Milvus stack)
└── docs/                 # Per-strategy deep-dive documentation
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Orchestration | LangGraph 0.2+ |
| LLM framework | LangChain 0.3+ |
| LLM providers | AWS Bedrock, Azure OpenAI |
| Vector database | Milvus 2.4 (HNSW, Cosine) |
| API | FastAPI + Uvicorn |
| Configuration | Pydantic Settings v2 |
| Logging | structlog |
| Testing | pytest + pytest-asyncio |
| Containers | Docker + docker-compose |

---

## License

MIT
