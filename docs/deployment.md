# Deployment Guide

## Local Development

### Prerequisites

- Python 3.11+
- Docker + Docker Compose (for Milvus)
- AWS credentials (if using Bedrock) or Azure OpenAI endpoint + key

### 1. Bootstrap environment

```bash
git clone https://github.com/your-org/chunking-engine
cd chunking-engine
bash setup.sh
source .venv/bin/activate
```

### 2. Configure credentials

```bash
cp .env.example .env
# Edit .env with your provider credentials
```

Minimum required variables:

**Azure OpenAI:**
```
LLM_PROVIDER=azure_openai
AZURE_OPENAI_ENDPOINT=https://YOUR_RESOURCE.openai.azure.com/
AZURE_OPENAI_API_KEY=your-key
AZURE_CHAT_DEPLOYMENT=gpt-4o
AZURE_EMBEDDING_DEPLOYMENT=text-embedding-3-large
```

**AWS Bedrock:**
```
LLM_PROVIDER=bedrock
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=AKIA...
AWS_SECRET_ACCESS_KEY=...
```

### 3. Start Milvus

```bash
docker compose -f docker/docker-compose.yml up -d
# Milvus is ready when:
docker logs milvus-standalone 2>&1 | grep "Milvus startup complete"
```

### 4. Start the API

```bash
uvicorn api.main:app --reload --port 8000
# Swagger UI: http://localhost:8000/docs
```

### 5. Verify

```bash
curl http://localhost:8000/health
# Expected: {"status":"ok","provider":"azure_openai","milvus":"connected","version":"1.0.0"}
```

---

## Docker Deployment

Build and run the full stack with Docker Compose:

```bash
docker compose -f docker/docker-compose.yml up --build
```

This starts:
- **etcd** on port 2379
- **MinIO** on ports 9000, 9001
- **Milvus** on port 19530
- **Chunking Engine API** on port 8000

Tear down:
```bash
docker compose -f docker/docker-compose.yml down -v
```

---

## Production Checklist

### Security

- [ ] Set `API_KEY` in `.env` — all `/api/v1/*` endpoints require `X-API-Key` header
- [ ] Use IAM roles instead of static AWS credentials (set no key/secret, use instance profile)
- [ ] Store secrets in your cloud provider's secret manager (AWS Secrets Manager, Azure Key Vault)
- [ ] Enable HTTPS termination at the load balancer / ingress

### Performance

- [ ] Set `UVICORN_WORKERS` to `2 * CPU_cores + 1` in the Dockerfile CMD
- [ ] Tune `MILVUS_EMBEDDING_DIM` to match your model (1536 for Titan v2 / text-embedding-3-large)
- [ ] Increase Milvus HNSW `efConstruction` for higher recall at the cost of slower index builds
- [ ] Use connection pooling for Milvus client in high-throughput deployments

### Observability

- [ ] Set `LOG_LEVEL=INFO` (or `DEBUG` for troubleshooting)
- [ ] Pipe `structlog` JSON output to your log aggregation platform
- [ ] Monitor Milvus collection stats via `GET /api/v1/collections` (add this endpoint)
- [ ] Alert on API error rate > 1%

### Scaling

The API is stateless — all state lives in Milvus. Scale horizontally:
```bash
docker compose -f docker/docker-compose.yml up --scale api=4
```

For Milvus at scale, migrate from standalone to Milvus cluster mode (separate index nodes, query nodes, data nodes).

---

## Switching LLM Provider

```bash
# .env
LLM_PROVIDER=bedrock  # was azure_openai

# Restart the API
uvicorn api.main:app --reload
```

No code changes. The `ProviderFactory` reads the env var at startup and returns the correct `LLMProvider` implementation.

---

## Embedding Dimension Mismatch

If you switch embedding models after the collection is created, you must recreate it (new dimension):

```bash
# Drop the existing collection via pymilvus REPL or Milvus Attu UI
# Then restart the API — it will recreate the collection with the new dimension
```

**Important:** Set `MILVUS_EMBEDDING_DIM` in `.env` to match the new model before restarting.

| Model | Dimension |
|---|---|
| `text-embedding-ada-002` | 1536 |
| `text-embedding-3-large` | 1536 (default), up to 3072 with `dimensions` param |
| `text-embedding-3-small` | 1536 |
| `amazon.titan-embed-text-v2:0` | 1024 or 1536 (configurable) |

---

## Milvus Backup

Milvus data is persisted in Docker volumes. Backup:

```bash
# Stop Milvus
docker compose stop milvus

# Backup MinIO data (Milvus object storage)
docker run --rm -v milvus_data:/data -v $(pwd)/backup:/backup alpine \
    tar czf /backup/milvus-backup-$(date +%Y%m%d).tar.gz /data

# Restart Milvus
docker compose start milvus
```
