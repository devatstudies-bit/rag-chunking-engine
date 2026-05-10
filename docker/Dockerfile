# ─────────────────────────────────────────────────────────────────────────────
# Multi-stage build for the Chunking Engine API
# ─────────────────────────────────────────────────────────────────────────────

# ── Stage 1: dependency builder ───────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: runtime image ────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL maintainer="Chunking Engine Team"
LABEL description="Production-grade adaptive document chunking engine"

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy source code
COPY src/ ./src/
COPY api/  ./api/
COPY pyproject.toml .

# Install the package itself
RUN pip install --no-cache-dir -e .

# Create non-root user
RUN groupadd --gid 1001 appgroup && \
    useradd  --uid 1001 --gid appgroup --shell /bin/bash --create-home appuser
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health').raise_for_status()"

CMD ["uvicorn", "api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "4", \
     "--loop", "uvloop"]
