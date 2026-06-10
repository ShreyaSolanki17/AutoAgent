# ── Stage 1: Builder ──────────────────────────────────────
# Install dependencies and pre-download the embedding model
FROM python:3.12-slim AS builder

WORKDIR /build

# Install system deps for building Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Pre-download the sentence-transformers model (~80MB)
# This avoids a 30-second cold start on every container startup
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"


# ── Stage 2: Runtime ─────────────────────────────────────
FROM python:3.12-slim

# Install runtime system deps only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

# Copy pre-downloaded model cache from builder
COPY --from=builder /root/.cache/torch /root/.cache/torch
COPY --from=builder /root/.cache/huggingface /root/.cache/huggingface

# Copy application code
COPY backend/ /app/backend/
COPY docker/ /app/docker/

WORKDIR /app

# Switch to non-root user
USER appuser

EXPOSE 8000

# Health check — verifies the API is responding
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')" || exit 1

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
