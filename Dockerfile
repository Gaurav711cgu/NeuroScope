# ─── Multi-Stage Production Dockerfile for NeuroScope v3 ────────────────────

# Stage 1: Build Dependencies
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Stage 2: Runtime Image
FROM python:3.11-slim AS runner

WORKDIR /app/backend

# Create non-root system user for security
RUN groupadd -g 10001 neuroscope && \
    useradd -u 10001 -g neuroscope -s /bin/bash -m neuroscope

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local
COPY backend/ /app/backend/

ENV PYTHONPATH=/app/backend
ENV PORT=8000
ENV PYTHONUNBUFFERED=1

RUN chown -R neuroscope:neuroscope /app
USER neuroscope

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
  CMD curl -f http://localhost:8000/api/health || exit 1

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
