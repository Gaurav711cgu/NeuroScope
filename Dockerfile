FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend source
COPY backend/ ./backend/

# HF Spaces requires port 7860
ENV PORT=7860
ENV PYTHONPATH=/app/backend

# Pre-download Gemma-2-2b-it weights at build time to avoid cold-start delays.
# HF_TOKEN must be set as a Space secret (NOT hardcoded here).
# In HF Space: Settings → Secrets → Add secret HF_TOKEN
RUN --mount=type=secret,id=HF_TOKEN \
    HF_TOKEN=$(cat /run/secrets/HF_TOKEN 2>/dev/null || echo "") \
    python -c "
import os
token = os.environ.get('HF_TOKEN', '')
if token:
    from huggingface_hub import snapshot_download
    print('Pre-downloading google/gemma-2-2b-it weights...')
    snapshot_download('google/gemma-2-2b-it', token=token, ignore_patterns=['*.msgpack', '*.h5'])
    print('Done.')
else:
    print('HF_TOKEN not set — weights will be downloaded on first request.')
"

WORKDIR /app/backend

# Run the FastAPI server
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
