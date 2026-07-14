# syntax=docker/dockerfile:1
FROM python:3.12-slim AS base

# faiss-cpu and mlflow pull in a few native deps; keep the image slim but complete.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY data/raw/ ./data/raw/

# Build the FAISS index at image-build time so the container starts ready to
# serve. Uses local-tfidf by default (no model download / no network needed
# during the build) -- override EMBEDDING_PROVIDER at build time if you want
# a sentence-transformers image instead (needs network during build).
ARG EMBEDDING_PROVIDER=local-tfidf
ENV EMBEDDING_PROVIDER=${EMBEDDING_PROVIDER}
RUN python -m src.ingest.build_index

ENV LLM_PROVIDER=anthropic
ENV PYTHONUNBUFFERED=1

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
