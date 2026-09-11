# RAGBench — HuggingFace Spaces image.
#
# Same pattern as mlserve: build the index at image-build time so the Space boots
# ready-to-query. We use **SciFact only** (5K corpus, ~30s build). FiQA's 57K
# corpus is too large/slow to index at Spaces build time — run it locally with
# `make index-fiqa` against a real Qdrant instead.
#
# The whole demo runs in ONE container: instead of a separate Qdrant service we
# point QDRANT_URL at a filesystem path, which QdrantClient serves as a persisted
# *embedded* instance (see ragbench.common.clients.get_qdrant_client). HF_SPACE=true
# switches the app to SciFact-only mode.
#
# Build context must be the repository ROOT:
#   docker build -f deploy/huggingface/Dockerfile -t ragbench-space .

# ---------------------------------------------------------------------------
# Stage 1 — build the React frontend
# ---------------------------------------------------------------------------
FROM node:20-slim AS frontend
WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm install
COPY frontend/ ./
RUN npm run build

# ---------------------------------------------------------------------------
# Stage 2 — Python app + prebuilt SciFact index
# ---------------------------------------------------------------------------
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    HF_SPACE=true \
    QDRANT_URL=/app/data/qdrant_local \
    RAGBENCH_DEVICE=cpu \
    HF_HOME=/app/.cache/huggingface

RUN pip install --no-cache-dir uv

WORKDIR /app

COPY pyproject.toml README.md ./
COPY uv.lock* ./
COPY src ./src
RUN uv sync --no-dev

COPY configs ./configs
COPY --from=frontend /frontend/dist ./frontend/dist

# Build the SciFact index (downloads bge-large + dataset, writes bm25s + embedded
# Qdrant vectors into /app/data). Baked into the image so the Space starts hot.
RUN uv run python -m ragbench.indexing.build --config configs/scifact.yaml

# HuggingFace Spaces expects the app on port 7860.
EXPOSE 7860
CMD ["uv", "run", "uvicorn", "ragbench.serving.app:app", "--host", "0.0.0.0", "--port", "7860"]
