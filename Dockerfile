# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------
# Stage 1 — build the React frontend into static assets
# ---------------------------------------------------------------------------
FROM node:20-slim AS frontend
WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm install
COPY frontend/ ./
RUN npm run build

# ---------------------------------------------------------------------------
# Stage 2 — Python app (FastAPI + serving), serves the built SPA on one port
# ---------------------------------------------------------------------------
FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_SYSTEM_PYTHON=1 \
    UV_LINK_MODE=copy

# uv for fast, reproducible installs.
RUN pip install --no-cache-dir uv

WORKDIR /app

# Resolve deps first for better layer caching.
COPY pyproject.toml README.md ./
COPY uv.lock* ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-dev --extra ragas --frozen

# Runtime assets.
COPY configs ./configs

COPY --from=frontend /frontend/dist ./frontend/dist
EXPOSE 8080
CMD ["uv", "run", "uvicorn", "ragbench.serving.app:app", "--host", "0.0.0.0", "--port", "8080"]
