"""Factories for external service clients (Qdrant, MLflow)."""

from __future__ import annotations

import os
from functools import lru_cache


@lru_cache(maxsize=4)
def get_qdrant_client(url: str | None = None):
    """Return a cached :class:`qdrant_client.QdrantClient`.

    ``QDRANT_URL`` resolution (arg -> env -> localhost) selects the mode:

    - ``http://...`` / ``https://...`` -> a real Qdrant server (Docker Compose).
    - ``:memory:``                     -> ephemeral embedded instance (tests).
    - any other value                  -> a filesystem path for a *persisted*
      embedded instance. This is what HuggingFace Spaces uses so the whole demo
      runs in a single container with no separate Qdrant service.
    """
    from qdrant_client import QdrantClient

    url = url or os.environ.get("QDRANT_URL", "http://localhost:6333")
    if url == ":memory:":
        return QdrantClient(location=":memory:")
    if url.startswith(("http://", "https://")):
        # prefer_grpc=False keeps the REST API path (matches the compose port map).
        return QdrantClient(url=url, prefer_grpc=False, timeout=120)
    # Embedded, persisted-to-disk mode.
    return QdrantClient(path=url)


def mlflow_tracking_uri() -> str:
    """Resolve the MLflow tracking URI from the environment."""
    return os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5001")
