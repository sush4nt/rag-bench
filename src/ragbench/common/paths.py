"""Centralized filesystem path resolution.

All runtime artifacts (downloaded datasets, bm25s indexes) live under a single
data root so that Docker volume mounts and HuggingFace Spaces builds stay simple.
"""

from __future__ import annotations

import os
from pathlib import Path

# repo root = three parents up from this file: common/ -> ragbench/ -> src/ -> root
_REPO_ROOT = Path(__file__).resolve().parents[3]


def repo_root() -> Path:
    """Absolute path to the repository root."""
    return _REPO_ROOT


def data_dir() -> Path:
    """Root directory for datasets and built indexes (override via RAGBENCH_DATA_DIR)."""
    override = os.environ.get("RAGBENCH_DATA_DIR")
    root = Path(override) if override else _REPO_ROOT / "data"
    if not root.is_absolute():
        root = _REPO_ROOT / root
    return root


def dataset_dir(dataset: str) -> Path:
    """Directory holding a single dataset's raw BEIR files (corpus/queries/qrels)."""
    return data_dir() / dataset


def bm25_index_dir(dataset: str) -> Path:
    """Directory where the persisted bm25s index for ``dataset`` lives."""
    return dataset_dir(dataset) / "bm25_index"


def ensure_dir(path: Path) -> Path:
    """Create ``path`` (and parents) if missing; return it for chaining."""
    path.mkdir(parents=True, exist_ok=True)
    return path
