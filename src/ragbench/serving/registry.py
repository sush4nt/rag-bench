"""Pipeline registry — lazy, cached construction per (dataset, pipeline).

Pipelines are expensive to build (they load models / open Qdrant), so we build
each one on first request and cache it. Configs are loaded once at import from
``configs/*.yaml``. Honors ``HF_SPACE=true`` by exposing only SciFact.
"""

from __future__ import annotations

import os
import threading

from ragbench.common.logging import get_logger
from ragbench.common.paths import bm25_index_dir, repo_root
from ragbench.config.schema import RagbenchConfig, load_config
from ragbench.retrieval.base import Pipeline
from ragbench.retrieval.factory import PIPELINE_CAPABILITIES, build_pipeline

log = get_logger(__name__)

_CONFIG_FILES = {"scifact": "configs/scifact.yaml", "fiqa": "configs/fiqa.yaml"}


def _hf_space_mode() -> bool:
    return os.environ.get("HF_SPACE", "false").lower() in ("1", "true", "yes")


class PipelineRegistry:
    def __init__(self):
        self._lock = threading.Lock()
        self._pipelines: dict[tuple[str, str], Pipeline] = {}
        self._configs: dict[str, RagbenchConfig] = {}
        for name, rel in _CONFIG_FILES.items():
            path = repo_root() / rel
            if path.exists():
                self._configs[name] = load_config(path)

    # --- datasets ------------------------------------------------------------
    def available_datasets(self) -> list[str]:
        datasets = list(self._configs)
        if _hf_space_mode():
            return [d for d in datasets if d == "scifact"]
        return datasets

    def is_available(self, dataset: str) -> bool:
        return dataset in self.available_datasets()

    def get_config(self, dataset: str) -> RagbenchConfig:
        if dataset not in self._configs:
            raise KeyError(f"Unknown dataset '{dataset}'")
        return self._configs[dataset]

    # --- pipelines -----------------------------------------------------------
    def get_pipeline(self, dataset: str, name: str) -> Pipeline:
        key = (dataset, name)
        if key in self._pipelines:
            return self._pipelines[key]
        with self._lock:
            if key in self._pipelines:  # double-checked locking
                return self._pipelines[key]
            cfg = self.get_config(dataset)
            log.info("Lazily building pipeline '%s' for dataset '%s'", name, dataset)
            pipeline = build_pipeline(name, cfg)
            self._pipelines[key] = pipeline
            return pipeline

    def pipeline_capabilities(self) -> dict[str, dict[str, str]]:
        return PIPELINE_CAPABILITIES

    # --- status --------------------------------------------------------------
    def status(self, dataset: str) -> dict:
        cfg = self.get_config(dataset)
        bm25_ready = (bm25_index_dir(dataset) / "params.index.json").exists() or (
            bm25_index_dir(dataset) / "data.csc.index.npy"
        ).exists()

        qdrant_count = 0
        qdrant_ready = False
        try:
            from ragbench.common.clients import get_qdrant_client

            client = get_qdrant_client()
            collection = cfg.qdrant.dense_collection
            if client.collection_exists(collection):
                qdrant_count = client.count(collection, exact=True).count
                qdrant_ready = qdrant_count > 0
        except Exception as exc:  # noqa: BLE001 - Qdrant may be down
            log.debug("Qdrant status check failed for %s: %s", dataset, exc)

        return {
            "dataset": dataset,
            "bm25_index_ready": bm25_ready,
            "qdrant_ready": qdrant_ready,
            "index_size_passages": qdrant_count,
            "embedding_model": cfg.indexing.embedding_model,
            "ready": bm25_ready or qdrant_ready,
        }


_REGISTRY: PipelineRegistry | None = None


def get_registry() -> PipelineRegistry:
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = PipelineRegistry()
    return _REGISTRY
