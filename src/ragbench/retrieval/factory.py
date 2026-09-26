"""Pipeline name -> class mapping and a small factory."""

from __future__ import annotations

from ragbench.config.schema import RagbenchConfig
from ragbench.retrieval.base import Pipeline
from ragbench.retrieval.bm25_pipeline import BM25Pipeline
from ragbench.retrieval.dense_pipeline import DensePipeline
from ragbench.retrieval.hybrid_pipeline import HybridPipeline
from ragbench.retrieval.reranked_pipeline import RerankedPipeline

PIPELINE_CLASSES: dict[str, type[Pipeline]] = {
    "bm25": BM25Pipeline,
    "dense": DensePipeline,
    "hybrid": HybridPipeline,
    "reranked": RerankedPipeline,
}

# Human-readable capabilities surfaced by GET /api/{dataset}/pipelines.
PIPELINE_CAPABILITIES: dict[str, dict[str, str]] = {
    "bm25": {"strategy": "Sparse lexical (bm25s)", "trade_off": "Fastest; misses semantics"},
    "dense": {"strategy": "Single-vector ANN (bge-large)", "trade_off": "Semantic; slower"},
    "hybrid": {"strategy": "BM25 + Dense via RRF", "trade_off": "Fuses lexical and semantic ranks"},
    "reranked": {
        "strategy": "Hybrid + cross-encoder",
        "trade_off": "Rescores top_k × multiplier candidates; latency scales with that set",
    },
}


def build_pipeline(name: str, config: RagbenchConfig) -> Pipeline:
    if name not in PIPELINE_CLASSES:
        raise ValueError(f"Unknown pipeline '{name}'. Valid: {sorted(PIPELINE_CLASSES)}")
    return PIPELINE_CLASSES[name](config)
