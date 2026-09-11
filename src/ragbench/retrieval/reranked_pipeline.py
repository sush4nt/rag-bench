"""Reranked pipeline — hybrid over-retrieval + cross-encoder reranking.

Step 1: run the hybrid pipeline with ``top_k * rerank_multiplier`` candidates.
Step 2: score each (query, passage) pair with a cross-encoder and keep top_k.

The cross-encoder adds ~60-120ms; we measure it separately and surface it as
``reranker_latency_ms`` for the Grafana "reranker overhead" panel.
"""

from __future__ import annotations

from functools import lru_cache
from time import perf_counter

from ragbench.common.logging import get_logger
from ragbench.common.protocol import RetrieveResult
from ragbench.retrieval.base import Pipeline
from ragbench.retrieval.hybrid_pipeline import HybridPipeline

log = get_logger(__name__)


@lru_cache(maxsize=2)
def get_cross_encoder(model_name: str):
    from sentence_transformers import CrossEncoder

    log.info("Loading cross-encoder '%s'", model_name)
    return CrossEncoder(model_name)


class RerankedPipeline(Pipeline):
    name = "reranked"

    def __init__(self, config):
        super().__init__(config)
        self.hybrid = HybridPipeline(config)
        self.reranker = get_cross_encoder(config.reranker.model)
        self.multiplier = config.serving.rerank_multiplier
        self._reranker_latency_ms: float | None = None

    def _search(self, query: str, top_k: int) -> list[RetrieveResult]:
        candidates = self.hybrid._search(query, top_k * self.multiplier)
        if not candidates:
            self._reranker_latency_ms = 0.0
            return []

        pairs = [(query, c.text) for c in candidates]
        t0 = perf_counter()
        scores = self.reranker.predict(pairs)
        self._reranker_latency_ms = (perf_counter() - t0) * 1000.0

        ranked = sorted(
            zip(candidates, scores, strict=True), key=lambda x: x[1], reverse=True
        )
        results: list[RetrieveResult] = []
        for cand, score in ranked[:top_k]:
            results.append(
                RetrieveResult(
                    doc_id=cand.doc_id,
                    score=float(score),
                    title=cand.title,
                    text=cand.text,
                )
            )
        return results

    def _pop_reranker_latency(self) -> float | None:
        val = self._reranker_latency_ms
        self._reranker_latency_ms = None
        return val
