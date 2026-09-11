"""Abstract retrieval pipeline.

The base class owns the cross-cutting concerns — timing, response assembly, and
the batch loop used by the eval harness — so each concrete pipeline only has to
implement ``_search(query, top_k) -> list[RetrieveResult]``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from time import perf_counter

from ragbench.common.protocol import RetrieveRequest, RetrieveResponse, RetrieveResult
from ragbench.config.schema import RagbenchConfig


class Pipeline(ABC):
    name: str  # "bm25" | "dense" | "hybrid" | "reranked"

    def __init__(self, config: RagbenchConfig):
        self.config = config
        self.dataset = config.dataset.name

    # --- to implement --------------------------------------------------------
    @abstractmethod
    def _search(self, query: str, top_k: int) -> list[RetrieveResult]:
        """Return ranked results (highest score first) for a single query."""
        ...

    # --- shared API ----------------------------------------------------------
    def retrieve(self, request: RetrieveRequest) -> RetrieveResponse:
        t0 = perf_counter()
        results = self._search(request.query, request.top_k)
        latency_ms = (perf_counter() - t0) * 1000.0
        return RetrieveResponse(
            query=request.query,
            pipeline=self.name,
            dataset=self.dataset,
            latency_ms=latency_ms,
            results=results,
            reranker_latency_ms=self._pop_reranker_latency(),
        )

    def retrieve_batch(
        self, queries: dict[str, str] | list[str], top_k: int
    ) -> dict[str, dict[str, float]]:
        """Run many queries; return ``{query_id: {doc_id: score}}``.

        This is the shape BEIR's evaluator expects. Accepts either a
        ``{query_id: text}`` mapping (preferred, from the eval harness) or a bare
        list of strings (keys become the string index).
        """
        items = queries.items() if isinstance(queries, dict) else enumerate(queries)
        out: dict[str, dict[str, float]] = {}
        for qid, text in items:
            results = self._search(text, top_k)
            out[str(qid)] = {r.doc_id: r.score for r in results}
        return out

    # --- optional hooks ------------------------------------------------------
    def _pop_reranker_latency(self) -> float | None:
        """Overridden by the reranked pipeline to report cross-encoder time."""
        return None

    def warmup(self) -> None:
        """Optionally run a throwaway query so first real request isn't cold."""
        try:
            self._search("warmup query", 1)
        except Exception:  # noqa: BLE001 - warmup is best-effort
            pass
