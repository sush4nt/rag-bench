"""Prometheus metrics for the serving layer.

All metrics are labelled by ``dataset`` and (where relevant) ``pipeline`` so the
Grafana dashboard can slice the side-by-side comparison. Eval gauges are seeded
from ``eval_latest.json`` on startup and refreshed by ``POST /eval/run``.
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

# Latency buckets tuned for retrieval (1ms .. 5s).
_LATENCY_BUCKETS = (
    0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0, 2.0, 5.0,
)

RETRIEVE_LATENCY = Histogram(
    "ragbench_retrieve_latency_seconds",
    "End-to-end retrieval latency per pipeline",
    labelnames=("dataset", "pipeline"),
    buckets=_LATENCY_BUCKETS,
)
RETRIEVE_REQUESTS = Counter(
    "ragbench_retrieve_requests_total",
    "Total retrieval requests per pipeline",
    labelnames=("dataset", "pipeline"),
)
RERANKER_LATENCY = Histogram(
    "ragbench_reranker_latency_seconds",
    "Cross-encoder reranking latency (reranked pipeline only)",
    labelnames=("dataset", "pipeline"),
    buckets=_LATENCY_BUCKETS,
)
INDEX_SIZE = Gauge(
    "ragbench_index_size_passages",
    "Number of indexed passages per dataset",
    labelnames=("dataset",),
)

# Last-eval-run gauges (per dataset+pipeline).
EVAL_NDCG10 = Gauge("ragbench_eval_ndcg_at_10", "NDCG@10 from last eval run", ("dataset", "pipeline"))
EVAL_MRR10 = Gauge("ragbench_eval_mrr_at_10", "MRR@10 from last eval run", ("dataset", "pipeline"))
EVAL_RECALL10 = Gauge(
    "ragbench_eval_recall_at_10", "Recall@10 from last eval run", ("dataset", "pipeline")
)
RAGAS_FAITHFULNESS = Gauge(
    "ragbench_ragas_faithfulness", "Faithfulness from last RAGAS run", ("dataset", "pipeline")
)
RAGAS_ANSWER_RELEVANCE = Gauge(
    "ragbench_ragas_answer_relevance", "Answer relevance from last RAGAS run", ("dataset", "pipeline")
)


def observe_retrieval(
    dataset: str, pipeline: str, latency_s: float, reranker_latency_s: float | None = None
) -> None:
    RETRIEVE_REQUESTS.labels(dataset, pipeline).inc()
    RETRIEVE_LATENCY.labels(dataset, pipeline).observe(latency_s)
    if reranker_latency_s is not None:
        RERANKER_LATENCY.labels(dataset, pipeline).observe(reranker_latency_s)


def set_index_size(dataset: str, passages: int) -> None:
    INDEX_SIZE.labels(dataset).set(passages)


def set_eval_gauges(summary: dict) -> None:
    """Seed the eval/RAGAS gauges from an ``eval_latest.json`` summary dict."""
    dataset = summary.get("dataset")
    if not dataset:
        return
    for pipeline, data in summary.get("pipelines", {}).items():
        EVAL_NDCG10.labels(dataset, pipeline).set(data.get("ndcg@10", 0.0))
        EVAL_MRR10.labels(dataset, pipeline).set(data.get("mrr@10", 0.0))
        EVAL_RECALL10.labels(dataset, pipeline).set(data.get("recall@10", 0.0))
        ragas = data.get("ragas") or {}
        if "faithfulness" in ragas:
            RAGAS_FAITHFULNESS.labels(dataset, pipeline).set(ragas["faithfulness"])
        if "answer_relevancy" in ragas:
            RAGAS_ANSWER_RELEVANCE.labels(dataset, pipeline).set(ragas["answer_relevancy"])


def setup_instrumentator(app) -> None:
    """Attach the default FastAPI Prometheus instrumentator and expose /metrics."""
    from prometheus_fastapi_instrumentator import Instrumentator

    Instrumentator(
        should_group_status_codes=True,
        excluded_handlers=["/metrics", "/health"],
    ).instrument(app).expose(app, include_in_schema=False, endpoint="/metrics")
