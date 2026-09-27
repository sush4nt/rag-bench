"""End-to-end eval runner: retrieval metrics (+ optional RAGAS) -> MLflow + JSON.

Each pipeline produces one MLflow run under experiment ``ragbench/{dataset}``.
Results are also written to ``data/{dataset}/eval_latest.json`` so the serving
layer can seed its Prometheus gauges and the frontend MetricsPanel on startup.

    python -m ragbench.evaluation.runner --config configs/fiqa.yaml
    python -m ragbench.evaluation.runner --config configs/scifact.yaml --no-ragas
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from ragbench.common.clients import mlflow_tracking_uri
from ragbench.common.logging import get_logger
from ragbench.common.paths import dataset_dir, ensure_dir
from ragbench.common.protocol import RetrieveRequest
from ragbench.config.schema import RagbenchConfig, load_config
from ragbench.data.loader import load_dataset, load_qa_answers
from ragbench.evaluation.retrieval_eval import evaluate_retrieval, latency_percentiles
from ragbench.retrieval.factory import build_pipeline

log = get_logger(__name__)


def eval_result_path(dataset: str) -> Path:
    return dataset_dir(dataset) / "eval_latest.json"


def _headline_metrics(metrics: dict[str, dict[str, float]], latency: dict[str, float]) -> dict:
    """Flatten the metrics needed by gauges / the frontend."""
    return {
        "ndcg@10": metrics["ndcg"].get("NDCG@10", 0.0),
        "mrr@10": metrics["mrr"].get("MRR@10", 0.0),
        "recall@10": metrics["recall"].get("Recall@10", 0.0),
        "recall@100": metrics["recall"].get("Recall@100", 0.0),
        "map@100": metrics["map"].get("MAP@100", 0.0),
        "precision@10": metrics["precision"].get("Precision@10", 0.0),
        **latency,
    }


def _run_pipeline_retrieval(pipeline, queries: dict[str, str], qrels, k_values):
    """Run every query once via ``retrieve`` to capture BOTH results and latency."""
    results: dict[str, dict[str, float]] = {}
    latencies: list[float] = []
    top_k = max(k_values)
    for qid, text in queries.items():
        resp = pipeline.retrieve(
            RetrieveRequest(query=text, pipeline=pipeline.name, top_k=top_k)
        )
        results[qid] = {r.doc_id: r.score for r in resp.results}
        latencies.append(resp.latency_ms)
    metrics = evaluate_retrieval(qrels, results, k_values)
    return metrics, latency_percentiles(latencies)


def _log_to_mlflow(cfg: RagbenchConfig, pipeline_name: str, headline: dict, ragas_means: dict):
    try:
        import mlflow

        mlflow.set_tracking_uri(mlflow_tracking_uri())
        mlflow.set_experiment(f"ragbench/{cfg.dataset.name}")
        with mlflow.start_run(run_name=f"{cfg.dataset.name}_{pipeline_name}"):
            mlflow.set_tag("dataset", cfg.dataset.name)
            mlflow.set_tag("pipeline", pipeline_name)
            mlflow.set_tag("embedding_model", cfg.indexing.embedding_model)
            for key, val in headline.items():
                mlflow.log_metric(key.replace("@", "_at_"), float(val))
            for key, val in ragas_means.items():
                mlflow.log_metric(key, float(val))
    except Exception as exc:  # noqa: BLE001 - MLflow is best-effort in CLI/CI
        log.warning("MLflow logging skipped (%s)", exc)


def run_eval(
    config_path: str,
    pipelines: list[str] | None = None,
    with_ragas: bool = True,
    log_mlflow: bool = True,
) -> dict:
    cfg = load_config(config_path)
    pipeline_names = pipelines or cfg.evaluation.pipelines
    k_values = cfg.evaluation.k_values

    corpus, queries, qrels = load_dataset(
        cfg.dataset.beir_name, split=cfg.dataset.split, local_path=cfg.dataset.local_path
    )

    summary = {
        "dataset": cfg.dataset.name,
        "timestamp": datetime.now(UTC).isoformat(),
        "num_queries": len(queries),
        "num_passages": len(corpus),
        "pipelines": {},
    }

    for name in pipeline_names:
        log.info("--- Evaluating pipeline '%s' on '%s' ---", name, cfg.dataset.name)
        t0 = time.perf_counter()
        pipeline = build_pipeline(name, cfg)
        metrics, latency = _run_pipeline_retrieval(pipeline, queries, qrels, k_values)
        headline = _headline_metrics(metrics, latency)
        log.info(
            "  '%s': NDCG@10=%.4f MRR@10=%.4f Recall@10=%.4f p95=%.1fms (%.1fs)",
            name,
            headline["ndcg@10"],
            headline["mrr@10"],
            headline["recall@10"],
            headline["p95_ms"],
            time.perf_counter() - t0,
        )

        ragas_means: dict[str, float] = {}
        if with_ragas:
            ragas_means = _maybe_run_ragas(cfg, pipeline, corpus, queries, qrels)

        if log_mlflow:
            _log_to_mlflow(cfg, name, headline, ragas_means)

        summary["pipelines"][name] = {
            **headline,
            "metrics": metrics,
            "ragas": ragas_means,
        }

    out_path = eval_result_path(cfg.dataset.name)
    ensure_dir(out_path.parent)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info("Wrote eval summary -> %s", out_path)
    return summary


def _maybe_run_ragas(cfg, pipeline, corpus, queries, qrels) -> dict[str, float]:
    try:
        from ragbench.evaluation.generation_eval import build_qa_samples, run_ragas_eval

        answers = load_qa_answers(cfg.dataset.name, cfg.dataset.qa_hf_dataset)
        samples = build_qa_samples(
            queries, qrels, corpus, answers, limit=cfg.evaluation.ragas_sample_size
        )
        df = run_ragas_eval(
            pipeline,
            samples,
            ragas_llm_model=cfg.evaluation.ragas_llm,
            generation_cfg=cfg.generation,
            sample_size=cfg.evaluation.ragas_sample_size,
        )
        cols = ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]
        return {c: float(df[c].mean()) for c in cols if c in df.columns}
    except Exception as exc:  # noqa: BLE001 - RAGAS is opt-in / may lack creds
        log.warning("RAGAS eval skipped for '%s' (%s)", pipeline.name, exc)
        return {}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run RAGBench evaluation.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--pipelines", nargs="*", default=None)
    parser.add_argument("--no-ragas", action="store_true", help="Skip RAGAS generation eval")
    parser.add_argument("--no-mlflow", action="store_true", help="Skip MLflow logging")
    args = parser.parse_args(argv)
    run_eval(
        args.config,
        pipelines=args.pipelines,
        with_ragas=not args.no_ragas,
        log_mlflow=not args.no_mlflow,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
