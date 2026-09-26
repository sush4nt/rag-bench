"""Serving-performance benchmark, separate from retrieval-quality evaluation.

Quality evaluation retrieves at ``max(k_values)`` (usually 100) so Recall@100
is defined. A serving request uses a small ``top_k``. This module measures
latency, throughput, error rate, CPU, and process memory at serving cutoffs
under a fixed concurrency. It does not compute NDCG or recall.

    python -m ragbench.evaluation.serving_bench --config configs/scifact.yaml --pipelines bm25
"""

from __future__ import annotations

import argparse
import json
import random
import resource
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

from ragbench.common.logging import get_logger
from ragbench.common.paths import dataset_dir, ensure_dir
from ragbench.common.protocol import RetrieveRequest
from ragbench.config.schema import load_config
from ragbench.data.loader import load_dataset
from ragbench.evaluation.retrieval_eval import latency_percentiles
from ragbench.retrieval.factory import build_pipeline

log = get_logger(__name__)

DEFAULT_TOP_K = (5, 10, 20)
# 50 in-flight CPU embeddings oversubscribed this machine (one dense cell of 100
# queries took about 7 minutes). Pass ``--concurrency 1 10 50`` to include it.
DEFAULT_CONCURRENCY = (1, 10)


def serving_result_path(dataset: str) -> Path:
    return dataset_dir(dataset) / "serving_latest.json"


def _peak_rss_mb() -> float:
    """Process high-water RSS in MiB. On macOS ``ru_maxrss`` is bytes; on Linux, KB."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return rss / (1024 * 1024)
    return rss / 1024


def select_queries(queries: dict[str, str], max_queries: int | None, seed: int) -> dict[str, str]:
    """Reproducible sample. ``max_queries`` of 0 or None keeps every query."""
    items = sorted(queries.items())
    if max_queries and 0 < max_queries < len(items):
        picked = random.Random(seed).sample(items, max_queries)
        items = sorted(picked)
    return dict(items)


def _measure_cell(
    pipeline, query_texts: list[str], top_k: int, concurrency: int, warmup: int
) -> dict:
    warm_n = min(max(warmup, 0), len(query_texts))
    for text in query_texts[:warm_n]:
        try:
            pipeline.retrieve(RetrieveRequest(query=text, pipeline=pipeline.name, top_k=top_k))
        except Exception as exc:  # noqa: BLE001 - warmup is best-effort
            log.debug("Warmup query failed for '%s': %s", pipeline.name, exc)

    latencies: list[float] = []
    errors = 0
    cpu0 = time.process_time()
    wall0 = time.perf_counter()

    def _service(text: str) -> float:
        """Service time once a worker has the request. Excludes time waiting to start."""
        t0 = time.perf_counter()
        pipeline.retrieve(RetrieveRequest(query=text, pipeline=pipeline.name, top_k=top_k))
        return (time.perf_counter() - t0) * 1000.0

    if concurrency <= 1:
        for text in query_texts:
            try:
                latencies.append(_service(text))
            except Exception as exc:  # noqa: BLE001
                errors += 1
                log.debug("Serving request failed (%s): %s", pipeline.name, exc)
    else:
        # ``concurrency`` workers pull the query list, so latency is service time
        # while that many requests are in flight. Throughput uses the wall clock.
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [pool.submit(_service, text) for text in query_texts]
            for fut in as_completed(futures):
                try:
                    latencies.append(fut.result())
                except Exception as exc:  # noqa: BLE001
                    errors += 1
                    log.debug("Serving request failed (%s): %s", pipeline.name, exc)

    wall = max(time.perf_counter() - wall0, 1e-9)
    cpu = time.process_time() - cpu0
    stats = latency_percentiles(latencies)
    stats.update(
        {
            "qps": round(len(latencies) / wall, 3),
            "error_rate": round(errors / max(len(query_texts), 1), 5),
            "errors": errors,
            "requests": len(query_texts),
            "cpu_util": round(cpu / wall, 3),
            "process_peak_rss_mb": round(_peak_rss_mb(), 1),
        }
    )
    return stats


def benchmark_pipeline(
    pipeline,
    queries: dict[str, str],
    top_ks: list[int],
    concurrencies: list[int],
    warmup: int = 2,
) -> dict:
    """Run every (top_k, concurrency) cell for one already-built pipeline."""
    texts = list(queries.values())
    out: dict[str, dict] = {}
    for top_k in top_ks:
        cells: dict[str, dict] = {}
        for concurrency in concurrencies:
            log.info(
                "  serving '%s' top_k=%s concurrency=%s queries=%s",
                pipeline.name,
                top_k,
                concurrency,
                len(texts),
            )
            cells[str(concurrency)] = _measure_cell(pipeline, texts, top_k, concurrency, warmup)
        out[str(top_k)] = {"concurrency": cells}
    return {"top_k": out}


def run_serving_bench(
    config_path: str,
    pipelines: list[str] | None = None,
    top_ks: list[int] | None = None,
    concurrencies: list[int] | None = None,
    max_queries: int | None = 100,
    seed: int = 42,
    warmup: int = 2,
) -> dict:
    """Load the dataset, benchmark each pipeline, and write ``serving_latest.json``."""
    cfg = load_config(config_path)
    names = pipelines or cfg.evaluation.pipelines
    top_ks = list(top_ks or DEFAULT_TOP_K)
    concurrencies = list(concurrencies or DEFAULT_CONCURRENCY)
    _corpus, queries, _qrels = load_dataset(
        cfg.dataset.beir_name, split=cfg.dataset.split, local_path=cfg.dataset.local_path
    )
    chosen = select_queries(queries, max_queries, seed)
    summary = {
        "dataset": cfg.dataset.name,
        "benchmark": "serving_performance",
        "timestamp": datetime.now(UTC).isoformat(),
        "seed": seed,
        "num_queries": len(chosen),
        "warmup": warmup,
        "top_k": top_ks,
        "concurrency": concurrencies,
        "pipelines": {},
    }
    out_path = serving_result_path(cfg.dataset.name)
    ensure_dir(out_path.parent)

    def _write() -> None:
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    for name in names:
        log.info("--- Serving bench '%s' on '%s' ---", name, cfg.dataset.name)
        try:
            pipeline = build_pipeline(name, cfg)
        except Exception as exc:  # noqa: BLE001 - one missing index should not drop the rest
            log.warning("Skipping pipeline '%s': %s", name, exc)
            summary["pipelines"][name] = {"error": str(exc)}
            _write()
            continue
        summary["pipelines"][name] = benchmark_pipeline(
            pipeline, chosen, top_ks, concurrencies, warmup=warmup
        )
        _write()

    log.info("Wrote serving summary -> %s", out_path)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the RAGBench serving-performance benchmark.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--pipelines", nargs="*", default=None)
    parser.add_argument("--top-k", nargs="*", type=int, default=list(DEFAULT_TOP_K))
    parser.add_argument("--concurrency", nargs="*", type=int, default=list(DEFAULT_CONCURRENCY))
    parser.add_argument(
        "--max-queries",
        type=int,
        default=100,
        help="Query sample size. 0 uses every query.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=2)
    args = parser.parse_args(argv)
    if any(k < 1 or k > 100 for k in args.top_k):
        parser.error("--top-k values must be between 1 and 100")
    if any(c < 1 for c in args.concurrency):
        parser.error("--concurrency values must be >= 1")
    run_serving_bench(
        args.config,
        pipelines=args.pipelines,
        top_ks=args.top_k,
        concurrencies=args.concurrency,
        max_queries=args.max_queries,
        seed=args.seed,
        warmup=args.warmup,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
