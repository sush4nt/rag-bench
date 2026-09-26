"""Serving-performance benchmark: small top_k and concurrency, no IR metrics."""

from __future__ import annotations

import json

from ragbench.evaluation.serving_bench import (
    benchmark_pipeline,
    select_queries,
    serving_result_path,
)


def test_select_queries_is_reproducible():
    queries = {f"q{i}": f"question {i}" for i in range(20)}
    a = select_queries(queries, max_queries=5, seed=42)
    b = select_queries(queries, max_queries=5, seed=42)
    assert a == b
    assert len(a) == 5
    assert select_queries(queries, max_queries=0, seed=1) == dict(sorted(queries.items()))


def test_benchmark_pipeline_separates_top_k_and_concurrency(bm25_pipeline):
    queries = {
        "q1": "aspirin colorectal cancer",
        "q2": "vitamin d infections",
        "q3": "mediterranean diet",
        "q4": "statins cholesterol",
    }
    summary = benchmark_pipeline(
        bm25_pipeline,
        queries,
        top_ks=[2, 4],
        concurrencies=[1, 2],
        warmup=0,
    )
    assert set(summary["top_k"]) == {"2", "4"}
    for top_k in ("2", "4"):
        cells = summary["top_k"][top_k]["concurrency"]
        assert set(cells) == {"1", "2"}
        for stats in cells.values():
            assert stats["requests"] == 4
            assert stats["errors"] == 0
            assert stats["error_rate"] == 0
            assert stats["p50_ms"] <= stats["p95_ms"] <= stats["p99_ms"]
            assert stats["qps"] > 0
            assert "process_peak_rss_mb" in stats
            assert "cpu_util" in stats
    # IR metrics do not belong on a serving cell.
    assert "ndcg@10" not in summary["top_k"]["2"]["concurrency"]["1"]


def test_serving_latest_endpoint(api_client):
    missing = api_client.get("/api/scifact/serving/latest")
    assert missing.status_code == 200
    assert missing.json()["available"] is False

    path = serving_result_path("scifact")
    path.write_text(
        json.dumps(
            {
                "dataset": "scifact",
                "benchmark": "serving_performance",
                "pipelines": {
                    "bm25": {"top_k": {"10": {"concurrency": {"1": {"p95_ms": 1.2, "qps": 100}}}}}
                },
            }
        ),
        encoding="utf-8",
    )
    found = api_client.get("/api/scifact/serving/latest")
    assert found.status_code == 200
    body = found.json()
    assert body["available"] is True
    assert body["benchmark"] == "serving_performance"
    assert body["pipelines"]["bm25"]["top_k"]["10"]["concurrency"]["1"]["qps"] == 100
