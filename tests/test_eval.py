"""Evaluation harness tests: metric correctness + RAGAS plumbing."""

from __future__ import annotations

import pytest

from ragbench.evaluation.retrieval_eval import (
    evaluate_retrieval,
    latency_percentiles,
)

K = [1, 5, 10]


def test_perfect_ranking_scores_one():
    qrels = {"q1": {"a": 1, "b": 1}}
    results = {"q1": {"a": 3.0, "b": 2.0, "c": 1.0}}
    m = evaluate_retrieval(qrels, results, K)
    assert m["ndcg"]["NDCG@10"] == pytest.approx(1.0)
    assert m["mrr"]["MRR@10"] == pytest.approx(1.0)
    assert m["recall"]["Recall@10"] == pytest.approx(1.0)


def test_metrics_bounded_zero_to_one():
    qrels = {"q1": {"a": 1}, "q2": {"z": 1}}
    results = {"q1": {"b": 3.0, "a": 1.0}, "q2": {"y": 2.0, "x": 1.0}}
    m = evaluate_retrieval(qrels, results, K)
    for family in m.values():
        for val in family.values():
            assert 0.0 <= val <= 1.0


def test_mrr_first_relevant_at_rank_two():
    # Relevant doc 'a' sits at rank 2 -> MRR = 1/2.
    qrels = {"q1": {"a": 1}}
    results = {"q1": {"b": 5.0, "a": 4.0, "c": 1.0}}
    m = evaluate_retrieval(qrels, results, K)
    assert m["mrr"]["MRR@10"] == pytest.approx(0.5)


def test_recall_at_k_cutoff():
    # Two relevant docs; only one appears within top-1.
    qrels = {"q1": {"a": 1, "b": 1}}
    results = {"q1": {"a": 3.0, "c": 2.0, "b": 1.0}}
    m = evaluate_retrieval(qrels, results, [1, 5])
    assert m["recall"]["Recall@1"] == pytest.approx(0.5)
    assert m["recall"]["Recall@5"] == pytest.approx(1.0)


def test_ndcg_penalizes_lower_rank():
    qrels = {"q1": {"a": 1}}
    high = evaluate_retrieval(qrels, {"q1": {"a": 9.0, "b": 1.0}}, K)["ndcg"]["NDCG@10"]
    low = evaluate_retrieval(qrels, {"q1": {"b": 9.0, "a": 1.0}}, K)["ndcg"]["NDCG@10"]
    assert high > low


def test_latency_percentiles_ordering():
    p = latency_percentiles([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    assert p["p50_ms"] <= p["p95_ms"] <= p["p99_ms"]
    assert p["mean_ms"] == pytest.approx(55.0)


def test_latency_percentiles_empty():
    p = latency_percentiles([])
    assert p == {"p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0, "mean_ms": 0.0}


def test_quality_headline_labels_depth_latency():
    from ragbench.evaluation.runner import quality_headline

    headline = quality_headline(
        {"ndcg": {"NDCG@10": 0.5}, "mrr": {}, "recall": {}, "map": {}, "precision": {}},
        {"p50_ms": 1.0, "p95_ms": 2.0, "p99_ms": 3.0, "mean_ms": 1.5},
        retrieval_depth=100,
    )
    assert headline["latency_scope"] == "quality"
    assert headline["retrieval_depth"] == 100
    assert headline["p95_ms"] == 2.0
    assert headline["ndcg@10"] == 0.5


def test_quality_statements_follow_measured_direction():
    """SciFact numbers: dense leads @10; hybrid leads Recall@100; reranking is lower."""
    from ragbench.evaluation.report import quality_statements

    pipelines = {
        "bm25": {"ndcg@10": 0.6863, "recall@10": 0.81867, "recall@100": 0.91267},
        "dense": {"ndcg@10": 0.74099, "recall@10": 0.87656, "recall@100": 0.95167},
        "hybrid": {"ndcg@10": 0.72049, "recall@10": 0.84522, "recall@100": 0.95833},
        "reranked": {"ndcg@10": 0.66624, "recall@10": 0.79556, "recall@100": 0.937},
    }
    lines = quality_statements(pipelines, retrieval_depth=100, rerank_multiplier=5)
    text = " ".join(lines)
    assert "Dense NDCG@10 is 8.0% higher than BM25" in text
    assert "Dense Recall@10 is 7.1% higher than BM25" in text
    assert "Hybrid Recall@10 is 3.6% lower than dense" in text
    assert "Hybrid Recall@100 is 0.7% higher than dense" in text
    assert "Reranked NDCG@10 is 7.5% lower than hybrid" in text
    assert "retrieval depth 100" in text
    assert "500" in text
    assert "top_k 5, 10, and 20" in text


def test_build_qa_samples():
    from ragbench.evaluation.generation_eval import build_qa_samples

    queries = {"q1": "does aspirin reduce cancer?"}
    qrels = {"q1": {"sf_1": 1, "sf_x": 0}}
    corpus = {"sf_1": {"title": "Aspirin", "text": "Aspirin reduces cancer risk."}}
    samples = build_qa_samples(queries, qrels, corpus, answers={"q1": "Yes."})
    assert len(samples) == 1
    assert samples[0]["ground_truth"] == "Yes."
    assert samples[0]["gold_contexts"]


def test_ragas_requires_api_key(monkeypatch):
    """run_ragas_eval fails gracefully (clear error) without an API key."""
    from ragbench.evaluation.generation_eval import run_ragas_eval

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        run_ragas_eval(pipeline=object(), qa_samples=[], ragas_llm_model="x", sample_size=5)
