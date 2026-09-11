"""Pipeline behaviour tests (bm25 always runs; heavy pipelines self-skip)."""

from __future__ import annotations

import pytest

from conftest import HEAVY
from ragbench.common.protocol import RetrieveRequest, RetrieveResult


def test_bm25_returns_top_k(bm25_pipeline):
    req = RetrieveRequest(query="aspirin colorectal cancer", pipeline="bm25", top_k=3)
    resp = bm25_pipeline.retrieve(req)

    assert resp.pipeline == "bm25"
    assert resp.dataset == "scifact"
    assert len(resp.results) == 3
    assert resp.latency_ms >= 0.0


def test_bm25_result_fields_and_ranking(bm25_pipeline):
    req = RetrieveRequest(query="aspirin colorectal cancer", pipeline="bm25", top_k=5)
    resp = bm25_pipeline.retrieve(req)

    top = resp.results[0]
    assert isinstance(top, RetrieveResult)
    assert top.doc_id and top.text
    # The aspirin/cancer passage should rank first for this query.
    assert top.doc_id == "sf_1"
    # Scores are monotonically non-increasing.
    scores = [r.score for r in resp.results]
    assert scores == sorted(scores, reverse=True)


def test_bm25_top_k_capped_to_corpus(bm25_pipeline):
    req = RetrieveRequest(query="cancer", pipeline="bm25", top_k=100)
    resp = bm25_pipeline.retrieve(req)
    assert 0 < len(resp.results) <= 10  # tiny corpus has 10 docs


def test_bm25_retrieve_batch_shape(bm25_pipeline):
    queries = {"q1": "aspirin cancer", "q2": "vitamin d infections"}
    out = bm25_pipeline.retrieve_batch(queries, top_k=3)
    assert set(out) == {"q1", "q2"}
    for _qid, docs in out.items():
        assert isinstance(docs, dict)
        assert all(isinstance(v, float) for v in docs.values())


def test_reranked_reports_reranker_latency_is_none_for_bm25(bm25_pipeline):
    resp = bm25_pipeline.retrieve(
        RetrieveRequest(query="statins cholesterol", pipeline="bm25", top_k=2)
    )
    assert resp.reranker_latency_ms is None


@HEAVY
@pytest.mark.parametrize("name", ["dense", "hybrid", "reranked"])
def test_heavy_pipelines(name, scifact_config):
    from ragbench.retrieval.factory import build_pipeline

    pipeline = build_pipeline(name, scifact_config)
    resp = pipeline.retrieve(RetrieveRequest(query="aspirin cancer", pipeline=name, top_k=5))
    assert len(resp.results) <= 5
    if name == "reranked":
        assert resp.reranker_latency_ms is not None
