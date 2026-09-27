"""API contract tests via FastAPI TestClient (bm25 over the tiny fixture index)."""

from __future__ import annotations

from conftest import HEAVY


def test_health(api_client):
    r = api_client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_retrieve_returns_valid_schema(api_client):
    r = api_client.post(
        "/api/scifact/retrieve",
        json={"query": "aspirin colorectal cancer", "pipeline": "bm25", "top_k": 3},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["pipeline"] == "bm25"
    assert body["dataset"] == "scifact"
    assert "latency_ms" in body
    assert len(body["results"]) == 3
    first = body["results"][0]
    assert {"doc_id", "score", "title", "text"} <= set(first)


def test_status_returns_index_stats(api_client):
    r = api_client.get("/api/scifact/status")
    assert r.status_code == 200
    body = r.json()
    assert body["dataset"] == "scifact"
    assert body["bm25_index_ready"] is True
    assert "index_size_passages" in body


def test_pipelines_lists_all_four(api_client):
    r = api_client.get("/api/scifact/pipelines")
    assert r.status_code == 200
    names = {p["name"] for p in r.json()["pipelines"]}
    assert names == {"bm25", "dense", "hybrid", "reranked"}


def test_unknown_pipeline_is_422(api_client):
    r = api_client.post(
        "/api/scifact/retrieve",
        json={"query": "x", "pipeline": "does-not-exist", "top_k": 3},
    )
    assert r.status_code == 422


def test_batch_runs_selected_pipelines(api_client):
    r = api_client.post(
        "/api/scifact/retrieve/batch",
        json={"query": "vitamin d infections", "pipelines": ["bm25"], "top_k": 2},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["dataset"] == "scifact"
    assert len(body["responses"]) == 1
    assert body["responses"][0]["pipeline"] == "bm25"


def test_datasets_meta_endpoint(api_client):
    r = api_client.get("/api/datasets")
    assert r.status_code == 200
    names = {d["name"] for d in r.json()["datasets"]}
    assert "scifact" in names


def test_generation_info(api_client):
    r = api_client.get("/api/scifact/generation")
    assert r.status_code == 200
    body = r.json()
    assert body["available"] is True
    assert body["provider"] == "fake"
    assert body["model"] == "gpt-5-nano"


def test_ask_returns_cited_answer(api_client):
    r = api_client.post(
        "/api/scifact/ask",
        json={"query": "aspirin colorectal cancer", "pipeline": "bm25", "top_k": 3},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["pipeline"] == "bm25"
    assert body["dataset"] == "scifact"
    assert body["answer"]
    assert body["abstained"] is False
    assert len(body["contexts"]) == 3
    context_ids = [c["doc_id"] for c in body["contexts"]]
    for cit in body["citations"]:
        assert context_ids[cit["context_index"]] == cit["doc_id"]
    assert body["invalid_citations"] == []
    t = body["timings"]
    assert t["rerank_ms"] is None
    assert {"retrieval_ms", "generation_ms", "total_ms"} <= set(t)
    assert body["token_usage"]["input_tokens"] > 0


def test_ask_uses_config_top_k_by_default(api_client):
    r = api_client.post("/api/scifact/ask", json={"query": "vitamin d", "pipeline": "bm25"})
    assert r.status_code == 200
    assert len(r.json()["contexts"]) == 5


def test_ask_rejects_unknown_pipeline(api_client):
    r = api_client.post("/api/scifact/ask", json={"query": "x", "pipeline": "hybrid_reranked"})
    assert r.status_code == 422


def test_ask_503_when_generation_disabled(api_client, monkeypatch):
    from ragbench.serving.registry import get_registry

    registry = get_registry()
    cfg = registry._configs["scifact"]
    disabled = cfg.model_copy(
        update={"generation": cfg.generation.model_copy(update={"enabled": False})}
    )
    monkeypatch.setitem(registry._configs, "scifact", disabled)
    monkeypatch.setattr(registry, "_generators", {})

    r = api_client.post("/api/scifact/ask", json={"query": "x", "pipeline": "bm25"})
    assert r.status_code == 503
    assert "disabled" in r.json()["detail"]
    assert api_client.get("/api/scifact/generation").json()["available"] is False


@HEAVY
def test_ask_reranked_reports_rerank_stage(api_client):
    r = api_client.post(
        "/api/scifact/ask", json={"query": "aspirin colorectal cancer", "pipeline": "reranked"}
    )
    assert r.status_code == 200
    assert r.json()["timings"]["rerank_ms"] > 0


def test_missing_index_returns_503(api_client):
    # FiQA bm25 index was never built in the test data dir -> pipeline build fails.
    r = api_client.post(
        "/api/fiqa/retrieve",
        json={"query": "etf vs index fund", "pipeline": "bm25", "top_k": 3},
    )
    assert r.status_code == 503
