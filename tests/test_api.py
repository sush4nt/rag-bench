"""API contract tests via FastAPI TestClient (bm25 over the tiny fixture index)."""

from __future__ import annotations


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


def test_missing_index_returns_503(api_client):
    # FiQA bm25 index was never built in the test data dir -> pipeline build fails.
    r = api_client.post(
        "/api/fiqa/retrieve",
        json={"query": "etf vs index fund", "pipeline": "bm25", "top_k": 3},
    )
    assert r.status_code == 503
