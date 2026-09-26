"""POST /api/{dataset}/ask contract tests. Generation is stubbed; retrieval is real BM25."""

from __future__ import annotations

from ragbench.generation.answer import GenerationFailed, GenerationResult


class _FakeGenerator:
    def generate(self, question: str, contexts: list[str]) -> GenerationResult:
        cited = " ".join(f"[{i}]" for i in range(1, len(contexts) + 1))
        return GenerationResult(
            text=f"Grounded answer {cited}".strip(),
            model="fake-model",
            input_tokens=11,
            output_tokens=7,
        )


def test_ask_returns_answer_citations_and_timings(api_client, monkeypatch):
    monkeypatch.setattr("ragbench.serving.ask.build_generator", lambda model: _FakeGenerator())
    r = api_client.post(
        "/api/scifact/ask",
        json={"query": "aspirin colorectal cancer", "pipeline": "bm25", "top_k": 3},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["pipeline"] == "bm25"
    assert body["dataset"] == "scifact"
    assert body["generation_model"] == "fake-model"
    assert body["answer"].startswith("Grounded answer")
    assert len(body["citations"]) == 3
    assert len(body["retrieved_chunks"]) == 3
    assert [c["index"] for c in body["citations"]] == [1, 2, 3]
    assert body["citations"][0]["doc_id"] == body["retrieved_chunks"][0]["doc_id"]
    assert body["rerank_ms"] == 0
    assert body["retrieval_ms"] >= 0
    assert body["generation_ms"] >= 0
    assert body["total_ms"] >= 0
    assert body["token_usage"] == {
        "input_tokens": 11,
        "output_tokens": 7,
        "total_tokens": 18,
    }


def test_ask_without_api_key_is_503(api_client, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    r = api_client.post(
        "/api/scifact/ask",
        json={"query": "aspirin colorectal cancer", "pipeline": "bm25", "top_k": 2},
    )
    assert r.status_code == 503
    assert "ANTHROPIC_API_KEY" in r.json()["detail"]


def test_ask_generation_failure_is_502(api_client, monkeypatch):
    class _Boom:
        def generate(self, question: str, contexts: list[str]):
            raise GenerationFailed("provider down")

    monkeypatch.setattr("ragbench.serving.ask.build_generator", lambda model: _Boom())
    r = api_client.post(
        "/api/scifact/ask",
        json={"query": "aspirin colorectal cancer", "pipeline": "bm25", "top_k": 2},
    )
    assert r.status_code == 502
    assert "provider down" in r.json()["detail"]


def test_ask_unknown_pipeline_is_422(api_client):
    r = api_client.post(
        "/api/scifact/ask",
        json={"query": "aspirin", "pipeline": "hybrid_reranked", "top_k": 2},
    )
    assert r.status_code == 422
