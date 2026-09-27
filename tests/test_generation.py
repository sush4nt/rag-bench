"""Unit tests for the shared generation module (no network, fake LLM)."""

from __future__ import annotations

import pytest

from ragbench.common.protocol import AskRequest, RetrieveResult
from ragbench.config.schema import GenerationConfig
from ragbench.generation import (
    ABSTAIN_TOKEN,
    FakeGenerator,
    GenerationUnavailable,
    answer_question,
    build_generator,
    build_prompt,
    is_abstention,
    parse_citations,
)

CONTEXTS = [
    RetrieveResult(doc_id="d1", score=3.0, title="Aspirin", text="Aspirin reduces cancer."),
    RetrieveResult(doc_id="d2", score=2.0, title="", text="Vitamin D prevents infections."),
    RetrieveResult(doc_id="d3", score=1.0, title="Diet", text="Diet reduces risk."),
]


def test_parse_citations_valid_invalid_and_order():
    citations, invalid = parse_citations("Claim A [2]. Claim B [1][2]. Bogus [7].", CONTEXTS)
    assert [c.marker for c in citations] == [2, 1]
    assert [c.doc_id for c in citations] == ["d2", "d1"]
    assert citations[0].context_index == 1
    assert invalid == [7]


def test_parse_citations_grouped_markers():
    citations, invalid = parse_citations("Both [1, 3] and [0].", CONTEXTS)
    assert [c.marker for c in citations] == [1, 3]
    assert invalid == [0]


def test_parse_citations_none():
    assert parse_citations("No citations here.", CONTEXTS) == ([], [])


def test_build_prompt_numbers_and_truncates():
    long = [RetrieveResult(doc_id="x", score=1.0, title="T", text="word " * 500)]
    prompt = build_prompt("q?", CONTEXTS + long, max_chars=50)
    assert "[1] Aspirin\nAspirin reduces cancer." in prompt.user
    assert "[2] Vitamin D prevents infections." in prompt.user
    assert "[4] T\n" in prompt.user
    assert prompt.user.count("word") < 20
    assert ABSTAIN_TOKEN in prompt.system
    assert prompt.user.endswith("Question: q?\nAnswer:")


def test_is_abstention():
    assert is_abstention(f"  {ABSTAIN_TOKEN}")
    assert not is_abstention("Aspirin helps [1].")


def test_build_generator_disabled_raises():
    with pytest.raises(GenerationUnavailable, match="disabled"):
        build_generator(GenerationConfig(enabled=False))


def test_build_generator_anthropic_without_key(monkeypatch):
    monkeypatch.setenv("RAGBENCH_GENERATION_PROVIDER", "anthropic")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(GenerationUnavailable, match="ANTHROPIC_API_KEY"):
        build_generator(GenerationConfig())


def test_answer_question_end_to_end(bm25_pipeline):
    cfg = GenerationConfig(top_k_default=3)
    resp = answer_question(
        bm25_pipeline,
        FakeGenerator(),
        AskRequest(query="aspirin colorectal cancer", pipeline="bm25"),
        cfg,
    )
    assert resp.pipeline == "bm25"
    assert len(resp.contexts) == 3
    assert not resp.abstained
    assert [c.marker for c in resp.citations] == [1, 2]
    assert resp.citations[0].doc_id == resp.contexts[0].doc_id
    assert resp.invalid_citations == []
    assert resp.timings.rerank_ms is None
    assert resp.timings.total_ms >= resp.timings.retrieval_ms
    assert resp.token_usage.input_tokens > 0


def test_answer_question_skips_llm_when_nothing_retrieved(bm25_pipeline, monkeypatch):
    class ExplodingGenerator(FakeGenerator):
        def generate(self, prompt):  # pragma: no cover - must not be called
            raise AssertionError("LLM called with no context")

    monkeypatch.setattr(bm25_pipeline, "_search", lambda q, k: [])
    resp = answer_question(
        bm25_pipeline, ExplodingGenerator(), AskRequest(query="x", pipeline="bm25"),
        GenerationConfig(),
    )
    assert resp.abstained
    assert resp.token_usage.input_tokens == 0
    assert resp.timings.generation_ms == 0.0
