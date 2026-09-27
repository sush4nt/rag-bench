"""End-to-end ask: retrieve -> (rerank) -> build context -> generate -> cite."""

from __future__ import annotations

from time import perf_counter

from ragbench.common.protocol import (
    AskRequest,
    AskResponse,
    RetrieveRequest,
    StageTimings,
    TokenUsage,
)
from ragbench.config.schema import GenerationConfig
from ragbench.generation.citations import parse_citations
from ragbench.generation.generator import Generator
from ragbench.generation.prompt import ABSTAIN_TOKEN, build_prompt, is_abstention
from ragbench.retrieval.base import Pipeline


def answer_question(
    pipeline: Pipeline, generator: Generator, request: AskRequest, cfg: GenerationConfig
) -> AskResponse:
    t0 = perf_counter()
    top_k = request.top_k or cfg.top_k_default

    retrieval = pipeline.retrieve(
        RetrieveRequest(query=request.query, pipeline=request.pipeline, top_k=top_k)
    )
    rerank_ms = retrieval.reranker_latency_ms
    retrieval_ms = retrieval.latency_ms - (rerank_ms or 0.0)
    contexts = retrieval.results

    if contexts:
        result = generator.generate(build_prompt(request.query, contexts, cfg.max_context_chars))
        answer, model = result.text, result.model
        generation_ms = result.latency_ms
        usage = TokenUsage(input_tokens=result.input_tokens, output_tokens=result.output_tokens)
    else:
        # Nothing to ground on: skip the (paid) LLM call entirely.
        answer, model, generation_ms, usage = ABSTAIN_TOKEN, generator.model, 0.0, TokenUsage()

    abstained = is_abstention(answer)
    citations, invalid = ([], []) if abstained else parse_citations(answer, contexts)

    return AskResponse(
        query=request.query,
        dataset=retrieval.dataset,
        pipeline=retrieval.pipeline,
        model=model,
        answer=answer,
        abstained=abstained,
        citations=citations,
        invalid_citations=invalid,
        contexts=contexts,
        timings=StageTimings(
            retrieval_ms=retrieval_ms,
            rerank_ms=rerank_ms,
            generation_ms=generation_ms,
            total_ms=(perf_counter() - t0) * 1000.0,
        ),
        token_usage=usage,
    )
