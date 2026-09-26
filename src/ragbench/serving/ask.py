"""Online retrieve → generate → cite path used by ``POST /ask``."""

from __future__ import annotations

from time import perf_counter

from ragbench.common.protocol import (
    AskRequest,
    AskResponse,
    Citation,
    RetrieveRequest,
    TokenUsage,
)
from ragbench.generation.answer import GenerationResult, build_generator
from ragbench.serving import metrics
from ragbench.serving.registry import get_registry


class PipelineUnavailable(RuntimeError):
    """The requested retrieval pipeline could not run."""


def run_ask(dataset: str, request: AskRequest, generator=None) -> AskResponse:
    """Retrieve, generate a grounded answer, and return citations plus timings."""
    registry = get_registry()
    cfg = registry.get_config(dataset)
    model = (request.generation_model or "").strip() or cfg.evaluation.ragas_llm

    try:
        pipeline = registry.get_pipeline(dataset, request.pipeline)
    except Exception as exc:  # noqa: BLE001 - missing index or model
        raise PipelineUnavailable(
            f"Pipeline '{request.pipeline}' unavailable for '{dataset}': {exc}"
        ) from exc

    if generator is None:
        generator = build_generator(model)

    t_total = perf_counter()
    try:
        retrieved = pipeline.retrieve(
            RetrieveRequest(query=request.query, pipeline=request.pipeline, top_k=request.top_k)
        )
    except Exception as exc:  # noqa: BLE001
        raise PipelineUnavailable(f"Retrieval failed: {exc}") from exc

    rerank_ms = float(retrieved.reranker_latency_ms or 0.0)
    retrieval_ms = max(retrieved.latency_ms - rerank_ms, 0.0)

    citations = [
        Citation(
            index=i,
            doc_id=hit.doc_id,
            title=hit.title,
            score=hit.score,
            text=hit.text,
        )
        for i, hit in enumerate(retrieved.results, start=1)
    ]

    if citations:
        t_gen = perf_counter()
        generated: GenerationResult = generator.generate(request.query, [c.text for c in citations])
        generation_ms = (perf_counter() - t_gen) * 1000.0
        answer = generated.text
        usage = TokenUsage(
            input_tokens=generated.input_tokens,
            output_tokens=generated.output_tokens,
            total_tokens=generated.input_tokens + generated.output_tokens,
        )
        model = generated.model or model
    else:
        generation_ms = 0.0
        answer = "No passages were retrieved for this query."
        usage = TokenUsage()

    total_ms = (perf_counter() - t_total) * 1000.0
    metrics.observe_retrieval(
        dataset,
        request.pipeline,
        retrieved.latency_ms / 1000.0,
        (rerank_ms / 1000.0) if retrieved.reranker_latency_ms is not None else None,
    )
    metrics.observe_ask(
        dataset,
        request.pipeline,
        retrieval_s=retrieval_ms / 1000.0,
        rerank_s=(rerank_ms / 1000.0) if rerank_ms else None,
        generation_s=generation_ms / 1000.0,
        total_s=total_ms / 1000.0,
    )
    return AskResponse(
        query=request.query,
        answer=answer,
        pipeline=request.pipeline,
        dataset=dataset,
        generation_model=model,
        citations=citations,
        retrieved_chunks=retrieved.results,
        retrieval_ms=round(retrieval_ms, 3),
        rerank_ms=round(rerank_ms, 3),
        generation_ms=round(generation_ms, 3),
        total_ms=round(total_ms, 3),
        token_usage=usage,
    )
