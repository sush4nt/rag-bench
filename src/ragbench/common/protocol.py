"""V2-style request/response protocol shared by the server and eval harness.

Mirrors mlserve's typed contract: a single ``RetrieveRequest`` in, a
``RetrieveResponse`` out, with per-result scoring and timing baked in.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

PipelineName = Literal["bm25", "dense", "hybrid", "reranked"]


class RetrieveRequest(BaseModel):
    """A single retrieval request against one pipeline."""

    query: str = Field(..., min_length=1, description="Natural-language query text")
    pipeline: PipelineName = Field(..., description="Which retrieval strategy to run")
    top_k: int = Field(10, ge=1, le=100, description="Number of results to return")


class BatchRetrieveRequest(BaseModel):
    """Run one query across several pipelines in a single round trip."""

    query: str = Field(..., min_length=1)
    pipelines: list[PipelineName] = Field(
        default_factory=lambda: ["bm25", "dense", "hybrid", "reranked"]
    )
    top_k: int = Field(10, ge=1, le=100)


class RetrieveResult(BaseModel):
    """A single retrieved passage."""

    doc_id: str
    score: float
    title: str = ""
    text: str = ""


class RetrieveResponse(BaseModel):
    """Ranked results plus timing/metadata for one pipeline run."""

    query: str
    pipeline: str
    dataset: str
    latency_ms: float
    results: list[RetrieveResult]
    # Populated only by the reranked pipeline; None otherwise.
    reranker_latency_ms: float | None = None


class BatchRetrieveResponse(BaseModel):
    """Side-by-side results for several pipelines on the same query."""

    query: str
    dataset: str
    responses: list[RetrieveResponse]


class AskRequest(BaseModel):
    """Retrieve with one pipeline, then generate a cited answer.

    The generation model is fixed server-side (``generation.model`` in the
    dataset config) so every pipeline is compared with the same LLM.
    """

    query: str = Field(..., min_length=1)
    pipeline: PipelineName
    top_k: int | None = Field(None, ge=1, le=20, description="Passages fed to the LLM")


class Citation(BaseModel):
    """A ``[n]`` marker in the answer, resolved to a retrieved passage."""

    marker: int
    doc_id: str
    context_index: int


class TokenUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0


class StageTimings(BaseModel):
    retrieval_ms: float  # first-stage retrieval only (excludes reranking)
    rerank_ms: float | None = None
    generation_ms: float
    total_ms: float


class AskResponse(BaseModel):
    query: str
    dataset: str
    pipeline: str
    model: str
    answer: str
    abstained: bool
    citations: list[Citation]
    invalid_citations: list[int]  # markers that point at no retrieved passage
    contexts: list[RetrieveResult]
    timings: StageTimings
    token_usage: TokenUsage
