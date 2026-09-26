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
    """Online RAG request: retrieve, then generate a cited answer."""

    query: str = Field(..., min_length=1, description="Natural-language question")
    pipeline: PipelineName = Field(
        "reranked",
        description="Retrieval strategy. 'reranked' is hybrid retrieval plus a cross-encoder.",
    )
    top_k: int = Field(5, ge=1, le=100, description="Passages passed to the generator")
    generation_model: str | None = Field(
        None,
        description="Chat model for the answer. Defaults to evaluation.ragas_llm in the dataset config.",
    )


class Citation(BaseModel):
    """A retrieved passage the answer may cite. ``index`` matches ``[n]`` in the answer."""

    index: int = Field(..., ge=1)
    doc_id: str
    title: str = ""
    score: float
    text: str = ""


class TokenUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


class AskResponse(BaseModel):
    """Grounded answer, the passages behind it, and a stage-level timing breakdown."""

    query: str
    answer: str
    pipeline: str
    dataset: str
    generation_model: str
    citations: list[Citation]
    retrieved_chunks: list[RetrieveResult]
    retrieval_ms: float
    rerank_ms: float
    generation_ms: float
    total_ms: float
    token_usage: TokenUsage
