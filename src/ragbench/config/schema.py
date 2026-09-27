"""Typed configuration models loaded from ``configs/*.yaml``.

One YAML file fully describes a dataset: how to download it, how to index it,
how to evaluate it, and how to serve it. Everything downstream (indexers,
pipelines, eval, routers) is driven by this object.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class DatasetConfig(BaseModel):
    name: str
    beir_name: str
    split: str = "test"
    local_path: str
    qa_hf_dataset: str | None = None


class QdrantConfig(BaseModel):
    collection_prefix: str

    @property
    def dense_collection(self) -> str:
        """Physical collection holding both named 'dense' and 'sparse' vectors."""
        return f"{self.collection_prefix}_dense"

    @property
    def sparse_collection(self) -> str:
        """Logical name for the sparse vectors (co-located in the dense collection)."""
        return f"{self.collection_prefix}_sparse"


class IndexingConfig(BaseModel):
    embedding_model: str = "BAAI/bge-large-en-v1.5"
    vector_size: int = 1024
    batch_size: int = 128
    device: str = "cpu"
    sparse_model: str = "Qdrant/bm25"


class EvaluationConfig(BaseModel):
    k_values: list[int] = Field(default_factory=lambda: [1, 5, 10, 100])
    ragas_llm: str = "claude-haiku-4-5-20251001"
    ragas_sample_size: int = 100
    pipelines: list[str] = Field(
        default_factory=lambda: ["bm25", "dense", "hybrid", "reranked"]
    )


class ServingConfig(BaseModel):
    router_prefix: str
    top_k_default: int = 10
    rerank_multiplier: int = 5


class RerankerConfig(BaseModel):
    model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class GenerationConfig(BaseModel):
    """Answer generation for ``POST /ask``.

    One pinned model serves every pipeline so answers differ only by retrieval.
    Offline RAGAS stays on ``evaluation.ragas_llm`` (Anthropic) and does not
    follow this provider.
    """

    enabled: bool = True
    provider: Literal["anthropic", "openai", "fake"] = "anthropic"
    model: str = "claude-haiku-4-5-20251001"
    top_k_default: int = 5
    max_tokens: int = 400
    temperature: float = 0.0
    # gpt-5-nano spends this budget on hidden reasoning unless effort is minimal.
    reasoning_effort: Literal["none", "minimal", "low", "medium", "high"] | None = None
    max_context_chars: int = 2000


class RagbenchConfig(BaseModel):
    dataset: DatasetConfig
    qdrant: QdrantConfig
    indexing: IndexingConfig
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    serving: ServingConfig
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)

    @property
    def name(self) -> str:
        return self.dataset.name


def load_config(path: str | Path) -> RagbenchConfig:
    """Parse and validate a dataset config YAML into a :class:`RagbenchConfig`."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return RagbenchConfig.model_validate(raw)
