"""Shared dense + sparse encoders.

These are cached process-wide so that a single ``bge-large-en-v1.5`` (1.3 GB)
is loaded once and reused by both the indexer and every dense/hybrid pipeline
across datasets. That mirrors mlserve's "load the weights once, compare the
serving path" philosophy.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from ragbench.common.logging import get_logger

log = get_logger(__name__)

# --- bge-large-en-v1.5 asymmetric encoding -------------------------------------
# NOTE: The BGE v1.5 family expects the retrieval *instruction* on the QUERY side
# and NO instruction on passages. (The project spec's table lists these swapped;
# we follow the model card, which is what produces correct MTEB-level numbers.)
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
BGE_PASSAGE_PREFIX = ""


class DenseEmbedder:
    """Wraps a SentenceTransformer with L2-normalized, prefixed encoding."""

    def __init__(self, model_name: str, device: str = "cpu"):
        from sentence_transformers import SentenceTransformer

        log.info("Loading dense model '%s' on %s", model_name, device)
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        self.use_bge_prefix = "bge" in model_name.lower()

    def _prefix(self, texts: list[str], prefix: str) -> list[str]:
        if not (self.use_bge_prefix and prefix):
            return texts
        return [prefix + t for t in texts]

    def encode_passages(self, texts: list[str], batch_size: int = 128) -> np.ndarray:
        return self.model.encode(
            self._prefix(texts, BGE_PASSAGE_PREFIX),
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 1000,
        )

    def encode_query(self, text: str) -> np.ndarray:
        vec = self.model.encode(
            self._prefix([text], BGE_QUERY_PREFIX),
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vec[0]

    def encode_queries(self, texts: list[str], batch_size: int = 128) -> np.ndarray:
        return self.model.encode(
            self._prefix(texts, BGE_QUERY_PREFIX),
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 1000,
        )


class SparseEncoder:
    """FastEmbed BM25 sparse encoder used for Qdrant server-side hybrid fusion."""

    def __init__(self, model_name: str = "Qdrant/bm25"):
        from fastembed import SparseTextEmbedding

        log.info("Loading sparse model '%s'", model_name)
        self.model_name = model_name
        self.model = SparseTextEmbedding(model_name=model_name)

    def encode_documents(self, texts: list[str]):
        """Yield fastembed ``SparseEmbedding`` objects (``.indices`` / ``.values``)."""
        return self.model.embed(texts)

    def encode_query(self, text: str):
        return next(iter(self.model.query_embed(text)))


@lru_cache(maxsize=4)
def get_dense_embedder(model_name: str, device: str = "cpu") -> DenseEmbedder:
    return DenseEmbedder(model_name, device=device)


@lru_cache(maxsize=4)
def get_sparse_encoder(model_name: str = "Qdrant/bm25") -> SparseEncoder:
    return SparseEncoder(model_name)
