"""Hybrid pipeline — dense + sparse fused server-side by Qdrant (RRF).

Both the dense and sparse vectors live in the same collection, so a single
``query_points`` round trip with two prefetches + a ``FusionQuery(RRF)`` does the
whole job. No client-side fusion glue.
"""

from __future__ import annotations

from ragbench.common.clients import get_qdrant_client
from ragbench.common.protocol import RetrieveResult
from ragbench.indexing.qdrant_indexer import DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME
from ragbench.retrieval.base import Pipeline
from ragbench.retrieval.dense_pipeline import _to_result
from ragbench.retrieval.embedder import get_dense_embedder, get_sparse_encoder

# How many candidates each branch contributes before fusion.
MIN_PREFETCH = 50


class HybridPipeline(Pipeline):
    name = "hybrid"

    def __init__(self, config):
        super().__init__(config)
        self.collection = config.qdrant.dense_collection
        self.client = get_qdrant_client()
        self.embedder = get_dense_embedder(
            config.indexing.embedding_model, device=config.indexing.device
        )
        self.sparse = get_sparse_encoder(config.indexing.sparse_model)

    def _search(self, query: str, top_k: int) -> list[RetrieveResult]:
        from qdrant_client.models import Fusion, FusionQuery, Prefetch, SparseVector

        prefetch_limit = max(MIN_PREFETCH, top_k)
        dense_vec = self.embedder.encode_query(query)
        sparse_emb = self.sparse.encode_query(query)

        response = self.client.query_points(
            collection_name=self.collection,
            prefetch=[
                Prefetch(
                    query=dense_vec.tolist(),
                    using=DENSE_VECTOR_NAME,
                    limit=prefetch_limit,
                ),
                Prefetch(
                    query=SparseVector(
                        indices=sparse_emb.indices.tolist(),
                        values=sparse_emb.values.tolist(),
                    ),
                    using=SPARSE_VECTOR_NAME,
                    limit=prefetch_limit,
                ),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=top_k,
            with_payload=True,
        )
        return [_to_result(p) for p in response.points]
