"""Dense pipeline — single-vector ANN over Qdrant.

Uses ``BAAI/bge-large-en-v1.5`` (1024-dim, cosine) with the asymmetric query
instruction applied by :class:`DenseEmbedder`.
"""

from __future__ import annotations

from ragbench.common.clients import get_qdrant_client
from ragbench.common.protocol import RetrieveResult
from ragbench.indexing.qdrant_indexer import DENSE_VECTOR_NAME
from ragbench.retrieval.base import Pipeline
from ragbench.retrieval.embedder import get_dense_embedder


def _to_result(point) -> RetrieveResult:
    payload = point.payload or {}
    return RetrieveResult(
        doc_id=str(payload.get("doc_id", point.id)),
        score=float(point.score),
        title=payload.get("title", ""),
        text=payload.get("text", ""),
    )


class DensePipeline(Pipeline):
    name = "dense"

    def __init__(self, config):
        super().__init__(config)
        self.collection = config.qdrant.dense_collection
        self.client = get_qdrant_client()
        self.embedder = get_dense_embedder(
            config.indexing.embedding_model, device=config.indexing.device
        )

    def _search(self, query: str, top_k: int) -> list[RetrieveResult]:
        vec = self.embedder.encode_query(query)
        response = self.client.query_points(
            collection_name=self.collection,
            query=vec.tolist(),
            using=DENSE_VECTOR_NAME,
            limit=top_k,
            with_payload=True,
        )
        return [_to_result(p) for p in response.points]
