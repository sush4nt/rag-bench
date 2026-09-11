"""Qdrant index builder — dense + sparse vectors in one collection.

Design note (reconciling the spec)
-----------------------------------
The spec's naming table lists two physical collections (``*_dense`` and
``*_sparse``), but the hybrid query sample fuses named "dense" and "sparse"
vectors *within a single collection* (``ragbench_fiqa_dense``). Qdrant's
server-side RRF fusion requires both vectors to co-exist in one collection, so
we create a single collection named ``{prefix}_dense`` that carries BOTH a named
dense vector and a named sparse vector. This matches the working hybrid sample
and is the only layout that supports built-in fusion.
"""

from __future__ import annotations

from ragbench.common.clients import get_qdrant_client
from ragbench.common.logging import get_logger
from ragbench.data.chunker import iter_passages
from ragbench.data.loader import Corpus
from ragbench.indexing.base import Indexer
from ragbench.retrieval.embedder import get_dense_embedder, get_sparse_encoder

log = get_logger(__name__)

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"


class QdrantIndexer(Indexer):
    def __init__(self, config):
        super().__init__(config)
        self.collection = config.qdrant.dense_collection
        self.client = get_qdrant_client()

    def exists(self) -> bool:
        try:
            return self.client.collection_exists(self.collection)
        except Exception:  # noqa: BLE001 - Qdrant may be unreachable
            return False

    def _create_collection(self) -> None:
        from qdrant_client.models import (
            Distance,
            SparseVectorParams,
            VectorParams,
        )

        if self.client.collection_exists(self.collection):
            log.info("Dropping existing collection '%s'", self.collection)
            self.client.delete_collection(self.collection)

        self.client.create_collection(
            collection_name=self.collection,
            vectors_config={
                DENSE_VECTOR_NAME: VectorParams(
                    size=self.config.indexing.vector_size,
                    distance=Distance.COSINE,
                    on_disk=False,  # in-memory for benchmark latency
                )
            },
            sparse_vectors_config={SPARSE_VECTOR_NAME: SparseVectorParams()},
        )
        log.info("Created collection '%s'", self.collection)

    def build(self, corpus: Corpus) -> int:
        from qdrant_client.models import PointStruct, SparseVector

        self._create_collection()

        dense = get_dense_embedder(
            self.config.indexing.embedding_model, device=self.config.indexing.device
        )
        sparse = get_sparse_encoder(self.config.indexing.sparse_model)

        doc_ids, titles, bodies, texts = [], [], [], []
        for doc_id, title, body, combined in iter_passages(corpus):
            doc_ids.append(doc_id)
            titles.append(title)
            bodies.append(body)
            texts.append(combined)

        batch_size = self.config.indexing.batch_size
        total = len(texts)
        log.info("Encoding + upserting %d passages into '%s'", total, self.collection)

        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch_texts = texts[start:end]

            dense_vecs = dense.encode_passages(batch_texts, batch_size=batch_size)
            sparse_vecs = list(sparse.encode_documents(batch_texts))

            points = []
            for i, sparse_emb in enumerate(sparse_vecs):
                idx = start + i
                points.append(
                    PointStruct(
                        id=idx,
                        vector={
                            DENSE_VECTOR_NAME: dense_vecs[i].tolist(),
                            SPARSE_VECTOR_NAME: SparseVector(
                                indices=sparse_emb.indices.tolist(),
                                values=sparse_emb.values.tolist(),
                            ),
                        },
                        payload={
                            "doc_id": doc_ids[idx],
                            "title": titles[idx],
                            "text": bodies[idx],
                            "dataset": self.dataset,
                        },
                    )
                )
            self.client.upsert(collection_name=self.collection, points=points, wait=True)
            if start % (batch_size * 20) == 0:
                log.info("  upserted %d / %d", end, total)

        log.info("Finished indexing '%s' -> %d points", self.collection, total)
        return total

    def count(self) -> int:
        """Number of points currently stored (used for status/metrics)."""
        try:
            return self.client.count(self.collection, exact=True).count
        except Exception:  # noqa: BLE001
            return 0
