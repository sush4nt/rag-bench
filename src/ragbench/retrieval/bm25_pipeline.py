"""BM25 pipeline — sparse lexical retrieval via bm25s.

Loads the persisted bm25s index (with corpus payloads) at construction time.
Fast, no GPU, fully offline — the latency floor of the benchmark.
"""

from __future__ import annotations

from ragbench.common.logging import get_logger
from ragbench.common.paths import bm25_index_dir
from ragbench.common.protocol import RetrieveResult
from ragbench.indexing.bm25_indexer import _make_stemmer, tokenize
from ragbench.retrieval.base import Pipeline

log = get_logger(__name__)


class BM25Pipeline(Pipeline):
    name = "bm25"

    def __init__(self, config):
        super().__init__(config)
        import bm25s

        path = bm25_index_dir(self.dataset)
        log.info("Loading bm25s index for '%s' from %s", self.dataset, path)
        self.retriever = bm25s.BM25.load(str(path), load_corpus=True)
        self.stemmer = _make_stemmer()

    def _search(self, query: str, top_k: int) -> list[RetrieveResult]:
        query_tokens = tokenize([query], stemmer=self.stemmer)
        # k cannot exceed the corpus size.
        k = min(top_k, self.retriever.scores["num_docs"])
        docs, scores = self.retriever.retrieve(query_tokens, k=k, show_progress=False)

        results: list[RetrieveResult] = []
        for doc, score in zip(docs[0], scores[0], strict=False):
            results.append(
                RetrieveResult(
                    doc_id=str(doc["doc_id"]),
                    score=float(score),
                    title=doc.get("title", ""),
                    text=doc.get("text", ""),
                )
            )
        return results
