"""bm25s lexical index builder.

We use ``bm25s`` (not Qdrant sparse vectors) for the pure-lexical BM25 pipeline:
it's faster to build, fully offline, and more faithful to classical BM25. The
index and a parallel corpus of ``{doc_id, title, text}`` records are persisted so
the serving pipeline can return payloads without a second lookup.
"""

from __future__ import annotations

from ragbench.common.logging import get_logger
from ragbench.common.paths import bm25_index_dir, ensure_dir
from ragbench.data.chunker import iter_passages
from ragbench.data.loader import Corpus
from ragbench.indexing.base import Indexer

log = get_logger(__name__)

STEMMER_LANG = "english"
STOPWORDS = "en"


def _make_stemmer():
    try:
        import Stemmer  # PyStemmer

        return Stemmer.Stemmer(STEMMER_LANG)
    except ImportError:  # pragma: no cover - PyStemmer is a declared dep
        log.warning("PyStemmer not available; tokenizing without stemming")
        return None


def tokenize(texts, stemmer=None):
    """Tokenize with bm25s using the shared stopword list + stemmer."""
    import bm25s

    return bm25s.tokenize(texts, stopwords=STOPWORDS, stemmer=stemmer, show_progress=False)


class BM25Indexer(Indexer):
    def index_path(self):
        return bm25_index_dir(self.dataset)

    def exists(self) -> bool:
        return (self.index_path() / "params.index.json").exists() or (
            self.index_path() / "data.csc.index.npy"
        ).exists()

    def build(self, corpus: Corpus) -> int:
        import bm25s

        path = ensure_dir(self.index_path())
        log.info("Building bm25s index for '%s' (%d passages)", self.dataset, len(corpus))

        records: list[dict] = []
        texts: list[str] = []
        for doc_id, title, body, combined in iter_passages(corpus):
            records.append({"doc_id": doc_id, "title": title, "text": body})
            texts.append(combined)

        stemmer = _make_stemmer()
        corpus_tokens = tokenize(texts, stemmer=stemmer)

        retriever = bm25s.BM25(corpus=records)
        retriever.index(corpus_tokens)
        retriever.save(str(path), corpus=records)

        log.info("Saved bm25s index -> %s", path)
        return len(records)
