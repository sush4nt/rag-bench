"""Passage normalization.

BEIR passages are already document-sized "chunks", so we don't split further —
we normalize them into a single searchable string. Keeping this in one place
guarantees the *same* text is fed to bm25s, the dense embedder, and the sparse
encoder, which is essential for a fair side-by-side benchmark.
"""

from __future__ import annotations

import re

_WS_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Collapse whitespace and strip; keep content otherwise intact."""
    if not text:
        return ""
    return _WS_RE.sub(" ", text).strip()


def passage_text(doc: dict[str, str]) -> str:
    """Combine a passage's title and body into one normalized string.

    ``"{title}. {text}"`` when a title is present, else just the body. This is
    the canonical text used for indexing and for the ``text`` field returned to
    clients.
    """
    title = normalize_text(doc.get("title", ""))
    body = normalize_text(doc.get("text", ""))
    if title and body:
        return f"{title}. {body}"
    return body or title


def iter_passages(corpus: dict[str, dict[str, str]]):
    """Yield ``(doc_id, title, body, combined_text)`` for every passage.

    Deterministic ordering (sorted by doc_id) so that index build order and the
    integer point ids assigned in Qdrant are reproducible across runs.
    """
    for doc_id in sorted(corpus.keys()):
        doc = corpus[doc_id]
        title = normalize_text(doc.get("title", ""))
        body = normalize_text(doc.get("text", ""))
        yield doc_id, title, body, passage_text(doc)
