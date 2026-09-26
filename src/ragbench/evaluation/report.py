"""Plain-language comparisons computed from a retrieval-quality summary.

Statements follow the measured direction. A negative change is reported as
lower, which is what the SciFact run actually shows for hybrid-vs-dense at
cutoff 10 and for reranking-vs-hybrid.
"""

from __future__ import annotations


def _rel(new: float, base: float) -> str:
    if base == 0:
        return "undefined relative to"
    pct = (new - base) / base * 100.0
    direction = "higher" if pct >= 0 else "lower"
    return f"{abs(pct):.1f}% {direction}"


def quality_statements(
    pipelines: dict[str, dict],
    *,
    retrieval_depth: int | None = None,
    rerank_multiplier: int | None = None,
) -> list[str]:
    """Compare the pipelines that are present. Missing pipelines are skipped."""
    lines: list[str] = []

    dense = pipelines.get("dense")
    bm25 = pipelines.get("bm25")
    hybrid = pipelines.get("hybrid")
    reranked = pipelines.get("reranked")

    if dense and bm25:
        lines.append(
            "Dense NDCG@10 is "
            f"{_rel(dense['ndcg@10'], bm25['ndcg@10'])} than BM25 "
            f"({dense['ndcg@10']:.3f} vs {bm25['ndcg@10']:.3f})."
        )
        lines.append(
            "Dense Recall@10 is "
            f"{_rel(dense['recall@10'], bm25['recall@10'])} than BM25 "
            f"({dense['recall@10']:.3f} vs {bm25['recall@10']:.3f})."
        )
    if hybrid and dense:
        lines.append(
            "Hybrid Recall@10 is "
            f"{_rel(hybrid['recall@10'], dense['recall@10'])} than dense "
            f"({hybrid['recall@10']:.3f} vs {dense['recall@10']:.3f})."
        )
        if "recall@100" in hybrid and "recall@100" in dense:
            lines.append(
                "Hybrid Recall@100 is "
                f"{_rel(hybrid['recall@100'], dense['recall@100'])} than dense "
                f"({hybrid['recall@100']:.3f} vs {dense['recall@100']:.3f})."
            )
    if reranked and hybrid:
        lines.append(
            "Reranked NDCG@10 is "
            f"{_rel(reranked['ndcg@10'], hybrid['ndcg@10'])} than hybrid "
            f"({reranked['ndcg@10']:.3f} vs {hybrid['ndcg@10']:.3f})."
        )

    depth = retrieval_depth
    if depth is None:
        for data in pipelines.values():
            if data.get("retrieval_depth"):
                depth = int(data["retrieval_depth"])
                break
    if depth:
        scored = ""
        if rerank_multiplier:
            scored = (
                f" The reranked pipeline scores about {depth * rerank_multiplier} "
                f"candidates at that depth."
            )
        lines.append(
            f"Latency figures on the quality run are measured at retrieval depth {depth}."
            f"{scored} Serving latency is measured separately at top_k 5, 10, and 20."
        )
    return lines
