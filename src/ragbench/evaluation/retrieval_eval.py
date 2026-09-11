"""BEIR-style retrieval metrics.

We implement the standard trec_eval / BEIR metric definitions in pure Python so
the harness has no heavy transitive dependency (BEIR pins old libs that clash
with a modern stack). Outputs match ``EvaluateRetrieval.evaluate`` / pytrec_eval:

- ``NDCG@k``      graded, log2 discount, normalized by the ideal DCG
- ``Recall@k``    fraction of relevant docs retrieved in the top-k
- ``Precision@k`` relevant docs in top-k divided by k
- ``MAP@k``       mean average precision, cut at k
- ``MRR@k``       reciprocal rank of the first relevant doc within top-k

A doc is "relevant" iff its qrel score > 0.
"""

from __future__ import annotations

import math

Qrels = dict[str, dict[str, int]]
Results = dict[str, dict[str, float]]


def _ranked_doc_ids(doc_scores: dict[str, float]) -> list[str]:
    """Descending by score, ties broken by doc_id for determinism."""
    return [d for d, _ in sorted(doc_scores.items(), key=lambda x: (-x[1], x[0]))]


def _dcg(gains: list[float]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def _query_metrics(
    ranked: list[str], rels: dict[str, int], k_values: list[int]
) -> dict[str, float]:
    relevant = {d for d, s in rels.items() if s > 0}
    num_relevant = len(relevant)
    out: dict[str, float] = {}

    for k in k_values:
        top = ranked[:k]

        # NDCG@k (graded)
        gains = [max(rels.get(d, 0), 0) for d in top]
        ideal = sorted((max(s, 0) for s in rels.values()), reverse=True)[:k]
        idcg = _dcg(ideal)
        out[f"NDCG@{k}"] = _dcg(gains) / idcg if idcg > 0 else 0.0

        # Recall@k / Precision@k
        hits = sum(1 for d in top if d in relevant)
        out[f"Recall@{k}"] = hits / num_relevant if num_relevant else 0.0
        out[f"P@{k}"] = hits / k if k else 0.0

        # MAP@k
        if num_relevant:
            running_hits = 0
            ap = 0.0
            for i, d in enumerate(top, start=1):
                if d in relevant:
                    running_hits += 1
                    ap += running_hits / i
            out[f"MAP@{k}"] = ap / num_relevant
        else:
            out[f"MAP@{k}"] = 0.0

        # MRR@k
        rr = 0.0
        for i, d in enumerate(top, start=1):
            if d in relevant:
                rr = 1.0 / i
                break
        out[f"MRR@{k}"] = rr

    return out


def evaluate_retrieval(
    qrels: Qrels, results: Results, k_values: list[int]
) -> dict[str, dict[str, float]]:
    """Aggregate metrics over all queries present in ``qrels``.

    Returns a dict grouped by metric family, each mapping ``"<METRIC>@k" -> value``::

        {"ndcg": {"NDCG@10": 0.71, ...}, "recall": {...}, "precision": {...},
         "map": {...}, "mrr": {...}}
    """
    families = {"ndcg": "NDCG", "recall": "Recall", "precision": "P", "map": "MAP", "mrr": "MRR"}
    accum: dict[str, list[float]] = {}

    qids = [q for q in qrels if q in results] or list(qrels)
    for qid in qids:
        ranked = _ranked_doc_ids(results.get(qid, {}))
        qm = _query_metrics(ranked, qrels[qid], k_values)
        for key, val in qm.items():
            accum.setdefault(key, []).append(val)

    n = max(len(qids), 1)
    grouped: dict[str, dict[str, float]] = {fam: {} for fam in families}
    for key, vals in accum.items():
        prefix = key.split("@")[0]
        k = key.split("@")[1]
        fam = next(f for f, p in families.items() if p == prefix)
        # Rename P@k -> Precision@k in the precision family for readability.
        label = f"Precision@{k}" if prefix == "P" else key
        grouped[fam][label] = round(sum(vals) / n, 5)
    return grouped


def run_retrieval_eval(pipeline, queries: dict[str, str], qrels: Qrels, k_values: list[int]):
    """Convenience wrapper: run all queries through the pipeline, then score."""
    results = pipeline.retrieve_batch(queries, top_k=max(k_values))
    return evaluate_retrieval(qrels, results, k_values)


def latency_percentiles(latencies_ms: list[float]) -> dict[str, float]:
    """p50 / p95 / p99 (and mean) from a list of per-query latencies (ms)."""
    if not latencies_ms:
        return {"p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0, "mean_ms": 0.0}
    ordered = sorted(latencies_ms)

    def pct(p: float) -> float:
        idx = min(int(round(p / 100 * (len(ordered) - 1))), len(ordered) - 1)
        return round(ordered[idx], 3)

    return {
        "p50_ms": pct(50),
        "p95_ms": pct(95),
        "p99_ms": pct(99),
        "mean_ms": round(sum(ordered) / len(ordered), 3),
    }
