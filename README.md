# RAGBench

> A production retrieval-strategy benchmarking platform. Same production patterns as
> [mlserve](#) (FastAPI · Prometheus · Grafana · k6 · HuggingFace) — but instead of
> comparing two inference runtimes, it compares **four retrieval pipelines** on the
> same corpus and queries. The delta is the pipeline.

**Stack:** Qdrant · FastAPI · React · BEIR metrics · RAGAS · MLflow · Prometheus / Grafana

---

## The four pipelines

| Pipeline   | Strategy                          | Expected trade-off                         |
|------------|-----------------------------------|--------------------------------------------|
| `bm25`     | Sparse lexical (`bm25s`)          | Fastest; best on keywords, misses semantics |
| `dense`    | Single-vector ANN (`bge-large`)   | Semantic understanding; slower indexing     |
| `hybrid`   | BM25 + Dense via Qdrant RRF fusion | Best quality/latency balance                |
| `reranked` | Hybrid + cross-encoder reranker   | Highest quality; +60–120 ms                 |

Identical queries, identical corpus, four strategies → measurable deltas in
**quality** (NDCG@10, MRR@10, Recall@10) and **latency** (p50/p95/p99). Generation
quality is measured separately with **RAGAS**. Everything feeds Prometheus; Grafana
renders the side-by-side dashboard whose thesis chart is *quality vs latency*.

---

## Quickstart

```bash
# 1. Install (core). Add the RAGAS extra for generation eval.
make install                 # uv sync
make install-ragas           # uv sync --extra ragas --extra dev

# 2. Bring up Qdrant + MLflow + Prometheus + Grafana (+ the app)
make docker-up               # or: docker compose up --build

# 3. Index a dataset (SciFact is fast; FiQA takes 8–12 min)
make index-scifact

# 4. Run the API locally (without Docker)
make serve                   # http://localhost:8080  (docs at /docs)

# 5. Evaluate + log to MLflow, then view the dashboard
make eval-scifact
```

Ports: **app** 8080 · **Qdrant** 6333 · **MLflow** 5001 · **Prometheus** 9090 · **Grafana** 3001.

---

## Datasets

| | SciFact | FiQA |
|---|---|---|
| Role | Dev / fast iteration | Primary benchmark |
| Corpus | 5,183 passages | 57,638 passages |
| Test queries | 300 | 648 |
| Domain | Scientific claims | Financial QA |

Both use the BEIR format (`corpus.jsonl`, `queries.jsonl`, `qrels/test.tsv`) and are
downloaded automatically on first index.

---

## API

All routes are mirrored under `/api/scifact` and `/api/fiqa`:

| Method & path | Purpose |
|---|---|
| `POST /api/{ds}/retrieve` | Single query, single pipeline |
| `POST /api/{ds}/retrieve/batch` | One query across several pipelines (parallel) |
| `GET  /api/{ds}/status` | Index readiness + stats |
| `GET  /api/{ds}/pipelines` | List pipelines + capabilities |
| `POST /api/{ds}/eval/run` | Trigger an offline eval run → MLflow + gauges |
| `GET  /api/datasets` | Available datasets + example queries |
| `GET  /metrics` | Prometheus scrape endpoint |

```bash
curl -s localhost:8080/api/fiqa/retrieve \
  -H 'content-type: application/json' \
  -d '{"query":"How does dollar cost averaging work?","pipeline":"hybrid","top_k":5}' | jq
```

---

## Repository layout

```
src/ragbench/
  common/      paths, logging, protocol types, service clients
  config/      pydantic config models + YAML loader
  data/        BEIR loader + passage normalization
  indexing/    bm25s + Qdrant index builders, build CLI
  retrieval/   base + bm25/dense/hybrid/reranked pipelines, shared embedder
  evaluation/  BEIR metrics, RAGAS gen eval, MLflow runner
  serving/     FastAPI app, dataset routers, registry, Prometheus metrics
frontend/      React + Vite + Tailwind (two dataset pages, side-by-side grid)
monitoring/    Prometheus config + provisioned Grafana dashboard
load_testing/  k6 scripts
deploy/        HuggingFace Spaces Dockerfile (SciFact-only build)
tests/         pytest suite (pipelines, eval, api)
```

### A note on two spec reconciliations
- **Qdrant collections:** server-side RRF fusion needs the dense and sparse vectors
  in *one* collection, so we store both named vectors inside `{prefix}_dense`
  (matches the spec's hybrid query sample). See `indexing/qdrant_indexer.py`.
- **BGE prefixes:** the BGE v1.5 model card puts the retrieval instruction on the
  *query*, not the passage. We follow the model card for correct scores. See
  `retrieval/embedder.py`.

---

## Evaluation

- **Retrieval:** NDCG@k, MRR@k, Recall@k, Precision@k, MAP@k (trec_eval/BEIR
  definitions, implemented dependency-free in `evaluation/retrieval_eval.py`).
- **Generation (RAGAS, opt-in):** Faithfulness, Answer Relevance, Context Recall,
  Context Precision. Runs on a sample (default 100 queries) with `claude-3-haiku`
  as judge. Requires `--extra ragas` and `ANTHROPIC_API_KEY`.

Every eval run logs one MLflow run per pipeline under `ragbench/{dataset}` and
writes `data/{dataset}/eval_latest.json`, which seeds the Prometheus gauges the
Grafana dashboard reads.

---

## Testing & load testing

```bash
make test          # pytest (pipelines, eval, api)
make load-test     # k6 against local serving
```

The test suite runs without downloading `bge-large`: it builds a tiny bm25s index
over a tuned SciFact-style fixture and exercises the API through it. Dense/hybrid/
reranked tests self-skip when heavy models or Qdrant are unavailable.

---

## Deployment (HuggingFace Spaces)

`deploy/huggingface/Dockerfile` builds the **SciFact** index at image-build time and
runs the app in `HF_SPACE=true` (SciFact-only) mode. FiQA is too large to index at
Spaces build time — see the deploy Dockerfile comments.
