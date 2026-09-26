# RAGBench Project Evaluation & Improvement Plan

## Purpose

This document evaluates the current `rag-bench` project from the perspective of a **Senior Data Scientist / Senior ML-AI Engineer interview**.

The strongest positioning is:

> **RAGBench is a controlled experimentation and evaluation platform for comparing retrieval strategies in Retrieval-Augmented Generation systems.**

It is stronger to describe it this way than as a generic "chat with PDFs" system.

---

# 1. Recommended Mental Model

A production RAG system is better understood as **two ML systems plus an operational layer**.

## Offline / Indexing Path

```text
Documents
   ↓
Parsing / Cleaning
   ↓
Chunking + Metadata
   ↓
Embedding / Sparse Representation
   ↓
Index
```

## Online / Retrieval + Generation Path

```text
User Question
   ↓
Query Understanding / Rewriting
   ↓
Candidate Retrieval          ← optimize recall
BM25 / Dense / Hybrid
   ↓
Reranking                    ← optimize precision
   ↓
Context Selection
   ↓
Prompt Construction
   ↓
LLM Generation
   ↓
Grounded Answer + Citations
```

## RAGOps / Evaluation Layer

```text
Retrieval quality
Generation quality
Latency / throughput
Cost
Freshness
Security / ACL
Observability
```

The senior-level idea is that **retrieval and generation should be evaluated independently before evaluating the end-to-end system**.

---

# 2. What RAGBench Currently Demonstrates

The project compares multiple retrieval strategies under a shared evaluation setup.

| Pipeline | Mechanism | Main idea |
|---|---|---|
| BM25 | Sparse lexical retrieval | Exact / keyword-based matching |
| Dense | Embedding + ANN retrieval | Semantic similarity |
| Hybrid | Dense + sparse + RRF | Combine lexical and semantic strengths |
| Reranked | Hybrid candidates + cross-encoder | Retrieve broadly, then score more accurately |

The project also includes:

- BEIR-style retrieval evaluation
- Metrics such as NDCG, Recall, Precision, MAP and MRR
- RAGAS-based generation evaluation
- Qdrant
- FastAPI
- MLflow
- Prometheus / Grafana
- Docker
- k6 load testing

The central experimental question is:

> **How much retrieval quality do I gain from more sophisticated retrieval strategies, and what latency/complexity do I pay for that improvement?**

That is a strong ML systems problem.

---

# 3. Strong Aspects of the Project

## 3.1 Controlled Experimentation

The strongest part of the project is that multiple retrieval strategies are compared using the same:

- corpus
- queries
- relevance labels
- evaluation metrics

This isolates the retrieval approach as the major experimental variable.

That makes the project more scientifically defensible than a demo where every pipeline differs in several ways at once.

---

## 3.2 BM25 Is Treated as a Serious Baseline

A common beginner mistake is assuming dense retrieval automatically dominates lexical retrieval.

BM25 remains valuable for:

- model names
- error codes
- product IDs
- acronyms
- uncommon proper nouns
- exact technical terms

Dense retrieval is stronger when the wording differs but meaning is similar.

A senior answer should therefore be:

> Sparse and dense retrieval fail differently, which is exactly why hybrid retrieval is useful.

---

## 3.3 Hybrid Retrieval Is a Good Design Choice

Dense similarity scores and BM25 scores live on different scales.

Example:

```text
BM25 score = 13.8
Cosine similarity = 0.82
```

Directly averaging them is not necessarily meaningful.

Reciprocal Rank Fusion avoids this by combining **rank positions** rather than raw scores.

\[
RRF(d) = \sum_i \frac{1}{k + rank_i(d)}
\]

This makes hybrid retrieval comparatively simple and robust.

---

## 3.4 Query and Passage Encoding Are Treated Carefully

For BGE-style embedding models, query instructions should generally be applied to queries rather than blindly to passages.

That implementation detail shows awareness that embedding-model usage matters, not just model selection.

---

## 3.5 Retrieval and Generation Metrics Are Separated

This is a particularly good architectural decision.

A bad final answer can happen because:

```text
A. Retrieval failed
or
B. Retrieval succeeded but generation failed
```

If only end-to-end answer quality is measured, those failure modes become hard to separate.

The current project measures retrieval using IR metrics and generation using RAG-oriented metrics.

That is conceptually correct.

---

# 4. Major Gaps and Improvements

## Gap 1 — No Real Document Chunking Pipeline

The current project largely treats BEIR passages as already prepared retrieval units.

That is valid for a retrieval benchmark, but it does **not** demonstrate full ingestion engineering.

Missing areas include:

```text
PDF / HTML / Markdown
       ↓
parser
       ↓
layout / section extraction
       ↓
chunking
       ↓
metadata creation
       ↓
embedding
       ↓
indexing
```

### Why an interviewer may care

Chunking can strongly affect retrieval.

Too-small chunks:

- improve specificity
- lose surrounding context

Too-large chunks:

- preserve context
- reduce retrieval precision
- increase prompt tokens
- create more irrelevant content

### Recommended improvement

Build one real ingestion path for PDF and Markdown.

Store metadata such as:

```text
chunk_id
document_id
title
section
page
source
created_at
updated_at
```

Then compare:

- 256-token chunks
- 512-token chunks
- 1024-token chunks
- overlapping fixed chunks
- semantic / structural chunking
- parent-child retrieval

Evaluate the effect on:

- Recall@k
- NDCG@k
- generation quality
- token cost
- latency

---

## Gap 2 — No Normal End-to-End `/ask` Pipeline

The project is currently stronger as a **retrieval benchmark** than as an online RAG application.

The API focuses on retrieval and evaluation, while generation mainly appears in offline evaluation.

### Recommended improvement

Add an endpoint such as:

```text
POST /api/{dataset}/ask
```

Input:

```json
{
  "query": "...",
  "pipeline": "hybrid_reranked",
  "top_k": 5,
  "generation_model": "..."
}
```

Output:

```json
{
  "answer": "...",
  "citations": [],
  "retrieved_chunks": [],
  "retrieval_ms": 0,
  "rerank_ms": 0,
  "generation_ms": 0,
  "total_ms": 0,
  "token_usage": {}
}
```

This would demonstrate:

```text
retrieve
→ rerank
→ build context
→ generate
→ cite
→ observe
```

---

## Gap 3 — FiQA Generation Ground Truth Should Be Rechecked

The current FiQA generation-evaluation path appears to treat generated-query dataset fields as if they were QA answer labels.

That risks conflating:

- document text
- generated query text
- relevance IDs
- answer ground truth

### Why this matters

If evaluation references are incorrectly mapped, generation metrics become unreliable even if the RAGAS code itself is correct.

### Recommended improvement

Separate evaluation datasets by purpose.

For retrieval:

```text
query
+
qrels
+
corpus
```

For generation:

```text
question
+
reference answer
+
optional reference evidence
```

If proper reference answers are unavailable, evaluate only metrics that do not falsely assume answer ground truth.

For example:

- faithfulness
- answer relevancy
- context precision
- context recall where valid

---

## Gap 4 — Publish Actual Benchmark Results

The current conceptual trade-offs are sensible, but senior interviewers will ask:

> What did the experiment actually show?

A benchmark project should publish a central result table.

Example:

| Method | NDCG@10 | Recall@10 | MRR@10 | p50 ms | p95 ms | QPS |
|---|---:|---:|---:|---:|---:|---:|
| BM25 |  |  |  |  |  |  |
| Dense |  |  |  |  |  |  |
| Hybrid |  |  |  |  |  |  |
| Reranked |  |  |  |  |  |  |

The project should be able to make concrete statements such as:

```text
Hybrid improved Recall@10 by X%
over dense retrieval,
at Y ms additional p95 latency.
```

and:

```text
Cross-encoder reranking improved NDCG@10 by X%,
but increased p95 latency by Y%.
```

That is the thesis of the project.

---

## Gap 5 — Quality Benchmarking and Serving Benchmarking Are Mixed

Offline retrieval evaluation may retrieve a large number of candidates because metrics are measured at large K values.

For example:

```text
Recall@100
```

may require retrieving 100 documents.

But a normal serving request might only need:

```text
top_k = 5 or 10
```

If reranking over-retrieves by 5x, then:

```text
quality benchmark:
top_k = 100
→ rerank ~500 candidates

serving:
top_k = 10
→ rerank ~50 candidates
```

These have very different latency profiles.

### Recommended improvement

Create separate benchmark suites.

### Retrieval-quality benchmark

```text
top 100 retrieval
metrics @1 / @5 / @10 / @100
```

### Serving-performance benchmark

Test:

```text
top_k = 5
top_k = 10
top_k = 20
```

under concurrency:

```text
1 user
10 users
50 users
```

Measure:

- p50 latency
- p95 latency
- p99 latency
- throughput
- CPU
- memory
- error rate

---

## Gap 6 — Evaluation Sampling Could Be More Rigorous

If generation evaluation simply takes the first N examples, the sample may be biased by dataset ordering.

### Recommended improvement

Use:

- fixed random seed
- reproducible random sampling
- stratification by query type
- difficulty buckets
- number of relevant documents
- domain / topic

For stronger experimentation, report confidence intervals.

Example:

```text
NDCG@10 = 0.487 ± 0.012
```

rather than only:

```text
NDCG@10 = 0.487
```

---

## Gap 7 — Same Family of Models for Generation and Evaluation

Using an LLM to generate answers and an LLM judge to evaluate them is common, but it can create evaluator dependence.

### Recommended improvement

Use:

```text
Generator = Model A
Judge = Model B
```

and validate the judge against a small human-reviewed dataset.

Good senior interview answer:

> LLM-as-a-judge is scalable, but I would not treat it as objective ground truth. I would calibrate it on human labels and measure judge agreement.

---

## Gap 8 — No Incremental Indexing / Freshness Strategy

The current benchmark-style indexing workflow can rebuild collections from scratch.

That is acceptable for experiments, but production systems need to handle:

```text
new document
updated document
deleted document
```

without a full rebuild.

### Recommended improvement

Add:

- stable document IDs
- chunk IDs
- content hashes
- incremental embedding
- upserts
- delete propagation
- document versioning
- index versioning
- blue/green index swap

This gives the project a stronger **RAGOps** story.

---

## Gap 9 — Metadata Filtering and Authorization

Enterprise RAG needs access control during retrieval.

Example:

```text
retrieve where
tenant_id = user.tenant
AND department IN user.allowed_departments
AND access_group IN user.groups
```

The key security principle is:

> Do not retrieve unauthorized content and then expect the LLM to keep it secret.

### Recommended improvement

Add metadata such as:

```text
tenant
department
document_type
owner
access_groups
effective_date
created_at
updated_at
source
```

and demonstrate filtered retrieval.

---

## Gap 10 — Production-Like Infrastructure Is Not the Same as Production Security

Docker Compose, Grafana and Qdrant make the system easy to demonstrate locally.

However, development defaults such as weak credentials or anonymous dashboards should not be described as production-ready security.

Better wording:

> **Production-patterned observability and serving architecture**

instead of:

> **Production-secure deployment**

until authentication, TLS, secrets, ACLs and hardening are added.

---

## Gap 11 — No Query Transformation Layer

Current flow:

```text
query
↓
retrieve
```

Advanced RAG often uses:

```text
query
↓
rewrite / expand / decompose
↓
retrieve
```

Useful techniques to understand and potentially implement:

- conversational query rewriting
- multi-query retrieval
- query expansion
- HyDE
- step-back prompting
- query decomposition
- sub-question retrieval

### Example

User query:

```text
"Why is it happening again?"
```

Conversation context:

```text
Repeated Kubernetes OOMKilled pod issue
```

Retrieval query:

```text
"Why does the Kubernetes pod repeatedly become OOMKilled?"
```

This is much more retrievable.

---

## Gap 12 — No Parent-Child / Contextual Retrieval

Small chunks retrieve well but may lack context.

Large chunks retain context but are harder to retrieve precisely.

Parent-child retrieval solves this by separating:

```text
retrieval unit
from
generation context
```

Example:

```text
Parent:
complete document section

Children:
individual paragraphs / sentences
```

Search the child.

Return the parent or neighboring context to the LLM.

Intuition:

> **Retrieve small, reason large.**

---

## Gap 13 — No Explicit Prompt-Injection / Poisoned-Document Testing

Retrieved content itself can be malicious.

A document can contain instructions such as:

```text
Ignore prior instructions...
Reveal confidential information...
```

That makes RAG vulnerable to indirect prompt injection.

### Recommended improvement

Add security tests covering:

- malicious retrieved documents
- prompt injection
- document poisoning
- PII leakage
- cross-tenant leakage
- ACL bypass
- malicious citations
- untrusted HTML / script-like content

---

# 5. Recommended Upgrade Priority

| Priority | Improvement | Reason |
|---|---|---|
| P0 | Validate / fix FiQA answer evaluation | Evaluation correctness |
| P0 | Publish benchmark results | Gives project a measurable conclusion |
| P0 | Position project as a retrieval benchmark | Prevents overclaiming |
| P1 | Add `/ask` generation + citation endpoint | Makes system truly end-to-end |
| P1 | Add real ingestion + chunking experiments | Covers major lifecycle gap |
| P1 | Separate quality and latency benchmarking | Better experimental methodology |
| P1 | Random / stratified evaluation | Stronger statistical validity |
| P2 | Metadata filters / ACL | Production readiness |
| P2 | Incremental indexing | RAGOps / freshness |
| P2 | Query transformation | Advanced retrieval |
| P2 | Parent-child / contextual retrieval | Advanced context design |
| P3 | ColBERT / late interaction | Advanced information retrieval |
| P3 | Prompt-injection tests | Security maturity |

---

# 6. Recommended Interview Positioning

Use a description similar to:

> **RAGBench is an experimentation platform for evaluating quality-latency trade-offs across RAG retrieval strategies. I keep the corpus and queries fixed and compare BM25, dense retrieval, hybrid dense+sparse retrieval using Reciprocal Rank Fusion, and hybrid retrieval followed by cross-encoder reranking. Retrieval is evaluated independently using metrics such as NDCG, Recall, MAP and MRR, while generation is evaluated separately using RAG-oriented metrics. The serving layer uses FastAPI, Qdrant provides vector retrieval, MLflow tracks experiments, and Prometheus/Grafana provide runtime observability.**

The important senior-level message is:

> **I deliberately separate retrieval evaluation from generation evaluation so I can diagnose where failures originate.**

---

# 7. How to Make the Project Stronger for Senior Data Scientist Roles

Emphasize:

- experimental design
- controlled comparisons
- evaluation methodology
- ablation studies
- statistical confidence
- error analysis
- hyperparameter trade-offs
- retrieval metric intuition
- business / product implications of latency and quality

Best extension:

> **Run controlled chunking, embedding-model, top-k and reranker experiments and publish the measured quality-latency frontier.**

---

# 8. How to Make the Project Stronger for Senior ML / AI Engineer Roles

Emphasize:

- serving architecture
- online retrieval
- caching
- ANN index behavior
- concurrency
- p95 / p99 latency
- throughput
- incremental indexing
- metadata filtering
- ACLs
- observability
- tracing
- prompt injection defenses
- versioned indexes
- cost monitoring

Best extension:

> **Turn the benchmark into a complete retrieve → rerank → generate → cite service with incremental indexing and production-style observability.**

---

# 9. Recommended Learning Order

Study RAG in this order:

1. Original RAG architecture
2. Sparse vs dense retrieval
3. BEIR retrieval evaluation
4. Hybrid retrieval
5. Reciprocal Rank Fusion
6. Cross-encoder reranking
7. Chunking strategies
8. Query transformation
9. Parent-child / contextual retrieval
10. RAGAS and generation evaluation
11. RAGOps
12. Security and ACL-aware retrieval

---

# 10. Core Senior-Level Takeaway

The strongest evolution of this project is **not adding more frameworks**.

The highest-value improvements are:

1. Fix evaluation correctness.
2. Publish measured results.
3. Add a real ingestion + chunking experiment.
4. Add a complete `/ask` path with citations.
5. Separate retrieval-quality testing from serving-performance testing.
6. Add freshness, authorization and security controls.

Those improvements make the project more defensible in both Senior Data Scientist and Senior ML/AI Engineer interviews.
