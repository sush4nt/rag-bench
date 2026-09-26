// Typed client for the RAGBench API (same-origin; proxied to :8080 in dev).

export type PipelineName = "bm25" | "dense" | "hybrid" | "reranked";

export const ALL_PIPELINES: PipelineName[] = ["bm25", "dense", "hybrid", "reranked"];

export const PIPELINE_META: Record<
  PipelineName,
  { label: string; color: string; blurb: string }
> = {
  bm25: { label: "BM25", color: "#38bdf8", blurb: "Sparse lexical" },
  dense: { label: "Dense", color: "#a78bfa", blurb: "Single-vector ANN" },
  hybrid: { label: "Hybrid", color: "#34d399", blurb: "BM25 + Dense (RRF)" },
  reranked: { label: "Reranked", color: "#fbbf24", blurb: "Hybrid + cross-encoder" },
};

export interface RetrieveResult {
  doc_id: string;
  score: number;
  title: string;
  text: string;
}

export interface RetrieveResponse {
  query: string;
  pipeline: PipelineName;
  dataset: string;
  latency_ms: number;
  results: RetrieveResult[];
  reranker_latency_ms: number | null;
}

export interface BatchRetrieveResponse {
  query: string;
  dataset: string;
  responses: RetrieveResponse[];
}

export interface DatasetStatus {
  dataset: string;
  bm25_index_ready: boolean;
  qdrant_ready: boolean;
  index_size_passages: number;
  embedding_model: string;
  ready: boolean;
}

export interface DatasetInfo {
  name: string;
  router_prefix: string;
  example_queries: string[];
}

export interface EvalPipeline {
  "ndcg@10": number;
  "mrr@10": number;
  "recall@10": number;
  "recall@100": number;
  p95_ms: number;
  p50_ms: number;
  latency_scope?: string;
  retrieval_depth?: number;
  ragas?: Record<string, number>;
}

export interface EvalSummary {
  dataset: string;
  available: boolean;
  timestamp?: string;
  pipelines: Record<string, EvalPipeline>;
}

async function jsonFetch<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init);
  if (!res.ok) {
    const detail = await res.text().catch(() => res.statusText);
    throw new Error(`${res.status}: ${detail}`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  datasets: () => jsonFetch<{ datasets: DatasetInfo[] }>("/api/datasets"),

  status: (dataset: string) => jsonFetch<DatasetStatus>(`/api/${dataset}/status`),

  batch: (dataset: string, query: string, pipelines: PipelineName[], topK: number) =>
    jsonFetch<BatchRetrieveResponse>(`/api/${dataset}/retrieve/batch`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, pipelines, top_k: topK }),
    }),

  evalLatest: (dataset: string) => jsonFetch<EvalSummary>(`/api/${dataset}/eval/latest`),
};
