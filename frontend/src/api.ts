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

export interface Citation {
  marker: number;
  doc_id: string;
  context_index: number;
}

export interface StageTimings {
  retrieval_ms: number;
  rerank_ms: number | null;
  generation_ms: number;
  total_ms: number;
}

export interface AskResponse {
  query: string;
  dataset: string;
  pipeline: PipelineName;
  model: string;
  answer: string;
  abstained: boolean;
  citations: Citation[];
  invalid_citations: number[];
  contexts: RetrieveResult[];
  timings: StageTimings;
  token_usage: { input_tokens: number; output_tokens: number };
}

export interface GenerationInfo {
  dataset: string;
  enabled: boolean;
  available: boolean;
  reason: string | null;
  provider: string;
  model: string;
  top_k_default: number;
  max_tokens: number;
  temperature: number;
}

export type AskState =
  | { status: "loading" }
  | { status: "done"; data: AskResponse }
  | { status: "error"; error: string };

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

  generationInfo: (dataset: string) =>
    jsonFetch<GenerationInfo>(`/api/${dataset}/generation`),

  ask: (dataset: string, query: string, pipeline: PipelineName) =>
    jsonFetch<AskResponse>(`/api/${dataset}/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, pipeline }),
    }),
};
