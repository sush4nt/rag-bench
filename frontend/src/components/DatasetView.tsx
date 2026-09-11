import { useEffect, useMemo, useState } from "react";
import {
  ALL_PIPELINES,
  api,
  DatasetInfo,
  DatasetStatus,
  PipelineName,
  RetrieveResponse,
} from "../api";
import MetricsPanel from "./MetricsPanel";
import PipelineSelector from "./PipelineSelector";
import QueryInput from "./QueryInput";
import ResultsGrid from "./ResultsGrid";

const FALLBACK_EXAMPLES: Record<string, string[]> = {
  fiqa: [
    "What are the risks of investing in REITs?",
    "How does dollar cost averaging work?",
    "Difference between ETF and index fund?",
    "How to evaluate a company's P/E ratio?",
  ],
  scifact: [
    "Aspirin lowers the risk of colorectal cancer.",
    "The Mediterranean diet reduces cardiovascular disease.",
    "Vitamin D supplementation prevents respiratory infections.",
  ],
};

export default function DatasetView({ dataset }: { dataset: string }) {
  const [selected, setSelected] = useState<PipelineName[]>([...ALL_PIPELINES]);
  const [responses, setResponses] = useState<RetrieveResponse[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [status, setStatus] = useState<DatasetStatus | null>(null);
  const [info, setInfo] = useState<DatasetInfo | null>(null);

  useEffect(() => {
    setResponses([]);
    setError(null);
    api.status(dataset).then(setStatus).catch(() => setStatus(null));
    api
      .datasets()
      .then((d) => setInfo(d.datasets.find((x) => x.name === dataset) ?? null))
      .catch(() => setInfo(null));
  }, [dataset]);

  const examples = useMemo(
    () => info?.example_queries ?? FALLBACK_EXAMPLES[dataset] ?? [],
    [info, dataset]
  );

  const run = async (query: string) => {
    if (selected.length === 0) {
      setError("Select at least one pipeline.");
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const res = await api.batch(dataset, query, selected, 5);
      setResponses(res.responses);
    } catch (e) {
      setError(String(e instanceof Error ? e.message : e));
      setResponses([]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col gap-6">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h1 className="text-2xl font-bold capitalize">{dataset}</h1>
          <p className="text-sm text-slate-500">
            {dataset === "fiqa"
              ? "Financial QA — primary benchmark (57,638 passages)"
              : "Scientific claims — dev dataset (5,183 passages)"}
          </p>
        </div>
        {status && (
          <div className="flex items-center gap-2 text-xs">
            <span
              className={`h-2 w-2 rounded-full ${
                status.ready ? "bg-emerald-400" : "bg-red-400"
              }`}
            />
            <span className="text-slate-400">
              {status.ready
                ? `index ready · ${status.index_size_passages.toLocaleString()} passages`
                : "index not built"}
            </span>
          </div>
        )}
      </div>

      <div className="flex flex-col gap-4 rounded-2xl border border-slate-800 bg-slate-900/30 p-4">
        <QueryInput examples={examples} loading={loading} onSubmit={run} />
        <PipelineSelector selected={selected} onChange={setSelected} />
      </div>

      {error && (
        <div className="rounded-lg border border-red-900 bg-red-950/40 px-4 py-3 text-sm text-red-300">
          {error}
        </div>
      )}

      <ResultsGrid responses={responses} loading={loading} selectedCount={selected.length} />

      <div>
        <h2 className="mb-2 text-sm font-semibold uppercase tracking-wide text-slate-400">
          Last eval run
        </h2>
        <MetricsPanel dataset={dataset} />
      </div>
    </div>
  );
}
