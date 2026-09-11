import { PIPELINE_META, PipelineName, RetrieveResponse } from "../api";
import LatencyBadge from "./LatencyBadge";
import ResultCard from "./ResultCard";

interface Props {
  responses: RetrieveResponse[];
  loading: boolean;
  selectedCount: number;
}

export default function ResultsGrid({ responses, loading, selectedCount }: Props) {
  const cols =
    selectedCount >= 4 ? "xl:grid-cols-4" : selectedCount === 3 ? "xl:grid-cols-3" : "xl:grid-cols-2";

  if (loading) {
    return (
      <div className={`grid grid-cols-1 gap-4 md:grid-cols-2 ${cols}`}>
        {Array.from({ length: selectedCount }).map((_, i) => (
          <div key={i} className="h-64 animate-pulse rounded-xl bg-slate-900/60" />
        ))}
      </div>
    );
  }

  if (responses.length === 0) {
    return (
      <div className="rounded-xl border border-dashed border-slate-800 py-16 text-center text-slate-500">
        Enter a query and hit <span className="text-slate-300">Compare</span> to see all
        pipelines side-by-side.
      </div>
    );
  }

  return (
    <div className={`grid grid-cols-1 gap-4 md:grid-cols-2 ${cols}`}>
      {responses.map((resp) => {
        const meta = PIPELINE_META[resp.pipeline as PipelineName];
        return (
          <div
            key={resp.pipeline}
            className="flex flex-col gap-3 rounded-xl border border-slate-800 bg-slate-900/30 p-3"
          >
            <div className="flex items-center justify-between border-b border-slate-800 pb-2">
              <div className="flex items-center gap-2">
                <span
                  className="h-3 w-3 rounded-full"
                  style={{ backgroundColor: meta.color }}
                />
                <span className="text-sm font-semibold">{meta.label}</span>
              </div>
              <LatencyBadge
                latencyMs={resp.latency_ms}
                rerankerMs={resp.reranker_latency_ms}
              />
            </div>
            <div className="flex flex-col gap-2">
              {resp.results.map((r, i) => (
                <ResultCard key={r.doc_id + i} rank={i + 1} result={r} />
              ))}
            </div>
          </div>
        );
      })}
    </div>
  );
}
