import { useEffect, useState } from "react";
import { api, EvalSummary, PIPELINE_META, PipelineName } from "../api";

const ORDER: PipelineName[] = ["bm25", "dense", "hybrid", "reranked"];

export default function MetricsPanel({ dataset }: { dataset: string }) {
  const [summary, setSummary] = useState<EvalSummary | null>(null);

  useEffect(() => {
    setSummary(null);
    api.evalLatest(dataset).then(setSummary).catch(() => setSummary(null));
  }, [dataset]);

  if (!summary || !summary.available) {
    return (
      <div className="rounded-xl border border-slate-800 bg-slate-900/30 p-4 text-sm text-slate-500">
        No eval run yet. Run <code className="text-slate-300">make eval-{dataset}</code> (or
        POST <code className="text-slate-300">/api/{dataset}/eval/run</code>) to populate
        NDCG@10 / MRR@10.
      </div>
    );
  }

  const rows = ORDER.filter((p) => summary.pipelines[p]);

  return (
    <div className="overflow-hidden rounded-xl border border-slate-800">
      <table className="w-full text-sm">
        <thead className="bg-slate-900/60 text-left text-xs uppercase tracking-wide text-slate-500">
          <tr>
            <th className="px-4 py-2">Pipeline</th>
            <th className="px-4 py-2">NDCG@10</th>
            <th className="px-4 py-2">MRR@10</th>
            <th className="px-4 py-2">Recall@10</th>
            <th className="px-4 py-2">p95 (ms)</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((p) => {
            const m = summary.pipelines[p];
            return (
              <tr key={p} className="border-t border-slate-800">
                <td className="px-4 py-2">
                  <span className="inline-flex items-center gap-2">
                    <span
                      className="h-2.5 w-2.5 rounded-full"
                      style={{ backgroundColor: PIPELINE_META[p].color }}
                    />
                    {PIPELINE_META[p].label}
                  </span>
                </td>
                <td className="px-4 py-2 font-mono">{m["ndcg@10"].toFixed(4)}</td>
                <td className="px-4 py-2 font-mono">{m["mrr@10"].toFixed(4)}</td>
                <td className="px-4 py-2 font-mono">{m["recall@10"].toFixed(4)}</td>
                <td className="px-4 py-2 font-mono">{m.p95_ms.toFixed(1)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
