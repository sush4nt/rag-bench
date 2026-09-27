import { ReactNode } from "react";
import { AskResponse, AskState, PIPELINE_META, PipelineName } from "../api";

interface Props {
  pipelines: PipelineName[];
  states: Partial<Record<PipelineName, AskState>>;
}

type Row = { pipeline: PipelineName; data: AskResponse };

function bestBy(rows: Row[], value: (r: Row) => number): PipelineName | null {
  if (rows.length < 2) return null;
  return rows.reduce((a, b) => (value(b) < value(a) ? b : a)).pipeline;
}

function Cell({ best, children }: { best: boolean; children: ReactNode }) {
  return (
    <td className={`px-3 py-2 font-mono ${best ? "text-emerald-400" : "text-slate-300"}`}>
      {children}
    </td>
  );
}

export default function AnswerComparison({ pipelines, states }: Props) {
  const rows: Row[] = pipelines.flatMap((p) => {
    const s = states[p];
    return s?.status === "done" ? [{ pipeline: p, data: s.data }] : [];
  });
  if (rows.length < 2) return null;

  const fastest = bestBy(rows, (r) => r.data.timings.total_ms);
  const cheapest = bestBy(rows, (r) => r.data.token_usage.input_tokens);

  // Evidence overlap: every passage cited by at least one pipeline.
  const docs = new Map<string, { title: string; citedBy: number }>();
  rows.forEach(({ data }) =>
    data.citations.forEach((c) => {
      const prev = docs.get(c.doc_id);
      const title = data.contexts[c.context_index]?.title || data.contexts[c.context_index]?.text;
      docs.set(c.doc_id, { title: prev?.title || title || "", citedBy: (prev?.citedBy ?? 0) + 1 });
    })
  );
  const docRows = [...docs.entries()].sort((a, b) => b[1].citedBy - a[1].citedBy);

  return (
    <div className="flex flex-col gap-4">
      <div className="overflow-x-auto rounded-xl border border-slate-800">
        <table className="w-full text-sm">
          <thead className="bg-slate-900/60 text-left text-xs uppercase tracking-wide text-slate-500">
            <tr>
              <th className="px-3 py-2">Pipeline</th>
              <th className="px-3 py-2">Total ms</th>
              <th className="px-3 py-2">Retrieve</th>
              <th className="px-3 py-2">Rerank</th>
              <th className="px-3 py-2">Generate</th>
              <th className="px-3 py-2">Tokens in / out</th>
              <th className="px-3 py-2">Cited</th>
              <th className="px-3 py-2">Invalid</th>
              <th className="px-3 py-2">Outcome</th>
            </tr>
          </thead>
          <tbody>
            {rows.map(({ pipeline, data }) => {
              const t = data.timings;
              const cited = new Set(data.citations.map((c) => c.doc_id)).size;
              return (
                <tr key={pipeline} className="border-t border-slate-800">
                  <td className="px-3 py-2">
                    <span className="inline-flex items-center gap-2">
                      <span
                        className="h-2.5 w-2.5 rounded-full"
                        style={{ backgroundColor: PIPELINE_META[pipeline].color }}
                      />
                      {PIPELINE_META[pipeline].label}
                    </span>
                  </td>
                  <Cell best={pipeline === fastest}>{t.total_ms.toFixed(0)}</Cell>
                  <Cell best={false}>{t.retrieval_ms.toFixed(0)}</Cell>
                  <Cell best={false}>{t.rerank_ms != null ? t.rerank_ms.toFixed(0) : "—"}</Cell>
                  <Cell best={false}>{t.generation_ms.toFixed(0)}</Cell>
                  <Cell best={pipeline === cheapest}>
                    {data.token_usage.input_tokens.toLocaleString()} / {data.token_usage.output_tokens}
                  </Cell>
                  <Cell best={false}>
                    {cited}/{data.contexts.length}
                  </Cell>
                  <td
                    className={`px-3 py-2 font-mono ${
                      data.invalid_citations.length ? "text-red-400" : "text-slate-500"
                    }`}
                  >
                    {data.invalid_citations.length}
                  </td>
                  <td className="px-3 py-2 text-xs">
                    {data.abstained ? (
                      <span className="rounded bg-amber-950/60 px-2 py-0.5 text-amber-300">
                        abstained
                      </span>
                    ) : (
                      <span className="rounded bg-emerald-950/60 px-2 py-0.5 text-emerald-300">
                        answered
                      </span>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {docRows.length > 0 && (
        <div className="overflow-x-auto rounded-xl border border-slate-800">
          <div className="border-b border-slate-800 bg-slate-900/60 px-3 py-2 text-xs uppercase tracking-wide text-slate-500">
            Evidence overlap — which passages each pipeline grounded its answer in
          </div>
          <table className="w-full text-sm">
            <thead className="text-left text-xs text-slate-500">
              <tr>
                <th className="px-3 py-2 font-medium">Passage</th>
                {rows.map(({ pipeline }) => (
                  <th key={pipeline} className="px-3 py-2 text-center font-medium">
                    {PIPELINE_META[pipeline].label}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {docRows.map(([docId, { title }]) => (
                <tr key={docId} className="border-t border-slate-800">
                  <td className="max-w-md px-3 py-2">
                    <div className="truncate text-slate-300" title={title}>
                      {title}
                    </div>
                    <div className="font-mono text-[10px] text-slate-600">{docId}</div>
                  </td>
                  {rows.map(({ pipeline, data }) => {
                    const color = PIPELINE_META[pipeline].color;
                    const cite = data.citations.find((c) => c.doc_id === docId);
                    const rank = data.contexts.findIndex((c) => c.doc_id === docId);
                    return (
                      <td key={pipeline} className="px-3 py-2 text-center">
                        {cite ? (
                          <span
                            className="inline-block h-3 w-3 rounded-full"
                            style={{ backgroundColor: color }}
                            title={`cited as [${cite.marker}]`}
                          />
                        ) : rank >= 0 ? (
                          <span
                            className="inline-block h-3 w-3 rounded-full border-2"
                            style={{ borderColor: color }}
                            title={`retrieved at #${rank + 1} but not cited`}
                          />
                        ) : (
                          <span className="text-slate-700" title="not retrieved">
                            ·
                          </span>
                        )}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
          <div className="border-t border-slate-800 px-3 py-2 text-[10px] text-slate-500">
            ● cited &nbsp; ○ retrieved, not cited &nbsp; · not retrieved
          </div>
        </div>
      )}
    </div>
  );
}
