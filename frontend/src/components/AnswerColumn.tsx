import { useState } from "react";
import { AskState, PIPELINE_META, PipelineName } from "../api";
import AnswerCard from "./AnswerCard";
import LatencyBadge from "./LatencyBadge";
import ResultCard from "./ResultCard";
import StageLatencyBar from "./StageLatencyBar";

interface Props {
  pipeline: PipelineName;
  state: AskState;
  scaleMs: number;
}

export default function AnswerColumn({ pipeline, state, scaleMs }: Props) {
  const meta = PIPELINE_META[pipeline];
  const [highlighted, setHighlighted] = useState<number | null>(null);

  const jumpTo = (contextIndex: number) => {
    setHighlighted(contextIndex);
    document
      .getElementById(`ctx-${pipeline}-${contextIndex}`)
      ?.scrollIntoView({ behavior: "smooth", block: "nearest" });
    window.setTimeout(() => setHighlighted((h) => (h === contextIndex ? null : h)), 1800);
  };

  const data = state.status === "done" ? state.data : null;
  const markersByIndex = new Map<number, number[]>();
  data?.citations.forEach((c) => {
    markersByIndex.set(c.context_index, [...(markersByIndex.get(c.context_index) ?? []), c.marker]);
  });

  return (
    <div className="flex flex-col gap-3 rounded-xl border border-slate-800 bg-slate-900/30 p-3">
      <div className="flex items-center justify-between border-b border-slate-800 pb-2">
        <div className="flex items-center gap-2">
          <span className="h-3 w-3 rounded-full" style={{ backgroundColor: meta.color }} />
          <span className="text-sm font-semibold">{meta.label}</span>
        </div>
        {data && <LatencyBadge latencyMs={data.timings.total_ms} />}
      </div>

      {state.status === "loading" && (
        <div className="flex flex-col gap-2">
          <div className="h-2.5 animate-pulse rounded-full bg-slate-800" />
          <div className="h-28 animate-pulse rounded-lg bg-slate-900/60" />
          <div className="h-40 animate-pulse rounded-lg bg-slate-900/60" />
        </div>
      )}

      {state.status === "error" && (
        <div className="rounded-lg border border-red-900 bg-red-950/40 px-3 py-2 text-xs text-red-300">
          {state.error}
        </div>
      )}

      {data && (
        <>
          <StageLatencyBar timings={data.timings} scaleMs={scaleMs} />
          <AnswerCard response={data} accent={meta.color} onCite={jumpTo} />
          <div className="text-[10px] font-semibold uppercase tracking-wide text-slate-500">
            Context passages
          </div>
          <div className="flex flex-col gap-2">
            {data.contexts.map((r, i) => {
              const markers = markersByIndex.get(i) ?? [];
              return (
                <ResultCard
                  key={r.doc_id + i}
                  id={`ctx-${pipeline}-${i}`}
                  rank={i + 1}
                  result={r}
                  citedMarkers={markers}
                  accent={meta.color}
                  dimmed={!data.abstained && markers.length === 0}
                  highlighted={highlighted === i}
                />
              );
            })}
          </div>
        </>
      )}
    </div>
  );
}
