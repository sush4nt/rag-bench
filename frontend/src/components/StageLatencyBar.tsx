import { StageTimings } from "../api";

export const STAGE_COLORS = {
  retrieval: "#38bdf8",
  rerank: "#fbbf24",
  generation: "#818cf8",
};

interface Props {
  timings: StageTimings;
  // Shared x-axis across columns so bar lengths are directly comparable.
  scaleMs: number;
}

export default function StageLatencyBar({ timings, scaleMs }: Props) {
  const rerank = timings.rerank_ms ?? 0;
  const other = Math.max(
    0,
    timings.total_ms - timings.retrieval_ms - rerank - timings.generation_ms
  );
  const segments = [
    { key: "retrieval", ms: timings.retrieval_ms, color: STAGE_COLORS.retrieval },
    { key: "rerank", ms: rerank, color: STAGE_COLORS.rerank },
    { key: "generation", ms: timings.generation_ms, color: STAGE_COLORS.generation },
    { key: "overhead", ms: other, color: "#475569" },
  ].filter((s) => s.ms > 0);
  const scale = Math.max(scaleMs, timings.total_ms, 1);

  return (
    <div className="flex flex-col gap-1">
      <div className="flex h-2.5 w-full overflow-hidden rounded-full bg-slate-800">
        {segments.map((s) => (
          <div
            key={s.key}
            title={`${s.key}: ${s.ms.toFixed(1)} ms`}
            style={{ width: `${(s.ms / scale) * 100}%`, backgroundColor: s.color }}
          />
        ))}
      </div>
      <div className="flex flex-wrap gap-x-3 font-mono text-[10px] text-slate-500">
        <span>
          <span style={{ color: STAGE_COLORS.retrieval }}>■</span> retrieve{" "}
          {timings.retrieval_ms.toFixed(0)}ms
        </span>
        {timings.rerank_ms != null && (
          <span>
            <span style={{ color: STAGE_COLORS.rerank }}>■</span> rerank {rerank.toFixed(0)}ms
          </span>
        )}
        <span>
          <span style={{ color: STAGE_COLORS.generation }}>■</span> generate{" "}
          {timings.generation_ms.toFixed(0)}ms
        </span>
      </div>
    </div>
  );
}
