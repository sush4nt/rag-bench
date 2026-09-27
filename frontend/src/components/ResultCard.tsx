import { RetrieveResult } from "../api";

interface Props {
  rank: number;
  result: RetrieveResult;
  id?: string;
  // Generate mode only: which [n] markers in the answer point at this passage.
  citedMarkers?: number[];
  accent?: string;
  dimmed?: boolean;
  highlighted?: boolean;
}

export default function ResultCard({
  rank,
  result,
  id,
  citedMarkers,
  accent,
  dimmed,
  highlighted,
}: Props) {
  const cited = citedMarkers && citedMarkers.length > 0;
  return (
    <div
      id={id}
      className={`rounded-lg border bg-slate-900/60 p-3 transition-all ${
        highlighted ? "ring-2" : ""
      } ${dimmed ? "opacity-50" : ""}`}
      style={{
        borderColor: cited && accent ? `${accent}66` : "#1e293b",
        ...(highlighted && accent ? { ["--tw-ring-color" as string]: accent } : {}),
      }}
    >
      <div className="mb-1 flex items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <span className="text-xs font-semibold text-slate-400">#{rank}</span>
          {cited && (
            <span
              className="rounded px-1.5 py-0.5 font-mono text-[10px] font-semibold text-slate-950"
              style={{ backgroundColor: accent }}
            >
              cited {citedMarkers.map((m) => `[${m}]`).join("")}
            </span>
          )}
          {citedMarkers && !cited && (
            <span className="text-[10px] uppercase tracking-wide text-slate-600">not cited</span>
          )}
        </div>
        <span className="font-mono text-xs text-slate-500">{result.score.toFixed(4)}</span>
      </div>
      {result.title && (
        <div className="mb-1 text-sm font-medium text-slate-200 line-clamp-2">
          {result.title}
        </div>
      )}
      <p className="text-xs leading-relaxed text-slate-400 line-clamp-4">{result.text}</p>
      <div className="mt-2 font-mono text-[10px] text-slate-600">{result.doc_id}</div>
    </div>
  );
}
