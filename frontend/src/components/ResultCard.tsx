import { RetrieveResult } from "../api";

interface Props {
  rank: number;
  result: RetrieveResult;
}

export default function ResultCard({ rank, result }: Props) {
  return (
    <div className="rounded-lg border border-slate-800 bg-slate-900/60 p-3">
      <div className="mb-1 flex items-center justify-between">
        <span className="text-xs font-semibold text-slate-400">#{rank}</span>
        <span className="font-mono text-xs text-slate-500">
          {result.score.toFixed(4)}
        </span>
      </div>
      {result.title && (
        <div className="mb-1 text-sm font-medium text-slate-200 line-clamp-2">
          {result.title}
        </div>
      )}
      <p className="text-xs leading-relaxed text-slate-400 line-clamp-4">
        {result.text}
      </p>
      <div className="mt-2 font-mono text-[10px] text-slate-600">{result.doc_id}</div>
    </div>
  );
}
