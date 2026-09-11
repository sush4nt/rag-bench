interface Props {
  latencyMs: number;
  rerankerMs?: number | null;
}

export default function LatencyBadge({ latencyMs, rerankerMs }: Props) {
  const color =
    latencyMs < 100 ? "#34d399" : latencyMs < 500 ? "#fbbf24" : "#f87171";
  return (
    <span className="inline-flex items-center gap-1 font-mono text-xs">
      <span style={{ color }}>{latencyMs.toFixed(1)} ms</span>
      {rerankerMs != null && rerankerMs > 0 && (
        <span className="text-slate-500">(+{rerankerMs.toFixed(0)} rerank)</span>
      )}
    </span>
  );
}
