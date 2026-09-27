import { GenerationInfo } from "../api";

export type Mode = "retrieve" | "generate";

interface Props {
  mode: Mode;
  onChange: (mode: Mode) => void;
  generation: GenerationInfo | null;
}

export default function ModeToggle({ mode, onChange, generation }: Props) {
  const canGenerate = generation?.available ?? false;
  const option = (value: Mode, label: string, disabled = false, title?: string) => (
    <button
      onClick={() => onChange(value)}
      disabled={disabled}
      title={title}
      className={`rounded-md px-3 py-1.5 text-sm transition-colors disabled:cursor-not-allowed disabled:opacity-40 ${
        mode === value ? "bg-slate-700 text-white" : "text-slate-400 hover:text-slate-200"
      }`}
    >
      {label}
    </button>
  );

  return (
    <div className="flex flex-wrap items-center gap-3">
      <div className="inline-flex rounded-lg border border-slate-800 bg-slate-900 p-0.5">
        {option("retrieve", "Retrieve")}
        {option(
          "generate",
          "Retrieve + Generate",
          !canGenerate,
          canGenerate ? undefined : `Generation unavailable: ${generation?.reason ?? "unknown"}`
        )}
      </div>
      {generation && mode === "generate" && (
        <span className="text-xs text-slate-500">
          LLM <span className="font-mono text-slate-300">{generation.model}</span> · fixed for all
          pipelines · top-{generation.top_k_default} passages · temp {generation.temperature}
        </span>
      )}
      {generation && !canGenerate && (
        <span className="text-xs text-slate-600">generation off: {generation.reason}</span>
      )}
    </div>
  );
}
