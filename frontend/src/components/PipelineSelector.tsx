import { ALL_PIPELINES, PIPELINE_META, PipelineName } from "../api";

interface Props {
  selected: PipelineName[];
  onChange: (next: PipelineName[]) => void;
}

export default function PipelineSelector({ selected, onChange }: Props) {
  const toggle = (p: PipelineName) => {
    if (selected.includes(p)) {
      onChange(selected.filter((x) => x !== p));
    } else {
      onChange([...ALL_PIPELINES].filter((x) => selected.includes(x) || x === p));
    }
  };

  return (
    <div className="flex flex-wrap gap-2">
      {ALL_PIPELINES.map((p) => {
        const meta = PIPELINE_META[p];
        const active = selected.includes(p);
        return (
          <button
            key={p}
            onClick={() => toggle(p)}
            className={`flex items-center gap-2 rounded-lg border px-3 py-1.5 text-sm transition-colors ${
              active
                ? "border-slate-600 bg-slate-800 text-white"
                : "border-slate-800 bg-slate-900 text-slate-500 hover:text-slate-300"
            }`}
            title={meta.blurb}
          >
            <span
              className="h-2.5 w-2.5 rounded-full"
              style={{ backgroundColor: active ? meta.color : "#475569" }}
            />
            {meta.label}
          </button>
        );
      })}
    </div>
  );
}
