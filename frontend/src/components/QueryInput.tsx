import { useState } from "react";

interface Props {
  examples: string[];
  loading: boolean;
  onSubmit: (query: string) => void;
}

export default function QueryInput({ examples, loading, onSubmit }: Props) {
  const [value, setValue] = useState("");

  const submit = () => {
    if (value.trim()) onSubmit(value.trim());
  };

  return (
    <div className="flex flex-col gap-3">
      <div className="flex gap-2">
        <input
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && submit()}
          placeholder="Ask a question…"
          className="flex-1 rounded-lg border border-slate-800 bg-slate-900 px-4 py-2.5 text-sm outline-none focus:border-slate-600"
        />
        <button
          onClick={submit}
          disabled={loading || !value.trim()}
          className="rounded-lg bg-indigo-600 px-5 py-2.5 text-sm font-medium text-white transition-colors hover:bg-indigo-500 disabled:opacity-40"
        >
          {loading ? "Running…" : "Compare"}
        </button>
      </div>
      <div className="flex flex-wrap gap-2">
        {examples.map((ex) => (
          <button
            key={ex}
            onClick={() => {
              setValue(ex);
              onSubmit(ex);
            }}
            className="rounded-full border border-slate-800 bg-slate-900/60 px-3 py-1 text-xs text-slate-400 hover:text-slate-200"
          >
            {ex}
          </button>
        ))}
      </div>
    </div>
  );
}
