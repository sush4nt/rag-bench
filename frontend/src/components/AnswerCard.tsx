import { Fragment, ReactNode } from "react";
import { AskResponse } from "../api";

// Mirrors the backend parser: "[1]", "[2, 3]"; adjacent "[2][3]" match separately.
const MARKER_RE = /\[(\d+(?:\s*,\s*\d+)*)\]/g;

interface Props {
  response: AskResponse;
  accent: string;
  onCite: (contextIndex: number) => void;
}

function renderWithChips(
  answer: string,
  nContexts: number,
  accent: string,
  onCite: (i: number) => void
): ReactNode[] {
  const out: ReactNode[] = [];
  let last = 0;
  for (const match of answer.matchAll(MARKER_RE)) {
    const start = match.index ?? 0;
    out.push(answer.slice(last, start));
    const markers = match[1].split(",").map((m) => parseInt(m.trim(), 10));
    out.push(
      <Fragment key={start}>
        {markers.map((m) => {
          const valid = m >= 1 && m <= nContexts;
          return valid ? (
            <button
              key={m}
              onClick={() => onCite(m - 1)}
              title={`Jump to passage #${m}`}
              className="mx-0.5 rounded px-1 font-mono text-[10px] font-semibold text-slate-950 hover:brightness-110"
              style={{ backgroundColor: accent }}
            >
              {m}
            </button>
          ) : (
            <span
              key={m}
              title="Cites a passage that was never retrieved"
              className="mx-0.5 rounded bg-red-500/80 px-1 font-mono text-[10px] font-semibold text-white"
            >
              {m}?
            </span>
          );
        })}
      </Fragment>
    );
    last = start + match[0].length;
  }
  out.push(answer.slice(last));
  return out;
}

export default function AnswerCard({ response, accent, onCite }: Props) {
  const { answer, abstained, citations, invalid_citations, contexts, token_usage, model } =
    response;
  const citedDocs = new Set(citations.map((c) => c.doc_id)).size;

  return (
    <div className="rounded-lg border border-slate-800 bg-slate-950/60 p-3">
      {abstained ? (
        <div className="rounded-md border border-amber-900 bg-amber-950/40 px-3 py-2 text-xs text-amber-300">
          <span className="font-semibold">Insufficient context.</span> The model declined to
          answer from these passages.
        </div>
      ) : (
        <p className="whitespace-pre-wrap text-sm leading-relaxed text-slate-200">
          {renderWithChips(answer, contexts.length, accent, onCite)}
        </p>
      )}
      <div className="mt-3 flex flex-wrap items-center gap-x-3 gap-y-1 border-t border-slate-800 pt-2 font-mono text-[10px] text-slate-500">
        <span>
          cited {citedDocs}/{contexts.length}
        </span>
        {invalid_citations.length > 0 && (
          <span className="text-red-400">{invalid_citations.length} invalid</span>
        )}
        <span>
          {token_usage.input_tokens.toLocaleString()} in · {token_usage.output_tokens} out
        </span>
        <span className="ml-auto truncate" title={model}>
          {model}
        </span>
      </div>
    </div>
  );
}
