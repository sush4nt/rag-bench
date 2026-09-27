import { AskState, PipelineName } from "../api";
import AnswerColumn from "./AnswerColumn";

interface Props {
  pipelines: PipelineName[];
  states: Partial<Record<PipelineName, AskState>>;
}

export default function AnswerGrid({ pipelines, states }: Props) {
  const shown = pipelines.filter((p) => states[p]);
  if (shown.length === 0) {
    return (
      <div className="rounded-xl border border-dashed border-slate-800 py-16 text-center text-slate-500">
        Enter a question and hit <span className="text-slate-300">Ask</span> to generate one
        answer per retrieval strategy with the same LLM.
      </div>
    );
  }

  const scaleMs = Math.max(
    0,
    ...shown.map((p) => {
      const s = states[p];
      return s?.status === "done" ? s.data.timings.total_ms : 0;
    })
  );
  const cols =
    shown.length >= 4 ? "xl:grid-cols-4" : shown.length === 3 ? "xl:grid-cols-3" : "xl:grid-cols-2";

  return (
    <div className={`grid grid-cols-1 items-start gap-4 md:grid-cols-2 ${cols}`}>
      {shown.map((p) => (
        <AnswerColumn key={p} pipeline={p} state={states[p]!} scaleMs={scaleMs} />
      ))}
    </div>
  );
}
