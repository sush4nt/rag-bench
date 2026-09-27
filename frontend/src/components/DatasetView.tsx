import { useEffect, useMemo, useRef, useState } from "react";
import {
  ALL_PIPELINES,
  api,
  AskState,
  DatasetInfo,
  DatasetStatus,
  GenerationInfo,
  PipelineName,
  RetrieveResponse,
} from "../api";
import AnswerComparison from "./AnswerComparison";
import AnswerGrid from "./AnswerGrid";
import MetricsPanel from "./MetricsPanel";
import ModeToggle, { Mode } from "./ModeToggle";
import PipelineSelector from "./PipelineSelector";
import QueryInput from "./QueryInput";
import ResultsGrid from "./ResultsGrid";

const FALLBACK_EXAMPLES: Record<string, string[]> = {
  fiqa: [
    "What are the risks of investing in REITs?",
    "How does dollar cost averaging work?",
    "Difference between ETF and index fund?",
    "How to evaluate a company's P/E ratio?",
  ],
  scifact: [
    "Aspirin lowers the risk of colorectal cancer.",
    "The Mediterranean diet reduces cardiovascular disease.",
    "Vitamin D supplementation prevents respiratory infections.",
  ],
};

export default function DatasetView({ dataset }: { dataset: string }) {
  const [selected, setSelected] = useState<PipelineName[]>([...ALL_PIPELINES]);
  const [responses, setResponses] = useState<RetrieveResponse[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [status, setStatus] = useState<DatasetStatus | null>(null);
  const [info, setInfo] = useState<DatasetInfo | null>(null);
  const [mode, setMode] = useState<Mode>("retrieve");
  const [generation, setGeneration] = useState<GenerationInfo | null>(null);
  const [askStates, setAskStates] = useState<Partial<Record<PipelineName, AskState>>>({});
  const [askPipelines, setAskPipelines] = useState<PipelineName[]>([]);
  // Ignores late responses from a previous question.
  const runId = useRef(0);

  useEffect(() => {
    setResponses([]);
    setAskStates({});
    setError(null);
    runId.current += 1;
    api.status(dataset).then(setStatus).catch(() => setStatus(null));
    api
      .generationInfo(dataset)
      .then((g) => {
        setGeneration(g);
        if (!g.available) setMode("retrieve");
      })
      .catch(() => {
        setGeneration(null);
        setMode("retrieve");
      });
    api
      .datasets()
      .then((d) => setInfo(d.datasets.find((x) => x.name === dataset) ?? null))
      .catch(() => setInfo(null));
  }, [dataset]);

  const examples = useMemo(
    () => info?.example_queries ?? FALLBACK_EXAMPLES[dataset] ?? [],
    [info, dataset]
  );

  const ask = async (query: string) => {
    const id = ++runId.current;
    const pipelines = [...selected];
    setError(null);
    setAskPipelines(pipelines);
    setAskStates(Object.fromEntries(pipelines.map((p) => [p, { status: "loading" }])));
    setLoading(true);
    // One request per pipeline so each column fills in as soon as it's ready
    // and one failure doesn't blank the others.
    await Promise.allSettled(
      pipelines.map((p) =>
        api.ask(dataset, query, p).then(
          (data) => {
            if (runId.current === id)
              setAskStates((s) => ({ ...s, [p]: { status: "done", data } }));
          },
          (e) => {
            if (runId.current === id)
              setAskStates((s) => ({
                ...s,
                [p]: { status: "error", error: String(e instanceof Error ? e.message : e) },
              }));
          }
        )
      )
    );
    if (runId.current === id) setLoading(false);
  };

  const run = async (query: string) => {
    if (selected.length === 0) {
      setError("Select at least one pipeline.");
      return;
    }
    if (mode === "generate") return ask(query);
    setLoading(true);
    setError(null);
    try {
      const res = await api.batch(dataset, query, selected, 5);
      setResponses(res.responses);
    } catch (e) {
      setError(String(e instanceof Error ? e.message : e));
      setResponses([]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col gap-6">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h1 className="text-2xl font-bold capitalize">{dataset}</h1>
          <p className="text-sm text-slate-500">
            {dataset === "fiqa"
              ? "Financial QA — primary benchmark (57,638 passages)"
              : "Scientific claims — dev dataset (5,183 passages)"}
          </p>
        </div>
        {status && (
          <div className="flex items-center gap-2 text-xs">
            <span
              className={`h-2 w-2 rounded-full ${
                status.ready ? "bg-emerald-400" : "bg-red-400"
              }`}
            />
            <span className="text-slate-400">
              {status.ready
                ? `index ready · ${status.index_size_passages.toLocaleString()} passages`
                : "index not built"}
            </span>
          </div>
        )}
      </div>

      <div className="flex flex-col gap-4 rounded-2xl border border-slate-800 bg-slate-900/30 p-4">
        <ModeToggle mode={mode} onChange={setMode} generation={generation} />
        <QueryInput
          examples={examples}
          loading={loading}
          onSubmit={run}
          submitLabel={mode === "generate" ? "Ask" : "Compare"}
        />
        <PipelineSelector selected={selected} onChange={setSelected} />
      </div>

      {error && (
        <div className="rounded-lg border border-red-900 bg-red-950/40 px-4 py-3 text-sm text-red-300">
          {error}
        </div>
      )}

      {mode === "generate" ? (
        <>
          <AnswerComparison pipelines={askPipelines} states={askStates} />
          <AnswerGrid pipelines={askPipelines} states={askStates} />
        </>
      ) : (
        <ResultsGrid responses={responses} loading={loading} selectedCount={selected.length} />
      )}

      <div>
        <h2 className="mb-2 text-sm font-semibold uppercase tracking-wide text-slate-400">
          Last eval run
        </h2>
        <MetricsPanel dataset={dataset} />
      </div>
    </div>
  );
}
