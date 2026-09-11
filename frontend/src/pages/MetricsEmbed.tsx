// Embeds the provisioned Grafana dashboard. In local dev Grafana runs on :3001
// (docker compose). Override with VITE_GRAFANA_URL if hosted elsewhere.

const GRAFANA_URL =
  (import.meta.env.VITE_GRAFANA_URL as string | undefined) ?? "http://localhost:3001";
const DASHBOARD_PATH = "/d/ragbench-main/ragbench-retrieval-strategy-comparison?kiosk";

export default function MetricsEmbed() {
  return (
    <div className="flex flex-col gap-3">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">Live metrics</h1>
          <p className="text-sm text-slate-500">
            Quality vs latency across all four pipelines — served by Prometheus + Grafana.
          </p>
        </div>
        <a
          href={`${GRAFANA_URL}${DASHBOARD_PATH}`}
          target="_blank"
          rel="noreferrer"
          className="text-xs text-slate-400 hover:text-slate-200"
        >
          Open in Grafana ↗
        </a>
      </div>
      <div className="overflow-hidden rounded-2xl border border-slate-800 bg-slate-900/30">
        <iframe
          title="RAGBench Grafana dashboard"
          src={`${GRAFANA_URL}${DASHBOARD_PATH}`}
          className="h-[80vh] w-full"
        />
      </div>
    </div>
  );
}
