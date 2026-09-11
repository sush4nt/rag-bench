import { NavLink, Outlet } from "react-router-dom";

const tabClass = ({ isActive }: { isActive: boolean }) =>
  `px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
    isActive ? "bg-slate-800 text-white" : "text-slate-400 hover:text-slate-200"
  }`;

export default function App() {
  return (
    <div className="min-h-screen">
      <header className="border-b border-slate-800 bg-slate-900/60 backdrop-blur sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-6 py-3 flex items-center gap-6">
          <div className="flex items-center gap-2">
            <span className="text-lg font-bold tracking-tight">RAGBench</span>
            <span className="text-xs text-slate-500 hidden sm:inline">
              retrieval strategy benchmark
            </span>
          </div>
          <nav className="flex items-center gap-1">
            <NavLink to="/fiqa" className={tabClass}>
              FiQA
            </NavLink>
            <NavLink to="/scifact" className={tabClass}>
              SciFact
            </NavLink>
            <NavLink to="/metrics" className={tabClass}>
              Metrics
            </NavLink>
          </nav>
          <a
            href="/docs"
            target="_blank"
            rel="noreferrer"
            className="ml-auto text-xs text-slate-400 hover:text-slate-200"
          >
            API docs ↗
          </a>
        </div>
      </header>
      <main className="max-w-7xl mx-auto px-6 py-6">
        <Outlet />
      </main>
    </div>
  );
}
