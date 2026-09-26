"""RAGBench FastAPI application.

Single-port service (mlserve pattern): dataset-namespaced JSON API under
``/api/*``, Prometheus at ``/metrics``, and the built React SPA served from
``frontend/dist`` when present.
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse

from ragbench.common.logging import get_logger
from ragbench.common.paths import dataset_dir, repo_root
from ragbench.serving import metrics
from ragbench.serving.registry import get_registry
from ragbench.serving.routers import fiqa, scifact

log = get_logger(__name__)

EXAMPLE_QUERIES = {
    "fiqa": [
        "What are the risks of investing in REITs?",
        "How does dollar cost averaging work?",
        "Difference between ETF and index fund?",
        "How to evaluate a company's P/E ratio?",
    ],
    "scifact": [
        "0-dimensional biomaterials show inductive properties.",
        "The Mediterranean diet reduces the risk of cardiovascular disease.",
        "Aspirin lowers the risk of colorectal cancer.",
        "Vitamin D supplementation prevents respiratory infections.",
    ],
}

_FRONTEND_DIST = repo_root() / "frontend" / "dist"


def _seed_gauges() -> None:
    """On startup, seed index-size + eval gauges from disk so panels aren't empty."""
    registry = get_registry()
    for dataset in registry.available_datasets():
        try:
            status = registry.status(dataset)
            metrics.set_index_size(dataset, status.get("index_size_passages", 0))
        except Exception as exc:  # noqa: BLE001
            log.debug("Could not seed index size for %s: %s", dataset, exc)

        eval_file = dataset_dir(dataset) / "eval_latest.json"
        if eval_file.exists():
            try:
                metrics.set_eval_gauges(json.loads(eval_file.read_text()))
                log.info("Seeded eval gauges for '%s' from %s", dataset, eval_file)
            except Exception as exc:  # noqa: BLE001
                log.warning("Failed to seed eval gauges for %s: %s", dataset, exc)

        serving_file = dataset_dir(dataset) / "serving_latest.json"
        if serving_file.exists():
            try:
                metrics.set_serving_gauges(json.loads(serving_file.read_text()))
                log.info("Seeded serving gauges for '%s' from %s", dataset, serving_file)
            except Exception as exc:  # noqa: BLE001
                log.warning("Failed to seed serving gauges for %s: %s", dataset, exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Starting RAGBench (available datasets: %s)", get_registry().available_datasets())
    _seed_gauges()
    yield


app = FastAPI(title="RAGBench", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(scifact.router, prefix="/api/scifact", tags=["scifact"])
app.include_router(fiqa.router, prefix="/api/fiqa", tags=["fiqa"])

# Prometheus /metrics + default HTTP instrumentation.
metrics.setup_instrumentator(app)


@app.get("/health", include_in_schema=False)
async def health() -> dict:
    return {"status": "ok"}


@app.get("/api/datasets", tags=["meta"])
async def datasets() -> dict:
    registry = get_registry()
    available = registry.available_datasets()
    return {
        "datasets": [
            {
                "name": d,
                "router_prefix": registry.get_config(d).serving.router_prefix,
                "example_queries": EXAMPLE_QUERIES.get(d, []),
            }
            for d in available
        ]
    }


# --- Frontend (SPA) -----------------------------------------------------------
if _FRONTEND_DIST.exists():
    from fastapi.staticfiles import StaticFiles

    app.mount("/assets", StaticFiles(directory=_FRONTEND_DIST / "assets"), name="assets")

    @app.get("/", include_in_schema=False)
    async def _root() -> FileResponse:
        return FileResponse(_FRONTEND_DIST / "index.html")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def _spa_fallback(full_path: str):
        # API + metrics are matched earlier; everything else -> SPA index.html.
        candidate = _FRONTEND_DIST / full_path
        if candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(_FRONTEND_DIST / "index.html")
else:

    @app.get("/", include_in_schema=False)
    async def _root_redirect() -> RedirectResponse:
        # No built frontend: point humans at the API docs.
        return RedirectResponse(url="/docs")
