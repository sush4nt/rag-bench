"""Factory that builds the symmetric per-dataset router.

Both ``/api/scifact/*`` and ``/api/fiqa/*`` expose identical routes; the only
difference is the bound dataset name. Keeping this in one factory guarantees the
two datasets never drift apart.
"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from ragbench.common.logging import get_logger
from ragbench.common.protocol import (
    BatchRetrieveRequest,
    BatchRetrieveResponse,
    RetrieveRequest,
    RetrieveResponse,
)
from ragbench.serving import metrics
from ragbench.serving.registry import get_registry

log = get_logger(__name__)


class EvalRunRequest(BaseModel):
    pipelines: list[str] | None = None
    with_ragas: bool = False
    background: bool = True


def _require_available(dataset: str) -> None:
    registry = get_registry()
    if dataset not in registry._configs:
        raise HTTPException(status_code=404, detail=f"Unknown dataset '{dataset}'")
    if not registry.is_available(dataset):
        raise HTTPException(
            status_code=503,
            detail=f"Dataset '{dataset}' is disabled (HF_SPACE mode serves SciFact only)",
        )


async def _run_single(dataset: str, request: RetrieveRequest) -> RetrieveResponse:
    registry = get_registry()
    try:
        pipeline = registry.get_pipeline(dataset, request.pipeline)
    except Exception as exc:  # noqa: BLE001 - index/model may be missing
        raise HTTPException(
            status_code=503,
            detail=f"Pipeline '{request.pipeline}' unavailable for '{dataset}': {exc}",
        ) from exc

    response: RetrieveResponse = await run_in_threadpool(pipeline.retrieve, request)
    metrics.observe_retrieval(
        dataset,
        request.pipeline,
        response.latency_ms / 1000.0,
        (response.reranker_latency_ms / 1000.0) if response.reranker_latency_ms else None,
    )
    return response


def make_router(dataset: str) -> APIRouter:
    router = APIRouter()

    @router.post("/retrieve", response_model=RetrieveResponse)
    async def retrieve(request: RetrieveRequest) -> RetrieveResponse:
        _require_available(dataset)
        return await _run_single(dataset, request)

    @router.post("/retrieve/batch", response_model=BatchRetrieveResponse)
    async def retrieve_batch(request: BatchRetrieveRequest) -> BatchRetrieveResponse:
        _require_available(dataset)
        subs = [
            RetrieveRequest(query=request.query, pipeline=p, top_k=request.top_k)
            for p in request.pipelines
        ]
        responses = await asyncio.gather(*(_run_single(dataset, s) for s in subs))
        return BatchRetrieveResponse(
            query=request.query, dataset=dataset, responses=list(responses)
        )

    @router.get("/status")
    async def status() -> dict:
        _require_available(dataset)
        return get_registry().status(dataset)

    @router.get("/pipelines")
    async def pipelines() -> dict:
        _require_available(dataset)
        return {
            "dataset": dataset,
            "pipelines": [
                {"name": name, **caps}
                for name, caps in get_registry().pipeline_capabilities().items()
            ],
        }

    @router.get("/eval/latest")
    async def eval_latest() -> dict:
        _require_available(dataset)
        import json

        from ragbench.evaluation.runner import eval_result_path

        path = eval_result_path(dataset)
        if not path.exists():
            return {"dataset": dataset, "pipelines": {}, "available": False}
        summary = json.loads(path.read_text())
        summary["available"] = True
        return summary

    @router.post("/eval/run")
    async def eval_run(req: EvalRunRequest, background_tasks: BackgroundTasks) -> dict:
        _require_available(dataset)
        from ragbench.common.paths import repo_root
        from ragbench.evaluation.runner import run_eval

        config_path = str(repo_root() / f"configs/{dataset}.yaml")

        def _job():
            summary = run_eval(
                config_path, pipelines=req.pipelines, with_ragas=req.with_ragas
            )
            metrics.set_eval_gauges(summary)
            return summary

        if req.background:
            background_tasks.add_task(_job)
            return {"status": "accepted", "dataset": dataset, "message": "eval running in background"}
        summary = await run_in_threadpool(_job)
        return {"status": "completed", "summary": summary}

    return router
