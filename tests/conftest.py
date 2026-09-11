"""Shared test fixtures.

We point ``RAGBENCH_DATA_DIR`` at a temp dir and build a *tiny* bm25s index for a
SciFact-shaped corpus. This lets the full API + pipeline path be exercised with
zero model downloads and no running Qdrant. Heavy pipelines (dense/hybrid/
reranked) self-skip unless ``RAGBENCH_RUN_HEAVY=1``.
"""

from __future__ import annotations

import os
import tempfile

# Must be set before any ragbench.common.paths call resolves the data dir.
_TMP_DATA = tempfile.mkdtemp(prefix="ragbench_test_data_")
os.environ["RAGBENCH_DATA_DIR"] = _TMP_DATA
os.environ.setdefault("HF_SPACE", "false")

import pytest  # noqa: E402

from ragbench.config.schema import load_config  # noqa: E402
from ragbench.indexing.bm25_indexer import BM25Indexer  # noqa: E402

# A tiny SciFact-style corpus: distinct keywords so BM25 ranking is predictable.
TINY_CORPUS = {
    "sf_1": {"title": "Aspirin and cancer", "text": "Aspirin reduces the risk of colorectal cancer in adults."},
    "sf_2": {"title": "Vitamin D", "text": "Vitamin D supplementation prevents respiratory tract infections."},
    "sf_3": {"title": "Mediterranean diet", "text": "The Mediterranean diet reduces cardiovascular disease risk."},
    "sf_4": {"title": "Biomaterials", "text": "0-dimensional biomaterials show inductive properties for bone growth."},
    "sf_5": {"title": "Statins", "text": "Statins lower LDL cholesterol and reduce heart attack incidence."},
    "sf_6": {"title": "Insulin", "text": "Insulin resistance is associated with type 2 diabetes mellitus."},
    "sf_7": {"title": "Exercise", "text": "Regular aerobic exercise improves cardiovascular fitness markedly."},
    "sf_8": {"title": "Smoking", "text": "Tobacco smoking is a leading cause of lung cancer worldwide."},
    "sf_9": {"title": "Sleep", "text": "Chronic sleep deprivation impairs immune system function."},
    "sf_10": {"title": "Antibiotics", "text": "Overuse of antibiotics drives bacterial resistance to treatment."},
}


@pytest.fixture(scope="session", autouse=True)
def _build_tiny_index():
    """Build a persisted bm25s index for the 'scifact' dataset in the temp data dir."""
    cfg = load_config("configs/scifact.yaml")
    BM25Indexer(cfg).build(TINY_CORPUS)
    yield


@pytest.fixture()
def scifact_config():
    return load_config("configs/scifact.yaml")


@pytest.fixture()
def bm25_pipeline(scifact_config):
    from ragbench.retrieval.bm25_pipeline import BM25Pipeline

    return BM25Pipeline(scifact_config)


@pytest.fixture()
def api_client():
    from fastapi.testclient import TestClient

    from ragbench.serving.app import app

    with TestClient(app) as client:
        yield client


HEAVY = pytest.mark.skipif(
    os.environ.get("RAGBENCH_RUN_HEAVY") != "1",
    reason="heavy pipeline (model download / Qdrant); set RAGBENCH_RUN_HEAVY=1 to run",
)
