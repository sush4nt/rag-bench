"""BEIR dataset loader.

The spec sketches ``beir.datasets.data_loader.GenericDataLoader``. In practice
BEIR pins old transitive deps that clash with a modern sentence-transformers /
qdrant stack, so we implement a small, dependency-free loader that reads the
*exact same* on-disk BEIR format (``corpus.jsonl`` / ``queries.jsonl`` /
``qrels/{split}.tsv``). Behaviour and return shapes are identical to BEIR's
``GenericDataLoader(...).load(split=...)``.

Return shapes
-------------
- ``corpus``:  ``{doc_id: {"title": str, "text": str}}``
- ``queries``: ``{query_id: str}``
- ``qrels``:   ``{query_id: {doc_id: relevance_score}}``
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import requests

from ragbench.common.logging import get_logger
from ragbench.common.paths import data_dir, ensure_dir

log = get_logger(__name__)

BEIR_URL_TEMPLATE = (
    "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"
)

Corpus = dict[str, dict[str, str]]
Queries = dict[str, str]
Qrels = dict[str, dict[str, int]]


def download_and_unzip(beir_name: str, out_root: Path | None = None) -> Path:
    """Download ``{beir_name}.zip`` from the BEIR mirror and unzip it.

    Returns the folder containing ``corpus.jsonl`` etc. Idempotent: if the folder
    already exists with a corpus file, the download is skipped.
    """
    out_root = out_root or data_dir()
    ensure_dir(out_root)
    target = out_root / beir_name

    if (target / "corpus.jsonl").exists():
        log.info("Dataset '%s' already present at %s", beir_name, target)
        return target

    url = BEIR_URL_TEMPLATE.format(name=beir_name)
    log.info("Downloading BEIR dataset '%s' from %s", beir_name, url)
    resp = requests.get(url, timeout=600, stream=True)
    resp.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        zf.extractall(out_root)

    if not (target / "corpus.jsonl").exists():
        # Some archives nest an extra folder; find the real corpus.jsonl.
        matches = list(out_root.glob(f"{beir_name}*/corpus.jsonl"))
        if matches:
            target = matches[0].parent
    log.info("Unzipped '%s' -> %s", beir_name, target)
    return target


def _read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def _load_corpus(folder: Path) -> Corpus:
    corpus: Corpus = {}
    for row in _read_jsonl(folder / "corpus.jsonl"):
        doc_id = str(row["_id"])
        corpus[doc_id] = {
            "title": row.get("title", "") or "",
            "text": row.get("text", "") or "",
        }
    return corpus


def _load_queries(folder: Path) -> dict[str, str]:
    queries: dict[str, str] = {}
    for row in _read_jsonl(folder / "queries.jsonl"):
        queries[str(row["_id"])] = row.get("text", "") or ""
    return queries


def _load_qrels(folder: Path, split: str) -> Qrels:
    qrels: Qrels = {}
    qrels_path = folder / "qrels" / f"{split}.tsv"
    with qrels_path.open("r", encoding="utf-8") as fh:
        header = fh.readline()  # "query-id\tcorpus-id\tscore"
        if "query-id" not in header:
            # No header — rewind and treat every line as data.
            fh.seek(0)
        for line in fh:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            qid, did, score = parts[0], parts[1], parts[2]
            qrels.setdefault(qid, {})[did] = int(float(score))
    return qrels


def load_dataset(
    beir_name: str,
    split: str = "test",
    local_path: str | Path | None = None,
) -> tuple[Corpus, Queries, Qrels]:
    """Return ``(corpus, queries, qrels)`` for a BEIR dataset.

    Downloads the dataset if it is not already present locally.
    """
    if local_path is not None and (Path(local_path) / "corpus.jsonl").exists():
        folder = Path(local_path)
    else:
        folder = download_and_unzip(beir_name)

    corpus = _load_corpus(folder)
    queries_all = _load_queries(folder)
    qrels = _load_qrels(folder, split)

    # Restrict queries to those with relevance judgments for this split — this is
    # exactly what BEIR does and keeps eval aligned with the ground truth.
    queries = {qid: text for qid, text in queries_all.items() if qid in qrels}

    log.info(
        "Loaded '%s' [%s]: %d passages, %d queries, %d qrels",
        beir_name,
        split,
        len(corpus),
        len(queries),
        len(qrels),
    )
    return corpus, queries, qrels


def load_qa_answers(dataset: str, qa_hf_dataset: str | None) -> dict[str, str]:
    """Best-effort load of free-text answers keyed by query id (for RAGAS).

    BEIR's FiQA ships relevance judgments but not answer strings; those live in
    the original FiQA QA dataset on HuggingFace. This is optional — RAGAS can
    fall back to using the gold passages as ``ground_truth`` when answers are
    unavailable (e.g. no ``datasets`` install or offline).
    """
    if not qa_hf_dataset:
        return {}
    try:
        from datasets import load_dataset as hf_load  # lazy: optional dep
    except ImportError:
        log.warning("`datasets` not installed; skipping free-text answer load for %s", dataset)
        return {}

    try:
        ds = hf_load(qa_hf_dataset, split="train")
    except Exception as exc:  # noqa: BLE001 - network/dataset availability is best-effort
        log.warning("Could not load QA answers from %s: %s", qa_hf_dataset, exc)
        return {}

    answers: dict[str, str] = {}
    for row in ds:
        qid = row.get("query_id") or row.get("_id") or row.get("id")
        ans = row.get("answer") or row.get("text") or row.get("output")
        if qid is not None and ans:
            answers[str(qid)] = str(ans)
    log.info("Loaded %d free-text answers for %s", len(answers), dataset)
    return answers
