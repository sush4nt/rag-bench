"""Index build CLI.

Usage:
    python -m ragbench.indexing.build --config configs/scifact.yaml
    python -m ragbench.indexing.build --config configs/fiqa.yaml [--only bm25|qdrant]

Downloads the dataset (if needed), builds the bm25s lexical index, and uploads
dense + sparse vectors to Qdrant.
"""

from __future__ import annotations

import argparse
import sys
import time

from ragbench.common.logging import get_logger
from ragbench.config.schema import load_config
from ragbench.data.loader import load_dataset
from ragbench.indexing.bm25_indexer import BM25Indexer
from ragbench.indexing.qdrant_indexer import QdrantIndexer

log = get_logger(__name__)


def build_index(config_path: str, only: str | None = None) -> None:
    cfg = load_config(config_path)
    log.info("=== Indexing '%s' from %s ===", cfg.dataset.name, config_path)

    corpus, queries, qrels = load_dataset(
        cfg.dataset.beir_name, split=cfg.dataset.split, local_path=cfg.dataset.local_path
    )

    if only in (None, "bm25"):
        t0 = time.perf_counter()
        n = BM25Indexer(cfg).build(corpus)
        log.info("bm25s: indexed %d passages in %.1fs", n, time.perf_counter() - t0)

    if only in (None, "qdrant"):
        t0 = time.perf_counter()
        n = QdrantIndexer(cfg).build(corpus)
        log.info("qdrant: indexed %d passages in %.1fs", n, time.perf_counter() - t0)

    log.info("=== Done indexing '%s' ===", cfg.dataset.name)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build RAGBench indexes.")
    parser.add_argument("--config", required=True, help="Path to dataset config YAML")
    parser.add_argument(
        "--only",
        choices=["bm25", "qdrant"],
        default=None,
        help="Build only one backend (default: both)",
    )
    args = parser.parse_args(argv)
    build_index(args.config, only=args.only)
    return 0


if __name__ == "__main__":
    sys.exit(main())
