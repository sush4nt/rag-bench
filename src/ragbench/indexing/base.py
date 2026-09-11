"""Abstract indexer interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ragbench.config.schema import RagbenchConfig
from ragbench.data.loader import Corpus


class Indexer(ABC):
    """Builds and persists a retrieval index for one dataset."""

    def __init__(self, config: RagbenchConfig):
        self.config = config
        self.dataset = config.dataset.name

    @abstractmethod
    def build(self, corpus: Corpus) -> int:
        """Build + persist the index from ``corpus``. Return #passages indexed."""
        ...

    @abstractmethod
    def exists(self) -> bool:
        """Whether a persisted index is already present."""
        ...
