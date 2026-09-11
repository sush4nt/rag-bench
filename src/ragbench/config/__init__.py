"""Pydantic config models + YAML loader."""

from ragbench.config.schema import RagbenchConfig, load_config

__all__ = ["RagbenchConfig", "load_config"]
