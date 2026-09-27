"""LLM generators behind a tiny protocol.

``anthropic`` is an optional dependency (``uv sync --extra ragas``); it is
imported lazily and any missing SDK / API key surfaces as
:class:`GenerationUnavailable` so the API can return a clear 503.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from time import perf_counter
from typing import Protocol

from ragbench.config.schema import GenerationConfig
from ragbench.generation.prompt import ABSTAIN_TOKEN, Prompt


class GenerationUnavailable(RuntimeError):
    """Generation cannot run (disabled, SDK missing, or no API key)."""


@dataclass(frozen=True)
class GenerationResult:
    text: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float


class Generator(Protocol):
    model: str

    def generate(self, prompt: Prompt) -> GenerationResult: ...


class AnthropicGenerator:
    def __init__(self, model: str, max_tokens: int = 400, temperature: float = 0.0):
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise GenerationUnavailable("ANTHROPIC_API_KEY is not set")
        try:
            import anthropic
        except ImportError as exc:
            raise GenerationUnavailable(
                "anthropic SDK not installed. Run: uv sync --extra ragas"
            ) from exc

        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    def generate(self, prompt: Prompt) -> GenerationResult:
        t0 = perf_counter()
        msg = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=prompt.system,
            messages=[{"role": "user", "content": prompt.user}],
            # anthropic>=1.0 dropped `temperature` from the typed signature.
            extra_body={"temperature": self.temperature},
        )
        latency_ms = (perf_counter() - t0) * 1000.0
        text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
        return GenerationResult(
            text=text.strip(),
            model=msg.model or self.model,
            input_tokens=msg.usage.input_tokens,
            output_tokens=msg.usage.output_tokens,
            latency_ms=latency_ms,
        )


class FakeGenerator:
    """Deterministic, offline generator for tests and keyless local demos.

    Cites the first two passages; abstains when the prompt has no passages.
    """

    def __init__(self, model: str = "fake"):
        self.model = model

    def generate(self, prompt: Prompt) -> GenerationResult:
        t0 = perf_counter()
        n_passages = len(re.findall(r"^\[\d+\] ", prompt.user, flags=re.MULTILINE))
        if n_passages == 0:
            text = ABSTAIN_TOKEN
        else:
            cited = "".join(f"[{i}]" for i in range(1, min(n_passages, 2) + 1))
            text = f"Based on the retrieved passages, here is a grounded answer {cited}."
        return GenerationResult(
            text=text,
            model=self.model,
            input_tokens=len(prompt.system.split()) + len(prompt.user.split()),
            output_tokens=len(text.split()),
            latency_ms=(perf_counter() - t0) * 1000.0,
        )


def resolve_provider(cfg: GenerationConfig) -> str:
    """``RAGBENCH_GENERATION_PROVIDER`` overrides the YAML (e.g. ``fake`` in tests)."""
    return os.environ.get("RAGBENCH_GENERATION_PROVIDER", cfg.provider).lower()


def build_generator(cfg: GenerationConfig) -> Generator:
    if not cfg.enabled:
        raise GenerationUnavailable("generation is disabled in the dataset config")
    provider = resolve_provider(cfg)
    if provider == "fake":
        return FakeGenerator(model=f"fake:{cfg.model}")
    if provider == "anthropic":
        return AnthropicGenerator(cfg.model, cfg.max_tokens, cfg.temperature)
    raise GenerationUnavailable(f"unknown generation provider '{provider}'")
