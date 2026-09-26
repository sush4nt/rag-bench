"""Grounded answer generation with inline citation markers.

The Anthropic SDK is an optional extra (``uv sync --extra ragas``). Callers
that cannot construct a generator receive :class:`GeneratorUnavailable` and
should surface that as a client error rather than inventing an answer.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


class GeneratorUnavailable(RuntimeError):
    """The configured generator cannot be constructed in this environment."""


class GenerationFailed(RuntimeError):
    """The generator was constructed but the completion call failed."""


@dataclass(frozen=True)
class GenerationResult:
    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0


def build_prompt(question: str, contexts: list[str]) -> str:
    """Prompt that numbers passages so the model can cite them as ``[n]``."""
    context_block = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(contexts))
    return (
        "Answer the question using ONLY the numbered context passages. "
        "Cite supporting passages inline as [1], [2], and so on. "
        "If the context is insufficient, say so.\n\n"
        f"Context:\n{context_block}\n\nQuestion: {question}\nAnswer:"
    )


class AnthropicGenerator:
    """Answer generator backed by the Anthropic Messages API."""

    def __init__(self, model: str, max_tokens: int = 400):
        import anthropic  # lazy: optional extra

        self.model = model
        self.max_tokens = max_tokens
        self.client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    def generate(self, question: str, contexts: list[str]) -> GenerationResult:
        try:
            msg = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=[{"role": "user", "content": build_prompt(question, contexts)}],
            )
        except Exception as exc:  # noqa: BLE001 - provider errors become a typed failure
            raise GenerationFailed(f"Generation failed ({self.model}): {exc}") from exc
        usage = getattr(msg, "usage", None)
        return GenerationResult(
            text=msg.content[0].text.strip(),
            model=self.model,
            input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
        )


def build_generator(model: str) -> AnthropicGenerator:
    """Construct the default generator or raise :class:`GeneratorUnavailable`."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise GeneratorUnavailable(
            "ANTHROPIC_API_KEY is required to generate an answer. "
            "Set it in the environment. The Anthropic SDK is installed with "
            "`uv sync --extra ragas`."
        )
    try:
        import anthropic  # noqa: F401
    except ImportError as exc:
        raise GeneratorUnavailable(
            "Anthropic SDK is not installed. Run: uv sync --extra ragas"
        ) from exc
    return AnthropicGenerator(model)
