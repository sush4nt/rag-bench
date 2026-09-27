"""Answer generation shared by the online ``/ask`` endpoint and offline RAGAS eval.

``/ask`` and RAGAS share the prompt. The ``/ask`` model comes from
``generation.provider``; RAGAS answers and the judge stay on Anthropic.
"""

from ragbench.generation.citations import parse_citations
from ragbench.generation.generator import (
    AnthropicGenerator,
    FakeGenerator,
    GenerationResult,
    GenerationUnavailable,
    Generator,
    OpenAIGenerator,
    build_generator,
)
from ragbench.generation.prompt import ABSTAIN_TOKEN, Prompt, build_prompt, is_abstention
from ragbench.generation.service import answer_question

__all__ = [
    "ABSTAIN_TOKEN",
    "AnthropicGenerator",
    "FakeGenerator",
    "OpenAIGenerator",
    "GenerationResult",
    "GenerationUnavailable",
    "Generator",
    "Prompt",
    "answer_question",
    "build_generator",
    "build_prompt",
    "is_abstention",
    "parse_citations",
]
