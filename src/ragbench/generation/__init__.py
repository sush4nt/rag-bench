"""Answer generation shared by the online ``/ask`` endpoint and offline RAGAS eval.

Keeping one prompt + one generator guarantees the eval scores exactly what is
served.
"""

from ragbench.generation.citations import parse_citations
from ragbench.generation.generator import (
    AnthropicGenerator,
    FakeGenerator,
    GenerationResult,
    GenerationUnavailable,
    Generator,
    build_generator,
)
from ragbench.generation.prompt import ABSTAIN_TOKEN, Prompt, build_prompt, is_abstention
from ragbench.generation.service import answer_question

__all__ = [
    "ABSTAIN_TOKEN",
    "AnthropicGenerator",
    "FakeGenerator",
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
