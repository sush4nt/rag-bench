"""The single grounded-answer prompt used for every pipeline."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ragbench.common.protocol import RetrieveResult

ABSTAIN_TOKEN = "INSUFFICIENT_CONTEXT"

SYSTEM_PROMPT = (
    "You answer questions using ONLY the numbered passages provided. "
    "Cite the passage(s) supporting each claim with bracketed numbers such as [1] or [2][3]. "
    "Only cite passage numbers that appear in the context. "
    f"If the passages do not contain enough information to answer, reply with exactly "
    f"{ABSTAIN_TOKEN} and nothing else. Be concise (at most 5 sentences)."
)


@dataclass(frozen=True)
class Prompt:
    system: str
    user: str


def truncate(text: str, max_chars: int) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + " …"


def context_texts(contexts: Sequence[RetrieveResult], max_chars: int) -> list[str]:
    """The passage strings exactly as the LLM sees them (title + truncated body)."""
    out: list[str] = []
    for c in contexts:
        body = truncate(c.text, max_chars)
        out.append(f"{c.title}\n{body}" if c.title else body)
    return out


def build_prompt(question: str, contexts: Sequence[RetrieveResult], max_chars: int) -> Prompt:
    blocks = [f"[{i + 1}] {t}" for i, t in enumerate(context_texts(contexts, max_chars))]
    context_block = "\n\n".join(blocks) if blocks else "(no passages retrieved)"
    user = f"Context:\n{context_block}\n\nQuestion: {question}\nAnswer:"
    return Prompt(system=SYSTEM_PROMPT, user=user)


def is_abstention(answer: str) -> bool:
    return answer.strip().upper().startswith(ABSTAIN_TOKEN)
