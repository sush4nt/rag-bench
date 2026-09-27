"""The single grounded-answer prompt used for every pipeline."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ragbench.common.protocol import RetrieveResult

ABSTAIN_TOKEN = "INSUFFICIENT_CONTEXT"

SYSTEM_PROMPT = (
    "You answer from the numbered passages only. Each passage is one source: "
    "its title and its body belong together. Words like \"this language\" or "
    "\"this study\" in the body refer to the title of that same passage.\n"
    "You may paraphrase, list, and combine facts that the passages state. "
    "You may not add facts from outside the passages, invent numbers, or cite "
    "a passage that does not support the claim.\n"
    "Cite the supporting passage after each claim, using markers like [1] or [2][3]. "
    "Only use passage numbers that appear in the context.\n"
    "If at least one passage is about the question, answer from it. "
    "If part of the question is not covered, say what is missing in one short clause "
    "and still give the supported part. Do not refuse the whole question in that case.\n"
    f"Reply with exactly {ABSTAIN_TOKEN} and nothing else only when no passage "
    "is about the subject of the question.\n"
    "Be concise: at most 5 sentences."
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
