"""RAGAS generation evaluation (opt-in).

Evaluates the full RAG loop: retrieve context with a pipeline, generate an
answer with an LLM, then score with RAGAS. Primary KPIs are **Faithfulness**
and **Context Recall** (they most directly differentiate retrieval quality);
**Answer Relevance** and **Context Precision** round out the picture.

Everything here imports its heavy/optional dependencies lazily so the rest of
the project installs and runs without ``ragas`` / ``anthropic``.
Install with: ``uv sync --extra ragas`` and set ``ANTHROPIC_API_KEY``.
"""

from __future__ import annotations

import os

from ragbench.common.logging import get_logger
from ragbench.common.protocol import RetrieveRequest
from ragbench.data.chunker import passage_text

log = get_logger(__name__)


def build_qa_samples(
    queries: dict[str, str],
    qrels: dict[str, dict[str, int]],
    corpus: dict[str, dict[str, str]],
    answers: dict[str, str] | None = None,
    limit: int | None = None,
) -> list[dict]:
    """Assemble ``[{question, ground_truth, gold_contexts}]`` for RAGAS.

    ``ground_truth`` prefers a free-text answer (FiQA QA) and falls back to the
    concatenated gold passages when no answer string is available.
    """
    answers = answers or {}
    samples: list[dict] = []
    for qid, question in queries.items():
        gold_doc_ids = [d for d, s in qrels.get(qid, {}).items() if s > 0]
        gold_contexts = [passage_text(corpus[d]) for d in gold_doc_ids if d in corpus]
        if not gold_contexts:
            continue
        ground_truth = answers.get(qid) or " ".join(gold_contexts)
        samples.append(
            {
                "query_id": qid,
                "question": question,
                "ground_truth": ground_truth,
                "gold_contexts": gold_contexts,
            }
        )
        if limit and len(samples) >= limit:
            break
    return samples


class AnthropicGenerator:
    """Thin wrapper around the Anthropic Messages API for answer generation."""

    def __init__(self, model: str):
        import anthropic  # lazy

        self.model = model
        self.client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    def generate(self, question: str, contexts: list[str]) -> str:
        context_block = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(contexts))
        prompt = (
            "Answer the question using ONLY the context below. "
            "If the context is insufficient, say so.\n\n"
            f"Context:\n{context_block}\n\nQuestion: {question}\nAnswer:"
        )
        msg = self.client.messages.create(
            model=self.model,
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text.strip()


def _ragas_llm(model: str):
    """Build a RAGAS-compatible LLM wrapper backed by Anthropic."""
    from langchain_anthropic import ChatAnthropic
    from ragas.llms import LangchainLLMWrapper

    return LangchainLLMWrapper(ChatAnthropic(model=model, max_tokens=1024, temperature=0.0))


def run_ragas_eval(
    pipeline,
    qa_samples: list[dict],
    ragas_llm_model: str,
    sample_size: int = 100,
    top_k: int = 5,
):
    """Run RAGAS over ``qa_samples`` for one pipeline; return a pandas DataFrame.

    Raises a clear error if the optional deps / API key are missing.
    """
    if "ANTHROPIC_API_KEY" not in os.environ:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is required for RAGAS generation eval. "
            "Set it in .env or export it before running eval."
        )
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
    except ImportError as exc:  # pragma: no cover - optional extra
        raise RuntimeError(
            "RAGAS extras not installed. Run: uv sync --extra ragas"
        ) from exc

    generator = AnthropicGenerator(ragas_llm_model)
    judge = _ragas_llm(ragas_llm_model)

    rows: list[dict] = []
    for sample in qa_samples[:sample_size]:
        resp = pipeline.retrieve(
            RetrieveRequest(query=sample["question"], pipeline=pipeline.name, top_k=top_k)
        )
        contexts = [r.text for r in resp.results] or [""]
        answer = generator.generate(sample["question"], contexts)
        rows.append(
            {
                "question": sample["question"],
                "answer": answer,
                "contexts": contexts,
                "ground_truth": sample["ground_truth"],
                "reference": sample["ground_truth"],
            }
        )

    log.info("Running RAGAS on %d samples for pipeline '%s'", len(rows), pipeline.name)
    result = evaluate(
        Dataset.from_list(rows),
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
        llm=judge,
    )
    return result.to_pandas()
