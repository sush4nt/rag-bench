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
from ragbench.config.schema import GenerationConfig
from ragbench.data.chunker import passage_text
from ragbench.generation.generator import AnthropicGenerator
from ragbench.generation.prompt import build_prompt, context_texts

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


def _ragas_llm(model: str):
    """Build a RAGAS-compatible LLM wrapper backed by Anthropic."""
    from langchain_anthropic import ChatAnthropic
    from ragas.llms import LangchainLLMWrapper

    return LangchainLLMWrapper(ChatAnthropic(model=model, max_tokens=1024, temperature=0.0))


def run_ragas_eval(
    pipeline,
    qa_samples: list[dict],
    ragas_llm_model: str,
    generation_cfg: GenerationConfig | None = None,
    sample_size: int = 100,
):
    """Run RAGAS over ``qa_samples`` for one pipeline; return a pandas DataFrame.

    Answers use the same prompt and ``top_k`` as ``/ask``, but always Claude via
    ``ragas_llm_model`` (judge and answer step). ``/ask`` may use a different
    provider, such as OpenAI, without changing this eval.

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

    generation_cfg = generation_cfg or GenerationConfig()
    generator = AnthropicGenerator(ragas_llm_model, max_tokens=400, temperature=0.0)
    judge = _ragas_llm(ragas_llm_model)

    rows: list[dict] = []
    for sample in qa_samples[:sample_size]:
        resp = pipeline.retrieve(
            RetrieveRequest(
                query=sample["question"],
                pipeline=pipeline.name,
                top_k=generation_cfg.top_k_default,
            )
        )
        prompt = build_prompt(sample["question"], resp.results, generation_cfg.max_context_chars)
        answer = generator.generate(prompt).text
        contexts = context_texts(resp.results, generation_cfg.max_context_chars) or [""]
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
