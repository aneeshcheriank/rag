"""
Evaluate a RAG pipeline using Ragas metrics.

Run:  python evaluate.py
"""

import json
import os
import logging

import instructor
import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI

from ragas import evaluate
from ragas.llms import llm_factory
from ragas.embeddings import HuggingFaceEmbeddings

# These private-module imports are the OLD-style classes that inherit from
# ``Metric`` (via SingleTurnMetric).  The public ``ragas.metrics.collections``
# variants use ``SimpleBaseMetric`` which is NOT recognised by
# ``evaluate()`` in ragas 0.4.x.  Remove the underscore-prefixed imports
# once ragas releases a version where evaluate() accepts SimpleBaseMetric.
from ragas.metrics._context_precision import ContextPrecision
from ragas.metrics._context_recall import ContextRecall
from ragas.metrics._faithfulness import Faithfulness
from ragas.metrics._answer_relevance import AnswerRelevancy

from src import config
from src.pipeline import rag

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

load_dotenv(find_dotenv())

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
EVAL_DATASET_PATH = "data/evaluation_dataset.json"
OUTPUT_PATH = "baseline_vectorstore_results.csv"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_eval_data(path: str) -> list[dict]:
    """Load the evaluation dataset from a JSON file."""
    with open(path) as f:
        return json.load(f)


def _serialise_ground_truth(gt) -> str:
    """Flatten a structured ground-truth dict into a string for Ragas v1.0+."""
    if isinstance(gt, dict):
        return "; ".join(f"{k}: {v}" for k, v in gt.items())
    return str(gt)


def run_pipeline_on_all(items: list[dict]) -> tuple[list, list, list, list]:
    """Run the RAG pipeline on every evaluation item.

    Returns (user_inputs, responses, retrieved_contexts, references).
    """
    user_inputs: list[str] = []
    responses: list[str] = []
    retrieved_contexts: list[list[str]] = []
    references: list[str] = []

    for i, item in enumerate(items):
        question = item["question"]
        gt = item["ground_truth"]

        result = rag(question=question, chat_history=[])

        user_inputs.append(question)
        responses.append(result["response"])
        retrieved_contexts.append(
            [doc.page_content for doc in result.get("context", [])]
        )
        references.append(_serialise_ground_truth(gt))

        logger.info("Processed %d/%d  |  %s", i + 1, len(items), question[:80])

    return user_inputs, responses, retrieved_contexts, references


def build_dataset(
    user_inputs: list[str],
    responses: list[str],
    retrieved_contexts: list[list[str]],
    references: list[str],
) -> Dataset:
    """Build a HuggingFace Dataset with the column names Ragas v1.0 expects."""
    return Dataset.from_dict(
        {
            "user_input": user_inputs,
            "response": responses,
            "retrieved_contexts": retrieved_contexts,
            "reference": references,
        }
    )


def build_evaluator_llm():
    """Return a Ragas-compatible LLM backed by DeepSeek.

    Three workarounds for Ragas 0.4.x bugs:

    1. Ragas' ``_patch_client_for_provider`` uses the manual
       ``instructor.Instructor(client, create=client.messages.create)``
       constructor which is broken — the raw ``create`` function doesn't
       handle ``response_model``.  We pre-patch the client with
       ``instructor.from_openai()`` and neuter the Ragas patcher.

    2. Ragas' ``_patch_client_for_provider`` hard-codes
       ``client.messages.create`` (Anthropic API shape) — DeepSeek uses the
       OpenAI-compatible ``client.chat.completions.create``.

    3. DeepSeek may need a higher ``max_tokens`` budget for structured
       output generation.
    """
    import ragas.llms.base as _ragas_base

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY not set — cannot create evaluator LLM.")

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")

    # Pre-patch the client properly (what _patch_client_for_provider *should* do).
    patched = instructor.from_openai(client, mode=instructor.Mode.JSON)

    # Neuter Ragas' broken patcher so llm_factory doesn't undo our fix.
    _ragas_base._patch_client_for_provider = lambda c, p: patched

    return llm_factory(
        config.LLM_MODEL,
        provider="deepseek",
        client=patched,
        max_tokens=2048,
    )


def build_evaluator_embeddings():
    """Return embeddings compatible with both old and new Ragas metric APIs.

    Ragas 0.4.x old-style metrics (``_answer_relevance``, etc.) call
    ``embed_query()`` / ``embed_documents()``, but the new-style
    ``HuggingFaceEmbeddings`` only exposes ``embed_text()`` /
    ``embed_texts()``.  We add the missing methods as aliases.
    """
    emb = HuggingFaceEmbeddings(model=config.EMBEDDING_MODEL)
    emb.embed_query = emb.embed_text  # old API → new API
    emb.embed_documents = emb.embed_texts  # old API → new API
    return emb


def report(results_df: pd.DataFrame):
    """Print mean scores and save CSV."""
    score_cols = [
        "context_precision",
        "context_recall",
        "faithfulness",
        "answer_relevancy",
    ]
    available = [c for c in score_cols if c in results_df.columns]

    results_df.to_csv(OUTPUT_PATH, index=False)
    logger.info("Results saved to %s", OUTPUT_PATH)

    print("\n📊  Mean Scores")
    print(results_df[available].mean().to_string())


# ---------------------------------------------------------------------------
# Main entry-point
# ---------------------------------------------------------------------------


def run_evaluation():
    # 1. Load data
    items = load_eval_data(EVAL_DATASET_PATH)
    logger.info("Loaded %d evaluation examples", len(items))

    # 2. Run RAG pipeline
    print("\n🔍  Running RAG pipeline on evaluation dataset...")
    user_inputs, responses, contexts, refs = run_pipeline_on_all(items)

    # 3. Build dataset
    dataset = build_dataset(user_inputs, responses, contexts, refs)

    # 4. Evaluator models
    evaluator_llm = build_evaluator_llm()
    evaluator_embeddings = build_evaluator_embeddings()

    # 5. Evaluate  —  pass llm / embeddings to evaluate(); it distributes them
    #    to metrics that need them (the old MetricWithLLM / MetricWithEmbeddings
    #    protocol).
    print("\n📊  Computing Ragas metrics...")
    result = evaluate(
        dataset=dataset,
        metrics=[
            ContextPrecision(),
            ContextRecall(),
            Faithfulness(),
            AnswerRelevancy(),
        ],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
    )

    # 6. Report
    report(result.to_pandas())


if __name__ == "__main__":
    run_evaluation()
