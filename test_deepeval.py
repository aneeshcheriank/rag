import os
import pandas as pd
import json
from dotenv import load_dotenv, find_dotenv

from deepeval.models import DeepSeekModel
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
)
from deepeval.test_case import LLMTestCase
from deepeval.errors import DeepEvalError

from src.pipeline import rag
from src import config

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(name)s] - %(levelname)s - %(message)s")

load_dotenv(find_dotenv())

# Increase per-attempt timeout — FaithfulnessMetric makes multiple LLM calls
# per test case and DeepSeek can take >88s on large contexts.
os.environ.setdefault("DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE", "300")

logger = logging.getLogger(__name__)

API_KEY = os.getenv(config.API_KEY, "")
if not API_KEY:
    raise ValueError("API KEY is not in the environment")

eval_data_path = "data/evaluation_dataset.json"

def load_eval_data(path: str) -> list[dict]:
    """Load the evaluation dataset from a JSON file."""
    with open(path, "r") as f:
        return json.load(f)

def ground_truth_to_str(ground_truth):
    return "," .join([f"{k}: {v}" for k, v in ground_truth.items()])

def run_deepeval(eval_data_path: str, output_path):
    # llm load
    model = DeepSeekModel(
        model = config.EVAL_MODEL,
        api_key = API_KEY,
        temperature=0.0
    )

    # evaluation data
    eval_data = load_eval_data(eval_data_path)
    logger.info(f"evaluation data has been loaded")

    if len(eval_data) == 0:
        logger.info(f"there is no data in {eval_data_path}")
        raise ValueError(f"no data available in {eval_data_path}")

    logger.info(f"Running evaluation on items")

    # 1. Run RAG pipeline on all items and build test cases
    test_cases: list[LLMTestCase] = []

    for item in eval_data:
        user_query = item["question"]
        ground_truth = item["ground_truth"]
        expected_answer = ground_truth_to_str(ground_truth)

        rag_out = rag(user_query, chat_history=[], k=config.TOP_K)
        actual_output = rag_out.get("response")
        retrieval_context = [doc.page_content for doc in rag_out.get("context", [])]

        test_cases.append(
            LLMTestCase(
                input=user_query,
                actual_output=actual_output,
                expected_output=expected_answer,
                retrieval_context=retrieval_context,
            )
        )

    # 2. Define metrics
    metric_factories = [
        ("answer_relevance", lambda: AnswerRelevancyMetric(threshold=0, model=model)),
        ("faithfulness", lambda: FaithfulnessMetric(threshold=0.0, model=model)),
        ("context_precision", lambda: ContextualPrecisionMetric(threshold=0.0, model=model)),
        ("context_recall", lambda: ContextualRecallMetric(threshold=0.0, model=model)),
    ]

    # 3. Evaluate one test case at a time — catch JSON/model failures per item
    records: list[dict] = []

    for i, tc in enumerate(test_cases):
        record: dict = {
            "query": tc.input,
            "actual_output": tc.actual_output,
            "expected_output": tc.expected_output,
            "retrieval_context": "\n---\n".join(tc.retrieval_context),
        }

        for metric_name, metric_builder in metric_factories:
            metric = metric_builder()
            try:
                metric.measure(tc)
                record[metric_name] = metric.score
            except DeepEvalError as e:
                logger.warning(
                    "Metric %s failed on item %d (%s…): %s",
                    metric_name, i + 1, str(tc.input)[:60], e,
                )
                record[metric_name] = None
            except Exception:
                logger.exception(
                    "Unexpected error — metric %s on item %d (%s…)",
                    metric_name, i + 1, str(tc.input)[:60],
                )
                record[metric_name] = None

        records.append(record)

    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)
    logger.info("Wrote %d rows to %s", len(df), output_path)
    return df
    

def aggregate_scores(df):
    cols = ["answer_relevance", "faithfulness", "context_precision", "context_recall"]
    sel_cols = [col for col in cols if col in df.columns]
    print("Mean values")
    print(df[sel_cols].mean().to_string())
    print("Median values")
    print(df[sel_cols].median().to_string())

if __name__ == "__main__":
    results = run_deepeval(eval_data_path, "evaluation_results.csv")
    aggregate_scores(results)