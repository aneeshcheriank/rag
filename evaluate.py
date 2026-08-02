import json
import pandas as pd
from datasets import Dataset
from ragas import evluate
from ragas.metrics import (
    context_precision,
    context_recall,
    faithfulness,
    answer_relevancy,
)

from src.pipeline import rag
from src.model import get_llm, get_embedding_model


def run_evaluation(output_filename="eval_results.csv"):

    # Load the evaluation dataset
    with open("data/evaluation_dataset.json", "r") as f:
        eval_data = json.load(f)

    questions = []
    answers = []
    contexts = []
    ground_truths = []

    # Get pipeline instance
    rag_chain = rag()

    print("🔍 Running RAG Pipeline on Evaluation Dataset...")

    # Collect predictions and contexts
    for idx, item in enumerate(eval_data):
        q = item["question"]
        gt = item["answer"]

        # Execute the RAG pipeline
        response = rag_chain.invoke({"question": q, "chat_history": [], "context": ""})

        # Pull text contents of context documents retriced bz the chain
        retrived_docs = [doc.page_content for doc in response.get]
