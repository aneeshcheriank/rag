import json
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    context_precision,
    context_recall,
    faithfulness,
    answer_relevancy,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from src.pipeline import rag
from src.model import get_llm, get_embeddings


def run_evaluation(output_filename="eval_results.csv"):

    # Load the evaluation dataset
    with open("data/evaluation_dataset.json", "r") as f:
        eval_data = json.load(f)

    questions = []
    answers = []
    contexts = []
    ground_truths = []

    # Get pipeline instance
    rag_chain = rag

    print("🔍 Running RAG Pipeline on Evaluation Dataset...")

    # Collect predictions and contexts
    for idx, item in enumerate(eval_data):
        q = item["question"]
        gt = item["ground_truth"]

        # Execute the RAG pipeline
        response = rag_chain(question = q, chat_history = [])
        answers.append(response.get("response", ""))
        
        # Pull text contents of context documents retriced bz the chain
        questions.append(q)
        retrived_docs = [doc.page_content for doc in response.get("context", [])]
        contexts.append(retrived_docs)
        ground_truths.append(gt)

    # Format as HuggingFace Dataset for evaluation
    data_dict = {
        "question": questions,
        "answer": answers,
        "context": contexts,
        "ground_truth": ground_truths,
    }
    dataset = Dataset.from_dict(data_dict)

    # Run Evaluation using LLM and Embeddings
    llm = LangchainLLMWrapper(get_llm())
    embedding_model = LangchainEmbeddingsWrapper(get_embeddings())

    print("📊 Evaluating Metrics...")
    results = evaluate(
        dataset = dataset,
        metrics = [
            context_precision,
            context_recall,
            faithfulness,
            answer_relevancy
        ],
        llm = llm,
        embeddings = embedding_model
    )

    # Export Results
    df = results.to_pandas()
    df.to_csv(output_filename, index=False)
    print(f"\n✅ Evaluation Completed! Results saved to {output_filename}")
    print("\nMean Scores:")
    print(df[["context_precision", "context_recall", "faithfulness", "answer_relevancy"]].mean())

if __name__ == "__main__":
    run_evaluation("baseline_vectorstore_results.csv")