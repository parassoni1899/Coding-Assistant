"""
evaluation/evaluator.py
========================
Runs automated checks on the Codebase Assistant RAG pipeline using DeepEval.
Evaluates Context Recall (Retrieval) and Faithfulness (LLM hallucination).
"""

import json
import os
import sys
from pathlib import Path

from deepeval import evaluate
from deepeval.metrics import ContextualRecallMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase
from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_llm
from retrieval.search import HybridRetriever

# Note: DeepEval uses OpenAI by default for its evaluator LLM.
# For local evaluation without an OpenAI key, you would instantiate an Ollama model:
# from deepeval.models import OllamaModel
# custom_model = OllamaModel(model_name="qwen2.5-coder:7b")


def load_benchmark_data(filepath: str) -> list:
    """Load the JSON benchmark questions."""
    with open(filepath, "r") as f:
        return json.load(f)


def run_evaluation(data_path: str):
    logger.info("Initializing Hybrid Retriever & LLM...")
    retriever = HybridRetriever()
    llm = get_llm()

    benchmark_data = load_benchmark_data(data_path)
    test_cases = []

    logger.info(f"Running pipeline for {len(benchmark_data)} test cases...")

    for item in benchmark_data:
        query = item["question"]
        expected_files = item["reference_files"]
        ground_truth = item["ground_truth"]

        # 1. Retrieve Context
        results = retriever.search(query, top_k=5)
        retrieved_texts = [r.content for r in results]
        
        # Format context for generation
        context_blocks = "\n".join(r.to_context_block() for r in results)

        # 2. Generate Answer
        messages = [
            SystemMessage(content="Answer the question based ONLY on the context."),
            HumanMessage(content=f"<context>\n{context_blocks}\n</context>\n\n<question>{query}</question>"),
        ]
        response = llm.invoke(messages)
        actual_output = response.content

        # 3. Build DeepEval TestCase
        test_case = LLMTestCase(
            input=query,
            actual_output=actual_output,
            expected_output=ground_truth,
            retrieval_context=retrieved_texts
        )
        test_cases.append(test_case)

    logger.info("Starting DeepEval metric calculations...")
    
    # Contextual Recall: Does the retrieved context contain the expected truth?
    recall_metric = ContextualRecallMetric(threshold=0.7)
    
    # Faithfulness: Is the generated answer hallucination-free?
    faithfulness_metric = FaithfulnessMetric(threshold=0.8)

    results = evaluate(
        test_cases=test_cases,
        metrics=[recall_metric, faithfulness_metric],
        print_results=True
    )

    logger.success("Evaluation complete.")
    return results


if __name__ == "__main__":
    benchmark_path = str(Path(__file__).parent / "benchmark.json")
    if not os.environ.get("OPENAI_API_KEY"):
        logger.warning("DeepEval requires an OPENAI_API_KEY for the evaluation model by default.")
        logger.warning("Set it in your .env, or modify evaluator.py to use a local Ollama model.")
    run_evaluation(benchmark_path)
