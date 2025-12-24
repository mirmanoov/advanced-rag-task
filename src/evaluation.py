"""
Evaluation module using RAGAS framework.
Measures Faithfulness and Answer Relevancy metrics.
"""
import json
import os
from typing import List, Dict, Any
from datetime import datetime
import pandas as pd
from datasets import Dataset

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from src.config import OPENAI_API_KEY, LLM_MODEL, EMBEDDING_MODEL, EVAL_RESULTS_PATH


def get_eval_llm():
    """Get LLM for RAGAS evaluation."""
    return LangchainLLMWrapper(ChatOpenAI(
        model=LLM_MODEL,
        openai_api_key=OPENAI_API_KEY,
        temperature=0
    ))


def get_eval_embeddings():
    """Get embeddings for RAGAS evaluation."""
    return LangchainEmbeddingsWrapper(OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        openai_api_key=OPENAI_API_KEY
    ))


def prepare_evaluation_dataset(results: List[Dict[str, Any]]) -> Dataset:
    """
    Prepare dataset for RAGAS evaluation.
    
    Args:
        results: List of dicts with keys: question, answer, contexts, ground_truth
    
    Returns:
        HuggingFace Dataset ready for RAGAS
    """
    data = {
        "question": [r["question"] for r in results],
        "answer": [r["answer"] for r in results],
        "contexts": [r["contexts"] for r in results],
        "ground_truth": [r.get("ground_truth", "") for r in results]
    }
    return Dataset.from_dict(data)


def run_evaluation(
    results: List[Dict[str, Any]],
    run_name: str = "evaluation"
) -> Dict[str, float]:
    """
    Run RAGAS evaluation on RAG results.
    
    Args:
        results: List of RAG query results
        run_name: Name for this evaluation run
    
    Returns:
        Dictionary with metric scores
    """
    # Prepare dataset
    dataset = prepare_evaluation_dataset(results)
    
    # Get evaluation components
    eval_llm = get_eval_llm()
    eval_embeddings = get_eval_embeddings()
    
    # Define metrics
    metrics = [faithfulness, answer_relevancy]
    
    # Run evaluation
    print(f"\nRunning RAGAS evaluation: {run_name}")
    print(f"Evaluating {len(results)} samples...")

    evaluation_result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=eval_llm,
        embeddings=eval_embeddings
    )

    # Extract scores - handle different RAGAS return formats
    def extract_score(result, key):
        value = result[key]
        if isinstance(value, list):
            # Filter out None values and calculate mean
            valid_values = [v for v in value if v is not None]
            return sum(valid_values) / len(valid_values) if valid_values else 0.0
        elif isinstance(value, (int, float)):
            return float(value)
        else:
            return float(value)

    scores = {
        "faithfulness": extract_score(evaluation_result, "faithfulness"),
        "answer_relevancy": extract_score(evaluation_result, "answer_relevancy"),
        "timestamp": datetime.now().isoformat(),
        "run_name": run_name,
        "num_samples": len(results)
    }

    return scores


def save_evaluation_results(
    scores: Dict[str, Any],
    filename: str
) -> str:
    """Save evaluation results to JSON file."""
    os.makedirs(EVAL_RESULTS_PATH, exist_ok=True)
    filepath = os.path.join(EVAL_RESULTS_PATH, filename)
    
    with open(filepath, 'w') as f:
        json.dump(scores, f, indent=2)
    
    print(f"Results saved to {filepath}")
    return filepath


def load_evaluation_results(filename: str) -> Dict[str, Any]:
    """Load evaluation results from JSON file."""
    filepath = os.path.join(EVAL_RESULTS_PATH, filename)
    with open(filepath, 'r') as f:
        return json.load(f)


def calculate_improvement(baseline: float, advanced: float) -> float:
    """Calculate percentage improvement."""
    if baseline == 0:
        return float('inf') if advanced > 0 else 0
    return ((advanced - baseline) / baseline) * 100


def print_comparison(baseline_scores: Dict, advanced_scores: Dict):
    """Print comparison between baseline and advanced scores."""
    print("\n" + "=" * 60)
    print("EVALUATION COMPARISON")
    print("=" * 60)
    
    for metric in ["faithfulness", "answer_relevancy"]:
        base = baseline_scores.get(metric, 0)
        adv = advanced_scores.get(metric, 0)
        improvement = calculate_improvement(base, adv)
        
        print(f"\n{metric.upper()}:")
        print(f"  Baseline:    {base:.4f}")
        print(f"  Advanced:    {adv:.4f}")
        print(f"  Improvement: {improvement:+.2f}%")
    
    print("\n" + "=" * 60)

