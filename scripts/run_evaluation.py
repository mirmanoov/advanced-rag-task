"""
Automated evaluation runner script.
Runs evaluation on baseline or advanced RAG system.
"""
import json
import sys
import argparse
from datetime import datetime
from typing import List, Dict, Any
import numpy as np

sys.path.insert(0, '.')

from src.rag_chain import BaselineRAG
from src.evaluation import run_evaluation, save_evaluation_results
from src.config import EVAL_RUNS


def load_eval_questions(path: str = "data/eval_questions.json") -> List[Dict]:
    """Load evaluation questions."""
    with open(path, 'r') as f:
        return json.load(f)


def generate_rag_results(
    rag_system,
    questions: List[Dict],
    system_name: str = "baseline"
) -> List[Dict[str, Any]]:
    """Generate results from RAG system for all questions."""
    results = []
    
    print(f"\nGenerating answers with {system_name} system...")
    for i, q_data in enumerate(questions, 1):
        question = q_data["question"]
        ground_truth = q_data["ground_truth"]
        
        print(f"  [{i}/{len(questions)}] {question[:50]}...")
        
        # Query RAG system
        response = rag_system.query(question)
        
        results.append({
            "question": question,
            "answer": response["answer"],
            "contexts": response["contexts"],
            "ground_truth": ground_truth
        })
    
    return results


def run_multiple_evaluations(
    rag_system,
    questions: List[Dict],
    system_name: str,
    num_runs: int = EVAL_RUNS
) -> Dict[str, Any]:
    """Run multiple evaluation runs and average results."""
    all_scores = []
    
    print(f"\n{'='*60}")
    print(f"EVALUATING {system_name.upper()} SYSTEM")
    print(f"Running {num_runs} evaluation(s)...")
    print(f"{'='*60}")
    
    for run in range(num_runs):
        print(f"\n--- Run {run + 1}/{num_runs} ---")
        
        # Generate results
        results = generate_rag_results(rag_system, questions, system_name)
        
        # Run evaluation
        scores = run_evaluation(results, f"{system_name}_run_{run + 1}")
        all_scores.append(scores)
        
        print(f"  Faithfulness: {scores['faithfulness']:.4f}")
        print(f"  Answer Relevancy: {scores['answer_relevancy']:.4f}")
    
    # Calculate averages and std
    avg_scores = {
        "system": system_name,
        "num_runs": num_runs,
        "num_questions": len(questions),
        "timestamp": datetime.now().isoformat(),
        "faithfulness": {
            "mean": float(np.mean([s["faithfulness"] for s in all_scores])),
            "std": float(np.std([s["faithfulness"] for s in all_scores])),
            "all_runs": [s["faithfulness"] for s in all_scores]
        },
        "answer_relevancy": {
            "mean": float(np.mean([s["answer_relevancy"] for s in all_scores])),
            "std": float(np.std([s["answer_relevancy"] for s in all_scores])),
            "all_runs": [s["answer_relevancy"] for s in all_scores]
        }
    }
    
    return avg_scores


def print_summary(scores: Dict[str, Any]):
    """Print evaluation summary."""
    print(f"\n{'='*60}")
    print(f"EVALUATION SUMMARY: {scores['system'].upper()}")
    print(f"{'='*60}")
    print(f"Questions evaluated: {scores['num_questions']}")
    print(f"Number of runs: {scores['num_runs']}")
    print(f"\nMetrics (mean ± std):")
    print(f"  Faithfulness:      {scores['faithfulness']['mean']:.4f} ± {scores['faithfulness']['std']:.4f}")
    print(f"  Answer Relevancy:  {scores['answer_relevancy']['mean']:.4f} ± {scores['answer_relevancy']['std']:.4f}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Run RAG evaluation")
    parser.add_argument("--system", choices=["baseline", "advanced", "both"], 
                       default="baseline", help="System to evaluate")
    parser.add_argument("--runs", type=int, default=EVAL_RUNS,
                       help="Number of evaluation runs")
    parser.add_argument("--questions", type=str, default="data/eval_questions.json",
                       help="Path to evaluation questions")
    args = parser.parse_args()
    
    # Load questions
    questions = load_eval_questions(args.questions)
    print(f"Loaded {len(questions)} evaluation questions")
    
    if args.system in ["baseline", "both"]:
        # Evaluate baseline
        rag = BaselineRAG()
        scores = run_multiple_evaluations(rag, questions, "baseline", args.runs)
        print_summary(scores)
        save_evaluation_results(scores, "baseline_results.json")
    
    if args.system in ["advanced", "both"]:
        # Import advanced RAG
        from src.rag_chain import AdvancedRAG
        rag = AdvancedRAG()
        scores = run_multiple_evaluations(rag, questions, "advanced", args.runs)
        print_summary(scores)
        save_evaluation_results(scores, "advanced_results.json")


if __name__ == "__main__":
    main()

