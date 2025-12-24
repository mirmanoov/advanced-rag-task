"""
Compare baseline and advanced RAG evaluation results.
"""
import json
import sys
sys.path.insert(0, '.')

from src.config import EVAL_RESULTS_PATH


def load_results(filename):
    """Load evaluation results."""
    with open(f"{EVAL_RESULTS_PATH}/{filename}", 'r') as f:
        return json.load(f)


def calculate_improvement(baseline, advanced):
    """Calculate percentage improvement."""
    return ((advanced - baseline) / baseline) * 100


def main():
    print("=" * 70)
    print("RAG EVALUATION COMPARISON")
    print("=" * 70)
    
    # Load results
    baseline = load_results("baseline_results.json")
    advanced = load_results("advanced_results.json")
    
    print(f"\nBaseline System:")
    print(f"  Questions: {baseline['num_questions']}")
    print(f"  Runs: {baseline['num_runs']}")
    print(f"  Faithfulness: {baseline['faithfulness']['mean']:.4f} ± {baseline['faithfulness']['std']:.4f}")
    print(f"  Answer Relevancy: {baseline['answer_relevancy']['mean']:.4f} ± {baseline['answer_relevancy']['std']:.4f}")
    
    print(f"\nAdvanced System:")
    print(f"  Questions: {advanced['num_questions']}")
    print(f"  Runs: {advanced['num_runs']}")
    print(f"  Faithfulness: {advanced['faithfulness']['mean']:.4f} ± {advanced['faithfulness']['std']:.4f}")
    print(f"  Answer Relevancy: {advanced['answer_relevancy']['mean']:.4f} ± {advanced['answer_relevancy']['std']:.4f}")
    
    # Calculate improvements
    faith_imp = calculate_improvement(
        baseline['faithfulness']['mean'],
        advanced['faithfulness']['mean']
    )
    relv_imp = calculate_improvement(
        baseline['answer_relevancy']['mean'],
        advanced['answer_relevancy']['mean']
    )
    
    print(f"\n{'='*70}")
    print("IMPROVEMENT SUMMARY")
    print("=" * 70)
    print(f"\n{'Metric':<25} {'Baseline':<12} {'Advanced':<12} {'Improvement':<15}")
    print("-" * 70)
    print(f"{'Faithfulness':<25} {baseline['faithfulness']['mean']:.4f}       {advanced['faithfulness']['mean']:.4f}       {faith_imp:+.2f}%")
    print(f"{'Answer Relevancy':<25} {baseline['answer_relevancy']['mean']:.4f}       {advanced['answer_relevancy']['mean']:.4f}       {relv_imp:+.2f}%")
    print("-" * 70)
    
    # Check if target achieved
    print(f"\n{'='*70}")
    print("TARGET ACHIEVEMENT")
    print("=" * 70)
    target = 30.0
    if faith_imp >= target:
        print(f"\n✅ FAITHFULNESS TARGET ACHIEVED!")
        print(f"   Target: +{target}% | Achieved: +{faith_imp:.2f}%")
    else:
        print(f"\n❌ Faithfulness target not achieved")
        print(f"   Target: +{target}% | Achieved: +{faith_imp:.2f}%")
    
    print("\n" + "=" * 70)
    
    return faith_imp, relv_imp


if __name__ == "__main__":
    main()

