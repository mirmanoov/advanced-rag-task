"""
Test script to verify advanced RAG system works end-to-end.
"""
import json
import sys
sys.path.insert(0, '.')

from src.rag_chain import BaselineRAG, AdvancedRAG
from src.evaluation import run_evaluation


def main():
    print("=" * 70)
    print("ADVANCED RAG SYSTEM - END-TO-END TEST")
    print("=" * 70)

    # Load evaluation questions
    with open('data/eval_questions.json', 'r') as f:
        questions = json.load(f)

    # Initialize both RAG systems
    print("\nInitializing Baseline RAG...")
    baseline_rag = BaselineRAG()

    print("\nInitializing Advanced RAG...")
    advanced_rag = AdvancedRAG()

    # Test with first 3 questions
    test_questions = questions[:3]

    print(f"\nComparing systems on {len(test_questions)} questions...\n")

    # Collect results for evaluation
    baseline_results = []
    advanced_results = []

    for i, q_data in enumerate(test_questions, 1):
        question = q_data['question']
        ground_truth = q_data['ground_truth']

        print(f"\n{'='*70}")
        print(f"Question {i}: {question}")
        print("-" * 70)

        # Baseline answer
        baseline_result = baseline_rag.query(question)
        baseline_results.append({
            'question': question,
            'answer': baseline_result['answer'],
            'contexts': baseline_result['contexts'],
            'ground_truth': ground_truth
        })
        print(f"\n[BASELINE] Answer:")
        print(baseline_result['answer'][:400])

        # Advanced answer
        advanced_result = advanced_rag.query(question)
        advanced_results.append({
            'question': question,
            'answer': advanced_result['answer'],
            'contexts': advanced_result['contexts'],
            'ground_truth': ground_truth
        })
        print(f"\n[ADVANCED] Answer:")
        print(advanced_result['answer'][:400])

        print(f"\n[GROUND TRUTH]:")
        print(ground_truth[:300])

    # Run evaluation
    print("\n" + "=" * 70)
    print("EVALUATING METRICS...")
    print("=" * 70)

    print("\nEvaluating Baseline...")
    baseline_metrics = run_evaluation(baseline_results, run_name="baseline_test")

    print("\nEvaluating Advanced...")
    advanced_metrics = run_evaluation(advanced_results, run_name="advanced_test")

    # Calculate improvement
    faith_improvement = ((advanced_metrics['faithfulness'] - baseline_metrics['faithfulness'])
                         / baseline_metrics['faithfulness'] * 100)
    relevancy_improvement = ((advanced_metrics['answer_relevancy'] - baseline_metrics['answer_relevancy'])
                             / baseline_metrics['answer_relevancy'] * 100)

    print("\n" + "=" * 70)
    print("METRICS COMPARISON")
    print("=" * 70)
    print(f"\n{'Metric':<20} {'Baseline':<12} {'Advanced':<12} {'Improvement':<12}")
    print("-" * 56)
    print(f"{'Faithfulness':<20} {baseline_metrics['faithfulness']:<12.4f} {advanced_metrics['faithfulness']:<12.4f} {faith_improvement:+.2f}%")
    print(f"{'Answer Relevancy':<20} {baseline_metrics['answer_relevancy']:<12.4f} {advanced_metrics['answer_relevancy']:<12.4f} {relevancy_improvement:+.2f}%")
    print("\n" + "=" * 70)
    print("ADVANCED TEST COMPLETE")
    print("=" * 70)

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

