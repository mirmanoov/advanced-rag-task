"""
Test script to verify baseline RAG system works end-to-end.
"""
import json
import sys
sys.path.insert(0, '.')

from src.rag_chain import BaselineRAG
from src.evaluation import run_evaluation


def main():
    print("=" * 70)
    print("BASELINE RAG SYSTEM - END-TO-END TEST")
    print("=" * 70)

    # Load evaluation questions
    with open('data/eval_questions.json', 'r') as f:
        questions = json.load(f)

    # Initialize RAG
    print("\nInitializing Baseline RAG...")
    rag = BaselineRAG()

    # Test with first 3 questions
    test_questions = questions[:3]

    print(f"\nTesting with {len(test_questions)} questions...\n")

    # Collect results for evaluation
    all_results = []

    for i, q_data in enumerate(test_questions, 1):
        question = q_data['question']
        ground_truth = q_data['ground_truth']

        print(f"\n{'='*70}")
        print(f"Question {i}: {question}")
        print("-" * 70)

        result = rag.query(question)
        all_results.append({
            'question': question,
            'answer': result['answer'],
            'contexts': result['contexts'],
            'ground_truth': ground_truth
        })

        print(f"\nGenerated Answer:")
        print(result['answer'][:600])

        print(f"\nGround Truth:")
        print(ground_truth[:300])

        print(f"\nContexts Retrieved: {len(result['contexts'])}")
        print(f"Context 1 Preview: {result['contexts'][0][:150]}...")

    # Run evaluation
    print("\n" + "=" * 70)
    print("EVALUATING METRICS...")
    print("=" * 70)

    metrics = run_evaluation(all_results, run_name="baseline_test")

    print("\n" + "=" * 70)
    print("BASELINE METRICS")
    print("=" * 70)
    print(f"\n  Faithfulness:      {metrics['faithfulness']:.4f}")
    print(f"  Answer Relevancy:  {metrics['answer_relevancy']:.4f}")
    print("\n" + "=" * 70)
    print("BASELINE TEST COMPLETE")
    print("=" * 70)

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

