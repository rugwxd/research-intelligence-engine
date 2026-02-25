"""Evaluation benchmark script.

Runs a set of predefined queries through the RAG pipeline and
computes aggregate evaluation metrics.

Usage:
    python scripts/evaluate.py [--config configs/default.yaml] [--ragas]
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import create_pipeline

logger = logging.getLogger(__name__)

BENCHMARK_QUERIES = [
    "What are the key innovations in transformer architectures since the original paper?",
    "How does RLHF improve large language model alignment?",
    "What are the main approaches to reducing hallucinations in LLMs?",
    "Compare vision transformers with convolutional neural networks for image classification",
    "What techniques are used for efficient fine-tuning of large language models?",
    "How do diffusion models work for image generation?",
    "What is chain-of-thought prompting and why does it improve reasoning?",
    "What are the latest advances in multi-modal AI models?",
    "How does retrieval-augmented generation improve factual accuracy?",
    "What methods exist for evaluating large language models?",
    "Explain the concept of attention mechanisms and their variants",
    "What are the challenges of training very large neural networks?",
    "How is reinforcement learning applied to robotics?",
    "What are the main approaches to neural architecture search?",
    "How do graph neural networks process structured data?",
]


def main():
    parser = argparse.ArgumentParser(description="Run RAG evaluation benchmark")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--ragas", action="store_true", help="Include RAGAS metrics")
    parser.add_argument("--output", type=str, default="data/processed/eval_results.json")
    args = parser.parse_args()

    pipeline = create_pipeline(args.config)
    pipeline.load_index()

    print(f"Running evaluation on {len(BENCHMARK_QUERIES)} queries...\n")

    all_scores: list[dict[str, float]] = []
    results_detail = []

    for i, query in enumerate(BENCHMARK_QUERIES, 1):
        print(f"[{i}/{len(BENCHMARK_QUERIES)}] {query[:70]}...")

        response = pipeline.query(query, use_ragas=args.ragas)
        all_scores.append(response.eval_scores)

        results_detail.append({
            "query": query,
            "answer_preview": response.answer[:200],
            "num_sources": len(response.sources),
            "latency_ms": response.latency_ms,
            "scores": response.eval_scores,
        })

        # Print inline scores
        score_str = " | ".join(f"{k}: {v:.3f}" for k, v in response.eval_scores.items())
        print(f"  Scores: {score_str}\n")

    # Aggregate results
    metric_names = all_scores[0].keys() if all_scores else []
    averages = {}
    for metric in metric_names:
        values = [s[metric] for s in all_scores if metric in s]
        averages[metric] = sum(values) / len(values) if values else 0.0

    print("\n" + "=" * 60)
    print("AGGREGATE RESULTS")
    print("=" * 60)
    for metric, avg in averages.items():
        print(f"  {metric:25s}: {avg:.4f}")
    print("=" * 60)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        "num_queries": len(BENCHMARK_QUERIES),
        "aggregate_scores": averages,
        "per_query_results": results_detail,
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nDetailed results saved to {output_path}")


if __name__ == "__main__":
    main()
