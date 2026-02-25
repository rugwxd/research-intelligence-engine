"""Ablation study for RAG pipeline components.

Systematically evaluates the impact of individual components
by varying one dimension at a time and measuring quality/latency
tradeoffs. Produces a results table for the README.

Usage:
    python scripts/ablation.py [--config configs/default.yaml]
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    ChunkingConfig,
    RetrievalConfig,
    Settings,
    load_config,
    setup_logging,
)
from src.data.arxiv_fetcher import ArxivFetcher
from src.data.models import Paper
from src.evaluation.metrics import HeuristicEvaluator
from src.generation.generator import Generator
from src.retrieval.retriever import Retriever
from src.vectorstore.chunker import SlidingWindowChunker, SemanticChunker, create_chunker
from src.vectorstore.embedder import Embedder
from src.vectorstore.faiss_store import FAISSStore

logger = logging.getLogger(__name__)

EVAL_QUERIES = [
    "What are the key innovations in transformer architectures?",
    "How does reinforcement learning from human feedback work?",
    "Compare vision transformers with convolutional neural networks",
    "What methods reduce hallucinations in large language models?",
    "Explain attention mechanisms and their variants",
]


def load_papers(config: Settings) -> list[Paper]:
    """Load previously ingested papers."""
    from src.config import PROJECT_ROOT
    papers_path = PROJECT_ROOT / config.arxiv.output_dir / "papers.json"
    return ArxivFetcher.load_papers(papers_path)


def run_experiment(
    name: str,
    papers: list[Paper],
    config: Settings,
    chunking_strategy: str = "sliding_window",
    chunk_size: int = 512,
    top_k: int = 5,
    use_reranker: bool = False,
    use_hybrid: bool = False,
) -> dict:
    """Run a single experiment configuration and collect metrics."""
    print(f"\n{'='*60}")
    print(f"Experiment: {name}")
    print(f"  Chunking: {chunking_strategy}, Chunk size: {chunk_size}")
    print(f"  Top-K: {top_k}, Reranker: {use_reranker}, Hybrid: {use_hybrid}")
    print(f"{'='*60}")

    # Configure chunking
    chunk_config = ChunkingConfig(
        chunk_size=chunk_size,
        chunk_overlap=chunk_size // 8,
        min_chunk_length=50,
    )
    chunker = create_chunker(chunk_config, strategy=chunking_strategy)
    chunks = chunker.chunk_papers(papers)

    # Build embeddings and index
    embedder = Embedder(config.embeddings)
    embeddings = embedder.embed_chunks(chunks)

    store = FAISSStore(config.vectordb, dimension=embedder.dimension)
    store.build_index(chunks, embeddings)

    # Configure retrieval
    retrieval_config = RetrievalConfig(
        top_k=top_k * 2,  # Over-fetch for reranking
        rerank_top_k=top_k,
        similarity_threshold=0.3,
    )

    retriever = Retriever(retrieval_config, store, embedder)
    generator = Generator(config.generation, config.anthropic_api_key)
    evaluator = HeuristicEvaluator()

    # Optional: hybrid search
    bm25_index = None
    if use_hybrid:
        from src.retrieval.bm25 import BM25Index
        bm25_index = BM25Index()
        bm25_index.build(chunks)

    # Optional: cross-encoder reranker
    reranker = None
    if use_reranker:
        from src.retrieval.reranker import CrossEncoderReranker
        reranker = CrossEncoderReranker()

    # Run evaluation queries
    all_scores = []
    all_latencies = []

    for query in EVAL_QUERIES:
        start = time.perf_counter()

        # Dense retrieval
        results = retriever.retrieve(query)

        # Hybrid fusion
        if bm25_index:
            from src.retrieval.hybrid import reciprocal_rank_fusion
            bm25_results = bm25_index.search(query, top_k=top_k * 2)
            results = reciprocal_rank_fusion(
                [results, bm25_results],
                weights=[0.6, 0.4],
            )[:top_k * 2]

        # Reranking
        if reranker:
            results = reranker.rerank(query, results, top_k=top_k)
        else:
            results = results[:top_k]

        # Generate and evaluate
        response = generator.generate_response(query, results)
        scores = evaluator.evaluate(response)

        latency = (time.perf_counter() - start) * 1000
        all_scores.append(scores)
        all_latencies.append(latency)

        print(f"  Query: {query[:50]}... | Overall: {scores['overall']:.3f} | {latency:.0f}ms")

    # Aggregate results
    avg_scores = {}
    for key in all_scores[0]:
        avg_scores[key] = sum(s[key] for s in all_scores) / len(all_scores)

    avg_latency = sum(all_latencies) / len(all_latencies)

    result = {
        "experiment": name,
        "config": {
            "chunking_strategy": chunking_strategy,
            "chunk_size": chunk_size,
            "top_k": top_k,
            "use_reranker": use_reranker,
            "use_hybrid": use_hybrid,
            "num_chunks": len(chunks),
        },
        "scores": avg_scores,
        "avg_latency_ms": avg_latency,
    }

    print(f"\n  Results: {json.dumps(avg_scores, indent=2)}")
    print(f"  Avg latency: {avg_latency:.0f}ms")
    print(f"  Total chunks: {len(chunks)}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Run RAG ablation study")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--output", type=str, default="data/processed/ablation_results.json")
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(config.logging)

    papers = load_papers(config)
    print(f"Loaded {len(papers)} papers for ablation study\n")

    experiments = []

    # Experiment 1: Baseline (sliding window, dense only, no reranker)
    experiments.append(run_experiment(
        "baseline",
        papers, config,
        chunking_strategy="sliding_window",
        chunk_size=512,
        top_k=5,
    ))

    # Experiment 2: Vary chunk size
    for size in [256, 512, 1024]:
        experiments.append(run_experiment(
            f"chunk_size_{size}",
            papers, config,
            chunk_size=size,
            top_k=5,
        ))

    # Experiment 3: Vary top-k
    for k in [3, 5, 10]:
        experiments.append(run_experiment(
            f"top_k_{k}",
            papers, config,
            top_k=k,
        ))

    # Experiment 4: Semantic chunking
    experiments.append(run_experiment(
        "semantic_chunking",
        papers, config,
        chunking_strategy="semantic",
        chunk_size=512,
        top_k=5,
    ))

    # Experiment 5: Hybrid search (BM25 + dense)
    experiments.append(run_experiment(
        "hybrid_search",
        papers, config,
        top_k=5,
        use_hybrid=True,
    ))

    # Experiment 6: Cross-encoder reranker
    experiments.append(run_experiment(
        "with_reranker",
        papers, config,
        top_k=5,
        use_reranker=True,
    ))

    # Experiment 7: Full pipeline (hybrid + reranker)
    experiments.append(run_experiment(
        "full_pipeline",
        papers, config,
        chunking_strategy="sliding_window",
        chunk_size=512,
        top_k=5,
        use_hybrid=True,
        use_reranker=True,
    ))

    # Print summary table
    print("\n" + "=" * 90)
    print("ABLATION STUDY RESULTS")
    print("=" * 90)
    print(f"{'Experiment':<25} {'Faithful':>10} {'Relevant':>10} {'Complete':>10} {'Overall':>10} {'Latency':>10}")
    print("-" * 90)
    for exp in experiments:
        s = exp["scores"]
        print(
            f"{exp['experiment']:<25} "
            f"{s.get('faithfulness', 0):>10.3f} "
            f"{s.get('relevance', 0):>10.3f} "
            f"{s.get('completeness', 0):>10.3f} "
            f"{s.get('overall', 0):>10.3f} "
            f"{exp['avg_latency_ms']:>8.0f}ms"
        )
    print("=" * 90)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(experiments, f, indent=2)
    print(f"\nFull results saved to {output_path}")


if __name__ == "__main__":
    main()
