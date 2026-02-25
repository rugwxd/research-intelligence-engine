"""RAG pipeline orchestrator.

Ties together all components (data ingestion, vector store, retrieval,
generation, evaluation) into a unified interface for both the CLI
and Streamlit UI.
"""

import logging
import time
from pathlib import Path

from src.config import Settings, load_config, setup_logging, PROJECT_ROOT
from src.data.arxiv_fetcher import ArxivFetcher
from src.data.models import RAGResponse
from src.evaluation.metrics import HeuristicEvaluator
from src.evaluation.ragas_eval import RagasEvaluator
from src.generation.generator import Generator
from src.retrieval.retriever import Retriever
from src.vectorstore.chunker import DocumentChunker
from src.vectorstore.embedder import Embedder
from src.vectorstore.faiss_store import FAISSStore

logger = logging.getLogger(__name__)


class RAGPipeline:
    """End-to-end RAG pipeline for research paper question answering."""

    def __init__(self, config: Settings | None = None) -> None:
        self.config = config or load_config()
        setup_logging(self.config.logging)

        self.fetcher = ArxivFetcher(self.config.arxiv)
        self.chunker = DocumentChunker(self.config.chunking)
        self.embedder = Embedder(self.config.embeddings)
        self.store = FAISSStore(self.config.vectordb, dimension=384)
        self.retriever = Retriever(self.config.retrieval, self.store, self.embedder)
        self.generator = Generator(self.config.generation, self.config.anthropic_api_key)
        self.heuristic_eval = HeuristicEvaluator()
        self.ragas_eval = RagasEvaluator()

    def ingest(self, target_papers: int = 500) -> int:
        """Run the full data ingestion pipeline.

        Fetches papers from arXiv, chunks them, generates embeddings,
        and builds the FAISS index.

        Args:
            target_papers: Minimum number of papers to fetch.

        Returns:
            Number of chunks indexed.
        """
        logger.info("Starting ingestion pipeline (target: %d papers)", target_papers)
        start = time.perf_counter()

        # Fetch papers
        papers = self.fetcher.fetch_papers(target_count=target_papers)
        self.fetcher.save_papers(papers)
        logger.info("Fetched %d papers", len(papers))

        # Chunk documents
        chunks = self.chunker.chunk_papers(papers)
        logger.info("Created %d chunks", len(chunks))

        # Generate embeddings and build index
        embeddings = self.embedder.embed_chunks(chunks)
        self.store.build_index(chunks, embeddings)
        self.store.save()

        elapsed = time.perf_counter() - start
        logger.info(
            "Ingestion complete: %d papers -> %d chunks indexed in %.1fs",
            len(papers), len(chunks), elapsed,
        )
        return len(chunks)

    def load_index(self) -> None:
        """Load a previously built FAISS index from disk."""
        self.store.load()
        logger.info("Index loaded: %d vectors", self.store.index.ntotal)

    def query(self, question: str, use_ragas: bool = False) -> RAGResponse:
        """Answer a question using the full RAG pipeline.

        Args:
            question: Natural language question about AI/ML research.
            use_ragas: Whether to run RAGAS evaluation (slower).

        Returns:
            RAGResponse with answer, sources, and evaluation scores.
        """
        if not self.store.is_ready:
            self.load_index()

        start = time.perf_counter()

        # Retrieve relevant chunks
        results = self.retriever.retrieve(question)

        # Generate answer
        response = self.generator.generate_response(question, results)

        # Evaluate
        response.eval_scores = self.heuristic_eval.evaluate(response)

        if use_ragas and self.ragas_eval.is_available:
            ragas_scores = self.ragas_eval.evaluate_single(response)
            response.eval_scores.update({f"ragas_{k}": v for k, v in ragas_scores.items()})

        response.latency_ms = (time.perf_counter() - start) * 1000

        logger.info(
            "Query answered in %.1fms: %s",
            response.latency_ms, question[:80],
        )
        return response


def create_pipeline(config_path: str | None = None) -> RAGPipeline:
    """Factory function to create a configured RAG pipeline."""
    config = load_config(config_path)
    return RAGPipeline(config)
