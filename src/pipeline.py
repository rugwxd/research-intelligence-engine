"""RAG pipeline orchestrator.

Ties together all components (data ingestion, vector store, retrieval,
generation, evaluation) into a unified interface for both the CLI
and Streamlit UI. Supports hybrid search, cross-encoder reranking,
and multi-layer evaluation.
"""

import logging
import time

from src.config import PROJECT_ROOT, Settings, load_config, setup_logging
from src.data.arxiv_fetcher import ArxivFetcher
from src.data.models import RAGResponse
from src.evaluation.metrics import HeuristicEvaluator
from src.evaluation.ragas_eval import RagasEvaluator
from src.generation.generator import Generator
from src.monitoring import LatencyTracker, metrics
from src.retrieval.bm25 import BM25Index
from src.retrieval.hybrid import reciprocal_rank_fusion
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.retriever import Retriever
from src.vectorstore.chunker import create_chunker
from src.vectorstore.embedder import Embedder
from src.vectorstore.faiss_store import FAISSStore

logger = logging.getLogger(__name__)


class RAGPipeline:
    """End-to-end RAG pipeline for research paper question answering.

    Supports three retrieval modes:
        - dense: FAISS-only semantic search (fast, good for most queries)
        - hybrid: BM25 + FAISS with reciprocal rank fusion (better for keyword-heavy queries)
        - full: hybrid + cross-encoder reranking (highest quality, higher latency)
    """

    def __init__(
        self,
        config: Settings | None = None,
        retrieval_mode: str = "hybrid",
        chunking_strategy: str = "sliding_window",
    ) -> None:
        self.config = config or load_config()
        setup_logging(self.config.logging)

        self.retrieval_mode = retrieval_mode
        self.chunking_strategy = chunking_strategy

        # Core components
        self.fetcher = ArxivFetcher(self.config.arxiv)
        self.chunker = create_chunker(self.config.chunking, strategy=chunking_strategy)
        self.embedder = Embedder(self.config.embeddings)
        self.store = FAISSStore(self.config.vectordb, dimension=384)
        self.retriever = Retriever(self.config.retrieval, self.store, self.embedder)
        self.generator = Generator(self.config.generation, self.config.anthropic_api_key)

        # Hybrid search components
        self.bm25_index: BM25Index | None = None
        if retrieval_mode in ("hybrid", "full"):
            self.bm25_index = BM25Index()

        # Cross-encoder reranker
        self.reranker: CrossEncoderReranker | None = None
        if retrieval_mode == "full":
            self.reranker = CrossEncoderReranker()

        # Evaluation
        self.heuristic_eval = HeuristicEvaluator()
        self.ragas_eval = RagasEvaluator()
        self._llm_judge = None

    @property
    def llm_judge(self):
        """Lazy-load LLM judge to avoid unnecessary API client init."""
        if self._llm_judge is None and self.config.anthropic_api_key:
            from src.evaluation.llm_judge import LLMJudge

            self._llm_judge = LLMJudge(self.config.anthropic_api_key)
        return self._llm_judge

    def ingest(
        self,
        target_papers: int = 500,
        extract_pdfs: bool = False,
        max_pdfs: int = 50,
    ) -> int:
        """Run the full data ingestion pipeline.

        Fetches papers from arXiv, optionally extracts full PDF text,
        chunks documents, generates embeddings, and builds indexes.

        Args:
            target_papers: Minimum number of papers to fetch.
            extract_pdfs: Whether to download and extract PDF full text.
            max_pdfs: Maximum number of PDFs to extract (rate-limited).

        Returns:
            Number of chunks indexed.
        """
        logger.info("Starting ingestion pipeline (target: %d papers)", target_papers)
        start = time.perf_counter()

        with LatencyTracker(metrics, "ingestion"):
            # Fetch papers
            papers = self.fetcher.fetch_papers(target_count=target_papers)
            self.fetcher.save_papers(papers)
            logger.info("Fetched %d papers", len(papers))
            metrics.set_gauge("papers.total", len(papers))

            # Optional: extract full PDF text
            if extract_pdfs:
                from src.data.pdf_extractor import PDFExtractor

                extractor = PDFExtractor(max_papers=max_pdfs)
                cache_dir = PROJECT_ROOT / "data" / "raw" / "pdfs"
                full_texts = extractor.extract_batch(papers, output_dir=cache_dir)

                for paper in papers:
                    if paper.arxiv_id in full_texts:
                        paper.full_text = full_texts[paper.arxiv_id]

                logger.info("Extracted full text for %d papers", len(full_texts))
                metrics.set_gauge("papers.with_full_text", len(full_texts))

            # Chunk documents
            chunks = self.chunker.chunk_papers(papers)
            logger.info("Created %d chunks", len(chunks))
            metrics.set_gauge("chunks.total", len(chunks))

            # Generate embeddings and build FAISS index
            embeddings = self.embedder.embed_chunks(chunks)
            self.store.build_index(chunks, embeddings)
            self.store.save()

            # Build BM25 index for hybrid search
            if self.bm25_index is not None:
                self.bm25_index.build(chunks)

        elapsed = time.perf_counter() - start
        logger.info(
            "Ingestion complete: %d papers -> %d chunks indexed in %.1fs",
            len(papers),
            len(chunks),
            elapsed,
        )
        metrics.observe("ingestion.duration_s", elapsed)
        return len(chunks)

    def load_index(self) -> None:
        """Load a previously built FAISS index from disk."""
        self.store.load()
        logger.info("Index loaded: %d vectors", self.store.index.ntotal)
        metrics.set_gauge("index.vectors", self.store.index.ntotal)

        # Rebuild BM25 index from loaded chunks
        if self.bm25_index is not None and self.store.chunks:
            self.bm25_index.build(self.store.chunks)

    def query(
        self,
        question: str,
        use_ragas: bool = False,
        use_llm_judge: bool = False,
    ) -> RAGResponse:
        """Answer a question using the full RAG pipeline.

        Args:
            question: Natural language question about AI/ML research.
            use_ragas: Whether to run RAGAS evaluation (slower).
            use_llm_judge: Whether to run LLM-as-judge evaluation (slowest, most accurate).

        Returns:
            RAGResponse with answer, sources, and evaluation scores.
        """
        if not self.store.is_ready:
            self.load_index()

        start = time.perf_counter()
        metrics.increment("queries.total")

        with LatencyTracker(metrics, "retrieval"):
            # Dense retrieval from FAISS
            dense_results = self.retriever.retrieve(question)

            # Hybrid: merge with BM25 results
            if self.bm25_index is not None:
                bm25_results = self.bm25_index.search(question, top_k=self.config.retrieval.top_k)
                results = reciprocal_rank_fusion(
                    [dense_results, bm25_results],
                    weights=[0.6, 0.4],
                )[: self.config.retrieval.top_k]
            else:
                results = dense_results

            # Cross-encoder reranking
            if self.reranker is not None:
                results = self.reranker.rerank(
                    question,
                    results,
                    top_k=self.config.retrieval.rerank_top_k,
                )
            else:
                results = results[: self.config.retrieval.rerank_top_k]

        # Generate answer
        with LatencyTracker(metrics, "generation"):
            response = self.generator.generate_response(question, results)

        # Evaluate: heuristic (always, fast)
        with LatencyTracker(metrics, "evaluation"):
            response.eval_scores = self.heuristic_eval.evaluate(response)

            # RAGAS evaluation (optional)
            if use_ragas and self.ragas_eval.is_available:
                ragas_scores = self.ragas_eval.evaluate_single(response)
                response.eval_scores.update({f"ragas_{k}": v for k, v in ragas_scores.items()})

            # LLM-as-judge evaluation (optional, highest quality)
            if use_llm_judge and self.llm_judge is not None:
                judge_result = self.llm_judge.evaluate(response)
                response.eval_scores.update(judge_result["scores"])

        response.latency_ms = (time.perf_counter() - start) * 1000
        metrics.observe("query.latency_ms", response.latency_ms)

        logger.info(
            "Query answered in %.1fms (mode=%s): %s",
            response.latency_ms,
            self.retrieval_mode,
            question[:80],
        )
        return response


def create_pipeline(
    config_path: str | None = None,
    retrieval_mode: str = "hybrid",
    chunking_strategy: str = "sliding_window",
) -> RAGPipeline:
    """Factory function to create a configured RAG pipeline.

    Args:
        config_path: Path to YAML config file.
        retrieval_mode: One of 'dense', 'hybrid', 'full'.
        chunking_strategy: One of 'sliding_window', 'semantic'.

    Returns:
        Configured RAGPipeline instance.
    """
    config = load_config(config_path)
    return RAGPipeline(
        config,
        retrieval_mode=retrieval_mode,
        chunking_strategy=chunking_strategy,
    )
