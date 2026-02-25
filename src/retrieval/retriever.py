"""Retrieval pipeline for finding relevant document chunks.

Orchestrates query embedding, FAISS search, optional re-ranking,
and context assembly for the generation stage.
"""

import logging
import time

import numpy as np

from src.config import RetrievalConfig
from src.data.models import DocumentChunk, RetrievalResult
from src.vectorstore.embedder import Embedder
from src.vectorstore.faiss_store import FAISSStore

logger = logging.getLogger(__name__)


class Retriever:
    """Retrieves and ranks relevant document chunks for a given query."""

    def __init__(
        self,
        config: RetrievalConfig,
        store: FAISSStore,
        embedder: Embedder,
    ) -> None:
        self.config = config
        self.store = store
        self.embedder = embedder

    def retrieve(self, query: str) -> list[RetrievalResult]:
        """Execute the full retrieval pipeline for a query.

        Steps:
            1. Embed the query
            2. Search FAISS index for top-k candidates
            3. Filter by similarity threshold
            4. Deduplicate by paper ID (keep best chunk per paper)
            5. Return top rerank_top_k results

        Args:
            query: Natural language query string.

        Returns:
            Ranked list of RetrievalResult objects.
        """
        start = time.perf_counter()

        query_embedding = self.embedder.embed_query(query)
        candidates = self.store.search(query_embedding, top_k=self.config.top_k)

        # Filter below threshold
        filtered = [
            r for r in candidates
            if r.score >= self.config.similarity_threshold
        ]

        if not filtered:
            logger.warning("No results above threshold %.2f for query: %s",
                           self.config.similarity_threshold, query[:80])
            # Fall back to top results regardless of threshold
            filtered = candidates[:self.config.rerank_top_k]

        # Deduplicate: keep only the best-scoring chunk per paper
        deduped = self._deduplicate_by_paper(filtered)

        # Take top rerank_top_k results
        results = deduped[:self.config.rerank_top_k]

        elapsed = (time.perf_counter() - start) * 1000
        logger.info(
            "Retrieved %d results for query (%.1fms): %s",
            len(results), elapsed, query[:80],
        )

        return results

    def _deduplicate_by_paper(self, results: list[RetrievalResult]) -> list[RetrievalResult]:
        """Keep only the highest-scoring chunk per paper."""
        best_by_paper: dict[str, RetrievalResult] = {}
        for result in results:
            pid = result.chunk.paper_id
            if pid not in best_by_paper or result.score > best_by_paper[pid].score:
                best_by_paper[pid] = result

        # Re-sort by score descending
        deduped = sorted(best_by_paper.values(), key=lambda r: r.score, reverse=True)
        return deduped

    @staticmethod
    def format_context(results: list[RetrievalResult]) -> str:
        """Format retrieval results into a context string for the generator.

        Args:
            results: Ranked retrieval results.

        Returns:
            Formatted context string with source attribution.
        """
        if not results:
            return "No relevant sources found."

        parts = []
        for i, result in enumerate(results, 1):
            meta = result.chunk.metadata
            title = meta.get("title", "Unknown")
            authors = meta.get("authors", [])
            author_str = ", ".join(authors[:3])
            if len(authors) > 3:
                author_str += " et al."
            arxiv_id = meta.get("arxiv_id", "")

            parts.append(
                f"[Source {i}] {title}\n"
                f"Authors: {author_str}\n"
                f"arXiv: {arxiv_id}\n"
                f"Relevance: {result.score:.3f}\n"
                f"Content: {result.chunk.text}\n"
            )

        return "\n---\n".join(parts)
