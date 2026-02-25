"""Cross-encoder reranking for improved retrieval precision.

Uses a cross-encoder model (ms-marco-MiniLM-L-6-v2) to rerank
candidate documents by jointly encoding query-document pairs.
This provides significantly better relevance scoring than
bi-encoder similarity at the cost of higher latency.
"""

import logging
import time

from sentence_transformers import CrossEncoder

from src.data.models import RetrievalResult

logger = logging.getLogger(__name__)

DEFAULT_CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker:
    """Reranks retrieval candidates using a cross-encoder model.

    Cross-encoders jointly encode (query, document) pairs and produce
    a relevance score, capturing fine-grained semantic interactions
    that bi-encoder dot-product similarity misses.

    Typical pipeline: FAISS top-50 -> CrossEncoder rerank -> top-5
    """

    def __init__(self, model_name: str = DEFAULT_CROSS_ENCODER) -> None:
        self.model_name = model_name
        self._model: CrossEncoder | None = None

    @property
    def model(self) -> CrossEncoder:
        """Lazy-load the cross-encoder model."""
        if self._model is None:
            logger.info("Loading cross-encoder: %s", self.model_name)
            self._model = CrossEncoder(self.model_name)
            logger.info("Cross-encoder loaded")
        return self._model

    def rerank(
        self,
        query: str,
        candidates: list[RetrievalResult],
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        """Rerank candidates using the cross-encoder.

        Args:
            query: The user query.
            candidates: Initial retrieval results from FAISS.
            top_k: Number of top results to return after reranking.

        Returns:
            Reranked list of RetrievalResult, sorted by cross-encoder score.
        """
        if not candidates:
            return []

        start = time.perf_counter()

        # Build query-document pairs for the cross-encoder
        pairs = [(query, result.chunk.text) for result in candidates]

        # Score all pairs
        scores = self.model.predict(pairs)

        # Attach cross-encoder scores and sort
        scored_results = []
        for result, ce_score in zip(candidates, scores):
            scored_results.append(
                RetrievalResult(
                    chunk=result.chunk,
                    score=float(ce_score),
                )
            )

        scored_results.sort(key=lambda r: r.score, reverse=True)

        elapsed = (time.perf_counter() - start) * 1000
        logger.info(
            "Reranked %d candidates -> top %d in %.1fms (best: %.3f, worst: %.3f)",
            len(candidates),
            top_k,
            elapsed,
            scored_results[0].score if scored_results else 0,
            scored_results[-1].score if scored_results else 0,
        )

        return scored_results[:top_k]
