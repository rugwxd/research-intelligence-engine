"""Hybrid retrieval combining dense (FAISS) and sparse (BM25) search.

Uses Reciprocal Rank Fusion (RRF) to merge ranked lists from
different retrieval systems, producing a unified ranking that
captures both semantic similarity and exact term matching.
"""

import logging
from collections import defaultdict

from src.data.models import DocumentChunk, RetrievalResult

logger = logging.getLogger(__name__)


def reciprocal_rank_fusion(
    ranked_lists: list[list[RetrievalResult]],
    k: int = 60,
    weights: list[float] | None = None,
) -> list[RetrievalResult]:
    """Merge multiple ranked lists using Reciprocal Rank Fusion.

    RRF score for document d = sum over lists L of: weight_L / (k + rank_L(d))

    This is robust to score scale differences between retrieval systems
    and consistently outperforms simple score interpolation.

    Reference: Cormack et al., "Reciprocal Rank Fusion outperforms
    Condorcet and individual Rank Learning Methods" (SIGIR 2009).

    Args:
        ranked_lists: List of ranked RetrievalResult lists from different systems.
        k: Smoothing constant (default 60, per original paper).
        weights: Optional per-system weights. Defaults to equal weighting.

    Returns:
        Fused list of RetrievalResult sorted by RRF score descending.
    """
    if not ranked_lists:
        return []

    if weights is None:
        weights = [1.0] * len(ranked_lists)

    if len(weights) != len(ranked_lists):
        raise ValueError(
            f"Weight count ({len(weights)}) must match list count ({len(ranked_lists)})"
        )

    # Accumulate RRF scores by chunk_id
    rrf_scores: dict[str, float] = defaultdict(float)
    chunk_map: dict[str, DocumentChunk] = {}

    for weight, ranked_list in zip(weights, ranked_lists):
        for rank, result in enumerate(ranked_list, start=1):
            cid = result.chunk.chunk_id
            rrf_scores[cid] += weight / (k + rank)
            chunk_map[cid] = result.chunk

    # Sort by fused score
    fused = [
        RetrievalResult(chunk=chunk_map[cid], score=score)
        for cid, score in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    ]

    logger.debug(
        "RRF fused %d lists -> %d unique results (top score: %.4f)",
        len(ranked_lists), len(fused), fused[0].score if fused else 0,
    )

    return fused
