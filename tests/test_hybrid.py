"""Tests for hybrid retrieval and reciprocal rank fusion."""

import pytest

from src.data.models import DocumentChunk, RetrievalResult
from src.retrieval.hybrid import reciprocal_rank_fusion


def _make_result(chunk_id: str, paper_id: str, score: float) -> RetrievalResult:
    chunk = DocumentChunk(
        chunk_id=chunk_id,
        paper_id=paper_id,
        text=f"Text for {chunk_id}",
        metadata={"title": f"Paper {paper_id}", "authors": []},
    )
    return RetrievalResult(chunk=chunk, score=score)


class TestReciprocalRankFusion:
    def test_single_list(self):
        results = [
            _make_result("c1", "p1", 0.9),
            _make_result("c2", "p2", 0.8),
        ]
        fused = reciprocal_rank_fusion([results])

        assert len(fused) == 2
        assert fused[0].chunk.chunk_id == "c1"

    def test_two_lists_merge(self):
        list1 = [
            _make_result("c1", "p1", 0.9),
            _make_result("c2", "p2", 0.8),
        ]
        list2 = [
            _make_result("c2", "p2", 0.95),  # Same chunk, different score
            _make_result("c3", "p3", 0.7),
        ]
        fused = reciprocal_rank_fusion([list1, list2])

        assert len(fused) == 3  # c1, c2, c3

    def test_document_in_both_lists_gets_boost(self):
        list1 = [
            _make_result("c1", "p1", 0.9),
            _make_result("c2", "p2", 0.5),
        ]
        list2 = [
            _make_result("c2", "p2", 0.9),
            _make_result("c3", "p3", 0.5),
        ]
        fused = reciprocal_rank_fusion([list1, list2])

        # c2 appears in both lists, so it should get a boost
        chunk_ids = [r.chunk.chunk_id for r in fused]
        c2_rank = chunk_ids.index("c2")
        assert c2_rank <= 1  # Should be ranked 1st or 2nd

    def test_weights_affect_ranking(self):
        list1 = [_make_result("c1", "p1", 0.9)]
        list2 = [_make_result("c2", "p2", 0.9)]

        # Heavily weight list2
        fused = reciprocal_rank_fusion([list1, list2], weights=[0.1, 10.0])
        assert fused[0].chunk.chunk_id == "c2"

    def test_empty_lists(self):
        fused = reciprocal_rank_fusion([])
        assert fused == []

    def test_weight_count_mismatch_raises(self):
        with pytest.raises(ValueError, match="Weight count"):
            reciprocal_rank_fusion(
                [[_make_result("c1", "p1", 0.9)]],
                weights=[1.0, 2.0],
            )
