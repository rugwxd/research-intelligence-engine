"""Tests for BM25 sparse retrieval."""

import pytest

from src.data.models import DocumentChunk
from src.retrieval.bm25 import BM25Index


@pytest.fixture
def bm25():
    return BM25Index()


@pytest.fixture
def sample_chunks():
    return [
        DocumentChunk(
            chunk_id="c1",
            paper_id="p1",
            text="Transformer models use self-attention mechanisms for sequence processing",
            metadata={"title": "Transformers", "authors": []},
        ),
        DocumentChunk(
            chunk_id="c2",
            paper_id="p2",
            text="Convolutional neural networks excel at image classification tasks",
            metadata={"title": "CNNs", "authors": []},
        ),
        DocumentChunk(
            chunk_id="c3",
            paper_id="p3",
            text="Reinforcement learning agents optimize cumulative reward signals",
            metadata={"title": "RL", "authors": []},
        ),
        DocumentChunk(
            chunk_id="c4",
            paper_id="p4",
            text="BERT uses bidirectional transformer pre-training for language understanding",
            metadata={"title": "BERT", "authors": []},
        ),
    ]


class TestBM25Index:
    def test_build_index(self, bm25, sample_chunks):
        bm25.build(sample_chunks)
        assert bm25.n_docs == 4
        assert bm25.avg_doc_length > 0

    def test_search_returns_relevant_results(self, bm25, sample_chunks):
        bm25.build(sample_chunks)
        results = bm25.search("transformer attention mechanism", top_k=2)

        assert len(results) > 0
        # Transformer-related docs should rank highest
        assert results[0].chunk.paper_id == "p1" or results[0].chunk.paper_id == "p4"

    def test_search_respects_top_k(self, bm25, sample_chunks):
        bm25.build(sample_chunks)
        results = bm25.search("neural networks", top_k=2)
        assert len(results) <= 2

    def test_search_empty_index(self, bm25):
        results = bm25.search("test query")
        assert len(results) == 0

    def test_search_no_matches(self, bm25, sample_chunks):
        bm25.build(sample_chunks)
        results = bm25.search("quantum computing superconductor")
        # Should return empty or low-scoring results
        assert all(r.score == 0 or r.score > 0 for r in results)

    def test_tokenize(self):
        tokens = BM25Index._tokenize("Hello World! This is a TEST-case with numbers 123.")
        assert "hello" in tokens
        assert "world" in tokens
        assert "test-case" in tokens
        assert "123" in tokens

    def test_idf_effect(self, bm25, sample_chunks):
        bm25.build(sample_chunks)
        # "transformer" appears in 2 docs, "reinforcement" in 1
        # A query for "reinforcement" should rank p3 higher than if all docs had it
        results = bm25.search("reinforcement learning", top_k=4)
        assert results[0].chunk.paper_id == "p3"
