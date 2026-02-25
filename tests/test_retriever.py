"""Tests for the retrieval pipeline."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from src.config import RetrievalConfig
from src.data.models import DocumentChunk, RetrievalResult
from src.retrieval.retriever import Retriever


@pytest.fixture
def config():
    return RetrievalConfig(top_k=10, rerank_top_k=5, similarity_threshold=0.3)


@pytest.fixture
def mock_store():
    store = MagicMock()
    store.is_ready = True
    return store


@pytest.fixture
def mock_embedder():
    embedder = MagicMock()
    embedder.embed_query.return_value = np.random.randn(1, 384).astype(np.float32)
    return embedder


def _make_result(paper_id: str, score: float) -> RetrievalResult:
    chunk = DocumentChunk(
        chunk_id=f"chunk_{paper_id}",
        paper_id=paper_id,
        text=f"Content from paper {paper_id}",
        metadata={"title": f"Paper {paper_id}", "authors": ["Author"], "arxiv_id": paper_id},
    )
    return RetrievalResult(chunk=chunk, score=score)


class TestRetriever:
    def test_retrieve_returns_results(self, config, mock_store, mock_embedder):
        mock_store.search.return_value = [
            _make_result("p1", 0.95),
            _make_result("p2", 0.85),
        ]
        retriever = Retriever(config, mock_store, mock_embedder)
        results = retriever.retrieve("test query")

        assert len(results) > 0
        mock_embedder.embed_query.assert_called_once()
        mock_store.search.assert_called_once()

    def test_retrieve_filters_below_threshold(self, config, mock_store, mock_embedder):
        mock_store.search.return_value = [
            _make_result("p1", 0.95),
            _make_result("p2", 0.10),  # Below threshold
        ]
        retriever = Retriever(config, mock_store, mock_embedder)
        results = retriever.retrieve("test query")

        assert len(results) == 1
        assert results[0].chunk.paper_id == "p1"

    def test_retrieve_deduplicates_by_paper(self, config, mock_store, mock_embedder):
        mock_store.search.return_value = [
            _make_result("p1", 0.95),
            _make_result("p1", 0.80),  # Same paper, lower score
            _make_result("p2", 0.85),
        ]
        # Fix: different chunk_ids for same paper
        mock_store.search.return_value[1].chunk.chunk_id = "chunk_p1_2"

        retriever = Retriever(config, mock_store, mock_embedder)
        results = retriever.retrieve("test query")

        paper_ids = [r.chunk.paper_id for r in results]
        assert paper_ids.count("p1") == 1

    def test_retrieve_respects_rerank_top_k(self, config, mock_store, mock_embedder):
        config.rerank_top_k = 2
        mock_store.search.return_value = [_make_result(f"p{i}", 0.9 - i * 0.1) for i in range(5)]
        retriever = Retriever(config, mock_store, mock_embedder)
        results = retriever.retrieve("test query")

        assert len(results) <= 2

    def test_format_context(self):
        results = [
            _make_result("p1", 0.95),
            _make_result("p2", 0.85),
        ]
        context = Retriever.format_context(results)

        assert "Source 1" in context
        assert "Source 2" in context
        assert "p1" in context

    def test_format_context_empty(self):
        context = Retriever.format_context([])
        assert "No relevant sources" in context

    def test_fallback_when_all_below_threshold(self, config, mock_store, mock_embedder):
        mock_store.search.return_value = [
            _make_result("p1", 0.10),
            _make_result("p2", 0.05),
        ]
        retriever = Retriever(config, mock_store, mock_embedder)
        results = retriever.retrieve("test query")

        # Should fall back to returning some results
        assert len(results) > 0
