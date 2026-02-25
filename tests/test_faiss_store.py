"""Tests for the FAISS vector store."""

import numpy as np
import pytest

from src.config import VectorDBConfig
from src.data.models import DocumentChunk
from src.vectorstore.faiss_store import FAISSStore


@pytest.fixture
def config():
    return VectorDBConfig(
        index_path="data/vectordb/test.index",
        metadata_path="data/vectordb/test_meta.pkl",
        index_type="flat",
    )


@pytest.fixture
def sample_chunks():
    return [
        DocumentChunk(
            chunk_id=f"c{i}",
            paper_id=f"p{i}",
            text=f"Content chunk {i}",
            metadata={"title": f"Paper {i}", "authors": [f"Author {i}"]},
        )
        for i in range(10)
    ]


@pytest.fixture
def sample_embeddings():
    np.random.seed(42)
    return np.random.randn(10, 384).astype(np.float32)


class TestFAISSStore:
    def test_build_index(self, config, sample_chunks, sample_embeddings):
        store = FAISSStore(config, dimension=384)
        store.build_index(sample_chunks, sample_embeddings)

        assert store.is_ready
        assert store.index.ntotal == 10

    def test_search_returns_results(self, config, sample_chunks, sample_embeddings):
        store = FAISSStore(config, dimension=384)
        store.build_index(sample_chunks, sample_embeddings)

        query = np.random.randn(1, 384).astype(np.float32)
        results = store.search(query, top_k=3)

        assert len(results) == 3
        assert all(r.score is not None for r in results)

    def test_search_respects_top_k(self, config, sample_chunks, sample_embeddings):
        store = FAISSStore(config, dimension=384)
        store.build_index(sample_chunks, sample_embeddings)

        query = np.random.randn(1, 384).astype(np.float32)

        results_3 = store.search(query, top_k=3)
        results_5 = store.search(query, top_k=5)

        assert len(results_3) == 3
        assert len(results_5) == 5

    def test_search_empty_index(self, config):
        store = FAISSStore(config, dimension=384)
        query = np.random.randn(1, 384).astype(np.float32)
        results = store.search(query, top_k=5)

        assert len(results) == 0

    def test_embedding_mismatch_raises(self, config, sample_chunks):
        store = FAISSStore(config, dimension=384)
        bad_embeddings = np.random.randn(5, 384).astype(np.float32)

        with pytest.raises(ValueError, match="Mismatch"):
            store.build_index(sample_chunks, bad_embeddings)

    def test_save_and_load(self, config, sample_chunks, sample_embeddings, tmp_path):
        config.index_path = str(tmp_path / "test.index")
        config.metadata_path = str(tmp_path / "test_meta.pkl")

        from unittest.mock import patch

        with patch("src.vectorstore.faiss_store.PROJECT_ROOT", tmp_path):
            store = FAISSStore(config, dimension=384)
            store.build_index(sample_chunks, sample_embeddings)
            store.save()

            store2 = FAISSStore(config, dimension=384)
            store2.load()

            assert store2.is_ready
            assert store2.index.ntotal == 10
            assert len(store2.chunks) == 10

    def test_is_ready_false_initially(self, config):
        store = FAISSStore(config, dimension=384)
        assert not store.is_ready

    def test_results_sorted_by_score(self, config, sample_chunks, sample_embeddings):
        store = FAISSStore(config, dimension=384)
        store.build_index(sample_chunks, sample_embeddings)

        query = np.random.randn(1, 384).astype(np.float32)
        results = store.search(query, top_k=5)

        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)
