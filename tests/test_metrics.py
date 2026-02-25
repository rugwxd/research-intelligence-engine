"""Tests for the heuristic evaluation metrics."""

import pytest

from src.data.models import DocumentChunk, RAGResponse, RetrievalResult
from src.evaluation.metrics import HeuristicEvaluator


@pytest.fixture
def evaluator():
    return HeuristicEvaluator()


@pytest.fixture
def sample_response():
    chunks = [
        DocumentChunk(
            chunk_id="c1",
            paper_id="p1",
            text=(
                "Transformer models use self-attention mechanisms to process "
                "sequential data. Multi-head attention allows the model to attend "
                "to different representation subspaces."
            ),
            metadata={
                "title": "Attention Is All You Need",
                "authors": ["Vaswani"],
                "arxiv_id": "1706.03762",
            },
        ),
        DocumentChunk(
            chunk_id="c2",
            paper_id="p2",
            text=(
                "BERT introduced bidirectional pre-training for language "
                "representations. Fine-tuning BERT achieves state-of-the-art "
                "results on many NLP benchmarks."
            ),
            metadata={
                "title": "BERT: Pre-training of Deep Bidirectional Transformers",
                "authors": ["Devlin"],
                "arxiv_id": "1810.04805",
            },
        ),
    ]
    sources = [
        RetrievalResult(chunk=chunks[0], score=0.92),
        RetrievalResult(chunk=chunks[1], score=0.85),
    ]
    return RAGResponse(
        query="How do transformer attention mechanisms work?",
        answer=(
            "Transformer models use self-attention mechanisms to process sequential "
            "data [Source 1]. Multi-head attention allows attending to different "
            "representation subspaces. BERT introduced bidirectional pre-training "
            "for language representations [Source 2], achieving state-of-the-art "
            "results on NLP benchmarks."
        ),
        sources=sources,
    )


class TestHeuristicEvaluator:
    def test_faithfulness_with_citations(self, evaluator, sample_response):
        score = evaluator.faithfulness_score(sample_response)
        assert 0.0 <= score <= 1.0
        assert score > 0.5  # Answer with citations should score well

    def test_faithfulness_no_citations(self, evaluator):
        response = RAGResponse(
            query="test",
            answer="This is completely made up information with no grounding.",
            sources=[
                RetrievalResult(
                    chunk=DocumentChunk(
                        chunk_id="c1",
                        paper_id="p1",
                        text="Actual source content about neural networks.",
                        metadata={"title": "T", "authors": []},
                    ),
                    score=0.9,
                )
            ],
        )
        score = evaluator.faithfulness_score(response)
        assert 0.0 <= score <= 1.0

    def test_relevance_score(self, evaluator, sample_response):
        score = evaluator.relevance_score(sample_response)
        assert 0.0 <= score <= 1.0
        assert score > 0.0  # Query terms should appear in sources

    def test_completeness_score(self, evaluator, sample_response):
        score = evaluator.completeness_score(sample_response)
        assert 0.0 <= score <= 1.0

    def test_evaluate_returns_all_metrics(self, evaluator, sample_response):
        scores = evaluator.evaluate(sample_response)
        assert "faithfulness" in scores
        assert "relevance" in scores
        assert "completeness" in scores
        assert "overall" in scores
        for value in scores.values():
            assert 0.0 <= value <= 1.0

    def test_empty_sources(self, evaluator):
        response = RAGResponse(
            query="test",
            answer="No sources available.",
            sources=[],
        )
        scores = evaluator.evaluate(response)
        assert all(0.0 <= v <= 1.0 for v in scores.values())

    def test_empty_answer(self, evaluator):
        response = RAGResponse(query="test", answer="", sources=[])
        scores = evaluator.evaluate(response)
        assert all(0.0 <= v <= 1.0 for v in scores.values())


class TestHelperMethods:
    def test_split_sentences(self):
        text = "First sentence. Second sentence! Third sentence?"
        sentences = HeuristicEvaluator._split_sentences(text)
        assert len(sentences) == 3

    def test_extract_content_words_filters_stopwords(self):
        text = "the quick brown fox jumps over the lazy dog"
        words = HeuristicEvaluator._extract_content_words(text)
        assert "the" not in words
        assert "quick" in words
        assert "brown" in words
