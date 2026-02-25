"""Tests for data models."""

from datetime import datetime

from src.data.models import DocumentChunk, Paper, RAGResponse


class TestPaper:
    def test_to_dict_roundtrip(self):
        paper = Paper(
            arxiv_id="2301.00001",
            title="Test Paper",
            abstract="This is a test.",
            authors=["Alice", "Bob"],
            categories=["cs.AI"],
            published=datetime(2023, 1, 1),
            updated=datetime(2023, 6, 1),
            pdf_url="https://arxiv.org/pdf/2301.00001",
            primary_category="cs.AI",
        )
        data = paper.to_dict()
        restored = Paper.from_dict(data)

        assert restored.arxiv_id == paper.arxiv_id
        assert restored.title == paper.title
        assert restored.abstract == paper.abstract
        assert restored.authors == paper.authors

    def test_to_dict_contains_all_fields(self):
        paper = Paper(
            arxiv_id="2301.00001",
            title="Title",
            abstract="Abstract",
            authors=["Author"],
            categories=["cs.AI"],
            published=datetime(2023, 1, 1),
            updated=datetime(2023, 1, 1),
            pdf_url="",
            primary_category="cs.AI",
        )
        data = paper.to_dict()
        expected_keys = {
            "arxiv_id",
            "title",
            "abstract",
            "authors",
            "categories",
            "published",
            "updated",
            "pdf_url",
            "primary_category",
        }
        assert set(data.keys()) == expected_keys


class TestDocumentChunk:
    def test_display_source_short_authors(self):
        chunk = DocumentChunk(
            chunk_id="abc",
            paper_id="123",
            text="test",
            metadata={"title": "My Paper", "authors": ["Alice", "Bob"]},
        )
        assert "My Paper" in chunk.display_source
        assert "Alice, Bob" in chunk.display_source

    def test_display_source_many_authors(self):
        chunk = DocumentChunk(
            chunk_id="abc",
            paper_id="123",
            text="test",
            metadata={
                "title": "Big Paper",
                "authors": ["A", "B", "C", "D", "E"],
            },
        )
        assert "et al." in chunk.display_source


class TestRAGResponse:
    def test_default_values(self):
        response = RAGResponse(
            query="test query",
            answer="test answer",
            sources=[],
        )
        assert response.eval_scores == {}
        assert response.latency_ms == 0.0
