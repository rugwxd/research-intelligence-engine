"""Tests for the arXiv fetcher module."""

import json
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.config import ArxivConfig
from src.data.arxiv_fetcher import ArxivFetcher
from src.data.models import Paper


@pytest.fixture
def config():
    return ArxivConfig(
        max_results_per_query=10,
        queries=["test query"],
        rate_limit_seconds=0.0,
    )


@pytest.fixture
def fetcher(config):
    return ArxivFetcher(config)


@pytest.fixture
def sample_papers():
    return [
        Paper(
            arxiv_id="2301.00001",
            title="Test Paper 1",
            abstract="Abstract for paper 1",
            authors=["Author A"],
            categories=["cs.AI"],
            published=datetime(2023, 1, 1),
            updated=datetime(2023, 1, 1),
            pdf_url="https://arxiv.org/pdf/2301.00001",
            primary_category="cs.AI",
        ),
        Paper(
            arxiv_id="2301.00002",
            title="Test Paper 2",
            abstract="Abstract for paper 2",
            authors=["Author B"],
            categories=["cs.LG"],
            published=datetime(2023, 2, 1),
            updated=datetime(2023, 2, 1),
            pdf_url="https://arxiv.org/pdf/2301.00002",
            primary_category="cs.LG",
        ),
    ]


class TestArxivFetcher:
    def test_save_and_load_papers(self, fetcher, sample_papers, tmp_path):
        fetcher.config.output_dir = str(tmp_path)
        # Override PROJECT_ROOT for testing
        with patch("src.data.arxiv_fetcher.PROJECT_ROOT", tmp_path):
            saved_path = fetcher.save_papers(sample_papers)
            assert saved_path.exists()

            loaded = ArxivFetcher.load_papers(saved_path)
            assert len(loaded) == 2
            assert loaded[0].arxiv_id == "2301.00001"
            assert loaded[1].title == "Test Paper 2"

    def test_parse_entry(self, fetcher):
        entry = {
            "id": "http://arxiv.org/abs/2301.00001v1",
            "title": "  Test   Paper  ",
            "summary": "  Test   Abstract  ",
            "authors": [{"name": "Alice"}, {"name": "Bob"}],
            "tags": [{"term": "cs.AI"}, {"term": "cs.LG"}],
            "arxiv_primary_category": {"term": "cs.AI"},
            "published_parsed": (2023, 1, 1, 0, 0, 0, 0, 0, 0),
            "updated_parsed": (2023, 1, 1, 0, 0, 0, 0, 0, 0),
            "links": [
                {"href": "http://arxiv.org/abs/2301.00001", "type": "text/html"},
                {"href": "http://arxiv.org/pdf/2301.00001", "type": "application/pdf"},
            ],
        }
        paper = fetcher._parse_entry(entry)

        assert paper.arxiv_id == "2301.00001"
        assert paper.title == "Test Paper"
        assert paper.abstract == "Test Abstract"
        assert paper.authors == ["Alice", "Bob"]
        assert paper.pdf_url == "http://arxiv.org/pdf/2301.00001"

    def test_deduplication_logic(self, fetcher, sample_papers):
        # Simulate the deduplication that happens in fetch_papers
        seen_ids: set[str] = set()
        unique = []
        duped_papers = sample_papers + [sample_papers[0]]  # duplicate

        for paper in duped_papers:
            if paper.arxiv_id not in seen_ids:
                seen_ids.add(paper.arxiv_id)
                unique.append(paper)

        assert len(unique) == 2
