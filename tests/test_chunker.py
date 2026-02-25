"""Tests for the document chunking module."""

from datetime import datetime

import pytest

from src.config import ChunkingConfig
from src.data.models import Paper
from src.vectorstore.chunker import DocumentChunker


@pytest.fixture
def chunker():
    config = ChunkingConfig(chunk_size=512, chunk_overlap=64, min_chunk_length=50)
    return DocumentChunker(config)


@pytest.fixture
def sample_paper():
    return Paper(
        arxiv_id="2301.00001",
        title="Attention Is All You Need: A Comprehensive Survey",
        abstract=(
            "The transformer architecture has revolutionized natural language processing "
            "and beyond. This survey covers the key innovations in attention mechanisms, "
            "including multi-head attention, sparse attention, and linear attention variants. "
            "We discuss applications across language modeling, machine translation, and "
            "computer vision. The paper also examines efficiency improvements such as "
            "FlashAttention, grouped query attention, and sliding window attention. "
            "Furthermore, we analyze the scaling laws governing transformer performance "
            "and discuss future research directions in architecture design."
        ),
        authors=["Alice Smith", "Bob Jones", "Charlie Brown"],
        categories=["cs.CL", "cs.AI"],
        published=datetime(2023, 1, 1),
        updated=datetime(2023, 6, 15),
        pdf_url="https://arxiv.org/pdf/2301.00001",
        primary_category="cs.CL",
    )


class TestDocumentChunker:
    def test_chunk_paper_produces_chunks(self, chunker, sample_paper):
        chunks = chunker.chunk_paper(sample_paper)
        assert len(chunks) > 0

    def test_chunk_ids_are_unique(self, chunker, sample_paper):
        chunks = chunker.chunk_paper(sample_paper)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_chunk_ids_are_deterministic(self, chunker, sample_paper):
        chunks1 = chunker.chunk_paper(sample_paper)
        chunks2 = chunker.chunk_paper(sample_paper)
        assert [c.chunk_id for c in chunks1] == [c.chunk_id for c in chunks2]

    def test_chunks_contain_paper_metadata(self, chunker, sample_paper):
        chunks = chunker.chunk_paper(sample_paper)
        for chunk in chunks:
            assert chunk.paper_id == sample_paper.arxiv_id
            assert chunk.metadata["title"] == sample_paper.title
            assert chunk.metadata["arxiv_id"] == sample_paper.arxiv_id

    def test_chunk_text_not_empty(self, chunker, sample_paper):
        chunks = chunker.chunk_paper(sample_paper)
        for chunk in chunks:
            assert len(chunk.text.strip()) > 0

    def test_min_chunk_length_respected(self, chunker, sample_paper):
        chunks = chunker.chunk_paper(sample_paper)
        for chunk in chunks:
            assert len(chunk.text) >= chunker.min_chunk_length

    def test_chunk_papers_multiple(self, chunker, sample_paper):
        papers = [sample_paper, sample_paper]
        # Modify second paper's ID so chunks differ
        papers[1] = Paper(**{**sample_paper.__dict__, "arxiv_id": "2301.00002"})
        chunks = chunker.chunk_papers(papers)
        paper_ids = set(c.paper_id for c in chunks)
        assert len(paper_ids) == 2

    def test_empty_abstract_handled(self, chunker):
        paper = Paper(
            arxiv_id="2301.00003",
            title="Short Paper",
            abstract="",
            authors=["Author"],
            categories=["cs.AI"],
            published=datetime(2023, 1, 1),
            updated=datetime(2023, 1, 1),
            pdf_url="",
            primary_category="cs.AI",
        )
        chunks = chunker.chunk_paper(paper)
        # Should still produce at least the title chunk
        assert isinstance(chunks, list)

    def test_small_chunk_size_config(self):
        config = ChunkingConfig(chunk_size=100, chunk_overlap=20, min_chunk_length=10)
        chunker = DocumentChunker(config)
        paper = Paper(
            arxiv_id="2301.00004",
            title="Test Paper",
            abstract="This is a test abstract with enough words to produce multiple chunks.",
            authors=["Author"],
            categories=["cs.AI"],
            published=datetime(2023, 1, 1),
            updated=datetime(2023, 1, 1),
            pdf_url="",
            primary_category="cs.AI",
        )
        chunks = chunker.chunk_paper(paper)
        assert isinstance(chunks, list)
