"""Data models for research papers and document chunks."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Paper:
    """Represents a single research paper from arXiv."""

    arxiv_id: str
    title: str
    abstract: str
    authors: list[str]
    categories: list[str]
    published: datetime
    updated: datetime
    pdf_url: str
    primary_category: str

    def to_dict(self) -> dict:
        return {
            "arxiv_id": self.arxiv_id,
            "title": self.title,
            "abstract": self.abstract,
            "authors": self.authors,
            "categories": self.categories,
            "published": self.published.isoformat(),
            "updated": self.updated.isoformat(),
            "pdf_url": self.pdf_url,
            "primary_category": self.primary_category,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Paper:
        return cls(
            arxiv_id=data["arxiv_id"],
            title=data["title"],
            abstract=data["abstract"],
            authors=data["authors"],
            categories=data["categories"],
            published=datetime.fromisoformat(data["published"]),
            updated=datetime.fromisoformat(data["updated"]),
            pdf_url=data["pdf_url"],
            primary_category=data["primary_category"],
        )


@dataclass
class DocumentChunk:
    """A chunk of text derived from a paper, ready for embedding."""

    chunk_id: str
    paper_id: str
    text: str
    metadata: dict = field(default_factory=dict)
    embedding: list[float] | None = None

    @property
    def display_source(self) -> str:
        title = self.metadata.get("title", "Unknown")
        authors = self.metadata.get("authors", [])
        author_str = ", ".join(authors[:3])
        if len(authors) > 3:
            author_str += " et al."
        return f"{title} ({author_str})"


@dataclass
class RetrievalResult:
    """A single retrieval result with score."""

    chunk: DocumentChunk
    score: float


@dataclass
class RAGResponse:
    """Complete RAG pipeline response."""

    query: str
    answer: str
    sources: list[RetrievalResult]
    eval_scores: dict[str, float] = field(default_factory=dict)
    latency_ms: float = 0.0
