"""Document chunking strategies for the RAG pipeline.

Splits paper abstracts and metadata into overlapping chunks suitable
for embedding and retrieval. Uses a sliding window approach with
configurable size and overlap.
"""

import hashlib
import logging
from typing import Iterator

from src.config import ChunkingConfig
from src.data.models import DocumentChunk, Paper

logger = logging.getLogger(__name__)


class DocumentChunker:
    """Splits papers into overlapping text chunks for embedding."""

    def __init__(self, config: ChunkingConfig) -> None:
        self.chunk_size = config.chunk_size
        self.chunk_overlap = config.chunk_overlap
        self.min_chunk_length = config.min_chunk_length

    def _generate_chunk_id(self, paper_id: str, chunk_index: int) -> str:
        """Generate a deterministic chunk ID."""
        raw = f"{paper_id}::chunk_{chunk_index}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def _build_document_text(self, paper: Paper) -> str:
        """Construct the full text to chunk from a paper's fields."""
        parts = [
            f"Title: {paper.title}",
            f"Authors: {', '.join(paper.authors[:5])}",
            f"Abstract: {paper.abstract}",
        ]
        return "\n\n".join(parts)

    def _split_text(self, text: str) -> Iterator[str]:
        """Split text into overlapping chunks using word boundaries."""
        words = text.split()
        if not words:
            return

        # Approximate chunk_size in words (avg 5 chars/word + space)
        words_per_chunk = max(1, self.chunk_size // 6)
        overlap_words = max(0, self.chunk_overlap // 6)

        start = 0
        while start < len(words):
            end = start + words_per_chunk
            chunk_text = " ".join(words[start:end])

            if len(chunk_text.strip()) >= self.min_chunk_length:
                yield chunk_text.strip()

            if end >= len(words):
                break
            start = end - overlap_words

    def chunk_paper(self, paper: Paper) -> list[DocumentChunk]:
        """Split a single paper into document chunks."""
        full_text = self._build_document_text(paper)
        chunks = []

        metadata = {
            "title": paper.title,
            "authors": paper.authors,
            "arxiv_id": paper.arxiv_id,
            "categories": paper.categories,
            "published": paper.published.isoformat(),
            "pdf_url": paper.pdf_url,
        }

        for i, chunk_text in enumerate(self._split_text(full_text)):
            chunk = DocumentChunk(
                chunk_id=self._generate_chunk_id(paper.arxiv_id, i),
                paper_id=paper.arxiv_id,
                text=chunk_text,
                metadata=metadata,
            )
            chunks.append(chunk)

        return chunks

    def chunk_papers(self, papers: list[Paper]) -> list[DocumentChunk]:
        """Split multiple papers into document chunks."""
        all_chunks = []
        for paper in papers:
            chunks = self.chunk_paper(paper)
            all_chunks.extend(chunks)

        logger.info(
            "Chunked %d papers into %d chunks (avg %.1f chunks/paper)",
            len(papers),
            len(all_chunks),
            len(all_chunks) / max(len(papers), 1),
        )
        return all_chunks
