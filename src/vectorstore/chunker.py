"""Document chunking strategies for the RAG pipeline.

Provides multiple chunking strategies:
- SlidingWindowChunker: Fixed-size overlapping windows (fast, predictable)
- SemanticChunker: Splits at natural topic boundaries using embedding similarity

Both conform to the same interface for easy comparison in ablation studies.
"""

import hashlib
import logging
from abc import ABC, abstractmethod
from typing import Iterator

import numpy as np

from src.config import ChunkingConfig
from src.data.models import DocumentChunk, Paper

logger = logging.getLogger(__name__)


class BaseChunker(ABC):
    """Abstract base class for document chunking strategies."""

    def __init__(self, config: ChunkingConfig) -> None:
        self.chunk_size = config.chunk_size
        self.chunk_overlap = config.chunk_overlap
        self.min_chunk_length = config.min_chunk_length

    @abstractmethod
    def _split_text(self, text: str) -> Iterator[str]:
        """Split text into chunks. Strategy-specific."""

    def _generate_chunk_id(self, paper_id: str, chunk_index: int, prefix: str = "") -> str:
        """Generate a deterministic chunk ID."""
        raw = f"{prefix}{paper_id}::chunk_{chunk_index}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def _build_document_text(self, paper: Paper) -> str:
        """Construct the full text to chunk from a paper's fields."""
        parts = [
            f"Title: {paper.title}",
            f"Authors: {', '.join(paper.authors[:5])}",
            f"Abstract: {paper.abstract}",
        ]
        if hasattr(paper, "full_text") and paper.full_text:
            parts.append(f"Content: {paper.full_text}")
        return "\n\n".join(parts)

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
            "Chunked %d papers into %d chunks (avg %.1f chunks/paper, strategy=%s)",
            len(papers),
            len(all_chunks),
            len(all_chunks) / max(len(papers), 1),
            self.__class__.__name__,
        )
        return all_chunks


class SlidingWindowChunker(BaseChunker):
    """Fixed-size sliding window chunker with configurable overlap.

    Splits text into equal-sized chunks with overlap to prevent
    information loss at chunk boundaries. Fast and predictable,
    but may split mid-sentence or mid-concept.

    Best for: Short texts (abstracts), uniform document lengths.
    """

    def _split_text(self, text: str) -> Iterator[str]:
        words = text.split()
        if not words:
            return

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


class SemanticChunker(BaseChunker):
    """Semantic chunker that splits at natural topic boundaries.

    Groups sentences into chunks by detecting topic shifts using
    embedding cosine similarity between consecutive sentence groups.
    When similarity drops below a threshold, a new chunk begins.

    This produces variable-length chunks that respect topical coherence,
    improving retrieval precision for complex documents.

    Best for: Long documents (full papers), multi-topic content.
    """

    def __init__(
        self,
        config: ChunkingConfig,
        similarity_threshold: float = 0.5,
        sentence_group_size: int = 3,
    ) -> None:
        super().__init__(config)
        self.similarity_threshold = similarity_threshold
        self.sentence_group_size = sentence_group_size
        self._embedder = None

    def _get_embedder(self):
        """Lazy-load a lightweight sentence embedder for boundary detection."""
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
        return self._embedder

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences using regex-based rules."""
        import re
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if len(s.strip()) > 10]

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def _split_text(self, text: str) -> Iterator[str]:
        sentences = self._split_sentences(text)
        if not sentences:
            return

        if len(sentences) <= self.sentence_group_size:
            yield " ".join(sentences)
            return

        # Group sentences into overlapping windows for embedding
        groups = []
        for i in range(0, len(sentences), self.sentence_group_size):
            group = " ".join(sentences[i:i + self.sentence_group_size])
            groups.append(group)

        if not groups:
            yield text
            return

        # Embed all sentence groups
        embedder = self._get_embedder()
        embeddings = embedder.encode(groups, normalize_embeddings=True)

        # Find breakpoints where similarity drops
        breakpoints = []
        for i in range(1, len(embeddings)):
            sim = self._cosine_similarity(embeddings[i - 1], embeddings[i])
            if sim < self.similarity_threshold:
                breakpoints.append(i)

        # Build chunks from breakpoints
        chunk_groups = []
        prev = 0
        for bp in breakpoints:
            chunk_groups.append(groups[prev:bp])
            prev = bp
        chunk_groups.append(groups[prev:])

        for chunk_group in chunk_groups:
            chunk_text = " ".join(chunk_group).strip()
            if len(chunk_text) >= self.min_chunk_length:
                # Enforce max chunk size by splitting oversized chunks
                if len(chunk_text) > self.chunk_size * 3:
                    words = chunk_text.split()
                    words_per_sub = max(1, self.chunk_size // 6)
                    for start in range(0, len(words), words_per_sub):
                        sub = " ".join(words[start:start + words_per_sub])
                        if len(sub) >= self.min_chunk_length:
                            yield sub
                else:
                    yield chunk_text


# Backward-compatible alias
DocumentChunker = SlidingWindowChunker


def create_chunker(config: ChunkingConfig, strategy: str = "sliding_window") -> BaseChunker:
    """Factory function for creating chunker instances.

    Args:
        config: Chunking configuration.
        strategy: One of 'sliding_window' or 'semantic'.

    Returns:
        Configured chunker instance.
    """
    if strategy == "semantic":
        return SemanticChunker(config)
    return SlidingWindowChunker(config)
