"""BM25 sparse retrieval for hybrid search.

Implements Okapi BM25 scoring from scratch to avoid heavy dependencies.
Used alongside FAISS dense retrieval for reciprocal rank fusion.
"""

import logging
import math
import re

from src.data.models import DocumentChunk, RetrievalResult

logger = logging.getLogger(__name__)


class BM25Index:
    """Okapi BM25 sparse retrieval index.

    BM25 captures exact term-matching signals that dense embeddings
    can miss, particularly for rare technical terms, acronyms, and
    specific model names common in ML research.

    Parameters:
        k1: Term frequency saturation parameter (default 1.5).
        b: Length normalization parameter (default 0.75).
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.chunks: list[DocumentChunk] = []
        self.doc_freqs: dict[str, int] = {}
        self.doc_lengths: list[int] = []
        self.avg_doc_length: float = 0.0
        self.tokenized_docs: list[list[str]] = []
        self.n_docs: int = 0

    def build(self, chunks: list[DocumentChunk]) -> None:
        """Build the BM25 index from document chunks.

        Args:
            chunks: List of document chunks to index.
        """
        self.chunks = chunks
        self.n_docs = len(chunks)
        self.tokenized_docs = [self._tokenize(c.text) for c in chunks]
        self.doc_lengths = [len(doc) for doc in self.tokenized_docs]
        self.avg_doc_length = sum(self.doc_lengths) / max(self.n_docs, 1)

        # Compute document frequencies
        self.doc_freqs = {}
        for doc_tokens in self.tokenized_docs:
            unique_terms = set(doc_tokens)
            for term in unique_terms:
                self.doc_freqs[term] = self.doc_freqs.get(term, 0) + 1

        logger.info(
            "Built BM25 index: %d docs, %d unique terms, avg length %.1f",
            self.n_docs,
            len(self.doc_freqs),
            self.avg_doc_length,
        )

    def search(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        """Search the BM25 index for relevant documents.

        Args:
            query: Natural language query string.
            top_k: Number of results to return.

        Returns:
            List of RetrievalResult sorted by BM25 score descending.
        """
        if not self.chunks:
            return []

        query_terms = self._tokenize(query)
        scores = self._score_documents(query_terms)

        # Get top-k indices
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append(
                    RetrievalResult(
                        chunk=self.chunks[idx],
                        score=scores[idx],
                    )
                )

        return results

    def _score_documents(self, query_terms: list[str]) -> list[float]:
        """Compute BM25 scores for all documents given query terms."""
        scores = [0.0] * self.n_docs

        for term in query_terms:
            if term not in self.doc_freqs:
                continue

            df = self.doc_freqs[term]
            idf = math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1.0)

            for i, doc_tokens in enumerate(self.tokenized_docs):
                tf = doc_tokens.count(term)
                if tf == 0:
                    continue

                doc_len = self.doc_lengths[i]
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_length)
                scores[i] += idf * (numerator / denominator)

        return scores

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Tokenize text into lowercase terms.

        Handles hyphenated terms (e.g., 'self-attention') by indexing both
        the full term and its components for better recall on technical text.
        """
        tokens = re.findall(r"\b[a-z0-9]+(?:[-'][a-z0-9]+)*\b", text.lower())
        expanded = []
        for token in tokens:
            expanded.append(token)
            if "-" in token:
                expanded.extend(token.split("-"))
        return expanded
