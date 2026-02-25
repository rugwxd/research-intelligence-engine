"""Embedding generation using sentence-transformers.

Wraps the sentence-transformers library to produce dense vector
representations of text chunks for similarity search.
"""

import logging

import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import EmbeddingsConfig
from src.data.models import DocumentChunk

logger = logging.getLogger(__name__)


class Embedder:
    """Generates embeddings for text using sentence-transformers."""

    def __init__(self, config: EmbeddingsConfig) -> None:
        self.config = config
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        """Lazy-load the embedding model."""
        if self._model is None:
            logger.info("Loading embedding model: %s", self.config.model_name)
            self._model = SentenceTransformer(self.config.model_name)
            logger.info("Embedding model loaded (dim=%d)", self.dimension)
        return self._model

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        return self.model.get_sentence_embedding_dimension()

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed.

        Returns:
            numpy array of shape (n_texts, embedding_dim).
        """
        logger.debug("Embedding %d texts", len(texts))

        embeddings = self.model.encode(
            texts,
            batch_size=self.config.batch_size,
            show_progress_bar=len(texts) > 100,
            normalize_embeddings=self.config.normalize,
        )

        return np.array(embeddings, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Generate embedding for a single query string.

        Returns:
            numpy array of shape (1, embedding_dim).
        """
        embedding = self.model.encode(
            [query],
            normalize_embeddings=self.config.normalize,
        )
        return np.array(embedding, dtype=np.float32)

    def embed_chunks(self, chunks: list[DocumentChunk]) -> np.ndarray:
        """Generate embeddings for document chunks and attach them.

        Args:
            chunks: List of DocumentChunk objects.

        Returns:
            numpy array of all embeddings.
        """
        texts = [chunk.text for chunk in chunks]
        embeddings = self.embed_texts(texts)

        for chunk, emb in zip(chunks, embeddings):
            chunk.embedding = emb.tolist()

        logger.info("Generated embeddings for %d chunks", len(chunks))
        return embeddings
