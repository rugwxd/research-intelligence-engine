"""FAISS-backed vector store for document chunk retrieval.

Manages a FAISS index alongside a metadata store, supporting
index creation, persistence, and similarity search operations.
"""

import logging
import pickle
from pathlib import Path

import faiss
import numpy as np

from src.config import VectorDBConfig, PROJECT_ROOT
from src.data.models import DocumentChunk, RetrievalResult

logger = logging.getLogger(__name__)


class FAISSStore:
    """Vector store backed by a FAISS index with metadata persistence."""

    def __init__(self, config: VectorDBConfig, dimension: int) -> None:
        self.config = config
        self.dimension = dimension
        self.index: faiss.Index | None = None
        self.chunks: list[DocumentChunk] = []

    def _create_index(self) -> faiss.Index:
        """Create a new FAISS index based on configuration."""
        index_type = self.config.index_type.lower()

        if index_type == "flat":
            index = faiss.IndexFlatIP(self.dimension)
        elif index_type == "ivf":
            quantizer = faiss.IndexFlatIP(self.dimension)
            nlist = min(100, max(1, len(self.chunks) // 10))
            index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
        elif index_type == "hnsw":
            index = faiss.IndexHNSWFlat(self.dimension, 32)
        else:
            logger.warning("Unknown index type '%s', falling back to flat", index_type)
            index = faiss.IndexFlatIP(self.dimension)

        return index

    def build_index(self, chunks: list[DocumentChunk], embeddings: np.ndarray) -> None:
        """Build the FAISS index from chunks and their embeddings.

        Args:
            chunks: Document chunks with metadata.
            embeddings: Pre-computed embeddings array of shape (n, dim).
        """
        if embeddings.shape[0] != len(chunks):
            raise ValueError(
                f"Mismatch: {embeddings.shape[0]} embeddings vs {len(chunks)} chunks"
            )

        self.chunks = chunks
        self.index = self._create_index()

        # Normalize for cosine similarity via inner product
        faiss.normalize_L2(embeddings)

        # IVF indexes require training
        if hasattr(self.index, "train") and not self.index.is_trained:
            logger.info("Training IVF index with %d vectors", len(embeddings))
            self.index.train(embeddings)

        self.index.add(embeddings)
        logger.info(
            "Built FAISS index: %d vectors, dim=%d, type=%s",
            self.index.ntotal,
            self.dimension,
            self.config.index_type,
        )

    def search(self, query_embedding: np.ndarray, top_k: int = 10) -> list[RetrievalResult]:
        """Search the index for the most similar chunks.

        Args:
            query_embedding: Query vector of shape (1, dim).
            top_k: Number of results to return.

        Returns:
            List of RetrievalResult sorted by descending similarity.
        """
        if self.index is None or self.index.ntotal == 0:
            logger.warning("Search called on empty index")
            return []

        query_vec = query_embedding.copy()
        faiss.normalize_L2(query_vec)

        scores, indices = self.index.search(query_vec, min(top_k, self.index.ntotal))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            results.append(RetrievalResult(
                chunk=self.chunks[idx],
                score=float(score),
            ))

        return results

    def save(self) -> None:
        """Persist the index and metadata to disk."""
        if self.index is None:
            raise RuntimeError("No index to save")

        index_path = PROJECT_ROOT / self.config.index_path
        metadata_path = PROJECT_ROOT / self.config.metadata_path

        index_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(index_path))

        with open(metadata_path, "wb") as f:
            pickle.dump(self.chunks, f)

        logger.info("Saved FAISS index (%d vectors) to %s", self.index.ntotal, index_path)

    def load(self) -> None:
        """Load a previously saved index and metadata from disk."""
        index_path = PROJECT_ROOT / self.config.index_path
        metadata_path = PROJECT_ROOT / self.config.metadata_path

        if not index_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(
                f"Index files not found at {index_path} / {metadata_path}. "
                "Run the indexing pipeline first."
            )

        self.index = faiss.read_index(str(index_path))

        with open(metadata_path, "rb") as f:
            self.chunks = pickle.load(f)

        logger.info(
            "Loaded FAISS index: %d vectors, %d chunks",
            self.index.ntotal,
            len(self.chunks),
        )

    @property
    def is_ready(self) -> bool:
        """Check if the store has a loaded index."""
        return self.index is not None and self.index.ntotal > 0
