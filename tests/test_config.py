"""Tests for configuration management."""

import pytest

from src.config import (
    ArxivConfig,
    ChunkingConfig,
    EmbeddingsConfig,
    GenerationConfig,
    Settings,
    load_config,
)


class TestSettings:
    def test_default_settings(self):
        settings = Settings()
        assert settings.arxiv.max_results_per_query == 150
        assert settings.chunking.chunk_size == 512
        assert settings.embeddings.model_name == "all-MiniLM-L6-v2"

    def test_load_config_with_defaults(self):
        config = load_config()
        assert isinstance(config, Settings)
        assert len(config.arxiv.queries) > 0

    def test_arxiv_config_defaults(self):
        config = ArxivConfig()
        assert config.rate_limit_seconds == 3.0
        assert len(config.queries) >= 3

    def test_chunking_config(self):
        config = ChunkingConfig(chunk_size=256, chunk_overlap=32)
        assert config.chunk_size == 256
        assert config.chunk_overlap == 32

    def test_embeddings_config(self):
        config = EmbeddingsConfig()
        assert config.batch_size == 64
        assert config.normalize is True

    def test_generation_config(self):
        config = GenerationConfig()
        assert config.max_tokens == 2048
        assert config.temperature == 0.1
