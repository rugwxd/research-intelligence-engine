"""Centralized configuration management for the Research Intelligence Engine."""

import logging
import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

PROJECT_ROOT = Path(__file__).parent.parent.resolve()


class ArxivConfig(BaseModel):
    max_results_per_query: int = 150
    queries: list[str] = Field(
        default_factory=lambda: [
            "large language models",
            "transformer architecture",
            "reinforcement learning",
            "computer vision deep learning",
            "neural network optimization",
        ]
    )
    rate_limit_seconds: float = 3.0
    output_dir: str = "data/raw"


class ChunkingConfig(BaseModel):
    chunk_size: int = 512
    chunk_overlap: int = 64
    min_chunk_length: int = 50


class EmbeddingsConfig(BaseModel):
    model_name: str = "all-MiniLM-L6-v2"
    batch_size: int = 64
    normalize: bool = True


class VectorDBConfig(BaseModel):
    index_path: str = "data/vectordb/faiss.index"
    metadata_path: str = "data/vectordb/metadata.pkl"
    index_type: str = "flat"


class RetrievalConfig(BaseModel):
    top_k: int = 10
    rerank_top_k: int = 5
    similarity_threshold: float = 0.3


class GenerationConfig(BaseModel):
    model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 2048
    temperature: float = 0.1


class EvaluationConfig(BaseModel):
    num_samples: int = 50
    faithfulness_threshold: float = 0.7
    relevance_threshold: float = 0.6


class LoggingConfig(BaseModel):
    level: str = "INFO"
    format: str = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    file: str = "logs/rie.log"


class Settings(BaseSettings):
    """Application settings loaded from YAML config and environment variables."""

    arxiv: ArxivConfig = Field(default_factory=ArxivConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    vectordb: VectorDBConfig = Field(default_factory=VectorDBConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")

    model_config = {"env_prefix": "", "env_nested_delimiter": "__"}


def load_config(config_path: str | None = None) -> Settings:
    """Load configuration from YAML file, with environment variable overrides."""
    if config_path is None:
        config_path = str(PROJECT_ROOT / "configs" / "default.yaml")

    data: dict[str, Any] = {}
    config_file = Path(config_path)
    if config_file.exists():
        with open(config_file) as f:
            data = yaml.safe_load(f) or {}

    # Environment variables override YAML values
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if api_key:
        data["anthropic_api_key"] = api_key

    return Settings(**data)


def setup_logging(config: LoggingConfig) -> None:
    """Configure application-wide logging."""
    log_file = PROJECT_ROOT / config.file
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=getattr(logging, config.level.upper()),
        format=config.format,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )
