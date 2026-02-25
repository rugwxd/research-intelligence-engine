"""Data ingestion script.

Fetches papers from arXiv, generates embeddings, and builds the FAISS index.
Run this before starting the Streamlit app.

Usage:
    python scripts/ingest.py [--papers 500] [--config configs/default.yaml]
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import create_pipeline

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Ingest arXiv papers into vector store")
    parser.add_argument(
        "--papers", type=int, default=500,
        help="Target number of papers to fetch (default: 500)",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to config YAML file",
    )
    args = parser.parse_args()

    pipeline = create_pipeline(args.config)

    print(f"Starting ingestion pipeline (target: {args.papers} papers)...")
    print("This will take several minutes due to arXiv rate limits.\n")

    num_chunks = pipeline.ingest(target_papers=args.papers)

    print(f"\nIngestion complete!")
    print(f"  Indexed chunks: {num_chunks}")
    print(f"\nRun the app with: streamlit run app.py")


if __name__ == "__main__":
    main()
