"""Data ingestion script.

Fetches papers from arXiv, generates embeddings, and builds the FAISS index.
Run this before starting the Streamlit app.

Usage:
    python scripts/ingest.py [--papers 500] [--extract-pdfs] [--mode hybrid]
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
    parser.add_argument(
        "--extract-pdfs", action="store_true",
        help="Download and extract full PDF text (slower, higher quality)",
    )
    parser.add_argument(
        "--max-pdfs", type=int, default=50,
        help="Max number of PDFs to extract (default: 50)",
    )
    parser.add_argument(
        "--mode", type=str, default="hybrid",
        choices=["dense", "hybrid", "full"],
        help="Retrieval mode to configure (default: hybrid)",
    )
    parser.add_argument(
        "--chunking", type=str, default="sliding_window",
        choices=["sliding_window", "semantic"],
        help="Chunking strategy (default: sliding_window)",
    )
    args = parser.parse_args()

    pipeline = create_pipeline(
        args.config,
        retrieval_mode=args.mode,
        chunking_strategy=args.chunking,
    )

    print(f"Starting ingestion pipeline (target: {args.papers} papers)...")
    print(f"  Retrieval mode: {args.mode}")
    print(f"  Chunking: {args.chunking}")
    print(f"  PDF extraction: {args.extract_pdfs}")
    print("This will take several minutes due to arXiv rate limits.\n")

    num_chunks = pipeline.ingest(
        target_papers=args.papers,
        extract_pdfs=args.extract_pdfs,
        max_pdfs=args.max_pdfs,
    )

    print(f"\nIngestion complete!")
    print(f"  Indexed chunks: {num_chunks}")
    print(f"\nRun the app with: streamlit run app.py")


if __name__ == "__main__":
    main()
