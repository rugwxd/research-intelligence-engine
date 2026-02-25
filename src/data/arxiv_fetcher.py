"""ArXiv API client for fetching AI/ML research papers.

Uses the arXiv Atom feed API to retrieve paper metadata and abstracts
across multiple research topics with built-in rate limiting and deduplication.
"""

import json
import logging
import re
import time
from datetime import datetime
from pathlib import Path

import feedparser
import requests
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

from src.config import ArxivConfig, PROJECT_ROOT
from src.data.models import Paper

logger = logging.getLogger(__name__)

ARXIV_API_BASE = "http://export.arxiv.org/api/query"


class ArxivFetcher:
    """Fetches research papers from the arXiv API with rate limiting and retry logic."""

    def __init__(self, config: ArxivConfig) -> None:
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "ResearchIntelligenceEngine/1.0"
        })

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
    )
    def _fetch_page(self, query: str, start: int, max_results: int) -> list[Paper]:
        """Fetch a single page of results from arXiv API."""
        params = {
            "search_query": f"all:{query}",
            "start": start,
            "max_results": max_results,
            "sortBy": "relevance",
            "sortOrder": "descending",
        }

        response = self.session.get(ARXIV_API_BASE, params=params, timeout=30)
        response.raise_for_status()

        feed = feedparser.parse(response.text)
        papers = []

        for entry in feed.entries:
            try:
                paper = self._parse_entry(entry)
                papers.append(paper)
            except (KeyError, ValueError) as e:
                logger.warning("Failed to parse entry %s: %s", entry.get("id", "unknown"), e)

        return papers

    def _parse_entry(self, entry: dict) -> Paper:
        """Parse a single arXiv feed entry into a Paper object."""
        arxiv_id = entry["id"].split("/abs/")[-1]
        # Clean version suffix for consistent IDs
        arxiv_id = re.sub(r"v\d+$", "", arxiv_id)

        title = re.sub(r"\s+", " ", entry["title"]).strip()
        abstract = re.sub(r"\s+", " ", entry["summary"]).strip()

        authors = [a.get("name", "") for a in entry.get("authors", [])]

        categories = [t["term"] for t in entry.get("tags", [])]
        primary_category = entry.get("arxiv_primary_category", {}).get("term", "")

        published = datetime(*entry["published_parsed"][:6])
        updated = datetime(*entry["updated_parsed"][:6])

        pdf_url = ""
        for link in entry.get("links", []):
            if link.get("type") == "application/pdf":
                pdf_url = link["href"]
                break

        return Paper(
            arxiv_id=arxiv_id,
            title=title,
            abstract=abstract,
            authors=authors,
            categories=categories,
            published=published,
            updated=updated,
            pdf_url=pdf_url,
            primary_category=primary_category,
        )

    def fetch_papers(self, target_count: int = 500) -> list[Paper]:
        """Fetch papers across all configured queries, deduplicated by arXiv ID.

        Args:
            target_count: Minimum number of unique papers to collect.

        Returns:
            List of unique Paper objects.
        """
        seen_ids: set[str] = set()
        all_papers: list[Paper] = []
        per_query = self.config.max_results_per_query

        logger.info(
            "Starting arXiv fetch: %d queries, up to %d per query",
            len(self.config.queries),
            per_query,
        )

        for query in tqdm(self.config.queries, desc="Fetching queries"):
            page_size = 100
            fetched_for_query = 0

            for start in range(0, per_query, page_size):
                batch_size = min(page_size, per_query - start)
                papers = self._fetch_page(query, start, batch_size)

                if not papers:
                    logger.info("No more results for query '%s' at offset %d", query, start)
                    break

                for paper in papers:
                    if paper.arxiv_id not in seen_ids:
                        seen_ids.add(paper.arxiv_id)
                        all_papers.append(paper)
                        fetched_for_query += 1

                logger.debug(
                    "Query '%s': fetched %d papers (offset %d)",
                    query, fetched_for_query, start,
                )

                # Respect arXiv rate limits
                time.sleep(self.config.rate_limit_seconds)

            logger.info(
                "Query '%s': %d unique papers collected", query, fetched_for_query
            )

        logger.info("Total unique papers fetched: %d", len(all_papers))
        return all_papers

    def save_papers(self, papers: list[Paper], filename: str = "papers.json") -> Path:
        """Persist fetched papers to JSON."""
        output_dir = PROJECT_ROOT / self.config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / filename

        data = [p.to_dict() for p in papers]
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

        logger.info("Saved %d papers to %s", len(papers), output_path)
        return output_path

    @staticmethod
    def load_papers(filepath: str | Path) -> list[Paper]:
        """Load papers from a previously saved JSON file."""
        with open(filepath) as f:
            data = json.load(f)

        papers = [Paper.from_dict(d) for d in data]
        logger.info("Loaded %d papers from %s", len(papers), filepath)
        return papers
