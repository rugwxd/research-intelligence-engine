"""PDF text extraction for full paper processing.

Downloads and extracts text from arXiv paper PDFs, providing
full document content beyond just abstracts for deeper retrieval.
Includes section detection and cleaning for academic paper formatting.
"""

import logging
import re
import tempfile
import time
from pathlib import Path

import requests
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

from src.data.models import Paper

logger = logging.getLogger(__name__)


class PDFExtractor:
    """Downloads and extracts text from arXiv PDF papers.

    Handles rate limiting, retries, and text cleaning specific
    to academic paper formatting (removing headers, footers,
    references sections, and LaTeX artifacts).
    """

    def __init__(self, rate_limit: float = 5.0, max_papers: int | None = None) -> None:
        self.rate_limit = rate_limit
        self.max_papers = max_papers
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "ResearchIntelligenceEngine/1.0"
        })

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=3, max=30),
    )
    def _download_pdf(self, url: str) -> bytes:
        """Download a PDF from the given URL."""
        response = self.session.get(url, timeout=60)
        response.raise_for_status()
        return response.content

    def _extract_text_from_bytes(self, pdf_bytes: bytes) -> str:
        """Extract text from PDF bytes using PyMuPDF or fallback."""
        try:
            import fitz  # PyMuPDF

            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            text_parts = []
            for page in doc:
                text_parts.append(page.get_text())
            doc.close()
            return "\n".join(text_parts)
        except ImportError:
            logger.warning("PyMuPDF not installed. Install with: pip install pymupdf")
            return ""
        except Exception as e:
            logger.error("PDF extraction failed: %s", e)
            return ""

    def _clean_academic_text(self, text: str) -> str:
        """Clean extracted text from academic paper formatting.

        Removes:
            - Page numbers and headers/footers
            - LaTeX artifacts and commands
            - References section (usually noise for retrieval)
            - Excessive whitespace and line breaks
        """
        # Remove common LaTeX artifacts
        text = re.sub(r"\\[a-zA-Z]+\{[^}]*\}", "", text)
        text = re.sub(r"\\[a-zA-Z]+", "", text)
        text = re.sub(r"\$[^$]+\$", "[equation]", text)

        # Remove page numbers (standalone numbers on their own line)
        text = re.sub(r"\n\s*\d+\s*\n", "\n", text)

        # Truncate at references section
        ref_patterns = [
            r"\n\s*References\s*\n",
            r"\n\s*REFERENCES\s*\n",
            r"\n\s*Bibliography\s*\n",
        ]
        for pattern in ref_patterns:
            match = re.search(pattern, text)
            if match:
                text = text[:match.start()]
                break

        # Normalize whitespace
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r" {2,}", " ", text)

        return text.strip()

    def extract_paper_text(self, paper: Paper) -> str:
        """Download and extract full text from a paper's PDF.

        Args:
            paper: Paper object with a pdf_url field.

        Returns:
            Cleaned full text string, or empty string on failure.
        """
        if not paper.pdf_url:
            logger.debug("No PDF URL for paper %s", paper.arxiv_id)
            return ""

        try:
            pdf_bytes = self._download_pdf(paper.pdf_url)
            raw_text = self._extract_text_from_bytes(pdf_bytes)
            cleaned = self._clean_academic_text(raw_text)

            logger.debug(
                "Extracted %d chars from %s (raw: %d)",
                len(cleaned), paper.arxiv_id, len(raw_text),
            )
            return cleaned
        except Exception as e:
            logger.warning("Failed to extract PDF for %s: %s", paper.arxiv_id, e)
            return ""

    def extract_batch(
        self,
        papers: list[Paper],
        output_dir: Path | None = None,
    ) -> dict[str, str]:
        """Extract full text from multiple papers.

        Args:
            papers: List of papers to process.
            output_dir: Optional directory to cache extracted text.

        Returns:
            Dictionary mapping arxiv_id -> extracted text.
        """
        papers_to_process = papers
        if self.max_papers:
            papers_to_process = papers[:self.max_papers]

        results = {}
        succeeded = 0
        failed = 0

        for paper in tqdm(papers_to_process, desc="Extracting PDFs"):
            # Check cache
            if output_dir:
                cache_path = output_dir / f"{paper.arxiv_id.replace('/', '_')}.txt"
                if cache_path.exists():
                    results[paper.arxiv_id] = cache_path.read_text()
                    succeeded += 1
                    continue

            text = self.extract_paper_text(paper)
            if text:
                results[paper.arxiv_id] = text
                succeeded += 1

                # Cache to disk
                if output_dir:
                    output_dir.mkdir(parents=True, exist_ok=True)
                    cache_path = output_dir / f"{paper.arxiv_id.replace('/', '_')}.txt"
                    cache_path.write_text(text)
            else:
                failed += 1

            time.sleep(self.rate_limit)

        logger.info(
            "PDF extraction complete: %d succeeded, %d failed out of %d",
            succeeded, failed, len(papers_to_process),
        )
        return results
