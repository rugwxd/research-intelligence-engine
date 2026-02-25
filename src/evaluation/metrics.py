"""RAG evaluation metrics: faithfulness, relevance, and completeness.

Provides both lightweight heuristic scoring for real-time feedback
and RAGAS-based evaluation for comprehensive benchmarking.
"""

import logging
import re
from collections import Counter

import numpy as np

from src.data.models import RAGResponse

logger = logging.getLogger(__name__)


class HeuristicEvaluator:
    """Fast heuristic evaluation metrics for real-time scoring.

    These metrics provide approximate quality signals without
    requiring additional LLM calls, suitable for live UI feedback.
    """

    def faithfulness_score(self, response: RAGResponse) -> float:
        """Estimate how well the answer sticks to the retrieved context.

        Measures the fraction of substantive answer sentences that
        contain references to source material (via citation markers
        or lexical overlap with source text).

        Returns:
            Score between 0.0 and 1.0.
        """
        answer = response.answer
        source_texts = [r.chunk.text.lower() for r in response.sources]

        sentences = self._split_sentences(answer)
        if not sentences:
            return 0.0

        grounded_count = 0
        for sentence in sentences:
            # Check for explicit citations
            if re.search(r"\[Source\s*\d+\]", sentence):
                grounded_count += 1
                continue

            # Check for lexical overlap with sources
            sentence_words = set(self._extract_content_words(sentence.lower()))
            if not sentence_words:
                grounded_count += 1  # Skip empty/trivial sentences
                continue

            max_overlap = 0.0
            for source_text in source_texts:
                source_words = set(self._extract_content_words(source_text))
                if source_words:
                    overlap = len(sentence_words & source_words) / len(sentence_words)
                    max_overlap = max(max_overlap, overlap)

            if max_overlap >= 0.3:
                grounded_count += 1

        return grounded_count / len(sentences)

    def relevance_score(self, response: RAGResponse) -> float:
        """Estimate how relevant the retrieved chunks are to the query.

        Measures lexical overlap between the query and retrieved chunks.

        Returns:
            Score between 0.0 and 1.0.
        """
        query_words = set(self._extract_content_words(response.query.lower()))
        if not query_words:
            return 0.0

        scores = []
        for result in response.sources:
            chunk_words = set(self._extract_content_words(result.chunk.text.lower()))
            if chunk_words:
                overlap = len(query_words & chunk_words) / len(query_words)
                scores.append(overlap)

        if not scores:
            return 0.0

        # Weighted average: top results matter more
        weights = [1.0 / (i + 1) for i in range(len(scores))]
        weighted_score = sum(s * w for s, w in zip(scores, weights)) / sum(weights)

        return min(1.0, weighted_score)

    def completeness_score(self, response: RAGResponse) -> float:
        """Estimate answer completeness relative to available sources.

        Checks whether the answer references a reasonable fraction
        of the retrieved sources and covers key topics.

        Returns:
            Score between 0.0 and 1.0.
        """
        answer_lower = response.answer.lower()

        # Check how many sources are referenced
        cited_sources = set(int(m) for m in re.findall(r"\[Source\s*(\d+)\]", response.answer))
        source_coverage = len(cited_sources) / max(len(response.sources), 1)

        # Check topic coverage: what fraction of source key terms appear in answer
        all_source_terms: Counter[str] = Counter()
        for result in response.sources:
            terms = self._extract_content_words(result.chunk.text.lower())
            all_source_terms.update(terms)

        top_terms = [term for term, _ in all_source_terms.most_common(20)]
        if top_terms:
            term_coverage = sum(1 for t in top_terms if t in answer_lower) / len(top_terms)
        else:
            term_coverage = 0.0

        # Answer length factor (penalize very short answers)
        word_count = len(response.answer.split())
        length_factor = min(1.0, word_count / 100)

        score = 0.4 * source_coverage + 0.35 * term_coverage + 0.25 * length_factor
        return min(1.0, score)

    def evaluate(self, response: RAGResponse) -> dict[str, float]:
        """Run all heuristic metrics and return scores."""
        scores = {
            "faithfulness": self.faithfulness_score(response),
            "relevance": self.relevance_score(response),
            "completeness": self.completeness_score(response),
        }

        scores["overall"] = np.mean(list(scores.values())).item()

        logger.info(
            "Heuristic eval scores - faithfulness: %.3f, relevance: %.3f, "
            "completeness: %.3f, overall: %.3f",
            scores["faithfulness"],
            scores["relevance"],
            scores["completeness"],
            scores["overall"],
        )

        return scores

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Split text into sentences."""
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if len(s.strip()) > 10]

    @staticmethod
    def _extract_content_words(text: str) -> list[str]:
        """Extract meaningful content words, filtering stop words."""
        stop_words = {
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "can",
            "shall",
            "to",
            "of",
            "in",
            "for",
            "on",
            "with",
            "at",
            "by",
            "from",
            "as",
            "into",
            "through",
            "during",
            "before",
            "after",
            "above",
            "below",
            "between",
            "and",
            "but",
            "or",
            "not",
            "no",
            "nor",
            "so",
            "yet",
            "both",
            "either",
            "neither",
            "each",
            "every",
            "all",
            "any",
            "few",
            "more",
            "most",
            "other",
            "some",
            "such",
            "than",
            "too",
            "very",
            "just",
            "also",
            "that",
            "this",
            "these",
            "those",
            "it",
            "its",
            "they",
            "them",
            "their",
            "we",
            "our",
            "you",
            "your",
            "he",
            "she",
            "his",
            "her",
            "which",
            "what",
            "who",
            "whom",
            "how",
            "when",
            "where",
            "why",
            "if",
            "then",
            "else",
            "about",
            "up",
            "out",
            "over",
        }
        words = re.findall(r"\b[a-z]{3,}\b", text.lower())
        return [w for w in words if w not in stop_words]
