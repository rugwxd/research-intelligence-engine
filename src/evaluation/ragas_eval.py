"""RAGAS-based evaluation for comprehensive RAG quality assessment.

Provides a wrapper around the RAGAS library to compute standard
RAG evaluation metrics: faithfulness, answer relevancy, and
context precision/recall.
"""

import logging
from typing import Any

from datasets import Dataset

from src.data.models import RAGResponse

logger = logging.getLogger(__name__)


def _import_ragas():
    """Import RAGAS components with graceful fallback."""
    try:
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )

        return evaluate, {
            "faithfulness": faithfulness,
            "answer_relevancy": answer_relevancy,
            "context_precision": context_precision,
            "context_recall": context_recall,
        }
    except ImportError:
        logger.warning("RAGAS library not available. Install with: pip install ragas")
        return None, {}


class RagasEvaluator:
    """Evaluate RAG responses using the RAGAS framework.

    RAGAS provides reference-free metrics for RAG quality:
    - Faithfulness: Is the answer grounded in the context?
    - Answer Relevancy: Does the answer address the question?
    - Context Precision: Are the retrieved contexts relevant?
    - Context Recall: Did we retrieve all necessary contexts?
    """

    def __init__(self) -> None:
        self._evaluate_fn, self._metrics = _import_ragas()

    @property
    def is_available(self) -> bool:
        return self._evaluate_fn is not None

    def evaluate_single(self, response: RAGResponse, ground_truth: str = "") -> dict[str, float]:
        """Evaluate a single RAG response.

        Args:
            response: The RAG pipeline response to evaluate.
            ground_truth: Optional reference answer for context recall.

        Returns:
            Dictionary of metric names to scores.
        """
        if not self.is_available:
            logger.warning("RAGAS not available, returning empty scores")
            return {}

        contexts = [r.chunk.text for r in response.sources]

        data = {
            "question": [response.query],
            "answer": [response.answer],
            "contexts": [contexts],
        }

        if ground_truth:
            data["ground_truth"] = [ground_truth]

        dataset = Dataset.from_dict(data)

        metrics_to_use = list(self._metrics.values())
        if not ground_truth and "context_recall" in self._metrics:
            # context_recall requires ground truth
            metrics_to_use = [m for k, m in self._metrics.items() if k != "context_recall"]

        try:
            result = self._evaluate_fn(dataset, metrics=metrics_to_use)
            scores = {k: float(v) for k, v in result.items() if isinstance(v, (int, float))}
            logger.info("RAGAS evaluation scores: %s", scores)
            return scores
        except Exception as e:
            logger.error("RAGAS evaluation failed: %s", e)
            return {}

    def evaluate_batch(
        self,
        responses: list[RAGResponse],
        ground_truths: list[str] | None = None,
    ) -> dict[str, Any]:
        """Evaluate a batch of RAG responses.

        Args:
            responses: List of RAG responses.
            ground_truths: Optional list of reference answers.

        Returns:
            Aggregated evaluation results with per-metric averages.
        """
        if not self.is_available:
            return {"error": "RAGAS not available"}

        questions = [r.query for r in responses]
        answers = [r.answer for r in responses]
        contexts = [[c.chunk.text for c in r.sources] for r in responses]

        data = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
        }

        if ground_truths:
            data["ground_truth"] = ground_truths

        dataset = Dataset.from_dict(data)

        metrics_to_use = list(self._metrics.values())
        if not ground_truths and "context_recall" in self._metrics:
            metrics_to_use = [m for k, m in self._metrics.items() if k != "context_recall"]

        try:
            result = self._evaluate_fn(dataset, metrics=metrics_to_use)
            scores = {k: float(v) for k, v in result.items() if isinstance(v, (int, float))}
            logger.info("RAGAS batch evaluation (%d samples): %s", len(responses), scores)
            return scores
        except Exception as e:
            logger.error("RAGAS batch evaluation failed: %s", e)
            return {"error": str(e)}
