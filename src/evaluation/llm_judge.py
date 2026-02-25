"""LLM-as-Judge evaluation for rigorous RAG quality assessment.

Uses Claude to evaluate faithfulness, relevance, and completeness
with structured rubrics. This replaces simple lexical overlap with
semantic understanding of answer-context alignment.

This is the gold-standard evaluation approach used in production
RAG systems (see: G-Eval, Prometheus, JudgeLM literature).
"""

import json
import logging
import re
from typing import Any

import anthropic

from src.data.models import RAGResponse

logger = logging.getLogger(__name__)

FAITHFULNESS_PROMPT = """You are evaluating the faithfulness of a RAG system's answer.

Faithfulness measures whether every claim in the answer is supported by the provided source documents.

## Source Documents
{context}

## Question
{question}

## Answer to Evaluate
{answer}

## Task
1. Extract each distinct factual claim from the answer.
2. For each claim, determine if it is SUPPORTED, PARTIALLY SUPPORTED, or NOT SUPPORTED by the sources.
3. Calculate the faithfulness score as: (supported + 0.5 * partially_supported) / total_claims

Respond in this exact JSON format:
{{
    "claims": [
        {{"claim": "...", "verdict": "SUPPORTED|PARTIALLY_SUPPORTED|NOT_SUPPORTED", "evidence": "..."}}
    ],
    "score": <float between 0.0 and 1.0>,
    "reasoning": "..."
}}"""

RELEVANCE_PROMPT = """You are evaluating the relevance of retrieved documents for a RAG system.

## Question
{question}

## Retrieved Documents
{context}

## Task
For each retrieved document, assess how relevant it is to answering the question.
Rate each as HIGHLY_RELEVANT, SOMEWHAT_RELEVANT, or NOT_RELEVANT.

Calculate the relevance score as: (highly * 1.0 + somewhat * 0.5) / total_documents

Respond in this exact JSON format:
{{
    "documents": [
        {{"source": "...", "verdict": "HIGHLY_RELEVANT|SOMEWHAT_RELEVANT|NOT_RELEVANT", "reason": "..."}}
    ],
    "score": <float between 0.0 and 1.0>,
    "reasoning": "..."
}}"""

COMPLETENESS_PROMPT = """You are evaluating the completeness of a RAG system's answer.

## Question
{question}

## Source Documents
{context}

## Answer to Evaluate
{answer}

## Task
Evaluate whether the answer:
1. Addresses all aspects of the question
2. Utilizes relevant information from all applicable sources
3. Provides sufficient depth and detail
4. Includes appropriate caveats or limitations

Score from 0.0 to 1.0 where:
- 1.0 = Comprehensive answer covering all aspects with proper source utilization
- 0.7 = Good answer but misses some relevant information from sources
- 0.4 = Partial answer, addresses the question but lacks depth
- 0.0 = Does not meaningfully answer the question

Respond in this exact JSON format:
{{
    "aspects_covered": ["..."],
    "aspects_missing": ["..."],
    "source_utilization": "...",
    "score": <float between 0.0 and 1.0>,
    "reasoning": "..."
}}"""


class LLMJudge:
    """LLM-based evaluation using Claude as a judge.

    Implements structured rubric evaluation for faithfulness, relevance,
    and completeness. Each metric uses a carefully designed prompt that
    forces claim-level analysis before scoring, reducing bias.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-20250514",
    ) -> None:
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def _call_judge(self, prompt: str) -> dict[str, Any]:
        """Make a judge call and parse JSON response."""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=2048,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )

        text = response.content[0].text

        # Extract JSON from response (handle markdown code blocks)
        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if json_match:
            text = json_match.group(1)

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to find any JSON object in the response
            brace_match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
            if brace_match:
                return json.loads(brace_match.group())
            logger.error("Failed to parse judge response: %s", text[:200])
            return {"score": 0.0, "reasoning": "Parse error"}

    def _format_context(self, response: RAGResponse) -> str:
        """Format source documents for judge prompts."""
        parts = []
        for i, result in enumerate(response.sources, 1):
            meta = result.chunk.metadata
            title = meta.get("title", "Unknown")
            parts.append(f"[Source {i}] {title}\n{result.chunk.text}")
        return "\n\n---\n\n".join(parts)

    def evaluate_faithfulness(self, response: RAGResponse) -> dict[str, Any]:
        """Evaluate answer faithfulness using claim-level analysis."""
        prompt = FAITHFULNESS_PROMPT.format(
            context=self._format_context(response),
            question=response.query,
            answer=response.answer,
        )
        result = self._call_judge(prompt)
        logger.info("LLM Judge faithfulness: %.3f", result.get("score", 0))
        return result

    def evaluate_relevance(self, response: RAGResponse) -> dict[str, Any]:
        """Evaluate retrieval relevance using per-document assessment."""
        prompt = RELEVANCE_PROMPT.format(
            question=response.query,
            context=self._format_context(response),
        )
        result = self._call_judge(prompt)
        logger.info("LLM Judge relevance: %.3f", result.get("score", 0))
        return result

    def evaluate_completeness(self, response: RAGResponse) -> dict[str, Any]:
        """Evaluate answer completeness against available sources."""
        prompt = COMPLETENESS_PROMPT.format(
            question=response.query,
            context=self._format_context(response),
            answer=response.answer,
        )
        result = self._call_judge(prompt)
        logger.info("LLM Judge completeness: %.3f", result.get("score", 0))
        return result

    def evaluate(self, response: RAGResponse) -> dict[str, Any]:
        """Run full LLM-as-judge evaluation.

        Returns:
            Dictionary with scores and detailed analysis for each metric.
        """
        faithfulness = self.evaluate_faithfulness(response)
        relevance = self.evaluate_relevance(response)
        completeness = self.evaluate_completeness(response)

        scores = {
            "llm_faithfulness": faithfulness.get("score", 0.0),
            "llm_relevance": relevance.get("score", 0.0),
            "llm_completeness": completeness.get("score", 0.0),
        }
        scores["llm_overall"] = sum(scores.values()) / len(scores)

        return {
            "scores": scores,
            "details": {
                "faithfulness": faithfulness,
                "relevance": relevance,
                "completeness": completeness,
            },
        }
