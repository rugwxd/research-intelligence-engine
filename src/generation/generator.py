"""Answer generation using Anthropic Claude API.

Generates grounded answers from retrieved context with structured
prompts that encourage faithfulness and source attribution.
"""

import logging
import time

import anthropic

from src.config import GenerationConfig
from src.data.models import RAGResponse, RetrievalResult
from src.retrieval.retriever import Retriever

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a research assistant specializing in AI and machine learning.
Your role is to answer questions using ONLY the provided source documents.

Rules:
1. Base your answer strictly on the provided sources. Do not use external knowledge.
2. Cite sources using [Source N] notation when referencing specific findings.
3. If the sources do not contain enough information, explicitly state what is missing.
4. Provide a structured, comprehensive answer with clear reasoning.
5. When sources disagree, present both perspectives with citations."""

USER_PROMPT_TEMPLATE = """Based on the following research paper excerpts, answer the question.

## Sources
{context}

## Question
{query}

## Instructions
Provide a detailed, well-structured answer grounded in the sources above.
Cite specific sources using [Source N] notation."""


class Generator:
    """Generates answers grounded in retrieved context using Claude."""

    def __init__(self, config: GenerationConfig, api_key: str) -> None:
        self.config = config
        self.client = anthropic.Anthropic(api_key=api_key)

    def generate(self, query: str, context: str) -> str:
        """Generate an answer for a query given retrieved context.

        Args:
            query: The user's question.
            context: Formatted context from the retriever.

        Returns:
            Generated answer string.
        """
        # Guard against excessively long context that may exceed token limits
        max_context_chars = self.config.max_tokens * 3  # rough char-to-token ratio
        if len(context) > max_context_chars:
            logger.warning(
                "Context truncated from %d to %d chars to fit token budget",
                len(context),
                max_context_chars,
            )
            context = context[:max_context_chars] + "\n\n[Context truncated...]"

        user_message = USER_PROMPT_TEMPLATE.format(
            context=context,
            query=query,
        )

        start = time.perf_counter()

        response = self.client.messages.create(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )

        answer = response.content[0].text
        elapsed = (time.perf_counter() - start) * 1000
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens

        logger.info(
            "Generated answer (%.1fms, %d input + %d output tokens): %s...",
            elapsed,
            input_tokens,
            output_tokens,
            answer[:100],
        )

        return answer

    def generate_response(
        self,
        query: str,
        results: list[RetrievalResult],
    ) -> RAGResponse:
        """Run the full generation pipeline: format context, generate, package response.

        Args:
            query: User query.
            results: Retrieved document chunks with scores.

        Returns:
            Complete RAGResponse with answer and sources.
        """
        start = time.perf_counter()

        context = Retriever.format_context(results)
        answer = self.generate(query, context)

        latency = (time.perf_counter() - start) * 1000

        return RAGResponse(
            query=query,
            answer=answer,
            sources=results,
            latency_ms=latency,
        )
