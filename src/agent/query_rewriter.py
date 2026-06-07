"""
Agentic RAG System — Agentic Layer
Query rewriter: reformulates a failed query for retry retrieval (Gemini).
This is Step C of the corrective RAG loop.
"""

from __future__ import annotations

import os

from loguru import logger


REWRITE_PROMPT = """You are an expert at improving search queries to find relevant information in a document corpus.

The following query failed to retrieve useful documents:
Original Query: "{query}"

Your task: Rewrite this query to be more effective for document retrieval.
Strategies to use:
1. Use different terminology or synonyms that might appear in the source documents
2. Break down complex questions into their core factual need
3. Add domain-specific keywords that would appear in expert documents
4. Make implicit assumptions explicit

Output ONLY the rewritten query (no explanation, no quotes, no preamble).

Rewritten Query:"""


EXPANSION_PROMPT = """You are helping improve document retrieval by generating query variations.

Original Query: "{query}"

Generate 3 alternative phrasings of this query that would help find relevant documents.
Use different vocabulary, level of specificity, and framing.
Output only the 3 queries, one per line, numbered 1. 2. 3.

Alternatives:"""


class QueryRewriter:
    """
    Rewrites a failed query to improve retrieval on retry.

    Implements two strategies:
    - `rewrite`: Single focused rewrite optimized for document matching.
    - `expand`: Generate multiple query variations (query expansion).

    Args:
        model:       Gemini model for rewriting. Cheap/fast model recommended.
        temperature: 0.3 for slight creativity in rewriting.
    """

    def __init__(self, model: str = "gemini-1.5-flash", temperature: float = 0.3):
        self.model = model
        self.temperature = temperature
        self.client = None
        self.types = None

    def _get_client(self):
        if self.client is None:
            from google import genai
            self.types = genai.types
            self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        return self.client

    def rewrite(self, query: str, context: str = "") -> str:
        """
        Generate a single improved query.

        Args:
            query:   Original query that failed retrieval.
            context: Optional extra context (e.g., domain hint).

        Returns:
            Rewritten query string.
        """
        client = self._get_client()
        prompt = REWRITE_PROMPT.format(query=query)
        if context:
            prompt += f"\n\nDomain context: {context}"

        response = client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=self.types.GenerateContentConfig(
                max_output_tokens=100,
                temperature=self.temperature,
            )
        )
        rewritten = response.text.strip()
        logger.info(f"QueryRewriter: '{query[:60]}' → '{rewritten[:60]}'")
        return rewritten

    def expand(self, query: str) -> list[str]:
        """
        Generate multiple query variations via query expansion.

        Args:
            query: Original query.

        Returns:
            List of alternative query strings.
        """
        client = self._get_client()
        response = client.models.generate_content(
            model=self.model,
            contents=EXPANSION_PROMPT.format(query=query),
            config=self.types.GenerateContentConfig(
                max_output_tokens=200,
                temperature=0.5,
            )
        )
        raw = response.text.strip()

        # Parse numbered lines
        alternatives = []
        for line in raw.split("\n"):
            line = line.strip()
            if line and line[0].isdigit():
                # Remove leading "1. ", "2. " etc.
                parts = line.split(".", 1)
                if len(parts) == 2:
                    alternatives.append(parts[1].strip())

        logger.info(f"QueryRewriter expanded to {len(alternatives)} alternatives")
        return alternatives or [query]  # fallback to original if parse fails
