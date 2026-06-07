"""
Agentic RAG System — Agentic Layer
Document grader: scores each retrieved chunk as relevant or irrelevant to the query (Gemini).
This is Step A of the corrective RAG loop.
"""

from __future__ import annotations

import os
from typing import List, Tuple

from loguru import logger


GRADER_PROMPT = """You are a relevance grader evaluating whether a retrieved document is useful for answering a question.

Question: {query}
Retrieved Document: {document}

Instructions:
- Answer ONLY with "yes" or "no"
- "yes" = the document contains information that is directly useful for answering the question
- "no" = the document is off-topic, too vague, or does not help answer the question
- Do not explain your reasoning

Answer:"""


class DocumentGrader:
    """
    LLM-based relevance grader for retrieved documents.

    Uses a cheap/fast model (gemini-1.5-flash) to classify each retrieved chunk
    as relevant or irrelevant before passing to the generator. This:
    1. Filters noise from retrieved context → fewer hallucinations.
    2. Provides a signal for the corrective loop to trigger query rewriting.
    3. Acts as a first line of defense against injected malicious chunks.

    Args:
        model:       Gemini model for grading. Use a cheap/fast model.
        temperature: 0 for deterministic grading.
    """

    def __init__(self, model: str = "gemini-1.5-flash", temperature: float = 0.0):
        try:
            from google import genai
        except ImportError:
            raise ImportError("Run: pip install google-genai")

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

    def grade_document(self, query: str, document: str) -> Tuple[bool, str]:
        """
        Grade a single document for relevance to the query.

        Args:
            query:    The user query.
            document: The retrieved chunk content.

        Returns:
            Tuple of (is_relevant: bool, raw_response: str).
        """
        client = self._get_client()
        response = client.models.generate_content(
            model=self.model,
            contents=GRADER_PROMPT.format(query=query, document=document[:1000]),
            config=self.types.GenerateContentConfig(
                max_output_tokens=5,
                temperature=self.temperature,
            )
        )
        raw = response.text.strip().lower()
        is_relevant = raw.startswith("yes")
        return is_relevant, raw

    def grade_documents(
        self,
        query: str,
        hits: List[dict],
    ) -> Tuple[List[dict], List[dict], dict]:
        """
        Grade all retrieved documents.

        Args:
            query: User query.
            hits:  List of retrieved hit dicts.

        Returns:
            Tuple of:
                - relevant_hits: List of hits graded as relevant.
                - irrelevant_hits: List of hits graded as irrelevant.
                - grading_summary: Dict with counts and relevance_ratio.
        """
        relevant: List[dict] = []
        irrelevant: List[dict] = []

        for hit in hits:
            is_rel, raw = self.grade_document(query, hit["content"])
            annotated = {**hit, "graded_relevant": is_rel, "grade_raw": raw}
            if is_rel:
                relevant.append(annotated)
            else:
                irrelevant.append(annotated)

        total = len(hits)
        summary = {
            "total": total,
            "relevant": len(relevant),
            "irrelevant": len(irrelevant),
            "relevance_ratio": len(relevant) / total if total > 0 else 0.0,
        }

        logger.info(
            f"Grading: {summary['relevant']}/{summary['total']} relevant "
            f"({summary['relevance_ratio']:.0%}) for query: '{query[:60]}…'"
        )
        return relevant, irrelevant, summary
