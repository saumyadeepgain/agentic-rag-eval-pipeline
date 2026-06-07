"""
Agentic RAG System — Generation Layer
Generator: answer synthesis with citations and self-check hallucination grading (Gemini).
"""

from __future__ import annotations

import os
from typing import List, Optional

from loguru import logger


# ──────────────────────────────────────────────────────────────────────────────
# Generation Prompts
# ──────────────────────────────────────────────────────────────────────────────

GENERATION_PROMPT = """You are a precise question-answering assistant. Answer the question using ONLY the provided context documents.

Instructions:
1. Base your answer exclusively on the provided context. Do not use outside knowledge.
2. For each factual claim in your answer, cite the source using [Doc N] notation where N is the document number.
3. If the context does not contain enough information to answer the question, say "INSUFFICIENT CONTEXT" and explain what is missing.
4. Be concise but complete. Prefer specific facts over vague generalities.

Context Documents:
{context}

Question: {question}

Answer (with citations):"""


SELF_CHECK_PROMPT = """You are a fact-checker verifying that an answer is supported by its source documents.

Answer: {answer}

Source Documents:
{context}

For each claim in the answer:
- Check if it is explicitly supported by the source documents
- Mark it as SUPPORTED or UNSUPPORTED

Then give an overall assessment:
- FAITHFUL: All main claims are supported by the context
- PARTIALLY_FAITHFUL: Some claims are supported, some are not
- UNFAITHFUL: Major claims are not supported or contradict the context

Output format (JSON):
{{
  "assessment": "<FAITHFUL|PARTIALLY_FAITHFUL|UNFAITHFUL>",
  "unsupported_claims": ["<claim1>", "<claim2>"],
  "faithfulness_score": <0.0-1.0>
}}"""


DIRECT_ANSWER_PROMPT = """Answer the following question directly using your general knowledge.
Be concise and accurate.

Question: {question}

Answer:"""


# ──────────────────────────────────────────────────────────────────────────────
# Generator
# ──────────────────────────────────────────────────────────────────────────────

class RAGGenerator:
    """
    Generates answers from retrieved context with citation tracking and
    optional hallucination self-check.

    The self-check implements a mini-faithfulness metric *inside* the
    pipeline (not just as offline eval) — a key talking point:
    "I built the eval metric into the production loop."

    Args:
        model:          LLM for answer generation (gemini-1.5-flash recommended).
        grader_model:   Cheap model for self-check grading (gemini-1.5-flash).
        temperature:    Generation temperature. 0.1 for factual Q&A.
        self_check:     Whether to run the post-generation faithfulness check.
        max_context_docs: Max documents to include in prompt (token control).
        max_context_chars: Max characters per doc (token control).
    """

    def __init__(
        self,
        model: str = "gemini-1.5-flash",
        grader_model: str = "gemini-1.5-flash",
        temperature: float = 0.1,
        self_check: bool = True,
        max_context_docs: int = 5,
        max_context_chars: int = 800,
    ):
        self.model = model
        self.grader_model = grader_model
        self.temperature = temperature
        self.self_check = self_check
        self.max_context_docs = max_context_docs
        self.max_context_chars = max_context_chars
        self.client = None
        self.types = None

    def _get_client(self):
        if self.client is None:
            from google import genai
            self.types = genai.types
            self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        return self.client

    def _format_context(self, hits: List[dict]) -> str:
        """Format retrieved hits as numbered context documents."""
        docs = hits[:self.max_context_docs]
        parts = []
        for i, hit in enumerate(docs, start=1):
            content = hit["content"][:self.max_context_chars]
            source = hit.get("metadata", {}).get("source_doc_id", f"doc_{i}")
            parts.append(f"[Doc {i}] (Source: {source})\n{content}")
        return "\n\n".join(parts)

    def _extract_citations(self, answer: str, hits: List[dict]) -> List[dict]:
        """
        Extract which documents were cited in the answer by [Doc N] markers.
        Returns list of {doc_index, chunk_id, content_snippet}.
        """
        import re
        citations = []
        doc_refs = set(re.findall(r'\[Doc (\d+)\]', answer))
        docs = hits[:self.max_context_docs]

        for ref in sorted(doc_refs):
            idx = int(ref) - 1
            if 0 <= idx < len(docs):
                citations.append({
                    "citation_label": f"[Doc {ref}]",
                    "chunk_id": docs[idx].get("chunk_id", ""),
                    "source_doc_id": docs[idx].get("metadata", {}).get("source_doc_id", ""),
                    "content_snippet": docs[idx]["content"][:200],
                })
        return citations

    def _run_self_check(self, answer: str, context_str: str) -> dict:
        """Run post-generation faithfulness self-check."""
        import json
        client = self._get_client()
        try:
            response = client.models.generate_content(
                model=self.grader_model,
                contents=SELF_CHECK_PROMPT.format(answer=answer, context=context_str),
                config=self.types.GenerateContentConfig(
                    max_output_tokens=300,
                    temperature=0.0,
                    response_mime_type="application/json",
                )
            )
            return json.loads(response.text)
        except Exception as e:
            logger.warning(f"Self-check failed: {e}")
            return {"assessment": "UNKNOWN", "faithfulness_score": None}

    def generate(self, query: str, hits: List[dict]) -> dict:
        """
        Generate an answer from a query and retrieved context.

        Args:
            query: User question.
            hits:  Retrieved and graded hits (relevant docs only).

        Returns:
            Dict with:
                - answer: Generated answer string with [Doc N] citations.
                - citations: List of cited document metadata.
                - self_check: Faithfulness self-check result (if enabled).
                - used_context: Number of context docs used.
                - is_direct: True if answered without context (direct mode).
        """
        client = self._get_client()

        # No context → direct answer mode
        if not hits:
            response = client.models.generate_content(
                model=self.model,
                contents=DIRECT_ANSWER_PROMPT.format(question=query),
                config=self.types.GenerateContentConfig(
                    max_output_tokens=500,
                    temperature=self.temperature,
                )
            )
            answer = response.text.strip()
            return {
                "answer": answer,
                "citations": [],
                "self_check": None,
                "used_context": 0,
                "is_direct": True,
            }

        # Format context
        context_str = self._format_context(hits)

        # Generate answer
        system_instruction = (
            "You are a precise, citation-aware question-answering assistant. "
            "Always ground your answers in the provided context documents."
        )
        
        response = client.models.generate_content(
            model=self.model,
            contents=GENERATION_PROMPT.format(
                context=context_str,
                question=query,
            ),
            config=self.types.GenerateContentConfig(
                system_instruction=system_instruction,
                max_output_tokens=600,
                temperature=self.temperature,
            )
        )
        answer = response.text.strip()

        # Check for INSUFFICIENT CONTEXT signal
        if "INSUFFICIENT CONTEXT" in answer:
            logger.info("Generator flagged INSUFFICIENT CONTEXT")
            citations = []
            self_check_result = None
        else:
            citations = self._extract_citations(answer, hits)
            self_check_result = (
                self._run_self_check(answer, context_str) if self.self_check else None
            )

        logger.debug(
            f"Generated answer ({len(answer)} chars), "
            f"citations={len(citations)}, "
            f"self_check={self_check_result.get('assessment') if self_check_result else 'skipped'}"
        )

        return {
            "answer": answer,
            "citations": citations,
            "self_check": self_check_result,
            "used_context": min(len(hits), self.max_context_docs),
            "is_direct": False,
        }
