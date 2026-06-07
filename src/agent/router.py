"""
Agentic RAG System — Agentic Layer
Router: classifies a query to determine the appropriate action before retrieval (Gemini).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from loguru import logger


class RouteAction(str, Enum):
    """Possible routing decisions."""
    RETRIEVE = "retrieve"          # Normal case: retrieve from corpus then generate
    DIRECT_ANSWER = "direct"       # Query is factual/general knowledge, no retrieval needed
    CLARIFY = "clarify"            # Query is too ambiguous to retrieve effectively
    OUT_OF_SCOPE = "out_of_scope"  # Query is clearly unrelated to available domains


@dataclass
class RoutingDecision:
    action: RouteAction
    confidence: float
    reasoning: str
    suggested_domain: Optional[str] = None


ROUTER_PROMPT = """You are a query router for a RAG (Retrieval-Augmented Generation) system.
The system has access to documents from these domains: {domains}

Analyze the following query and decide the best action:
Query: "{query}"

Choose ONE action:
- RETRIEVE: The query needs specific facts from the document corpus. Use for domain-specific questions.
- DIRECT: The query can be answered from general knowledge without document retrieval (e.g., basic math, common facts).
- CLARIFY: The query is too vague or ambiguous to retrieve useful documents.
- OUT_OF_SCOPE: The query is completely unrelated to the available domains.

Respond in this exact format (JSON):
{{
  "action": "<RETRIEVE|DIRECT|CLARIFY|OUT_OF_SCOPE>",
  "confidence": <0.0-1.0>,
  "reasoning": "<one sentence explanation>",
  "domain": "<finance|biomedical|legal|general|null>"
}}"""


class QueryRouter:
    """
    Classifies incoming queries to route them to the appropriate pipeline branch.

    The router runs BEFORE retrieval and enables the agent to:
    - Skip retrieval for simple factual queries (reduces latency/cost).
    - Decline ambiguous queries with a clarification request.
    - Refuse out-of-scope queries without wasting retrieval calls.

    Args:
        model:   Gemini model. Use cheap/fast model — routing is a simple classification.
        domains: List of available domain names (for the prompt).
    """

    def __init__(
        self,
        model: str = "gemini-1.5-flash",
        domains: list[str] | None = None,
    ):
        self.model = model
        self.domains = domains or ["finance", "biomedical", "legal"]
        self.client = None
        self.types = None

    def _get_client(self):
        if self.client is None:
            from google import genai
            self.types = genai.types
            self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        return self.client

    def route(self, query: str) -> RoutingDecision:
        """
        Determine the routing action for a query.

        Args:
            query: User query string.

        Returns:
            RoutingDecision with action, confidence, reasoning, and domain hint.
        """
        import json

        client = self._get_client()
        prompt = ROUTER_PROMPT.format(
            domains=", ".join(self.domains),
            query=query,
        )

        try:
            response = client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=self.types.GenerateContentConfig(
                    max_output_tokens=150,
                    temperature=0.0,
                    response_mime_type="application/json",
                )
            )
            raw = response.text
            data = json.loads(raw)

            action_str = data.get("action", "RETRIEVE").upper()
            action_map = {
                "RETRIEVE": RouteAction.RETRIEVE,
                "DIRECT": RouteAction.DIRECT_ANSWER,
                "CLARIFY": RouteAction.CLARIFY,
                "OUT_OF_SCOPE": RouteAction.OUT_OF_SCOPE,
            }
            action = action_map.get(action_str, RouteAction.RETRIEVE)

            decision = RoutingDecision(
                action=action,
                confidence=float(data.get("confidence", 0.8)),
                reasoning=data.get("reasoning", ""),
                suggested_domain=data.get("domain"),
            )

        except Exception as e:
            logger.warning(f"Router parsing failed ({e}), defaulting to RETRIEVE")
            decision = RoutingDecision(
                action=RouteAction.RETRIEVE,
                confidence=0.5,
                reasoning="Routing failed, defaulting to retrieval.",
            )

        logger.info(
            f"Router: '{query[:60]}' → {decision.action.value} "
            f"(confidence={decision.confidence:.2f}, domain={decision.suggested_domain})"
        )
        return decision
