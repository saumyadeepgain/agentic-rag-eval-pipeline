"""
Agentic RAG System — Retrieval Layer
Reranker module: cross-encoder reranking using local bge-reranker or Cohere API.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import List

from loguru import logger


# ──────────────────────────────────────────────────────────────────────────────
# Base Class
# ──────────────────────────────────────────────────────────────────────────────

class BaseReranker(ABC):
    """Abstract reranker: takes a query and candidate hits, returns re-ordered hits."""

    @abstractmethod
    def rerank(self, query: str, hits: List[dict], top_k: int = 5) -> List[dict]:
        """
        Rerank a list of hits for a given query.

        Args:
            query:  User query.
            hits:   Candidate hits from initial retrieval (each has 'content' key).
            top_k:  Number of top results to return after reranking.

        Returns:
            Re-ordered list of hit dicts, each augmented with 'rerank_score'.
        """
        ...


# ──────────────────────────────────────────────────────────────────────────────
# Local Cross-Encoder (BGE-Reranker)
# ──────────────────────────────────────────────────────────────────────────────

class CrossEncoderReranker(BaseReranker):
    """
    Local cross-encoder reranker using sentence-transformers CrossEncoder.

    The cross-encoder jointly encodes (query, document) pairs and scores
    their relevance — more accurate than bi-encoder similarity but slower
    (O(n) model inferences, not O(1) cosine comparisons).

    Recommended model: BAAI/bge-reranker-v2-m3 (multilingual, strong quality).

    Args:
        model_name: HuggingFace model identifier for the cross-encoder.
        device:     "cpu", "cuda", or "mps". Auto-detected if None.
        batch_size: Pairs per forward pass.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        device: str | None = None,
        batch_size: int = 32,
    ):
        try:
            from sentence_transformers import CrossEncoder
            import torch
        except ImportError:
            raise ImportError("Run: pip install sentence-transformers torch")

        import torch
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(f"Loading CrossEncoderReranker: {model_name} on {device}")
        self.model = CrossEncoder(model_name, device=device)
        self.batch_size = batch_size
        logger.info("CrossEncoderReranker ready.")

    def rerank(self, query: str, hits: List[dict], top_k: int = 5) -> List[dict]:
        if not hits:
            return []

        pairs = [(query, hit["content"]) for hit in hits]
        scores = self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)

        scored_hits = [
            {**hit, "rerank_score": float(score), "retriever": f"{hit.get('retriever','')}_reranked"}
            for hit, score in zip(hits, scores)
        ]
        scored_hits.sort(key=lambda x: x["rerank_score"], reverse=True)
        result = scored_hits[:top_k]

        logger.debug(
            f"CrossEncoderReranker: {len(hits)} → {len(result)} hits, "
            f"top score={result[0]['rerank_score']:.4f}"
        )
        return result


# ──────────────────────────────────────────────────────────────────────────────
# Cohere Reranker (API-based)
# ──────────────────────────────────────────────────────────────────────────────

class CohereReranker(BaseReranker):
    """
    Reranks using the Cohere Rerank API (rerank-english-v3.0).
    Requires COHERE_API_KEY environment variable.

    Higher quality than local bge-reranker but adds network latency + cost.
    Use for benchmarking or if you have Cohere credits.

    Args:
        model: Cohere reranking model name.
    """

    def __init__(self, model: str = "rerank-english-v3.0"):
        try:
            import cohere
        except ImportError:
            raise ImportError("Run: pip install cohere")

        api_key = os.getenv("COHERE_API_KEY")
        if not api_key:
            raise EnvironmentError("COHERE_API_KEY not set.")

        import cohere
        self.client = cohere.Client(api_key)
        self.model = model
        logger.info(f"CohereReranker initialized: model={model}")

    def rerank(self, query: str, hits: List[dict], top_k: int = 5) -> List[dict]:
        if not hits:
            return []

        documents = [hit["content"] for hit in hits]
        response = self.client.rerank(
            query=query,
            documents=documents,
            top_n=top_k,
            model=self.model,
        )

        result = []
        for r in response.results:
            hit = {
                **hits[r.index],
                "rerank_score": r.relevance_score,
                "retriever": f"{hits[r.index].get('retriever','')}_cohere_reranked",
            }
            result.append(hit)

        logger.debug(f"CohereReranker: {len(hits)} → {len(result)} hits")
        return result


# ──────────────────────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────────────────────

def get_reranker(provider: str = "cross_encoder", **kwargs) -> BaseReranker:
    """
    Factory for reranker instances.

    Args:
        provider: "cross_encoder" (local) or "cohere" (API).
        **kwargs: Passed to the reranker constructor.
    """
    providers = {
        "cross_encoder": CrossEncoderReranker,
        "cohere": CohereReranker,
    }
    if provider not in providers:
        raise ValueError(f"Unknown reranker '{provider}'. Choose from: {list(providers)}")
    return providers[provider](**kwargs)
