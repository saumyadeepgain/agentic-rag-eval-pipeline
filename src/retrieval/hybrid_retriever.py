"""
Agentic RAG System — Retrieval Layer
Hybrid retriever: BM25 + dense vector search fused via Reciprocal Rank Fusion (RRF).
"""

from __future__ import annotations

from collections import defaultdict
from typing import List

from loguru import logger

from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever


# ──────────────────────────────────────────────────────────────────────────────
# Reciprocal Rank Fusion
# ──────────────────────────────────────────────────────────────────────────────

def reciprocal_rank_fusion(
    ranked_lists: List[List[dict]],
    k: int = 60,
    weight_per_list: List[float] | None = None,
) -> List[dict]:
    """
    Combine multiple ranked lists via Reciprocal Rank Fusion (RRF).

    RRF score for document d across lists:
        RRF(d) = Σ_i  weight_i / (k + rank_i(d))

    Where rank_i(d) is d's 1-indexed position in list i (or absent → ignored).

    Args:
        ranked_lists:     Each sub-list is an ordered list of hit dicts.
        k:                RRF constant. Higher k = more weight to lower-ranked items.
                          Standard default is 60 (from the original RRF paper).
        weight_per_list:  Optional per-list weight multipliers (must sum to 1 if provided).

    Returns:
        Merged and re-ranked list of hit dicts with an "rrf_score" field.
    """
    if weight_per_list is None:
        weight_per_list = [1.0] * len(ranked_lists)

    rrf_scores: dict[str, float] = defaultdict(float)
    hit_registry: dict[str, dict] = {}   # chunk_id → hit dict

    for list_idx, (hits, weight) in enumerate(zip(ranked_lists, weight_per_list)):
        for rank, hit in enumerate(hits, start=1):
            cid = hit.get("chunk_id") or hit.get("content", "")[:50]
            rrf_scores[cid] += weight / (k + rank)
            if cid not in hit_registry:
                hit_registry[cid] = hit

    # Build final sorted list
    merged = []
    for cid, score in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True):
        entry = {**hit_registry[cid], "rrf_score": score, "retriever": "hybrid"}
        merged.append(entry)

    return merged


# ──────────────────────────────────────────────────────────────────────────────
# Hybrid Retriever
# ──────────────────────────────────────────────────────────────────────────────

class HybridRetriever:
    """
    Combines BM25 sparse retrieval and dense vector retrieval via RRF.

    Why hybrid?
    - Dense retrieval excels at semantic similarity (paraphrases, synonyms).
    - BM25 excels at exact-match, rare terms, and named entities.
    - Combining both via RRF consistently outperforms either alone.

    Args:
        dense_retriever: DenseRetriever instance.
        bm25_retriever:  BM25Retriever instance.
        top_k_each:      How many results to fetch from each retriever pre-fusion.
        top_k_final:     How many results to return after fusion.
        rrf_k:           RRF constant. Default 60.
        bm25_weight:     Weight for BM25 list (0–1). Dense weight = 1 - bm25_weight.
    """

    def __init__(
        self,
        dense_retriever: DenseRetriever,
        bm25_retriever: BM25Retriever,
        top_k_each: int = 20,
        top_k_final: int = 10,
        rrf_k: int = 60,
        bm25_weight: float = 0.4,
    ):
        self.dense = dense_retriever
        self.bm25 = bm25_retriever
        self.top_k_each = top_k_each
        self.top_k_final = top_k_final
        self.rrf_k = rrf_k
        self.bm25_weight = bm25_weight
        self.dense_weight = 1.0 - bm25_weight

    def retrieve(self, query: str, top_k: int | None = None) -> List[dict]:
        """
        Retrieve and fuse results from both retrievers.

        Args:
            query: User query string.
            top_k: Override top_k_final if provided.

        Returns:
            RRF-merged and re-ranked list of hit dicts.
        """
        k_final = top_k or self.top_k_final

        dense_hits = self.dense.retrieve(query, top_k=self.top_k_each)
        bm25_hits = self.bm25.retrieve(query, top_k=self.top_k_each)

        logger.debug(
            f"HybridRetriever: dense={len(dense_hits)} hits, "
            f"bm25={len(bm25_hits)} hits → fusing with RRF(k={self.rrf_k})"
        )

        fused = reciprocal_rank_fusion(
            ranked_lists=[dense_hits, bm25_hits],
            k=self.rrf_k,
            weight_per_list=[self.dense_weight, self.bm25_weight],
        )

        result = fused[:k_final]
        logger.debug(f"HybridRetriever: {len(result)} results after RRF fusion")
        return result
