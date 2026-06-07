"""
Agentic RAG System — Retrieval Layer
BM25 sparse retriever.
"""

from __future__ import annotations

from typing import List

from loguru import logger

from src.ingestion.indexer import BM25Indexer


class BM25Retriever:
    """
    Keyword-based sparse retrieval using BM25Okapi.

    BM25 excels at exact-match and rare-term queries where dense
    embeddings may underperform. Combined with dense retrieval via
    RRF in the HybridRetriever.

    Args:
        bm25_indexer: An initialized BM25Indexer instance.
        top_k:        Number of results to return.
    """

    def __init__(self, bm25_indexer: BM25Indexer, top_k: int = 10):
        self.indexer = bm25_indexer
        self.top_k = top_k

    def retrieve(self, query: str, top_k: int | None = None) -> List[dict]:
        """
        Perform BM25 keyword search.

        Args:
            query: The user query string.
            top_k: Override default top_k if provided.

        Returns:
            List of hit dicts sorted by BM25 score (descending).
        """
        k = top_k or self.top_k
        hits = self.indexer.query(query_text=query, top_k=k)
        logger.debug(f"BM25Retriever: '{query[:60]}…' → {len(hits)} hits")
        return hits
