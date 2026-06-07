"""
Agentic RAG System — Retrieval Layer
Dense retriever: pure vector similarity search via ChromaDB.
"""

from __future__ import annotations

from typing import List

from loguru import logger

from src.ingestion.indexer import ChromaIndexer


class DenseRetriever:
    """
    Retrieves documents using pure dense vector similarity search.

    This is the naive baseline retriever. All other retrieval strategies
    extend or replace this.

    Args:
        chroma_indexer: An initialized ChromaIndexer instance.
        top_k:          Number of top documents to return.
    """

    def __init__(self, chroma_indexer: ChromaIndexer, top_k: int = 10):
        self.indexer = chroma_indexer
        self.top_k = top_k

    def retrieve(self, query: str, top_k: int | None = None) -> List[dict]:
        """
        Perform vector similarity search.

        Args:
            query: The user query string.
            top_k: Override default top_k if provided.

        Returns:
            List of hit dicts with keys: content, metadata, chunk_id, score, retriever.
        """
        k = top_k or self.top_k
        hits = self.indexer.query(query_text=query, top_k=k)
        logger.debug(f"DenseRetriever: '{query[:60]}…' → {len(hits)} hits")
        return hits
