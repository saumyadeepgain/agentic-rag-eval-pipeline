"""
Agentic RAG System — Retrieval Layer
HyDE: Hypothetical Document Embeddings query transformation (Gemini).
"""

from __future__ import annotations

import os
from typing import List

from loguru import logger

from src.ingestion.embedder import BaseEmbedder
from src.ingestion.indexer import ChromaIndexer


class HyDERetriever:
    """
    Hypothetical Document Embeddings (HyDE) retrieval.

    Key idea (from Gao et al. 2022):
    Instead of embedding the raw query, instruct an LLM to generate a
    *hypothetical* answer document, then embed THAT document and use it as
    the query vector. The resulting embedding lives in document-space rather
    than query-space, improving recall for short/ambiguous queries.

    How it works:
        1. Generate a hypothetical answer via LLM (cheap model, ~50-100 tokens).
        2. Embed the hypothetical answer.
        3. Use that embedding to query the vector store.
        4. Optionally average multiple hypothetical embeddings for robustness.

    Args:
        embedder:      Embedding model (same as used for indexing).
        chroma_indexer: The ChromaDB index to query against.
        llm_model:     Model used to generate hypothetical answers.
        n_hypotheticals: Number of hypothetical docs to generate and average.
        top_k:         Documents to return.
    """

    HYDE_PROMPT = """Generate a concise, factual passage that would directly answer the following question.
Write it as if it were an excerpt from a relevant document. Do not mention that it's hypothetical.
Keep it under 100 words and focus on the core facts.

Question: {query}

Passage:"""

    def __init__(
        self,
        embedder: BaseEmbedder,
        chroma_indexer: ChromaIndexer,
        llm_model: str = "gemini-1.5-flash",
        n_hypotheticals: int = 1,
        top_k: int = 10,
    ):
        try:
            from google import genai
        except ImportError:
            raise ImportError("Run: pip install google-genai")

        self.embedder = embedder
        self.indexer = chroma_indexer
        self.llm_model = llm_model
        self.n_hypotheticals = n_hypotheticals
        self.top_k = top_k
        self.client = None   # lazy init
        self.types = None

    def _get_client(self):
        if self.client is None:
            from google import genai
            self.types = genai.types
            self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        return self.client

    def _generate_hypothetical(self, query: str) -> str:
        """Generate a single hypothetical answer document."""
        client = self._get_client()
        response = client.models.generate_content(
            model=self.llm_model,
            contents=self.HYDE_PROMPT.format(query=query),
            config=self.types.GenerateContentConfig(
                max_output_tokens=150,
                temperature=0.7,
            )
        )
        return response.text.strip()

    def retrieve(self, query: str, top_k: int | None = None) -> List[dict]:
        """
        Perform HyDE retrieval.

        Args:
            query: Original user query.
            top_k: Override default top_k.

        Returns:
            List of hit dicts from vector search on hypothetical embedding.
        """
        import numpy as np

        k = top_k or self.top_k

        # Step 1: Generate hypothetical document(s)
        hypotheticals = [self._generate_hypothetical(query) for _ in range(self.n_hypotheticals)]
        logger.debug(f"HyDE: generated {len(hypotheticals)} hypothetical doc(s)")

        # Step 2: Embed hypothetical(s)
        embeddings = [self.embedder.embed_query(h) for h in hypotheticals]

        # Step 3: Average embeddings if multiple hypotheticals
        avg_embedding = list(np.mean(embeddings, axis=0))

        # Step 4: Query ChromaDB with hypothetical embedding directly
        results = self.indexer.collection.query(
            query_embeddings=[avg_embedding],
            n_results=min(k, self.indexer.count()),
            include=["documents", "metadatas", "distances"],
        )

        hits = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            hits.append({
                "content": doc,
                "metadata": meta,
                "chunk_id": meta.get("source_doc_id", "") + "::" + str(meta.get("chunk_index", "")),
                "score": 1.0 - dist,
                "retriever": "hyde",
                "hypothetical_docs": hypotheticals,
            })

        logger.debug(f"HyDE: '{query[:60]}…' → {len(hits)} hits via hypothetical embedding")
        return hits
