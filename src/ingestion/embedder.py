"""
Agentic RAG System — Ingestion Layer
Embedder module: abstraction over Gemini and local sentence-transformer embeddings.
"""

from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from typing import List

import numpy as np
from loguru import logger


# ──────────────────────────────────────────────────────────────────────────────
# Base Class
# ──────────────────────────────────────────────────────────────────────────────

class BaseEmbedder(ABC):
    """Abstract base for all embedding providers."""

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of document texts."""
        ...

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query string (may use different prompt/model than docs)."""
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return embedding vector dimension."""
        ...


# ──────────────────────────────────────────────────────────────────────────────
# Gemini Embedder
# ──────────────────────────────────────────────────────────────────────────────

class GeminiEmbedder(BaseEmbedder):
    """
    Embeds text via the Google Gemini embeddings API.

    Args:
        model:      Gemini embedding model name. Default: text-embedding-004.
        batch_size: Number of texts per API call.
        dimensions: Optional output dimension truncation.
    """

    DIMENSION_MAP = {
        "text-embedding-004": 768,
    }

    def __init__(
        self,
        model: str = "text-embedding-004",
        batch_size: int = 100,
        dimensions: int | None = None,
    ):
        try:
            from google import genai
            from google.genai import types
        except ImportError:
            raise ImportError("Run: pip install google-genai")

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError("GEMINI_API_KEY not set. Check your .env file.")

        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.batch_size = batch_size
        self._dimensions = dimensions
        self._dim = dimensions or self.DIMENSION_MAP.get(model, 768)
        self.types = types
        logger.info(f"GeminiEmbedder initialized: model={model}, dim={self._dim}")

    @property
    def dimension(self) -> int:
        return self._dim

    def _call_api(self, texts: List[str], task_type: str) -> List[List[float]]:
        config = self.types.EmbedContentConfig(
            task_type=task_type,
            output_dimensionality=self._dimensions
        )
        response = self.client.models.embed_content(
            model=self.model,
            contents=texts,
            config=config
        )
        return [emb.values for emb in response.embeddings]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Batch embed documents with automatic batching."""
        if not texts:
            return []
        all_embeddings: List[List[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            logger.debug(f"Embedding batch {i // self.batch_size + 1}: {len(batch)} texts")
            embeddings = self._call_api(batch, task_type="RETRIEVAL_DOCUMENT")
            all_embeddings.extend(embeddings)
            # Sleep to respect rate limits if needed on free tier
            time.sleep(1)
        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        return self._call_api([text], task_type="RETRIEVAL_QUERY")[0]


# ──────────────────────────────────────────────────────────────────────────────
# Local Embedder (sentence-transformers)
# ──────────────────────────────────────────────────────────────────────────────

class LocalEmbedder(BaseEmbedder):
    """
    Embeds text using a local HuggingFace sentence-transformer model.
    Useful for zero-cost offline embedding (e.g., BAAI/bge-large-en-v1.5).

    Args:
        model_name: HuggingFace model identifier.
        device:     "cpu", "cuda", or "mps". Auto-detected if None.
        batch_size: Texts per forward pass.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-large-en-v1.5",
        device: str | None = None,
        batch_size: int = 64,
    ):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("Run: pip install sentence-transformers")

        import torch
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(f"Loading local embedder: {model_name} on {device}")
        self.model = SentenceTransformer(model_name, device=device)
        self.batch_size = batch_size
        self._dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"LocalEmbedder ready: model={model_name}, dim={self._dim}")

    @property
    def dimension(self) -> int:
        return self._dim

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=len(texts) > 100,
            normalize_embeddings=True,
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        # BGE models use a query prefix for better retrieval performance
        if "bge" in self.model.model_card_data.model_name.lower():
            text = f"Represent this sentence for searching relevant passages: {text}"
        embedding = self.model.encode(text, normalize_embeddings=True)
        return embedding.tolist()


# ──────────────────────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────────────────────

def get_embedder(provider: str = "gemini", **kwargs) -> BaseEmbedder:
    """
    Factory to instantiate an embedder by provider name.

    Args:
        provider: "gemini" or "local"
        **kwargs: Passed through to the embedder constructor.

    Returns:
        BaseEmbedder instance.
    """
    providers = {
        "gemini": GeminiEmbedder,
        "local": LocalEmbedder,
    }
    # Keep backwards compatibility for scripts that pass "openai" explicitly
    if provider == "openai":
        logger.warning("Provider 'openai' requested but system is configured for gemini. Using gemini.")
        provider = "gemini"
        
    if provider not in providers:
        raise ValueError(f"Unknown embedder provider '{provider}'. Choose from: {list(providers)}")
    return providers[provider](**kwargs)
