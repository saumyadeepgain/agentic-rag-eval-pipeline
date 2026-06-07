"""
Agentic RAG System — Ingestion Layer
Package init: exposes top-level ingestion API.
"""

from src.ingestion.chunker import Chunk, FixedSizeChunker, RecursiveChunker, get_chunker
from src.ingestion.embedder import BaseEmbedder, GeminiEmbedder, LocalEmbedder, get_embedder
from src.ingestion.indexer import ChromaIndexer, BM25Indexer

__all__ = [
    "Chunk",
    "FixedSizeChunker",
    "RecursiveChunker",
    "get_chunker",
    "BaseEmbedder",
    "GeminiEmbedder",
    "LocalEmbedder",
    "get_embedder",
    "ChromaIndexer",
    "BM25Indexer",
]
