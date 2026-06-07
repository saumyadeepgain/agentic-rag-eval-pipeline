"""
Agentic RAG System — Ingestion Layer
Indexer module: ChromaDB vector index + BM25 sparse index.
"""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import List

from loguru import logger

from src.ingestion.chunker import Chunk
from src.ingestion.embedder import BaseEmbedder


# ──────────────────────────────────────────────────────────────────────────────
# ChromaDB Vector Indexer
# ──────────────────────────────────────────────────────────────────────────────

class ChromaIndexer:
    """
    Manages a ChromaDB collection for vector search.

    Stores Chunk objects with their embeddings. Supports upsert (idempotent
    re-indexing), deletion, and query. Each collection is domain-scoped.

    Args:
        collection_name: ChromaDB collection identifier (e.g. "finance").
        persist_dir:     Directory where ChromaDB persists data.
        embedder:        BaseEmbedder instance for computing embeddings.
    """

    def __init__(
        self,
        collection_name: str,
        persist_dir: str,
        embedder: BaseEmbedder,
    ):
        try:
            import chromadb
        except ImportError:
            raise ImportError("Run: pip install chromadb")

        self.embedder = embedder
        self.collection_name = collection_name
        Path(persist_dir).mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},   # cosine similarity
        )
        logger.info(
            f"ChromaIndexer: collection='{collection_name}', "
            f"persist_dir='{persist_dir}', "
            f"existing_docs={self.collection.count()}"
        )

    def index_chunks(self, chunks: List[Chunk], batch_size: int = 128) -> None:
        """
        Upsert chunks into ChromaDB with pre-computed embeddings.
        Safe to call repeatedly — will overwrite existing chunks by ID.
        """
        if not chunks:
            return

        texts = [c.content for c in chunks]
        ids = [c.chunk_id for c in chunks]
        metadatas = [
            {**c.metadata, "source_doc_id": c.source_doc_id, "chunk_index": c.chunk_index}
            for c in chunks
        ]

        logger.info(f"Computing embeddings for {len(chunks)} chunks…")
        embeddings = self.embedder.embed_documents(texts)

        # Upsert in batches
        for i in range(0, len(chunks), batch_size):
            batch_slice = slice(i, i + batch_size)
            self.collection.upsert(
                ids=ids[batch_slice],
                documents=texts[batch_slice],
                embeddings=embeddings[batch_slice],
                metadatas=metadatas[batch_slice],
            )
        logger.info(f"Indexed {len(chunks)} chunks into collection '{self.collection_name}'")

    def query(self, query_text: str, top_k: int = 10) -> List[dict]:
        """
        Dense vector search. Returns list of dicts with content + metadata + score.
        """
        query_embedding = self.embedder.embed_query(query_text)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.collection.count()),
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
                "score": 1.0 - dist,   # convert cosine distance → similarity
                "retriever": "dense",
            })
        return hits

    def count(self) -> int:
        return self.collection.count()

    def reset(self) -> None:
        """Delete all documents in the collection (for re-indexing experiments)."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.warning(f"Collection '{self.collection_name}' reset.")


# ──────────────────────────────────────────────────────────────────────────────
# BM25 Sparse Indexer
# ──────────────────────────────────────────────────────────────────────────────

class BM25Indexer:
    """
    Builds and queries a BM25 sparse index using rank_bm25.

    Stores the raw corpus in memory (and on disk for persistence). Used alongside
    ChromaIndexer to enable hybrid retrieval via Reciprocal Rank Fusion.

    Args:
        index_path: Path to a .pkl file where the index is persisted.
    """

    def __init__(self, index_path: str):
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError("Run: pip install rank-bm25")

        self.BM25Okapi = BM25Okapi
        self.index_path = Path(index_path)
        self.index_path.parent.mkdir(parents=True, exist_ok=True)

        self.corpus: List[str] = []
        self.chunk_ids: List[str] = []
        self.metadatas: List[dict] = []
        self.bm25 = None

        if self.index_path.exists():
            self._load()
            logger.info(f"BM25Indexer: loaded {len(self.corpus)} docs from '{index_path}'")
        else:
            logger.info(f"BM25Indexer: new index at '{index_path}'")

    def _tokenize(self, text: str) -> List[str]:
        """Simple whitespace + lowercase tokenization."""
        return text.lower().split()

    def _build(self) -> None:
        tokenized_corpus = [self._tokenize(doc) for doc in self.corpus]
        self.bm25 = self.BM25Okapi(tokenized_corpus)

    def _save(self) -> None:
        with open(self.index_path, "wb") as f:
            pickle.dump({
                "corpus": self.corpus,
                "chunk_ids": self.chunk_ids,
                "metadatas": self.metadatas,
            }, f)

    def _load(self) -> None:
        with open(self.index_path, "rb") as f:
            data = pickle.load(f)
        self.corpus = data["corpus"]
        self.chunk_ids = data["chunk_ids"]
        self.metadatas = data["metadatas"]
        self._build()

    def index_chunks(self, chunks: List[Chunk]) -> None:
        """Add chunks to the BM25 index. Rebuilds the index after adding."""
        for chunk in chunks:
            if chunk.chunk_id not in self.chunk_ids:
                self.corpus.append(chunk.content)
                self.chunk_ids.append(chunk.chunk_id)
                self.metadatas.append({
                    **chunk.metadata,
                    "source_doc_id": chunk.source_doc_id,
                    "chunk_index": chunk.chunk_index,
                })
        self._build()
        self._save()
        logger.info(f"BM25Indexer: total corpus size = {len(self.corpus)} docs")

    def query(self, query_text: str, top_k: int = 10) -> List[dict]:
        """
        BM25 keyword search. Returns ranked list of hits with BM25 scores.
        """
        if not self.bm25:
            logger.warning("BM25 index is empty — no results.")
            return []

        tokenized_query = self._tokenize(query_text)
        scores = self.bm25.get_scores(tokenized_query)

        # Get top_k indices by score
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        hits = []
        for idx in ranked_indices:
            if scores[idx] > 0:   # filter zero-score results
                hits.append({
                    "content": self.corpus[idx],
                    "metadata": self.metadatas[idx],
                    "chunk_id": self.chunk_ids[idx],
                    "score": float(scores[idx]),
                    "retriever": "bm25",
                })
        return hits

    def reset(self) -> None:
        """Clear index."""
        self.corpus = []
        self.chunk_ids = []
        self.metadatas = []
        self.bm25 = None
        if self.index_path.exists():
            self.index_path.unlink()
        logger.warning("BM25 index reset.")
