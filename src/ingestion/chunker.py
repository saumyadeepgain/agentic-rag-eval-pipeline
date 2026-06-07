"""
Agentic RAG System — Ingestion Layer
Chunker module: Fixed-size and Recursive chunking strategies.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List

from loguru import logger


# ──────────────────────────────────────────────────────────────────────────────
# Data Model
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Chunk:
    """A single text chunk with metadata."""
    content: str
    chunk_id: str
    source_doc_id: str
    chunk_index: int
    metadata: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.content)


# ──────────────────────────────────────────────────────────────────────────────
# Base Class
# ──────────────────────────────────────────────────────────────────────────────

class BaseChunker(ABC):
    """Abstract base for all chunking strategies."""

    @abstractmethod
    def chunk(self, text: str, doc_id: str, metadata: dict | None = None) -> List[Chunk]:
        """Split text into a list of Chunk objects."""
        ...

    def _make_chunk_id(self, doc_id: str, index: int) -> str:
        return f"{doc_id}::chunk_{index:04d}"


# ──────────────────────────────────────────────────────────────────────────────
# Strategy 1: Fixed-Size Chunker (baseline)
# ──────────────────────────────────────────────────────────────────────────────

class FixedSizeChunker(BaseChunker):
    """
    Splits text into fixed character-length windows with configurable overlap.
    Simple, deterministic — used as the naive baseline.

    Args:
        chunk_size:    Maximum characters per chunk.
        chunk_overlap: Number of overlapping characters between adjacent chunks.
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, text: str, doc_id: str, metadata: dict | None = None) -> List[Chunk]:
        metadata = metadata or {}
        chunks: List[Chunk] = []
        start = 0
        index = 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            content = text[start:end].strip()
            if content:
                chunks.append(Chunk(
                    content=content,
                    chunk_id=self._make_chunk_id(doc_id, index),
                    source_doc_id=doc_id,
                    chunk_index=index,
                    metadata={**metadata, "chunker": "fixed_size",
                               "chunk_size": self.chunk_size,
                               "chunk_overlap": self.chunk_overlap},
                ))
                index += 1
            start += self.chunk_size - self.chunk_overlap

        logger.debug(f"FixedSizeChunker: doc '{doc_id}' → {len(chunks)} chunks")
        return chunks


# ──────────────────────────────────────────────────────────────────────────────
# Strategy 2: Recursive / Semantic Chunker (improved)
# ──────────────────────────────────────────────────────────────────────────────

class RecursiveChunker(BaseChunker):
    """
    Recursively splits text using a priority-ordered list of separators.
    Tries to keep semantically coherent units together (paragraphs > sentences > words).
    This is the LangChain RecursiveCharacterTextSplitter approach, reimplemented
    without the LangChain dependency so we can instrument every step.

    Args:
        chunk_size:    Target max characters per chunk.
        chunk_overlap: Characters to carry over between chunks.
        separators:    Ordered list of separator strings to try.
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        separators: List[str] | None = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or self.DEFAULT_SEPARATORS

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """Recursively split text using separator hierarchy."""
        final_chunks: List[str] = []
        separator = separators[-1]  # fallback: character-level

        for sep in separators:
            if sep == "":
                separator = sep
                break
            if sep in text:
                separator = sep
                break

        splits = text.split(separator) if separator else list(text)
        good_splits: List[str] = []

        for split in splits:
            if len(split) <= self.chunk_size:
                good_splits.append(split)
            else:
                # This split is still too large — recurse with next separators
                if good_splits:
                    final_chunks.extend(self._merge_splits(good_splits, separator))
                    good_splits = []
                remaining_seps = separators[separators.index(separator) + 1:] if separator in separators else [""]
                final_chunks.extend(self._split_text(split, remaining_seps or [""]))

        if good_splits:
            final_chunks.extend(self._merge_splits(good_splits, separator))

        return final_chunks

    def _merge_splits(self, splits: List[str], separator: str) -> List[str]:
        """Merge small splits into chunks up to chunk_size, with overlap."""
        merged: List[str] = []
        current_parts: List[str] = []
        current_len = 0
        sep_len = len(separator)

        for split in splits:
            split_len = len(split)
            if current_len + sep_len + split_len > self.chunk_size and current_parts:
                chunk_text = separator.join(current_parts).strip()
                if chunk_text:
                    merged.append(chunk_text)
                # Keep overlap: remove from the front until we're within overlap budget
                while current_parts and current_len > self.chunk_overlap:
                    removed = current_parts.pop(0)
                    current_len -= len(removed) + sep_len
            current_parts.append(split)
            current_len += split_len + sep_len

        if current_parts:
            chunk_text = separator.join(current_parts).strip()
            if chunk_text:
                merged.append(chunk_text)

        return merged

    def chunk(self, text: str, doc_id: str, metadata: dict | None = None) -> List[Chunk]:
        metadata = metadata or {}
        raw_splits = self._split_text(text, self.separators)
        chunks: List[Chunk] = []

        for index, content in enumerate(raw_splits):
            content = content.strip()
            if content:
                chunks.append(Chunk(
                    content=content,
                    chunk_id=self._make_chunk_id(doc_id, index),
                    source_doc_id=doc_id,
                    chunk_index=index,
                    metadata={**metadata, "chunker": "recursive",
                               "chunk_size": self.chunk_size,
                               "chunk_overlap": self.chunk_overlap},
                ))

        logger.debug(f"RecursiveChunker: doc '{doc_id}' → {len(chunks)} chunks")
        return chunks


# ──────────────────────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────────────────────

def get_chunker(strategy: str, chunk_size: int = 512, chunk_overlap: int = 64) -> BaseChunker:
    """
    Factory function to instantiate a chunker by strategy name.

    Args:
        strategy:      "fixed" or "recursive"
        chunk_size:    Max characters per chunk.
        chunk_overlap: Overlap between adjacent chunks.

    Returns:
        A BaseChunker subclass instance.
    """
    strategies = {
        "fixed": FixedSizeChunker,
        "recursive": RecursiveChunker,
    }
    if strategy not in strategies:
        raise ValueError(f"Unknown chunking strategy '{strategy}'. Choose from: {list(strategies)}")
    return strategies[strategy](chunk_size=chunk_size, chunk_overlap=chunk_overlap)
