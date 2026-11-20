"""Retrieval over the index built by ``src.ingestion``.

* Dense: exact cosine search. Document and query vectors are L2-normalised by
  the same embedding client, so a dot product is cosine similarity; the index
  manifest records the embedding model and loading refuses a mismatch.
* BM25: word tokens without punctuation or stop words and Lucene's IDF, which is
  always positive (Okapi's can be zero or negative on small corpora).
* Hybrid: Reciprocal Rank Fusion of the two rankings on the shared ``chunk_id``.
* Reranking: optional cross-encoder (or a lexical reranker for offline runs) over
  the fused candidates.
* Metadata filters are applied to every retriever before ranking, so a filter
  can never let an out-of-scope chunk through; no match means an empty result.
"""

from __future__ import annotations

import asyncio
import math
import threading
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .llm import LLMClient, content_terms
from .schemas import (
    AppConfig,
    DocumentChunk,
    IndexManifest,
    RetrievalOutcome,
    RetrievalResult,
    Scalar,
)
from .utils import get_logger, log_event, read_jsonl, write_json, write_jsonl

log = get_logger("retriever")
CHUNKS_FILE, EMBEDDINGS_FILE, MANIFEST_FILE = "chunks.jsonl", "embeddings.npy", "manifest.json"


class IndexNotFoundError(FileNotFoundError):
    """No complete index in ``ingestion.index_dir``."""


class IndexMismatchError(RuntimeError):
    """The index was built with a different embedding model or is inconsistent."""


# ──────────────────────────────────────────────────────────────── index files
def save_index(
    index_dir: str | Path, chunks: list[DocumentChunk], embeddings: np.ndarray, manifest: IndexManifest
) -> None:
    """Write chunks, vectors and manifest; the manifest goes last so a partial write is detectable."""
    directory = Path(index_dir)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / MANIFEST_FILE).unlink(missing_ok=True)
    write_jsonl(directory / CHUNKS_FILE, chunks)
    tmp = directory / f".{EMBEDDINGS_FILE}.tmp"
    with open(tmp, "wb") as f:
        np.save(f, np.asarray(embeddings, dtype=np.float32), allow_pickle=False)
    tmp.replace(directory / EMBEDDINGS_FILE)
    write_json(directory / MANIFEST_FILE, manifest)


def load_index(index_dir: str | Path) -> tuple[list[DocumentChunk], np.ndarray, IndexManifest]:
    directory = Path(index_dir)
    missing = [name for name in (CHUNKS_FILE, EMBEDDINGS_FILE, MANIFEST_FILE) if not (directory / name).is_file()]
    if missing:
        raise IndexNotFoundError(
            f"no complete index in {directory} (missing {', '.join(missing)}); run `make ingest` first"
        )
    manifest = IndexManifest.model_validate_json((directory / MANIFEST_FILE).read_text(encoding="utf-8"))
    chunks = read_jsonl(directory / CHUNKS_FILE, DocumentChunk)
    embeddings = np.load(directory / EMBEDDINGS_FILE, allow_pickle=False)
    if embeddings.ndim != 2 or embeddings.shape[0] != len(chunks) or len(chunks) != manifest.n_chunks:
        raise IndexMismatchError(
            f"index in {directory} is inconsistent: {len(chunks)} chunks, embeddings {embeddings.shape}, "
            f"manifest says {manifest.n_chunks}; rebuild it with `make ingest`"
        )
    if embeddings.shape[1] != manifest.dimension:
        raise IndexMismatchError(f"embedding dimension {embeddings.shape[1]} != manifest {manifest.dimension}")
    return chunks, embeddings.astype(np.float32, copy=False), manifest


def check_manifest(manifest: IndexManifest, cfg: AppConfig) -> None:
    """Refuse to query an index with a different embedding model than the config's."""
    if (manifest.embedding_provider, manifest.embedding_model) != (cfg.llm.provider, cfg.models.embedding):
        raise IndexMismatchError(
            f"index was built with {manifest.embedding_provider}:{manifest.embedding_model} but the config "
            f"embeds queries with {cfg.llm.provider}:{cfg.models.embedding}; rebuild with `make ingest` "
            "(or point ingestion.index_dir at a matching index)"
        )


# ──────────────────────────────────────────────────────────────── BM25
class BM25Index:
    """BM25 with Lucene's IDF ``log(1 + (N - df + 0.5) / (df + 0.5))`` (always positive)."""

    def __init__(self, texts: Sequence[str], k1: float = 1.2, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.n_docs = len(texts)
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        lengths = []
        for doc, text in enumerate(texts):
            terms = Counter(content_terms(text))
            lengths.append(sum(terms.values()))
            for term, tf in terms.items():
                self.postings[term].append((doc, tf))
        self.doc_len = np.asarray(lengths, dtype=np.float64)
        self.avg_len = float(self.doc_len.mean()) if self.n_docs and self.doc_len.sum() else 1.0
        self.idf = {
            term: math.log(1.0 + (self.n_docs - len(p) + 0.5) / (len(p) + 0.5)) for term, p in self.postings.items()
        }

    def scores(self, query: str) -> np.ndarray:
        out = np.zeros(self.n_docs, dtype=np.float64)
        for term in dict.fromkeys(content_terms(query)):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for doc, tf in self.postings[term]:
                norm = self.k1 * (1.0 - self.b + self.b * self.doc_len[doc] / self.avg_len)
                out[doc] += idf * tf * (self.k1 + 1.0) / (tf + norm)
        return out


# ──────────────────────────────────────────────────────────────── ranking helpers
def rank_indices(scores: np.ndarray, k: int, mask: np.ndarray | None = None, positive_only: bool = False) -> list[int]:
    """Indices of the ``k`` best scores (ties broken by index), restricted to ``mask``."""
    valid = np.ones(len(scores), dtype=bool) if mask is None else mask.copy()
    if positive_only:
        valid &= scores > 0
    candidates = np.flatnonzero(valid)
    if candidates.size == 0 or k <= 0:
        return []
    order = candidates[np.lexsort((candidates, -scores[candidates]))]
    return order[:k].tolist()


def rrf_fuse(
    rankings: Sequence[Sequence[str]], k: int = 60, weights: Sequence[float] | None = None
) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion: ``score(d) = Σ_i w_i / (k + rank_i(d))`` with 1-based ranks.

    An ID repeated inside one ranking counts once, at its first position (later
    items are not pushed down by the duplicate). Ties are broken by the best
    single-list rank, then by ID, so the output is deterministic."""
    weights = [1.0] * len(rankings) if weights is None else list(weights)
    if len(weights) != len(rankings):
        raise ValueError("rrf_fuse needs one weight per ranking")
    scores: dict[str, float] = {}
    best: dict[str, int] = {}
    for ranking, weight in zip(rankings, weights, strict=True):
        rank = 0
        seen: set[str] = set()
        for item in ranking:
            if item in seen:
                continue
            seen.add(item)
            rank += 1
            scores[item] = scores.get(item, 0.0) + weight / (k + rank)
            best[item] = min(best.get(item, rank), rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], best[kv[0]], kv[0]))


# ──────────────────────────────────────────────────────────────── rerankers
class Reranker(Protocol):
    name: str

    async def score(self, query: str, texts: list[str]) -> list[float]: ...


class LexicalReranker:
    """BM25 over the candidate set: deterministic and dependency-free (offline runs)."""

    name = "lexical"

    async def score(self, query: str, texts: list[str]) -> list[float]:
        return BM25Index(texts).scores(query).tolist()


class CrossEncoderReranker:
    """sentence-transformers cross-encoder (e.g. BAAI/bge-reranker-v2-m3), run in a worker thread."""

    name = "cross_encoder"

    def __init__(self, model_name: str, batch_size: int = 32) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise ImportError(
                "the cross-encoder reranker needs `pip install sentence-transformers`; "
                "or set retrieval.reranker=lexical or retrieval.rerank=false"
            ) from exc
        self._model = CrossEncoder(model_name)
        self._batch_size = batch_size
        self._lock = threading.Lock()

    def _predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        with self._lock:
            scores = self._model.predict(pairs, batch_size=self._batch_size, show_progress_bar=False)
        return [float(s) for s in scores]

    async def score(self, query: str, texts: list[str]) -> list[float]:
        return await asyncio.to_thread(self._predict, [(query, text) for text in texts])


def build_reranker(cfg: AppConfig) -> Reranker | None:
    if not cfg.retrieval.rerank:
        return None
    if cfg.llm.provider == "offline" or cfg.retrieval.reranker == "lexical":
        return LexicalReranker()
    return CrossEncoderReranker(cfg.models.reranker)


# ──────────────────────────────────────────────────────────────── retriever
_AUTO: Any = object()


class Retriever:
    def __init__(
        self,
        cfg: AppConfig,
        llm: LLMClient,
        chunks: list[DocumentChunk],
        embeddings: np.ndarray,
        manifest: IndexManifest,
        reranker: Reranker | None = None,
    ) -> None:
        if len(chunks) != embeddings.shape[0]:
            raise IndexMismatchError("one embedding per chunk is required")
        self.cfg, self.llm, self.manifest, self.reranker = cfg, llm, manifest, reranker
        self.chunks = chunks
        self.embeddings = np.asarray(embeddings, dtype=np.float32)
        self.bm25 = BM25Index([c.text for c in chunks])
        self._position = {c.chunk_id: i for i, c in enumerate(chunks)}
        if len(self._position) != len(chunks):
            raise IndexMismatchError("duplicate chunk IDs in the index")

    @classmethod
    def load(cls, cfg: AppConfig, llm: LLMClient, reranker: Any = _AUTO) -> Retriever:
        chunks, embeddings, manifest = load_index(cfg.ingestion.index_dir)
        check_manifest(manifest, cfg)
        chosen = build_reranker(cfg) if reranker is _AUTO else reranker
        log_event(
            log,
            "index loaded",
            chunks=len(chunks),
            model=manifest.embedding_model,
            mode=cfg.retrieval.mode,
            reranker=getattr(chosen, "name", None),
        )
        return cls(cfg, llm, chunks, embeddings, manifest, chosen)

    def filter_mask(self, filters: Mapping[str, Scalar] | None) -> np.ndarray | None:
        if not filters:
            return None
        return np.fromiter(
            (all(c.metadata.get(key) == value for key, value in filters.items()) for c in self.chunks),
            dtype=bool,
            count=len(self.chunks),
        )

    async def retrieve(
        self, query: str, *, top_k: int | None = None, filters: Mapping[str, Scalar] | None = None
    ) -> RetrievalOutcome:
        """Top ``top_k`` chunks for ``query``; an empty result is a normal outcome, not an error."""
        settings = self.cfg.retrieval
        k = top_k or settings.top_k
        outcome = RetrievalOutcome()
        started = time.perf_counter()
        mask = self.filter_mask(filters)
        if not self.chunks or not query.strip() or (mask is not None and not mask.any()):
            outcome.retrieval_ms = (time.perf_counter() - started) * 1000
            return outcome

        dense_scores: np.ndarray | None = None
        bm25_scores: np.ndarray | None = None
        rankings: list[list[str]] = []
        weights: list[float] = []
        if settings.mode in ("dense", "hybrid"):
            query_vec, usage = await self.llm.embed([query], task="embed_query")
            outcome.usage.append(usage)
            if query_vec.shape[1] != self.embeddings.shape[1]:
                raise IndexMismatchError(
                    f"query embedding has {query_vec.shape[1]} dimensions, index has {self.embeddings.shape[1]}"
                )
            dense_scores = self.embeddings @ query_vec[0]
            top = rank_indices(dense_scores, settings.candidates_k, mask)
            rankings.append([self.chunks[i].chunk_id for i in top])
            weights.append(settings.dense_weight)
        if settings.mode in ("bm25", "hybrid"):
            bm25_scores = self.bm25.scores(query)
            top = rank_indices(bm25_scores, settings.candidates_k, mask, positive_only=True)
            rankings.append([self.chunks[i].chunk_id for i in top])
            weights.append(settings.bm25_weight)

        if settings.mode == "hybrid":
            fused = rrf_fuse(rankings, k=settings.rrf_k, weights=weights)
        else:
            base = dense_scores if settings.mode == "dense" else bm25_scores
            assert base is not None
            fused = [(cid, float(base[self._position[cid]])) for cid in rankings[0]]
        candidates = fused[: settings.candidates_k]
        outcome.retrieval_ms = (time.perf_counter() - started) * 1000

        rerank_scores: dict[str, float] = {}
        if self.reranker is not None and candidates:
            rerank_started = time.perf_counter()
            texts = [self.chunks[self._position[cid]].text for cid, _ in candidates]
            scores = await self.reranker.score(query, texts)
            rerank_scores = {cid: float(s) for (cid, _), s in zip(candidates, scores, strict=True)}
            order = sorted(range(len(candidates)), key=lambda i: (-scores[i], i))
            candidates = [candidates[i] for i in order]
            outcome.rerank_ms = (time.perf_counter() - rerank_started) * 1000

        for rank, (cid, fused_score) in enumerate(candidates[:k], start=1):
            i = self._position[cid]
            outcome.results.append(
                RetrievalResult(
                    chunk=self.chunks[i],
                    rank=rank,
                    score=rerank_scores.get(cid, fused_score),
                    dense_score=None if dense_scores is None else float(dense_scores[i]),
                    bm25_score=None if bm25_scores is None else float(bm25_scores[i]),
                    rrf_score=fused_score if settings.mode == "hybrid" else None,
                    rerank_score=rerank_scores.get(cid),
                )
            )
        return outcome
