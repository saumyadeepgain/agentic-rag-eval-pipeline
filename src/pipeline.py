"""
Agentic RAG System — Pipeline Orchestrator
Single entry point that wires all components together based on a config.
Exposes 4 pipeline variants: naive, hybrid, hybrid_rerank, agentic_full.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

from dotenv import load_dotenv
from loguru import logger

load_dotenv()


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline Configuration
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class PipelineConfig:
    """
    Declarative configuration for a single pipeline variant.

    All boolean flags are cumulative — later variants layer on top of earlier ones.
    """
    name: str = "naive"
    description: str = ""

    # Ingestion
    chunking_strategy: str = "fixed"    # "fixed" | "recursive"
    chunk_size: int = 512
    chunk_overlap: int = 64

    # Retrieval
    retrieval_mode: str = "dense"       # "dense" | "hybrid"
    top_k_retrieval: int = 20
    top_k_final: int = 5
    rrf_k: int = 60
    bm25_weight: float = 0.4

    # Retrieval augmentation
    reranking: bool = False
    hyde: bool = False

    # Agentic
    agentic: bool = False
    max_retries: int = 2
    relevance_threshold: float = 0.3

    # Models
    generator_model: str = "gemini-1.5-flash"
    grader_model: str = "gemini-1.5-flash"
    embedding_model: str = "text-embedding-004"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"

    # Guardrails
    sandbox_context: bool = True
    output_filtering: bool = True
    self_check: bool = True

    # Storage
    chroma_persist_dir: str = "./data/chroma_db"
    bm25_index_path: str = "./data/bm25_index.pkl"


# Predefined variant configs
PIPELINE_CONFIGS: Dict[str, PipelineConfig] = {
    "naive": PipelineConfig(
        name="naive",
        description="Baseline: fixed-size chunking, dense vector search only, no agentic loop",
        chunking_strategy="fixed",
        retrieval_mode="dense",
        reranking=False,
        hyde=False,
        agentic=False,
    ),
    "hybrid": PipelineConfig(
        name="hybrid",
        description="Hybrid: recursive chunking, BM25+vector RRF fusion",
        chunking_strategy="recursive",
        retrieval_mode="hybrid",
        reranking=False,
        hyde=False,
        agentic=False,
    ),
    "hybrid_rerank": PipelineConfig(
        name="hybrid_rerank",
        description="Hybrid + cross-encoder reranking",
        chunking_strategy="recursive",
        retrieval_mode="hybrid",
        reranking=True,
        hyde=False,
        agentic=False,
    ),
    "agentic_full": PipelineConfig(
        name="agentic_full",
        description="Full agentic: hybrid + reranking + HyDE + corrective loop",
        chunking_strategy="recursive",
        retrieval_mode="hybrid",
        reranking=True,
        hyde=True,
        agentic=True,
    ),
}


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline Builder
# ──────────────────────────────────────────────────────────────────────────────

class RAGPipeline:
    """
    Assembled RAG pipeline for a given configuration.

    Lazy-initializes all components on first use to avoid loading heavy
    models (reranker, embedder) until actually needed.

    Usage:
        pipeline = RAGPipeline.from_config("hybrid_rerank", domain="finance")
        result = pipeline.run("What is the refund policy?")

        # For evaluation harness compatibility:
        pipeline_fn = pipeline.as_eval_fn()
        scores = run_ragas_eval(samples, pipeline_fn)
    """

    def __init__(self, config: PipelineConfig, domain: str = "general"):
        self.config = config
        self.domain = domain
        self._initialized = False

        # Component placeholders
        self._embedder = None
        self._chroma = None
        self._bm25 = None
        self._dense_retriever = None
        self._bm25_retriever = None
        self._hybrid_retriever = None
        self._reranker = None
        self._hyde_retriever = None
        self._generator = None
        self._grader = None
        self._rewriter = None
        self._router = None
        self._agent = None

    @classmethod
    def from_config(cls, variant: str, domain: str = "general") -> "RAGPipeline":
        """
        Instantiate a pipeline from a named variant.

        Args:
            variant: One of 'naive', 'hybrid', 'hybrid_rerank', 'agentic_full'.
            domain:  Domain name (used for namespacing ChromaDB collection + BM25 index).
        """
        if variant not in PIPELINE_CONFIGS:
            raise ValueError(f"Unknown variant '{variant}'. Choose from: {list(PIPELINE_CONFIGS)}")
        config = PIPELINE_CONFIGS[variant]
        # Override storage paths to be domain-namespaced
        config.chroma_persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")
        config.bm25_index_path = f"./data/bm25_{domain}.pkl"
        config.generator_model = os.getenv("GENERATOR_MODEL", config.generator_model)
        config.grader_model = os.getenv("GRADER_MODEL", config.grader_model)
        config.embedding_model = os.getenv("EMBEDDING_MODEL", config.embedding_model)
        return cls(config, domain)

    def _init_components(self) -> None:
        """Lazy-initialize all pipeline components."""
        if self._initialized:
            return
        cfg = self.config
        logger.info(f"Initializing pipeline '{cfg.name}' for domain '{self.domain}'")

        # ── Embedder ──────────────────────────────────────────────────────────
        from src.ingestion.embedder import GeminiEmbedder
        self._embedder = GeminiEmbedder(model=cfg.embedding_model)

        # ── Vector Store ──────────────────────────────────────────────────────
        from src.ingestion.indexer import ChromaIndexer, BM25Indexer
        collection_name = f"{self.domain}_{cfg.chunking_strategy}"
        self._chroma = ChromaIndexer(
            collection_name=collection_name,
            persist_dir=cfg.chroma_persist_dir,
            embedder=self._embedder,
        )

        # ── BM25 Index (always init — may be needed for hybrid) ───────────────
        self._bm25 = BM25Indexer(index_path=cfg.bm25_index_path)

        # ── Retrievers ────────────────────────────────────────────────────────
        from src.retrieval.dense_retriever import DenseRetriever
        from src.retrieval.bm25_retriever import BM25Retriever
        self._dense_retriever = DenseRetriever(self._chroma, top_k=cfg.top_k_retrieval)
        self._bm25_retriever = BM25Retriever(self._bm25, top_k=cfg.top_k_retrieval)

        if cfg.retrieval_mode == "hybrid":
            from src.retrieval.hybrid_retriever import HybridRetriever
            self._hybrid_retriever = HybridRetriever(
                dense_retriever=self._dense_retriever,
                bm25_retriever=self._bm25_retriever,
                top_k_each=cfg.top_k_retrieval,
                top_k_final=cfg.top_k_final * 3 if cfg.reranking else cfg.top_k_final,
                rrf_k=cfg.rrf_k,
                bm25_weight=cfg.bm25_weight,
            )

        # ── Reranker ──────────────────────────────────────────────────────────
        if cfg.reranking:
            from src.retrieval.reranker import CrossEncoderReranker
            self._reranker = CrossEncoderReranker(model_name=cfg.reranker_model)

        # ── HyDE ─────────────────────────────────────────────────────────────
        if cfg.hyde:
            from src.retrieval.hyde import HyDERetriever
            self._hyde_retriever = HyDERetriever(
                embedder=self._embedder,
                chroma_indexer=self._chroma,
                llm_model=cfg.grader_model,
                top_k=cfg.top_k_retrieval,
            )

        # ── Generator ────────────────────────────────────────────────────────
        from src.generation.generator import RAGGenerator
        self._generator = RAGGenerator(
            model=cfg.generator_model,
            grader_model=cfg.grader_model,
            self_check=cfg.self_check,
            max_context_docs=cfg.top_k_final,
        )

        # ── Agentic components ────────────────────────────────────────────────
        if cfg.agentic:
            from src.agent.grader import DocumentGrader
            from src.agent.query_rewriter import QueryRewriter
            from src.agent.router import QueryRouter
            from src.agent.corrective_loop import CorrectiveRAGLoop
            self._grader = DocumentGrader(model=cfg.grader_model)
            self._rewriter = QueryRewriter(model=cfg.grader_model)
            self._router = QueryRouter(model=cfg.grader_model)
            self._agent = CorrectiveRAGLoop(
                retriever_fn=self._retrieve,
                generator_fn=self._generate,
                grader=self._grader,
                rewriter=self._rewriter,
                router=self._router,
                max_retries=cfg.max_retries,
                relevance_threshold=cfg.relevance_threshold,
                top_k=cfg.top_k_retrieval,
            )

        self._initialized = True
        logger.info(f"Pipeline '{cfg.name}' ready.")

    def _retrieve(self, query: str, top_k: int = None) -> List[dict]:
        """Internal retrieval function, selects strategy based on config."""
        cfg = self.config
        top_k = top_k or cfg.top_k_retrieval

        # Query transformation: HyDE
        if cfg.hyde and self._hyde_retriever:
            hits = self._hyde_retriever.retrieve(query, top_k=top_k)
        elif cfg.retrieval_mode == "hybrid" and self._hybrid_retriever:
            hits = self._hybrid_retriever.retrieve(query, top_k=top_k)
        else:
            hits = self._dense_retriever.retrieve(query, top_k=top_k)

        # Reranking
        if cfg.reranking and self._reranker and hits:
            hits = self._reranker.rerank(query, hits, top_k=cfg.top_k_final)

        # Context sandboxing (defense against injection)
        if cfg.sandbox_context:
            from src.generation.guardrails import sandbox_context
            hits = sandbox_context(hits)

        return hits[:cfg.top_k_final]

    def _generate(self, query: str, hits: List[dict]) -> dict:
        """Internal generation function passed to the agentic loop."""
        result = self._generator.generate(query, hits)

        # Output filtering (injection defense)
        if self.config.output_filtering:
            from src.generation.guardrails import filter_output
            guard_result = filter_output(result["answer"])
            if not guard_result.passed:
                result["answer"] = guard_result.filtered_content
                result["guardrail_triggered"] = True
                result["guardrail_flags"] = guard_result.flags

        return result

    def ingest(self, documents: List[str], doc_ids: Optional[List[str]] = None) -> None:
        """
        Chunk and index a list of raw document strings.

        Args:
            documents: List of raw text strings to ingest.
            doc_ids:   Optional list of document IDs. Auto-generated if not provided.
        """
        self._init_components()
        from src.ingestion.chunker import get_chunker
        chunker = get_chunker(
            self.config.chunking_strategy,
            self.config.chunk_size,
            self.config.chunk_overlap,
        )

        all_chunks = []
        for i, doc_text in enumerate(documents):
            doc_id = doc_ids[i] if doc_ids else f"doc_{i:04d}"
            chunks = chunker.chunk(doc_text, doc_id=doc_id)
            all_chunks.extend(chunks)

        logger.info(f"Ingesting {len(documents)} docs → {len(all_chunks)} chunks")
        self._chroma.index_chunks(all_chunks)
        self._bm25.index_chunks(all_chunks)

    def run(self, query: str) -> dict:
        """
        Run the full pipeline for a single query.

        Args:
            query: User query string.

        Returns:
            Dict with: answer, citations, contexts, config_name, domain,
            was_abstained, self_check (if enabled), agent_trace (if agentic).
        """
        self._init_components()

        if self.config.agentic and self._agent:
            from src.agent.corrective_loop import AgentTrace
            trace: AgentTrace = self._agent.run(query)
            return {
                "answer": trace.final_answer,
                "citations": trace.metadata.get("citations", []),
                "contexts": [h["content"] for h in trace.final_context],
                "config_name": self.config.name,
                "domain": self.domain,
                "was_abstained": trace.was_abstained,
                "self_check": trace.metadata.get("self_check"),
                "agent_trace": {
                    "states_visited": trace.states_visited,
                    "retrieval_attempts": trace.retrieval_attempts,
                    "queries_tried": trace.queries_tried,
                    "total_latency_ms": trace.total_latency_ms,
                },
            }
        else:
            # Non-agentic: simple retrieve → generate
            hits = self._retrieve(query)
            result = self._generate(query, hits)
            return {
                "answer": result.get("answer", ""),
                "citations": result.get("citations", []),
                "contexts": [h["content"] for h in hits],
                "config_name": self.config.name,
                "domain": self.domain,
                "was_abstained": "INSUFFICIENT CONTEXT" in result.get("answer", ""),
                "self_check": result.get("self_check"),
            }

    def as_eval_fn(self) -> Callable:
        """
        Return a RAGAS-compatible evaluation function.

        Returns a callable(question, documents) → {answer, contexts} that:
        1. Ingests the provided documents (RAGBench source docs).
        2. Runs the pipeline to generate an answer.
        3. Returns RAGAS-compatible dict.
        """
        def eval_fn(question: str, documents: List[str]) -> Dict:
            self._init_components()
            # Temporarily ingest sample documents
            # (In a real eval, you'd pre-ingest the full corpus)
            self.ingest(documents)
            result = self.run(question)
            return {
                "answer": result["answer"],
                "contexts": result["contexts"],
            }
        return eval_fn
