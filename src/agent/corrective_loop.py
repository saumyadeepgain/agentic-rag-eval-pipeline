"""
Agentic RAG System — Agentic Layer
Corrective RAG loop: the core agentic state machine.

States:
    ROUTE → RETRIEVE → GRADE → GENERATE | REWRITE → RETRIEVE → GRADE → GENERATE | ABSTAIN

This is intentionally implemented as plain Python (while-loop + state enum),
not as a LangGraph or LangChain abstraction, so every step is explicit and
debuggable — and explainable in an interview without framework knowledge.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, List, Optional

from loguru import logger

from src.agent.grader import DocumentGrader
from src.agent.query_rewriter import QueryRewriter
from src.agent.router import QueryRouter, RouteAction


# ──────────────────────────────────────────────────────────────────────────────
# State Machine Definitions
# ──────────────────────────────────────────────────────────────────────────────

class AgentState(str, Enum):
    """All possible states in the corrective RAG state machine."""
    ROUTE = "route"
    RETRIEVE = "retrieve"
    GRADE = "grade"
    REWRITE = "rewrite"
    GENERATE = "generate"
    ABSTAIN = "abstain"
    DIRECT_ANSWER = "direct_answer"
    DONE = "done"


@dataclass
class AgentTrace:
    """
    Full execution trace for a single query run.
    Captured per-step for observability and debugging.
    """
    query: str
    states_visited: List[str] = field(default_factory=list)
    retrieval_attempts: int = 0
    queries_tried: List[str] = field(default_factory=list)
    grading_summaries: List[dict] = field(default_factory=list)
    final_context: List[dict] = field(default_factory=list)
    final_answer: str = ""
    was_abstained: bool = False
    route_action: str = ""
    total_latency_ms: float = 0.0
    step_latencies: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────────────────
# The Corrective Loop
# ──────────────────────────────────────────────────────────────────────────────

class CorrectiveRAGLoop:
    """
    Agentic corrective RAG state machine.

    Orchestrates the full query-to-answer pipeline with:
    - Routing: decide if retrieval is even needed
    - Retrieval: plug-in retriever function (dense, hybrid, or HyDE)
    - Grading: per-chunk relevance scoring
    - Query rewriting + retry on insufficient results
    - Abstention when context is persistently insufficient
    - Generation with citation grounding (via external generator)

    Design principle: This is NOT a framework — it's a while-loop with a state
    enum. Every transition is explicit and testable.

    Args:
        retriever_fn:    Callable(query: str, top_k: int) → List[dict hits]
        generator_fn:    Callable(query: str, context: List[dict]) → dict with 'answer', 'citations'
        grader:          DocumentGrader instance.
        rewriter:        QueryRewriter instance.
        router:          QueryRouter instance (None = skip routing, always retrieve).
        max_retries:     Max query rewrites before abstaining.
        relevance_threshold: Fraction of docs that must grade relevant to proceed.
        top_k:           Docs to retrieve per attempt.
    """

    def __init__(
        self,
        retriever_fn: Callable,
        generator_fn: Callable,
        grader: DocumentGrader,
        rewriter: QueryRewriter,
        router: Optional[QueryRouter] = None,
        max_retries: int = 2,
        relevance_threshold: float = 0.3,
        top_k: int = 10,
    ):
        self.retriever_fn = retriever_fn
        self.generator_fn = generator_fn
        self.grader = grader
        self.rewriter = rewriter
        self.router = router
        self.max_retries = max_retries
        self.relevance_threshold = relevance_threshold
        self.top_k = top_k

    def _time_step(self, step_name: str, trace: AgentTrace):
        """Context manager helper for timing individual steps."""
        return _StepTimer(step_name, trace)

    def run(self, query: str) -> AgentTrace:
        """
        Execute the corrective RAG loop for a single query.

        Args:
            query: User query string.

        Returns:
            AgentTrace with full execution details and the final answer.
        """
        trace = AgentTrace(query=query, queries_tried=[query])
        start_total = time.perf_counter()
        current_query = query
        state = AgentState.ROUTE
        retry_count = 0

        # ── State machine loop ────────────────────────────────────────────────
        while state != AgentState.DONE:
            trace.states_visited.append(state.value)
            logger.debug(f"[Agent] State: {state.value} | Query: '{current_query[:60]}'")

            # ── ROUTE ─────────────────────────────────────────────────────────
            if state == AgentState.ROUTE:
                if self.router is not None:
                    t0 = time.perf_counter()
                    decision = self.router.route(current_query)
                    trace.step_latencies["route"] = (time.perf_counter() - t0) * 1000
                    trace.route_action = decision.action.value

                    if decision.action == RouteAction.DIRECT_ANSWER:
                        state = AgentState.DIRECT_ANSWER
                    elif decision.action == RouteAction.CLARIFY:
                        trace.final_answer = (
                            "Your question is a bit ambiguous. Could you clarify "
                            "what specific aspect you're asking about?"
                        )
                        trace.was_abstained = True
                        state = AgentState.DONE
                    elif decision.action == RouteAction.OUT_OF_SCOPE:
                        trace.final_answer = (
                            "This question appears to be outside the scope of the "
                            "available document corpus (finance, biomedical, legal). "
                            "I cannot retrieve relevant information to answer it."
                        )
                        trace.was_abstained = True
                        state = AgentState.DONE
                    else:
                        state = AgentState.RETRIEVE
                else:
                    trace.route_action = RouteAction.RETRIEVE.value
                    state = AgentState.RETRIEVE

            # ── DIRECT ANSWER (no retrieval needed) ──────────────────────────
            elif state == AgentState.DIRECT_ANSWER:
                t0 = time.perf_counter()
                result = self.generator_fn(current_query, [])
                trace.step_latencies["generate_direct"] = (time.perf_counter() - t0) * 1000
                trace.final_answer = result.get("answer", "")
                state = AgentState.DONE

            # ── RETRIEVE ──────────────────────────────────────────────────────
            elif state == AgentState.RETRIEVE:
                t0 = time.perf_counter()
                hits = self.retriever_fn(current_query, self.top_k)
                elapsed = (time.perf_counter() - t0) * 1000
                attempt_key = f"retrieve_attempt_{trace.retrieval_attempts}"
                trace.step_latencies[attempt_key] = elapsed
                trace.retrieval_attempts += 1

                logger.debug(f"[Agent] Retrieved {len(hits)} docs in {elapsed:.1f}ms")
                # Store hits temporarily in metadata for grading
                trace.metadata["_current_hits"] = hits
                state = AgentState.GRADE

            # ── GRADE ─────────────────────────────────────────────────────────
            elif state == AgentState.GRADE:
                t0 = time.perf_counter()
                hits = trace.metadata.get("_current_hits", [])
                relevant, irrelevant, summary = self.grader.grade_documents(current_query, hits)
                trace.step_latencies[f"grade_attempt_{len(trace.grading_summaries)}"] = (
                    time.perf_counter() - t0
                ) * 1000
                trace.grading_summaries.append(summary)

                if summary["relevance_ratio"] >= self.relevance_threshold:
                    # Sufficient relevant docs — proceed to generation
                    trace.final_context = relevant
                    state = AgentState.GENERATE
                elif retry_count < self.max_retries:
                    # Insufficient — try rewriting the query
                    state = AgentState.REWRITE
                else:
                    # Exhausted retries — abstain
                    # Still use whatever relevant docs we have (may be empty)
                    trace.final_context = relevant
                    state = AgentState.ABSTAIN if not relevant else AgentState.GENERATE

            # ── REWRITE ───────────────────────────────────────────────────────
            elif state == AgentState.REWRITE:
                t0 = time.perf_counter()
                new_query = self.rewriter.rewrite(current_query)
                trace.step_latencies[f"rewrite_{retry_count}"] = (time.perf_counter() - t0) * 1000

                if new_query != current_query:
                    current_query = new_query
                    trace.queries_tried.append(current_query)
                    logger.info(f"[Agent] Rewritten query (attempt {retry_count + 1}): '{current_query[:60]}'")
                else:
                    logger.warning("[Agent] Rewriter returned same query — forcing abstain.")
                    state = AgentState.ABSTAIN
                    continue

                retry_count += 1
                state = AgentState.RETRIEVE

            # ── GENERATE ──────────────────────────────────────────────────────
            elif state == AgentState.GENERATE:
                t0 = time.perf_counter()
                result = self.generator_fn(current_query, trace.final_context)
                trace.step_latencies["generate"] = (time.perf_counter() - t0) * 1000
                trace.final_answer = result.get("answer", "")
                trace.metadata["citations"] = result.get("citations", [])
                trace.metadata["self_check"] = result.get("self_check", None)
                state = AgentState.DONE

            # ── ABSTAIN ───────────────────────────────────────────────────────
            elif state == AgentState.ABSTAIN:
                queries_tried_str = " → ".join(f'"{q}"' for q in trace.queries_tried)
                trace.final_answer = (
                    "I don't have sufficient information in the available documents to "
                    f"answer this question confidently.\n\n"
                    f"Queries tried: {queries_tried_str}\n\n"
                    "Please try rephrasing your question or providing more context."
                )
                trace.was_abstained = True
                state = AgentState.DONE

        # ── End of loop ───────────────────────────────────────────────────────
        trace.total_latency_ms = (time.perf_counter() - start_total) * 1000
        logger.info(
            f"[Agent] Done in {trace.total_latency_ms:.1f}ms | "
            f"States: {' → '.join(trace.states_visited)} | "
            f"Retries: {retry_count} | "
            f"Abstained: {trace.was_abstained}"
        )
        return trace


# ──────────────────────────────────────────────────────────────────────────────
# Step Timer Helper
# ──────────────────────────────────────────────────────────────────────────────

class _StepTimer:
    def __init__(self, name: str, trace: AgentTrace):
        self.name = name
        self.trace = trace
        self.t0 = None

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *args):
        elapsed = (time.perf_counter() - self.t0) * 1000
        self.trace.step_latencies[self.name] = elapsed
