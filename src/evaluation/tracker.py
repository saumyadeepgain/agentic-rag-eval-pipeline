"""
Agentic RAG System — Evaluation Layer
Cost and latency tracker for each pipeline stage.
"""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from loguru import logger


# ──────────────────────────────────────────────────────────────────────────────
# OpenAI Token Cost Table (per 1M tokens, as of mid-2025)
# ──────────────────────────────────────────────────────────────────────────────

COST_PER_1M_TOKENS: Dict[str, Dict[str, float]] = {
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "text-embedding-3-small": {"input": 0.02, "output": 0.0},
    "text-embedding-3-large": {"input": 0.13, "output": 0.0},
}


def estimate_cost_usd(model: str, input_tokens: int, output_tokens: int = 0) -> float:
    """
    Estimate API cost in USD for a given model and token counts.

    Args:
        model:         OpenAI model name.
        input_tokens:  Number of input/prompt tokens.
        output_tokens: Number of output/completion tokens.

    Returns:
        Estimated cost in USD.
    """
    pricing = COST_PER_1M_TOKENS.get(model, {"input": 0.0, "output": 0.0})
    cost = (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000
    return cost


# ──────────────────────────────────────────────────────────────────────────────
# Per-Query Stage Record
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class StageRecord:
    """Records latency and cost for a single pipeline stage call."""
    stage: str
    model: Optional[str]
    latency_ms: float
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


@dataclass
class QueryRecord:
    """Aggregated cost/latency record for a complete query run."""
    query: str
    pipeline_variant: str
    domain: str
    stages: List[StageRecord] = field(default_factory=list)
    total_latency_ms: float = 0.0
    total_cost_usd: float = 0.0
    retrieval_attempts: int = 0
    was_abstained: bool = False

    def add_stage(self, record: StageRecord) -> None:
        self.stages.append(record)
        self.total_cost_usd += record.cost_usd
        self.total_latency_ms += record.latency_ms

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "pipeline_variant": self.pipeline_variant,
            "domain": self.domain,
            "total_latency_ms": round(self.total_latency_ms, 2),
            "total_cost_usd": round(self.total_cost_usd, 6),
            "retrieval_attempts": self.retrieval_attempts,
            "was_abstained": self.was_abstained,
            "stages": [
                {
                    "stage": s.stage,
                    "model": s.model,
                    "latency_ms": round(s.latency_ms, 2),
                    "input_tokens": s.input_tokens,
                    "output_tokens": s.output_tokens,
                    "cost_usd": round(s.cost_usd, 6),
                }
                for s in self.stages
            ],
        }


# ──────────────────────────────────────────────────────────────────────────────
# Tracker
# ──────────────────────────────────────────────────────────────────────────────

class LatencyCostTracker:
    """
    Tracks per-stage latency and cost across multiple query runs.

    Usage:
        tracker = LatencyCostTracker()
        record = tracker.new_record("What is the refund policy?", "naive", "finance")
        with tracker.stage(record, "retrieval"):
            hits = retriever.retrieve(query)
        tracker.log_llm_call(record, "generate", "gpt-4o", input_tokens=800, output_tokens=200)
        tracker.finish_record(record)

    Then call tracker.summary() to get aggregate stats per pipeline variant.
    """

    def __init__(self):
        self.records: List[QueryRecord] = []
        self._active: Dict[str, float] = {}   # stage → start time

    def new_record(self, query: str, pipeline_variant: str, domain: str) -> QueryRecord:
        record = QueryRecord(query=query, pipeline_variant=pipeline_variant, domain=domain)
        record._start = time.perf_counter()
        return record

    @contextmanager
    def stage(self, record: QueryRecord, stage_name: str, model: Optional[str] = None):
        """Context manager to time a pipeline stage (non-LLM, e.g. retrieval, reranking)."""
        t0 = time.perf_counter()
        yield
        elapsed = (time.perf_counter() - t0) * 1000
        record.add_stage(StageRecord(
            stage=stage_name,
            model=model,
            latency_ms=elapsed,
        ))

    def log_llm_call(
        self,
        record: QueryRecord,
        stage_name: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
    ) -> None:
        """Log an LLM call with token counts for cost estimation."""
        cost = estimate_cost_usd(model, input_tokens, output_tokens)
        record.add_stage(StageRecord(
            stage=stage_name,
            model=model,
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
        ))

    def finish_record(self, record: QueryRecord) -> None:
        """Finalize a query record and add to the tracker."""
        if hasattr(record, "_start"):
            record.total_latency_ms = (time.perf_counter() - record._start) * 1000
        self.records.append(record)
        logger.debug(
            f"Tracked query | variant={record.pipeline_variant} | "
            f"latency={record.total_latency_ms:.0f}ms | "
            f"cost=${record.total_cost_usd:.5f}"
        )

    def summary(self) -> dict:
        """
        Compute per-variant aggregate statistics.

        Returns:
            Dict mapping pipeline_variant → {avg_latency_ms, avg_cost_usd, n_queries, ...}
        """
        from collections import defaultdict
        import statistics

        groups: Dict[str, List[QueryRecord]] = defaultdict(list)
        for r in self.records:
            groups[r.pipeline_variant].append(r)

        summary = {}
        for variant, records in groups.items():
            latencies = [r.total_latency_ms for r in records]
            costs = [r.total_cost_usd for r in records]
            summary[variant] = {
                "n_queries": len(records),
                "avg_latency_ms": round(statistics.mean(latencies), 1),
                "p95_latency_ms": round(sorted(latencies)[int(0.95 * len(latencies))], 1),
                "avg_cost_usd": round(statistics.mean(costs), 6),
                "total_cost_usd": round(sum(costs), 4),
                "abstention_rate": sum(1 for r in records if r.was_abstained) / len(records),
            }
        return summary

    def to_dataframe(self):
        """Export all records to a pandas DataFrame for analysis."""
        import pandas as pd
        return pd.DataFrame([r.to_dict() for r in self.records])

    def save(self, path: str) -> None:
        """Persist all records to JSON."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump([r.to_dict() for r in self.records], f, indent=2)
        logger.info(f"Saved {len(self.records)} tracking records to {path}")

    @classmethod
    def load(cls, path: str) -> "LatencyCostTracker":
        """Load records from a previously saved JSON file."""
        tracker = cls()
        with open(path) as f:
            data = json.load(f)
        # Reconstruct as minimal dicts (not full dataclasses — for analysis only)
        tracker._raw_data = data
        return tracker
