"""
Agentic RAG System — Evaluation Layer
Package init.
"""

from src.evaluation.ragas_eval import (
    load_ragbench_samples,
    build_ragas_dataset,
    run_ragas_eval,
    run_full_evaluation,
    print_comparison_table,
)
from src.evaluation.adversarial import (
    ADVERSARIAL_CASES,
    AdversarialTestCase,
    run_adversarial_eval,
    score_injection_resistance,
    print_defense_comparison,
)
from src.evaluation.tracker import LatencyCostTracker, estimate_cost_usd

__all__ = [
    "load_ragbench_samples",
    "build_ragas_dataset",
    "run_ragas_eval",
    "run_full_evaluation",
    "print_comparison_table",
    "ADVERSARIAL_CASES",
    "AdversarialTestCase",
    "run_adversarial_eval",
    "score_injection_resistance",
    "print_defense_comparison",
    "LatencyCostTracker",
    "estimate_cost_usd",
]
