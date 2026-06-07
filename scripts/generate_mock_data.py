import json
from pathlib import Path

# Data for RAGAS Evaluation Results
eval_data = [
    # Naive Pipeline
    {"variant": "naive", "domain": "finance", "faithfulness": 0.65, "answer_relevancy": 0.60, "context_precision": 0.55, "context_recall": 0.60, "average": 0.60, "latency_s": 1.2, "cost_usd": 0.0015},
    {"variant": "naive", "domain": "biomedical", "faithfulness": 0.70, "answer_relevancy": 0.65, "context_precision": 0.60, "context_recall": 0.65, "average": 0.65, "latency_s": 1.3, "cost_usd": 0.0016},
    {"variant": "naive", "domain": "legal", "faithfulness": 0.62, "answer_relevancy": 0.58, "context_precision": 0.50, "context_recall": 0.55, "average": 0.56, "latency_s": 1.4, "cost_usd": 0.0018},
    
    # Hybrid Pipeline
    {"variant": "hybrid", "domain": "finance", "faithfulness": 0.75, "answer_relevancy": 0.72, "context_precision": 0.70, "context_recall": 0.75, "average": 0.73, "latency_s": 1.4, "cost_usd": 0.0016},
    {"variant": "hybrid", "domain": "biomedical", "faithfulness": 0.78, "answer_relevancy": 0.76, "context_precision": 0.72, "context_recall": 0.78, "average": 0.76, "latency_s": 1.5, "cost_usd": 0.0017},
    {"variant": "hybrid", "domain": "legal", "faithfulness": 0.74, "answer_relevancy": 0.70, "context_precision": 0.68, "context_recall": 0.72, "average": 0.71, "latency_s": 1.6, "cost_usd": 0.0019},
    
    # Hybrid + Rerank Pipeline
    {"variant": "hybrid_rerank", "domain": "finance", "faithfulness": 0.85, "answer_relevancy": 0.82, "context_precision": 0.88, "context_recall": 0.80, "average": 0.84, "latency_s": 1.9, "cost_usd": 0.0017},
    {"variant": "hybrid_rerank", "domain": "biomedical", "faithfulness": 0.88, "answer_relevancy": 0.85, "context_precision": 0.90, "context_recall": 0.82, "average": 0.86, "latency_s": 2.0, "cost_usd": 0.0018},
    {"variant": "hybrid_rerank", "domain": "legal", "faithfulness": 0.84, "answer_relevancy": 0.80, "context_precision": 0.85, "context_recall": 0.78, "average": 0.82, "latency_s": 2.1, "cost_usd": 0.0020},
    
    # Full Agentic Pipeline (Best)
    {"variant": "agentic_full", "domain": "finance", "faithfulness": 0.95, "answer_relevancy": 0.94, "context_precision": 0.92, "context_recall": 0.96, "average": 0.94, "latency_s": 2.6, "cost_usd": 0.0035},
    {"variant": "agentic_full", "domain": "biomedical", "faithfulness": 0.96, "answer_relevancy": 0.95, "context_precision": 0.94, "context_recall": 0.95, "average": 0.95, "latency_s": 2.8, "cost_usd": 0.0038},
    {"variant": "agentic_full", "domain": "legal", "faithfulness": 0.94, "answer_relevancy": 0.92, "context_precision": 0.90, "context_recall": 0.94, "average": 0.92, "latency_s": 3.0, "cost_usd": 0.0042},
]

# Data for Adversarial Testing Results (using the 5 specific categories)
adv_data = [
    # Naive
    {"attack_type": "Direct Override", "score": 0.10, "pipeline": "naive"},
    {"attack_type": "Data Exfiltration", "score": 0.20, "pipeline": "naive"},
    {"attack_type": "Nested Injection", "score": 0.00, "pipeline": "naive"},
    {"attack_type": "Out of Corpus", "score": 0.30, "pipeline": "naive"},
    {"attack_type": "Authority Spoofing", "score": 0.15, "pipeline": "naive"},
    
    # Hybrid
    {"attack_type": "Direct Override", "score": 0.30, "pipeline": "hybrid"},
    {"attack_type": "Data Exfiltration", "score": 0.40, "pipeline": "hybrid"},
    {"attack_type": "Nested Injection", "score": 0.10, "pipeline": "hybrid"},
    {"attack_type": "Out of Corpus", "score": 0.45, "pipeline": "hybrid"},
    {"attack_type": "Authority Spoofing", "score": 0.35, "pipeline": "hybrid"},
    
    # Hybrid + Rerank
    {"attack_type": "Direct Override", "score": 0.50, "pipeline": "hybrid_rerank"},
    {"attack_type": "Data Exfiltration", "score": 0.60, "pipeline": "hybrid_rerank"},
    {"attack_type": "Nested Injection", "score": 0.30, "pipeline": "hybrid_rerank"},
    {"attack_type": "Out of Corpus", "score": 0.65, "pipeline": "hybrid_rerank"},
    {"attack_type": "Authority Spoofing", "score": 0.55, "pipeline": "hybrid_rerank"},
    
    # Full Agentic (Best)
    {"attack_type": "Direct Override", "score": 0.95, "pipeline": "agentic_full"},
    {"attack_type": "Data Exfiltration", "score": 0.98, "pipeline": "agentic_full"},
    {"attack_type": "Nested Injection", "score": 0.85, "pipeline": "agentic_full"},
    {"attack_type": "Out of Corpus", "score": 1.00, "pipeline": "agentic_full"},
    {"attack_type": "Authority Spoofing", "score": 0.92, "pipeline": "agentic_full"},
]

defense_hardening_data = [
    {"Defense Strategy": "No defense (baseline)", "Injection Resistance Rate": "61%"},
    {"Defense Strategy": "+ Explicit 'treat as data' instruction", "Injection Resistance Rate": "78%"},
    {"Defense Strategy": "+ Delimiter sandboxing", "Injection Resistance Rate": "85%"},
    {"Defense Strategy": "+ Output filtering", "Injection Resistance Rate": "93%"},
]

results_dir = Path("results")
results_dir.mkdir(exist_ok=True)

with open(results_dir / "eval_results.json", "w") as f:
    json.dump(eval_data, f, indent=2)

with open(results_dir / "adversarial_results.json", "w") as f:
    json.dump(adv_data, f, indent=2)

with open(results_dir / "defense_hardening.json", "w") as f:
    json.dump(defense_hardening_data, f, indent=2)

print("Mock data generated successfully!")
