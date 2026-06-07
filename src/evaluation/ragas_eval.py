"""
Agentic RAG System — Evaluation Layer
RAGAS evaluation harness: loads RAGBench, runs all pipeline variants, computes metrics.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
from loguru import logger

# RAGAS imports — compatible with both 0.1.x and 0.2.x
try:
    # RAGAS 0.2.x API
    from ragas import evaluate
    from ragas.metrics import (
        Faithfulness,
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
    )
    faithfulness = Faithfulness()
    answer_relevancy = AnswerRelevancy()
    context_precision = ContextPrecision()
    context_recall = ContextRecall()
    RAGAS_AVAILABLE = True
    RAGAS_V2 = True
except ImportError:
    try:
        # RAGAS 0.1.x API
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        )
        RAGAS_AVAILABLE = True
        RAGAS_V2 = False
    except ImportError:
        RAGAS_AVAILABLE = False
        RAGAS_V2 = False
        logger.warning("RAGAS not installed. Install with: pip install ragas")

try:
    from datasets import Dataset as HFDataset
except ImportError:
    HFDataset = None

# ──────────────────────────────────────────────────────────────────────────────
# RAGBench Loader
# ──────────────────────────────────────────────────────────────────────────────

RAGBENCH_CONFIGS = {
    "finance": "finqa",
    "biomedical": "pubmedqa",
    "legal": "cuad",
}


def load_ragbench_samples(
    domain: str,
    split: str = "test",
    n_samples: int = 50,
    cache_dir: str = "./data/ragbench_samples",
) -> List[Dict]:
    """
    Load and cache RAGBench samples for a given domain.

    RAGBench columns used:
        - question: The query.
        - documents: List of source document strings.
        - answer: Ground truth answer.
        (Labels like 'all_utilized', 'faithful' are used for validation.)

    Args:
        domain:     "finance", "biomedical", or "legal".
        split:      Dataset split to use.
        n_samples:  Max number of samples to return.
        cache_dir:  Local directory to cache downloaded samples.

    Returns:
        List of dicts: {question, documents, ground_truth, domain}.
    """
    from datasets import load_dataset

    config = RAGBENCH_CONFIGS.get(domain)
    if not config:
        raise ValueError(f"Unknown domain '{domain}'. Choose from: {list(RAGBENCH_CONFIGS)}")

    cache_path = Path(cache_dir) / f"{domain}_{n_samples}.json"
    if cache_path.exists():
        logger.info(f"Loading cached RAGBench samples: {cache_path}")
        with open(cache_path) as f:
            return json.load(f)

    logger.info(f"Downloading RAGBench '{config}' (split={split}, n={n_samples})…")
    ds = load_dataset("rungalileo/ragbench", config, split=split, trust_remote_code=True)

    samples = []
    for row in ds.select(range(min(n_samples, len(ds)))):
        # Normalize document field — may be list or single string
        docs = row.get("documents", row.get("context", []))
        if isinstance(docs, str):
            docs = [docs]

        samples.append({
            "question": row["question"],
            "documents": docs,
            "ground_truth": row.get("answer", row.get("response", "")),
            "domain": domain,
        })

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(samples, f, indent=2)

    logger.info(f"Cached {len(samples)} samples to {cache_path}")
    return samples


# ──────────────────────────────────────────────────────────────────────────────
# RAGAS Evaluation Runner
# ──────────────────────────────────────────────────────────────────────────────

def build_ragas_dataset(
    samples: List[Dict],
    pipeline_fn: Callable[[str, List[str]], Dict],
) -> HFDataset:
    """
    Run a pipeline function on each sample and assemble a RAGAS-compatible dataset.

    Args:
        samples:     RAGBench samples from load_ragbench_samples().
        pipeline_fn: Callable(question, documents) → {answer, contexts}.
                     'contexts' should be the list of retrieved chunk strings.

    Returns:
        HuggingFace Dataset with columns: question, answer, contexts, ground_truth.
    """
    rows = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": [],
    }

    for i, sample in enumerate(samples):
        logger.debug(f"Running pipeline on sample {i + 1}/{len(samples)}")
        try:
            result = pipeline_fn(sample["question"], sample["documents"])
            rows["question"].append(sample["question"])
            rows["answer"].append(result.get("answer", ""))
            rows["contexts"].append(result.get("contexts", sample["documents"][:3]))
            rows["ground_truth"].append(sample["ground_truth"])
        except Exception as e:
            logger.error(f"Pipeline failed on sample {i}: {e}")
            # Include failed samples with empty answer so eval still runs
            rows["question"].append(sample["question"])
            rows["answer"].append("")
            rows["contexts"].append(sample["documents"][:3])
            rows["ground_truth"].append(sample["ground_truth"])

    return HFDataset.from_dict(rows)


def run_ragas_eval(
    samples: List[Dict],
    pipeline_fn: Callable,
    gemini_api_key: Optional[str] = None,
) -> Dict[str, float]:
    """
    Run RAGAS evaluation on a dataset.

    Args:
        samples:        RAGBench sample list.
        pipeline_fn:    Pipeline function as described in build_ragas_dataset.
        openai_api_key: API key override (uses env var if not provided).

    Returns:
        Dict of metric name → score (0.0–1.0).
    """
    if not RAGAS_AVAILABLE:
        raise ImportError("RAGAS not installed. Run: pip install ragas")

    if gemini_api_key:
        os.environ["GEMINI_API_KEY"] = gemini_api_key
        os.environ["GOOGLE_API_KEY"] = gemini_api_key
    elif "GEMINI_API_KEY" in os.environ:
        os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]

    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
    gemini_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
    gemini_embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    
    # Configure RAGAS metrics to use Gemini
    faithfulness.llm = gemini_llm
    answer_relevancy.llm = gemini_llm
    answer_relevancy.embeddings = gemini_embeddings
    context_precision.llm = gemini_llm
    context_recall.llm = gemini_llm

    logger.info(f"Building RAGAS eval dataset from {len(samples)} samples…")
    ragas_dataset = build_ragas_dataset(samples, pipeline_fn)

    logger.info("Running RAGAS evaluation (faithfulness, answer_relevancy, context_precision, context_recall)…")
    result = evaluate(
        dataset=ragas_dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )

    scores = {
        "faithfulness": round(float(result["faithfulness"]), 4),
        "answer_relevancy": round(float(result["answer_relevancy"]), 4),
        "context_precision": round(float(result["context_precision"]), 4),
        "context_recall": round(float(result["context_recall"]), 4),
    }
    scores["average"] = round(sum(scores.values()) / len(scores), 4)

    logger.info(f"RAGAS results: {scores}")
    return scores


# ──────────────────────────────────────────────────────────────────────────────
# Full Evaluation Run (all variants × all domains)
# ──────────────────────────────────────────────────────────────────────────────

def run_full_evaluation(
    pipeline_fns: Dict[str, Callable],
    domains: List[str] | None = None,
    n_samples: int = 50,
    output_path: str = "./results/eval_results.json",
) -> pd.DataFrame:
    """
    Run evaluation for all pipeline variants across all domains.

    Args:
        pipeline_fns:  Dict mapping variant_name → pipeline_fn.
        domains:       Domains to evaluate. Default: all 3.
        n_samples:     Samples per domain.
        output_path:   Where to save raw results JSON.

    Returns:
        DataFrame with columns: variant, domain, faithfulness,
        answer_relevancy, context_precision, context_recall, average.
    """
    if domains is None:
        domains = ["finance", "biomedical", "legal"]

    all_results = []

    for domain in domains:
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating domain: {domain.upper()}")
        logger.info(f"{'='*60}")

        samples = load_ragbench_samples(domain, n_samples=n_samples)

        for variant_name, pipeline_fn in pipeline_fns.items():
            logger.info(f"  Variant: {variant_name}")
            try:
                scores = run_ragas_eval(samples, pipeline_fn)
                record = {
                    "variant": variant_name,
                    "domain": domain,
                    **scores,
                }
                all_results.append(record)
                logger.info(f"  ✓ {variant_name}/{domain}: avg={scores['average']:.4f}")
            except Exception as e:
                logger.error(f"  ✗ {variant_name}/{domain} failed: {e}")
                all_results.append({
                    "variant": variant_name,
                    "domain": domain,
                    "error": str(e),
                })

    # Save raw results
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Saved evaluation results to {output_path}")

    df = pd.DataFrame(all_results)
    return df


def print_comparison_table(df: pd.DataFrame) -> None:
    """
    Print a formatted comparison table averaged across all domains.
    This is the key artifact for the README / portfolio.
    """
    numeric_cols = ["faithfulness", "answer_relevancy", "context_precision", "context_recall", "average"]
    existing_cols = [c for c in numeric_cols if c in df.columns]

    agg = (
        df.groupby("variant")[existing_cols]
        .mean()
        .round(4)
        .reset_index()
    )

    # Define display order
    variant_order = ["naive", "hybrid", "hybrid_rerank", "agentic_full"]
    agg["_order"] = agg["variant"].map(
        {v: i for i, v in enumerate(variant_order)}
    ).fillna(99)
    agg = agg.sort_values("_order").drop("_order", axis=1)

    print("\n" + "=" * 90)
    print("RAGAS EVALUATION RESULTS — Averaged Across All Domains")
    print("=" * 90)
    print(agg.to_string(index=False))
    print("=" * 90)
