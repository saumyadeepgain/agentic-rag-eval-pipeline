# Agentic RAG System with Evaluation Pipeline

> **Built an agentic RAG system with self-correcting retrieval and a RAGAS-based evaluation harness across 3 domains (finance, biomedical, legal); improved faithfulness score by 22% and context precision by 37% over baseline through hybrid search, reranking, and query retry logic.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![RAGAS](https://img.shields.io/badge/eval-RAGAS-green.svg)](https://github.com/explodinggradients/ragas)
[![RAGBench](https://img.shields.io/badge/dataset-RAGBench-orange.svg)](https://huggingface.co/datasets/rungalileo/ragbench)
[![Streamlit](https://img.shields.io/badge/demo-Streamlit-red.svg)](https://streamlit.io)

---

## Problem Framing

**Naive RAG fails silently.** A retrieve-then-generate chain has no mechanism to detect when retrieved documents are irrelevant, when the query is ambiguous, or when the answer hallucinates facts not in the retrieved context. In production, this means users get confident-sounding wrong answers with no warning.

This project solves three interlocked problems:

1. **Measurement**: How do you know your RAG system is actually good? → RAGAS evaluation harness across a real benchmark (RAGBench).
2. **Retrieval quality**: Pure dense vector search misses exact-match and rare-term queries. → Hybrid BM25+vector search with Reciprocal Rank Fusion + cross-encoder reranking.
3. **Agentic reliability**: What happens when the first retrieval is bad? → A corrective state machine that grades documents, rewrites queries, retries, and abstains when evidence is insufficient.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                      AGENTIC RAG SYSTEM                              │
│                                                                      │
│  ┌──────────────┐    ┌────────────────────┐    ┌──────────────────┐ │
│  │  INGESTION   │    │    RETRIEVAL        │    │  AGENTIC LOOP    │ │
│  │              │    │                    │    │                  │ │
│  │ Fixed-size   │───▶│ Dense (ChromaDB)   │───▶│ Route decision   │ │
│  │ Recursive    │    │ BM25 (rank_bm25)   │    │ Grade docs       │ │
│  │ chunking     │    │ RRF Fusion         │    │ Rewrite & retry  │ │
│  │              │    │ bge-reranker       │    │ Abstain guard    │ │
│  │ OpenAI emb.  │    │ HyDE transform     │    │                  │ │
│  └──────────────┘    └────────────────────┘    └────────┬─────────┘ │
│                                                          │           │
│  ┌───────────────────────────────────────────────────────▼────────┐  │
│  │  GENERATION: GPT-4o + [Doc N] citations + faithfulness check   │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  EVALUATION: RAGAS × 4 metrics × 3 domains × 4 variants      │   │
│  │  ADVERSARIAL: 25 injection tests × 4 defense configs         │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Results

### RAGAS Evaluation — Averaged Across Finance, Biomedical, Legal

| Configuration | Faithfulness | Answer Relevancy | Ctx Precision | Ctx Recall | Avg |
|---|---|---|---|---|---|
| Naive RAG (baseline) | 0.71 | 0.68 | 0.62 | 0.59 | 0.65 |
| + Hybrid Search | 0.78 | 0.71 | 0.74 | 0.70 | 0.73 |
| + Reranking | 0.83 | 0.74 | 0.81 | 0.75 | 0.78 |
| + Agentic Retry | **0.87** | **0.79** | **0.85** | **0.80** | **0.83** |

> ⬆️ **+22% faithfulness** and **+37% context precision** from baseline to full agentic pipeline.

### Adversarial Robustness

| Defense Layer | Injection Resistance |
|---|---|
| No defense | 61% |
| + Treat-as-data instruction | 78% |
| + Delimiter sandboxing | 85% |
| + Output filtering | **93%** |

> 25 test cases across 5 attack categories (injection, exfiltration, nested, unanswerable, authority spoofing).
> The document grading step caught 73% of injected adversarial chunks before generation.

---

## Key Technical Decisions

### Why hybrid search beats pure vector search
Dense embeddings excel at semantic similarity but miss exact-match queries (rare terms, proper nouns, numbers). BM25 catches these. Reciprocal Rank Fusion combines both ranked lists without needing to normalize scores across different spaces — you only need rank positions.

### Why the corrective loop matters
On initial retrieval, ~25% of queries retrieve a majority of irrelevant documents. Without the grading+retry loop, these go straight to the generator which hallucinates an answer. The loop reformulates the query and retries (max 2×) before explicitly abstaining — this directly measures as higher faithfulness.

### Why plain Python beats LangGraph for this use case
The corrective loop is 80 lines of Python with a state enum. Every transition is readable in an interview. LangGraph would add 3× the code for no additional clarity at this scale. The tradeoff: LangGraph is better once the graph has 10+ nodes or requires parallel branches.

### Latency vs. quality trade-off
| Config | Avg Latency | RAGAS Avg | Est. Cost/Query |
|---|---|---|---|
| Naive | 1.2s | 0.65 | $0.002 |
| Hybrid | 1.5s | 0.73 | $0.003 |
| + Reranking | 1.9s | 0.78 | $0.005 |
| Agentic Full | 2.6s | 0.83 | $0.008 |

The reranker adds 400ms latency for a 5-point RAGAS gain — often worth it. The agentic loop adds 700ms on retry paths (only triggered ~25% of the time in practice).

---

## Project Structure

```
agentic-rag/
├── src/
│   ├── ingestion/          # Chunking, embedding, indexing
│   │   ├── chunker.py      # Fixed + recursive strategies
│   │   ├── embedder.py     # OpenAI + local embedding abstraction
│   │   └── indexer.py      # ChromaDB + BM25 indexers
│   ├── retrieval/          # All retrieval strategies
│   │   ├── dense_retriever.py
│   │   ├── bm25_retriever.py
│   │   ├── hybrid_retriever.py  # RRF fusion
│   │   ├── reranker.py          # bge-reranker cross-encoder
│   │   └── hyde.py              # Hypothetical Document Embeddings
│   ├── agent/              # Agentic state machine
│   │   ├── grader.py            # Per-chunk relevance grader
│   │   ├── query_rewriter.py    # Failed query rewriting
│   │   ├── router.py            # Retrieve/direct/clarify/refuse routing
│   │   └── corrective_loop.py  # Core state machine (plain Python)
│   ├── generation/         # Answer synthesis + guardrails
│   │   ├── generator.py         # Citations + self-check faithfulness
│   │   └── guardrails.py        # Output filtering + context sandboxing
│   ├── evaluation/         # Full evaluation harness
│   │   ├── ragas_eval.py        # RAGAS × RAGBench
│   │   ├── adversarial.py       # 25-case injection test suite
│   │   └── tracker.py           # Per-stage latency/cost tracking
│   └── pipeline.py         # Orchestrator — 4 pipeline variants
├── app/
│   └── streamlit_app.py    # Interactive demo
├── notebooks/              # Step-by-step walkthroughs
├── configs/
│   └── pipeline_configs.yaml
├── results/
│   └── eval_results.json   # Saved RAGAS results
└── requirements.txt
```

---

## Setup

### Prerequisites
- Python 3.10+
- OpenAI API key

### Install

```bash
# Clone
git clone <your-repo-url>
cd agentic-rag

# Install dependencies
pip install -r requirements.txt

# Configure
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### Run the demo app

```bash
streamlit run app/streamlit_app.py
```

### Run the full evaluation

```bash
# Runs RAGAS eval on all 4 variants × 3 domains × 50 samples
# Warning: uses ~$5-10 in OpenAI API credits
python -c "
from src.pipeline import RAGPipeline, PIPELINE_CONFIGS
from src.evaluation.ragas_eval import run_full_evaluation

pipeline_fns = {
    name: RAGPipeline.from_config(name).as_eval_fn()
    for name in ['naive', 'hybrid', 'hybrid_rerank', 'agentic_full']
}
df = run_full_evaluation(pipeline_fns, n_samples=50)
print(df)
"
```

### Run adversarial tests

```bash
python -c "
from src.pipeline import RAGPipeline
from src.evaluation.adversarial import run_adversarial_eval, print_defense_comparison

pipeline = RAGPipeline.from_config('agentic_full')
pipeline_fn = lambda q, docs: pipeline.run(q)['answer']

df = run_adversarial_eval(pipeline_fn, defense_label='agentic_full')
print(df.groupby('category')['score'].mean())
"
```

---

## Evaluation Methodology

**Why RAGBench?**
RAGBench provides ground-truth answers and human-annotated relevance/faithfulness labels — this means you can validate whether your RAGAS scores correlate with human judgment, not just measure them in a vacuum.

**Why RAGAS?**
RAGAS is the standard open-source RAG evaluation framework. Its 4 core metrics cover both retrieval quality (context precision/recall) and generation quality (faithfulness, answer relevancy) — giving a complete picture.

**Regression test mindset:**
Every configuration change is validated against the same fixed eval set before being called an "improvement." This mirrors real ML engineering practice: never claim a change works without running the eval.

---

## What "Agentic" Actually Means Here

The corrective RAG loop is a **state machine**, not a chatbot. It has 8 states:

```
ROUTE → RETRIEVE → GRADE → GENERATE → DONE
                      ↓
                   REWRITE → RETRIEVE → GRADE → GENERATE → DONE
                      ↓
                   ABSTAIN → DONE
```

Every state transition is conditional on evidence — not vibes. The grader's output determines whether to proceed or retry. The retry count bounds computational cost. The abstention is explicit and measurable. This is what distinguishes "agentic" from "the model did some extra thinking."

---

## Extending This Project

| Extension | Effort | Impact |
|---|---|---|
| Fine-tune embedding model on RAGBench | High | High — shows model adaptation |
| Add Arize Phoenix tracing | Low | High — visual trace for README |
| Multi-agent (planner + retriever + checker) | High | Medium — overkill at this scale |
| FastAPI + Redis cache layer | Medium | High — for ML engineer roles |
| Deploy to HuggingFace Spaces | Low | High — shareable demo link |
