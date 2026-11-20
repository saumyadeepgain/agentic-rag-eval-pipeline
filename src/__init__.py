"""Agentic RAG with an evaluation harness.

Pipeline: ``ingestion`` (chunk + index) → ``retriever`` (dense / BM25 / RRF /
rerank) → ``agent`` (bounded retrieve-grade-rewrite loop + budgeted generation)
→ ``evaluator`` (retrieval metrics + structured LLM judges). ``run_eval`` and
``compare_runs`` are the command-line entry points; ``schemas`` holds the shared
contracts.
"""

__version__ = "1.0.0"
