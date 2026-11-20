# make help lists the targets. Every target reads config.yaml; add overrides with
# ARGS, e.g. make eval MODE=naive ARGS="--top_k 3 --set retrieval.rrf_k=30"
PYTHON ?= python
MODE ?= agentic
ARGS ?=
RESULTS ?= results
# compare: defaults to the latest naive run (A) vs the latest agentic run (B)
A ?= $(lastword $(sort $(wildcard $(RESULTS)/*_naive*)))
B ?= $(lastword $(sort $(wildcard $(RESULTS)/*_agentic*)))
OFFLINE := --provider offline --set tokens.encoding=approx \
	--set ingestion.index_dir=.cache/smoke-index --set cache.path=.cache/smoke-cache.sqlite

.PHONY: help install install-dev ingest eval eval-all compare test lint format smoke ragbench clean

help:  ## list targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-12s %s\n", $$1, $$2}'

install:  ## pinned runtime dependencies
	$(PYTHON) -m pip install -r requirements.txt

install-dev:  ## pinned runtime + test/lint dependencies
	$(PYTHON) -m pip install -r requirements.txt -r requirements-dev.txt

ingest:  ## chunk, embed and index ingestion.corpus_path (needs the API key in .env)
	$(PYTHON) -m src.ingestion $(ARGS)

eval:  ## evaluate one mode: make eval MODE=naive|hybrid|hybrid_rerank|agentic
	$(PYTHON) -m src.run_eval --mode $(MODE) $(ARGS)

eval-all:  ## the full ablation: all four modes on the same index and dataset
	for mode in naive hybrid hybrid_rerank agentic; do $(PYTHON) -m src.run_eval --mode $$mode $(ARGS) || exit 1; done

compare:  ## compare two runs: make compare A=results/<run> B=results/<run>
	@test -n "$(A)" -a -n "$(B)" || (echo "no runs to compare: run make eval first or pass A= and B="; exit 2)
	$(PYTHON) -m src.compare_runs $(A) $(B)

test:  ## offline test suite (no API keys, no network)
	$(PYTHON) -m pytest -q

lint:  ## ruff lint + format check
	ruff check .
	ruff format --check .

format:  ## apply ruff fixes and formatting
	ruff check --fix .
	ruff format .

smoke:  ## keyless end-to-end run with deterministic stand-ins (scores are NOT quality results)
	$(PYTHON) -m src.ingestion $(OFFLINE)
	$(PYTHON) -m src.run_eval --mode naive $(OFFLINE) --results-dir $(RESULTS)/smoke
	$(PYTHON) -m src.run_eval --mode agentic $(OFFLINE) --results-dir $(RESULTS)/smoke
	$(PYTHON) -m src.compare_runs $$(ls -d $(RESULTS)/smoke/*_naive* | sort | tail -1) \
		$$(ls -d $(RESULTS)/smoke/*_agentic* | sort | tail -1)

ragbench:  ## 30-question RAGBench set (finance, biomedical, legal) from data/ragbench_samples
	$(PYTHON) scripts/prepare_ragbench.py --corpus-out data/ragbench_corpus.jsonl --eval-out data/ragbench_eval.jsonl
	@echo "then: make ingest ARGS=\"--corpus data/ragbench_corpus.jsonl --index-dir .cache/ragbench-index\""

clean:  ## remove indexes, caches and results
	rm -rf .cache $(RESULTS) .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
