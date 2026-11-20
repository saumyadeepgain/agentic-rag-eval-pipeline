# Audit: agentic-rag-eval-pipeline

Bug and risk report for the code as it was at commit
[`3675094`](https://github.com/saumyadeepgain/agentic-rag-eval-pipeline/tree/3675094c5a36b4df5aef1e7c5b3f41078960cd00)
(Sep 2026). All `file:line` references point to that commit. The refactor that
follows this report replaces those files; each finding ends with where it is
fixed.

**Method.** Every file was read. Findings marked **verified** were reproduced by
running the original code with a fake embedder and fake LLMs (no API calls) or
by regenerating its data; the rest come from reading the code.

**Bottom line.** None of the numbers in the README were measured. The committed
results are random numbers, and the evaluation harness has bugs that would have
made real measurements meaningless too: questions leak into each other's
retrieval corpus, "hybrid" search never fuses, the "full" variant never uses
BM25, and failures are scored with the gold documents.

---

## P0: results are not real, or retrieval is silently wrong

**P0-1 Every reported number is synthetic (verified).**
`results/eval_results.json`, `results/adversarial_results.json` and
`results/defense_hardening.json` are byte-for-byte the output of
`scripts/mock_data.py`, which draws `random.uniform(-0.1, 0.1)` around
hand-picked base scores with `random.seed(42)` (`scripts/mock_data.py:7-23`,
`:28-38`). The README tables match neither these files nor any code:
`README.md:57-64` (RAGAS table; its headline "+22% faithfulness / +37% context
precision" is just 0.87/0.71 and 0.85/0.62 from that table), `README.md:69-73`
(injection resistance 61→93%, hard-coded in `scripts/generate_mock_data.py:58-63`),
`README.md:92-99` (latency and cost per query) and the claims at
`README.md:76` ("grading caught 73% of injected chunks") and `README.md:86`
("~25% of queries retrieve mostly irrelevant documents"). No code computes any
of them; the cost/latency tracker is never instantiated
(`src/evaluation/tracker.py:106`, only re-exported in
`src/evaluation/__init__.py:20`). Even internally the file is inconsistent: the
`average` field is not the mean of the four metrics (agentic_full/finance stores
0.806, the four metrics average 0.927). The notebook plots these files (code
cell 1), shows a hand-written "sample record" (cell 2) and a hand-written
`mock_trace` as a "live trace" (cell 7).
*Fix:* mock scripts and results deleted; every number now comes from
`src/run_eval.py` run folders that record config, git commit and per-query traces.

**P0-2 The evaluation harness leaks documents between questions (verified).**
`as_eval_fn` ingests each question's documents into one persistent, shared index
before answering (`src/pipeline.py:378-388`). Document IDs restart at
`doc_0000` on every call (`src/pipeline.py:316`), so Chroma upserts overwrite
some chunks and leave the previous question's extra chunks behind
(`src/ingestion/indexer.py:84-89`), while BM25 skips IDs it already has and keeps
the *first* question's text forever (`src/ingestion/indexer.py:194`). The run
script builds one pipeline per variant with `domain="eval"` for all three
domains (`scripts/run_eval.py:30`), and both stores persist on disk, so this
accumulates across domains and across runs. Reproduced: question 2's retrieved
context contained question 1's document ("Apple sold more phones."), and the
BM25 corpus held only question 1's text.
*Fix:* the corpus is indexed once by `src/ingestion.py`; evaluation never writes
to the index; documents have stable IDs from the corpus file.

**P0-3 Hybrid search never fuses BM25 with dense results (verified).**
Dense hits are re-labelled `"<doc>::<index>"` (`src/ingestion/indexer.py:112`,
also `src/retrieval/hyde.py:131`) while BM25 hits keep the chunker's
`"<doc>::chunk_0000"` (`src/ingestion/chunker.py:45-46`,
`src/ingestion/indexer.py:226`). RRF keys on `chunk_id`
(`src/retrieval/hybrid_retriever.py:51-52`), so the same chunk is never summed.
With the default weights 0.6/0.4 and 20 dense candidates (`src/pipeline.py:41`,
`:44`), the 20th dense-only hit (0.6/80 = 0.0075) still beats the best
BM25-only hit (0.4/61 = 0.0066), so the hybrid top-5 is exactly the dense
top-5. Reproduced: fused top-5 contained only `doc_xxxx::0` IDs.
*Fix:* one chunk ID everywhere (`DocumentChunk.chunk_id`); `rrf_fuse` in
`src/retriever.py` is unit-tested against hand-computed scores.

**P0-4 Variants share one BM25 index, so "hybrid" searches the naive chunks
(verified).** Chroma collections are per domain *and* chunker
(`src/pipeline.py:185`) but the BM25 pickle is per domain only
(`src/pipeline.py:166`), and both chunkers produce identical IDs
(`src/ingestion/chunker.py:46`). Whichever variant ingests first owns BM25.
Reproduced: after naive and hybrid ingested the same document, hybrid's BM25
corpus equalled naive's fixed-size chunks (cut mid-word: `"...10 sets the payment term"`).
*Fix:* a single index directory with a manifest; ablations change retrieval
mode only, never the chunks.

**P0-5 `agentic_full` does not use hybrid search at all.** When `hyde=True`,
`_retrieve` takes the HyDE branch, which is dense-only, and skips BM25 and
fusion entirely (`src/pipeline.py:265-270`), contradicting its own description
(`src/pipeline.py:102`) and the README diagram. HyDE samples at temperature 0.7
(`src/retrieval/hyde.py:85`), so retrieval differs run to run, and it embeds the
hypothetical *document* with the query task type (`src/retrieval/hyde.py:110`).
*Fix:* `agentic` mode uses the same hybrid + rerank retriever as
`hybrid`; only the agent loop differs (HyDE dropped).

**P0-6 Failures are scored with the gold documents as context.** If the pipeline
raises, the row gets an empty answer and `contexts = sample["documents"][:3]`,
the question's own source documents (`src/evaluation/ragas_eval.py:160-166`);
the same fallback applies when a result has no `contexts` (`:158`). Context
precision and recall are therefore inflated exactly when the system fails.
*Fix:* errors are recorded in the trace, the query keeps its real (possibly
empty) context, and failures are counted in the report.

**P0-7 The adversarial suite cannot fail (verified).** The README's runner
`lambda q, docs: pipeline.run(q)['answer']` (`README.md:196`) discards the
injected malicious chunk, so no attack reaches the model. Scoring defaults to
1.0 when no leak marker is found (`src/evaluation/adversarial.py:283-284`), and
exceptions become `"[ERROR: ...]"` answers that are scored like any other
(`:330-336`). Reproduced: a pipeline that raises on every input scores
**90% "injection resistance"** (20/25 passed). No code builds the "4 defense
configs", and the hardened system prompt is never used
(`src/generation/guardrails.py:171`, only re-exported).
*Fix:* removed rather than kept as a misleading benchmark; prompt-injection
hygiene (documents delimited as data, closing tags escaped) lives in the
generation prompt in `src/agent.py`.

**P0-8 Following the README cannot work.** The README, `.env.example` and
`configs/pipeline_configs.yaml` describe OpenAI (`OPENAI_API_KEY`, `gpt-4o`,
`text-embedding-3-small`; `.env.example:3`, `:16-22`), but the code calls Gemini
and exits without `GEMINI_API_KEY` (`src/ingestion/embedder.py:71-73`,
`quickstart.py:17`, `scripts/run_eval.py:21`). The `.env.example` embedding
model is passed straight to the Gemini client (`src/pipeline.py:169`). The app
checks the other key (`app/streamlit_app.py:383`). `requirements.txt` lacks the
SDKs the code imports (`google-genai`: `src/ingestion/embedder.py:66`;
`langchain-google-genai`: `src/evaluation/ragas_eval.py:196`) while listing
unused ones (`openai`, `tiktoken`, `langchain-text-splitters`). The YAML config
is never loaded and disagrees with the code (top-k 10 vs 20; GPT-4o vs Gemini;
50 vs 10 samples). The default `gemini-1.5-flash` (`src/pipeline.py:56-57`) has
been discontinued by Google.
*Fix:* one provider-neutral client (any OpenAI-compatible API, including Gemini's
compatibility endpoint) configured only from `config.yaml` and environment
variables; requirements pinned and tested in CI.

## P1: silently wrong answers or crashes

**P1-1 The final answer answers the rewritten query, not the user (verified).**
Generation uses `current_query` (`src/agent/corrective_loop.py:235`), which after
a retry is the rewriter's output, and each rewrite rewrites the previous rewrite
(`:217`). Reproduced: asked "Can either party end the contract early, and how?",
the generator was called with "terminate agreement notice period days".
*Fix:* `src/agent.py` always generates for the original question; rewrites are
used for retrieval only (regression test in `tests/test_agent_loops.py`).

**P1-2 No context means an answer from general knowledge.** With no hits the
generator switches to a "use your general knowledge" prompt
(`src/generation/generator.py:181-198`); the router's DIRECT route does the same
(`src/agent/corrective_loop.py:146-147`, `:170-175`), and the non-agentic path
passes empty retrievals straight to it (`src/pipeline.py:356-358`). An empty or
failed retrieval becomes a confident, ungrounded answer.
*Fix:* empty retrieval or no relevant chunk always abstains with a fixed message.

**P1-3 An empty index crashes retrieval (verified).**
`n_results=min(top_k, count)` is 0 for an empty collection and Chroma raises
`TypeError: Number of requested results 0` (`src/ingestion/indexer.py:99`,
`src/retrieval/hyde.py:118`).
*Fix:* empty index and filters that match nothing return an empty list.

**P1-4 BM25 misses most question terms (verified).** Tokenisation is
`text.lower().split()` (`src/ingestion/indexer.py:167-169`), so `"ceo?"` never
matches `"ceo."`. On the tiny per-question corpora, `BM25Okapi` gives terms that
appear in half the chunks an IDF of 0 and the `scores > 0` filter
(`src/ingestion/indexer.py:222`) then drops them; on a two-document corpus BM25
returned nothing.
*Fix:* word-character tokenisation and Lucene's always-positive IDF in
`BM25Index` (`src/retriever.py`).

**P1-5 Creating one pipeline mutates another (verified).** `from_config` edits
the shared module-level `PipelineConfig` objects in place
(`src/pipeline.py:163-169`) and components initialise lazily. Reproduced:
creating a `legal` pipeline changed the existing `finance` pipeline's BM25 path
to `./data/bm25_legal.pkl`.
*Fix:* configuration is an immutable, validated Pydantic object; overrides
produce a new copy.

**P1-6 The index does not record its embedding model.** Collections are named
by domain and chunker only (`src/pipeline.py:185`), so changing
`EMBEDDING_MODEL` queries vectors from another model (silently, if the dimension
matches). `hnsw:space: cosine` only applies when a collection is first created
(`src/ingestion/indexer.py:53-56`).
*Fix:* the index manifest stores provider, model, dimension and
normalisation; loading with a different embedding model raises
`IndexMismatchError`. Vectors are L2-normalised so dot product equals cosine.

**P1-7 The judge grades its own model's answers.** RAGAS judges with
`gemini-1.5-flash` (`src/evaluation/ragas_eval.py:197`), the same default model
that generates answers (`src/pipeline.py:56`) and runs the "self-check"
(`src/generation/generator.py:91`), which invites self-preference bias. The
judges are wired by assigning raw LangChain objects to `metric.llm`
(`:201-205`) instead of RAGAS wrappers, against an unpinned `ragas>=0.1.21`
(`requirements.txt:16`) whose API has changed several times. No judge rationale
is stored.
*Fix:* separate `models.generator` and `models.judge` with a validation warning
when they are equal; structured judges with stored reasoning.

**P1-8 The reference answers and the retrieval set are not a benchmark.**
RAGBench has no `answer` column, so `ground_truth` silently falls back to
`response` (`src/evaluation/ragas_eval.py:114`): the LLM-generated answer that
RAGBench itself annotates for hallucinations. Retrieval runs over each question's
own source documents only (`src/pipeline.py:382`), so "context recall" tests
whether a search over the gold documents finds the gold documents. Nothing is
matched by document ID, so Recall@K, MRR or NDCG cannot be computed.
*Fix:* a pooled corpus with stable `doc_id`s; eval samples list
`gold_doc_ids`; deterministic retrieval metrics on IDs; references are labelled
by origin.

**P1-9 LLM outputs are parsed with string matching, and `None` crashes the run.**
The grader asks for "yes"/"no" with `max_output_tokens=5` and accepts
`raw.startswith("yes")` (`src/agent/grader.py:78-83`); the rewriter's raw text
becomes the next query (`src/agent/query_rewriter.py:92`); the router catches
every exception and silently retrieves (`src/agent/router.py:131-137`).
`response.text` is `None` for blocked or empty responses, and `.strip()` then
raises `AttributeError`, killing the whole evaluation
(`src/agent/grader.py:82`, `src/generation/generator.py:191`, `:221`,
`src/retrieval/hyde.py:88`, `src/agent/query_rewriter.py:92`).
*Fix:* every LLM call returns JSON validated by a Pydantic schema, with one
repair retry; each call site has a defined fallback that is recorded in the trace.

**P1-10 Context is budgeted in characters, and evaluation sees altered text.**
The generator keeps 5 documents cut at 800 characters each
(`src/generation/generator.py:115-118`), mid-sentence and with no token budget.
`<retrieved_context>` tags are written into the chunk text itself
(`src/generation/guardrails.py:119-129`, `src/pipeline.py:276-279`) and that
altered text is what RAGAS scores (`src/pipeline.py:343`, `:362`), while the
prompt that explains the tags is never used (`src/generation/guardrails.py:171`).
*Fix:* token-counted packing under `budgets.max_context_tokens` and a hard
prompt budget; delimiters are added only in the prompt; traces keep the original
chunk text.

## P2: reliability, cost and reproducibility

**P2-1 No retries, timeouts or concurrency.** Every API call is synchronous with
no retry, backoff or timeout; one rate-limit error ends a run. Embedding sleeps a
fixed second per batch (`src/ingestion/embedder.py:104-111`), grading runs one
chunk at a time (`src/agent/grader.py:107-113`), and samples run one at a time
(`src/evaluation/ragas_eval.py:152-166`).
*Fix:* async client, `asyncio.Semaphore` limits, `tenacity` exponential backoff
with jitter on rate limits, timeouts and 5xx.

**P2-2 No token or cost telemetry.** Responses' usage metadata is ignored
everywhere, and the price table only knows OpenAI models and returns $0 for
anything else without warning (`src/evaluation/tracker.py:22-27`, `:42`).
*Fix:* every call records prompt/completion tokens, latency, cost and cache
status into the trace; unknown models are flagged.

**P2-3 Runs are neither deterministic nor traceable.** Temperatures 0.1
(`src/generation/generator.py:92`), 0.3 and 0.5
(`src/agent/query_rewriter.py:55`, `:112`) and 0.7 (`src/retrieval/hyde.py:85`);
no response cache; results are overwritten in place with no config or commit
recorded (`src/evaluation/ragas_eval.py:283-285`).
*Fix:* temperature 0 for every call, SQLite cache keyed by
(model, prompt hash), timestamped run folders.

**P2-4 Too few samples, no uncertainty.** The run script evaluates 10 questions
per domain (`scripts/run_eval.py:39`) while the README says 50
(`README.md:173`); no confidence intervals; `average` mixes retrieval and
generation metrics (`src/evaluation/ragas_eval.py:222`).
*Fix:* bootstrap 95% CIs and `n` reported for every metric.

**P2-5 The ablation is confounded.** `naive` changes both the chunker and the
retriever (`src/pipeline.py:73-90`), so gains credited to hybrid search include
the chunking change.
*Fix:* all modes share one index; a mode changes exactly one stage.

**P2-6 Chunking breaks words, sentences and tables.** The fixed chunker cuts at
character offsets (`src/ingestion/chunker.py:75-89`); the recursive chunker
splits FinQA tables row by row on `"\n"` without their headers (`:112`); sizes
are characters, not tokens (`:59`, `:107`); overlap disappears whenever a split
is longer than the overlap (`:170-172`).
*Fix:* token-counted, sentence-aligned chunks with sentence overlap; tables
kept whole, or split by rows with the header repeated.

**P2-7 Security hygiene.** `pickle.load` of index files executes code if a file
is replaced (`src/ingestion/indexer.py:184-185`); `trust_remote_code=True` when
loading the dataset (`src/evaluation/ragas_eval.py:102`); binary vector stores are
committed (`data/chroma_db/chroma.sqlite3`, `notebooks/data/chroma_db/chroma.sqlite3`)
because `.gitignore` has no data, index or results rules. The output filter is
a regex list (`src/generation/guardrails.py:19-37`) and abstention detection is
substring matching (`:136-164`).
*Fix:* JSON + `.npy` index files, no pickle; `.gitignore` covers indexes,
caches and results; abstention is a structured field.

**P2-8 Agent-loop edge cases.** `while state != DONE`
(`src/agent/corrective_loop.py:134`) has no hard step cap; it terminates today
only because every branch eventually reaches `DONE`. The abstention text
embeds the internal queries tried into the user-facing answer
(`:244-250`), and that text is then scored as the answer.
*Fix:* `for` loop bounded by `agent.max_steps`; abstentions use one fixed
message and record their reason in the trace.

## P3: hygiene

- p95 is computed as `sorted(x)[int(0.95 * n)]`, which is the maximum for n ≤ 20
  (`src/evaluation/tracker.py:194`).
- O(n²) duplicate check over a list (`src/ingestion/indexer.py:194`).
- Dead code: `_StepTimer`/`_time_step` (`src/agent/corrective_loop.py:113-115`,
  `:269-281`), `QueryRewriter.expand` (`src/agent/query_rewriter.py:96`),
  duplicate imports (`src/retrieval/reranker.py:64-70`).
- No tests, no CI, no pinned versions.

---

## Target structure

```
agentic-rag-eval-pipeline/
├── config.yaml              # every tunable: models, chunking, retrieval, agent caps, budgets, judge, cache
├── .env.example             # variable names only
├── pyproject.toml           # package metadata, ruff and pytest settings
├── requirements.txt         # pinned runtime dependencies
├── requirements-dev.txt     # pinned test and lint dependencies
├── Makefile                 # ingest / eval / compare / test / lint / smoke
├── README.md
├── data/
│   ├── sample_corpus.jsonl  # small pooled corpus (RAGBench finance + biomedical documents)
│   └── sample_eval.jsonl    # 5 questions with gold document IDs (one unanswerable)
├── docs/
│   └── AUDIT.md             # this report
├── scripts/
│   └── prepare_ragbench.py  # RAGBench samples → pooled corpus + eval set with gold IDs
├── src/
│   ├── schemas.py           # Pydantic contracts shared by every module
│   ├── utils.py             # config + CLI overrides, logging, token counting, SQLite cache, git info
│   ├── llm.py               # async OpenAI-compatible and offline clients: JSON outputs, retries, usage, cost
│   ├── ingestion.py         # corpus loading, sentence/table-aware chunking, index build + manifest
│   ├── retriever.py         # cosine index, BM25, RRF fusion, reranking, metadata filters
│   ├── agent.py             # router, grader, rewriter, bounded loop, budgeted generation, trace
│   ├── evaluator.py         # Recall@K, MRR, NDCG + structured LLM judges, aggregation, CIs
│   ├── run_eval.py          # benchmark CLI → results/<timestamp>_<mode>/
│   └── compare_runs.py      # side-by-side deltas between two runs
├── tests/                   # offline: mocked LLMs, zero API calls
└── .github/workflows/ci.yml
```

`src/llm.py` is the one module beyond the requested list: retries, caching,
usage accounting and the offline backend have to live behind a single client so
that every call site gets them.
