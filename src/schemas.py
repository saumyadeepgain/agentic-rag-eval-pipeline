"""Pydantic contracts shared by every module.

Data flows ``Document`` → ``DocumentChunk`` (ingestion) → ``RetrievalResult``
(retriever) → ``AgentTrace`` (agent) → ``QueryEvaluation`` / ``EvalReport``
(evaluator, run_eval). LLM calls return the structured-output models in the
middle section; ``AppConfig`` mirrors ``config.yaml`` one section per class.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Scalar = str | int | float | bool
JudgeMetricName = Literal["faithfulness", "answer_relevancy", "context_precision"]


# ──────────────────────────────────────────────────────────────── configuration
class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RunConfig(_Section):
    mode: str = "agentic"
    seed: int = 13
    results_dir: str = "results"
    log_level: str = "INFO"
    log_format: Literal["text", "json"] = "text"


class LLMConfig(_Section):
    provider: Literal["openai", "offline"] = "openai"
    base_url: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    response_format: Literal["json_object", "json_schema"] = "json_object"
    timeout_s: float = Field(60.0, gt=0)
    max_retries: int = Field(5, ge=1)
    backoff_initial_s: float = Field(1.0, ge=0)
    backoff_max_s: float = Field(30.0, ge=0)
    max_concurrency: int = Field(8, ge=1)
    max_parse_retries: int = Field(1, ge=0)
    seed: int | None = 13


class ModelsConfig(_Section):
    embedding: str = "text-embedding-3-small"
    generator: str = "gpt-4o-mini"
    agent: str = "gpt-4o-mini"
    judge: str = "gpt-4o"
    reranker: str = "BAAI/bge-reranker-v2-m3"


class Price(_Section):
    input: float = Field(ge=0)
    output: float = Field(0.0, ge=0)


class TokensConfig(_Section):
    encoding: str = "o200k_base"


class IngestionConfig(_Section):
    corpus_path: str = "data/sample_corpus.jsonl"
    index_dir: str = ".cache/index"
    chunk_tokens: int = Field(200, ge=16)
    chunk_overlap_tokens: int = Field(40, ge=0)
    embed_batch_size: int = Field(64, ge=1)

    @model_validator(mode="after")
    def _overlap_smaller_than_chunk(self) -> IngestionConfig:
        if self.chunk_overlap_tokens >= self.chunk_tokens:
            raise ValueError("ingestion.chunk_overlap_tokens must be smaller than chunk_tokens")
        return self


class RetrievalConfig(_Section):
    mode: Literal["dense", "bm25", "hybrid"] = "hybrid"
    top_k: int = Field(5, ge=1)
    candidates_k: int = Field(20, ge=1)
    rrf_k: int = Field(60, ge=1)
    dense_weight: float = Field(1.0, ge=0)
    bm25_weight: float = Field(1.0, ge=0)
    rerank: bool = True
    reranker: Literal["cross_encoder", "lexical"] = "cross_encoder"

    @model_validator(mode="after")
    def _enough_candidates(self) -> RetrievalConfig:
        if self.candidates_k < self.top_k:
            raise ValueError("retrieval.candidates_k must be at least retrieval.top_k")
        return self


class AgentConfig(_Section):
    enabled: bool = True
    max_steps: int = Field(3, ge=1, le=10)
    min_relevant: int = Field(2, ge=1)
    use_router: bool = True
    abstain_message: str = "I don't know based on the available documents."


class BudgetsConfig(_Section):
    max_context_tokens: int = Field(2500, ge=1)
    max_prompt_tokens: int = Field(3500, ge=1)
    max_answer_tokens: int = Field(400, ge=1)
    max_agent_tokens: int = Field(300, ge=1)
    max_judge_tokens: int = Field(800, ge=1)
    max_grade_chunk_tokens: int = Field(600, ge=1)


class EvaluationConfig(_Section):
    dataset: str = "data/sample_eval.jsonl"
    limit: int | None = Field(None, ge=1)
    k_values: list[int] = Field(default_factory=lambda: [1, 3, 5])
    judges: list[JudgeMetricName] = Field(
        default_factory=lambda: ["faithfulness", "answer_relevancy", "context_precision"]
    )
    max_concurrent_queries: int = Field(4, ge=1)
    bootstrap_samples: int = Field(1000, ge=0)
    worst_cases: int = Field(5, ge=0)


class CacheConfig(_Section):
    enabled: bool = True
    path: str = ".cache/llm_cache.sqlite"


class AppConfig(_Section):
    run: RunConfig = Field(default_factory=RunConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    pricing: dict[str, Price] = Field(default_factory=dict)
    tokens: TokensConfig = Field(default_factory=TokensConfig)
    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    budgets: BudgetsConfig = Field(default_factory=BudgetsConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    modes: dict[str, dict[str, Any]] = Field(default_factory=dict)

    def warnings(self) -> list[str]:
        """Settings that are valid but probably not what was intended."""
        out: list[str] = []
        if self.models.judge == self.models.generator:
            out.append(
                f"models.judge equals models.generator ({self.models.judge}): "
                "the judge grades its own answers (self-preference bias)"
            )
        if self.budgets.max_context_tokens >= self.budgets.max_prompt_tokens:
            out.append("budgets.max_context_tokens leaves no room for instructions in max_prompt_tokens")
        if self.llm.provider == "offline" and self.retrieval.rerank and self.retrieval.reranker == "cross_encoder":
            out.append("offline provider: the lexical reranker replaces the cross-encoder")
        for role in ("generator", "agent", "judge", "embedding"):
            model = getattr(self.models, role)
            if self.llm.provider != "offline" and model not in self.pricing:
                out.append(f"no price for models.{role} '{model}': its cost is reported as unknown")
        return out


# ───────────────────────────────────────────────────────────── corpus and index
class Document(BaseModel):
    """One line of the corpus JSONL."""

    model_config = ConfigDict(extra="forbid")

    doc_id: str = Field(min_length=1)
    text: str
    title: str = ""
    metadata: dict[str, Scalar] = Field(default_factory=dict)


class DocumentChunk(BaseModel):
    """A token-bounded slice of a document. ``chunk_id`` is ``"<doc_id>::<position:04d>"``."""

    chunk_id: str
    doc_id: str
    position: int = Field(ge=0)
    text: str
    n_tokens: int = Field(ge=0)
    start_char: int = Field(ge=0)
    end_char: int = Field(ge=0)
    content_hash: str
    metadata: dict[str, Scalar] = Field(default_factory=dict)


class IndexManifest(BaseModel):
    """Written next to the index; checked against the config at load time."""

    embedding_provider: str
    embedding_model: str
    dimension: int
    normalized: bool = True
    token_encoding: str
    chunk_tokens: int
    chunk_overlap_tokens: int
    n_documents: int
    n_chunks: int
    corpus_hash: str
    created_at: str


class RetrievalResult(BaseModel):
    chunk: DocumentChunk
    rank: int = Field(ge=1)
    score: float
    dense_score: float | None = None
    bm25_score: float | None = None
    rrf_score: float | None = None
    rerank_score: float | None = None

    @property
    def chunk_id(self) -> str:
        return self.chunk.chunk_id

    @property
    def doc_id(self) -> str:
        return self.chunk.doc_id


# ─────────────────────────────────────────────────────── telemetry and usage
class LLMUsage(BaseModel):
    """One LLM or embedding call (or cache hit)."""

    task: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float | None = 0.0  # at list price, even when served from cache; None = unpriced
    latency_ms: float = 0.0
    cached: bool = False
    attempts: int = 0  # API attempts including retries (0 for a cache hit)
    parse_retries: int = 0


class TokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    embedding_tokens: int = 0
    cost_usd: float = 0.0  # list-price cost of the work, cached or not
    billed_cost_usd: float = 0.0  # excludes cache hits
    llm_calls: int = 0
    cache_hits: int = 0
    unpriced_calls: int = 0

    def add(self, usage: LLMUsage) -> None:
        if usage.task.startswith("embed"):
            self.embedding_tokens += usage.prompt_tokens
        else:
            self.prompt_tokens += usage.prompt_tokens
            self.completion_tokens += usage.completion_tokens
        self.llm_calls += 1
        if usage.cached:
            self.cache_hits += 1
        if usage.cost_usd is None:
            self.unpriced_calls += 1
        else:
            self.cost_usd += usage.cost_usd
            if not usage.cached:
                self.billed_cost_usd += usage.cost_usd

    @classmethod
    def total(cls, usages: list[LLMUsage]) -> TokenUsage:
        out = cls()
        for usage in usages:
            out.add(usage)
        return out


class StageTimings(BaseModel):
    retrieval_ms: float = 0.0  # query embedding + dense/BM25 search + fusion
    rerank_ms: float = 0.0
    agent_ms: float = 0.0  # router, grader and rewriter calls
    generation_ms: float = 0.0
    eval_ms: float = 0.0  # judges (filled in by the evaluator)
    total_ms: float = 0.0  # end-to-end pipeline time, excluding evaluation


# ─────────────────────────────────────────────────── LLM structured outputs
class RouteDecision(BaseModel):
    reasoning: str = Field(description="One or two sentences explaining the decision.")
    action: Literal["retrieve", "refuse"]
    search_query: str = Field(description="Standalone search query for the document index.")


class GradeDecision(BaseModel):
    reasoning: str = Field(description="One sentence on what the passage does or does not contain.")
    relevant: bool


class RewriteDecision(BaseModel):
    reasoning: str = Field(description="Why the previous queries failed and what to try instead.")
    query: str


class GeneratedAnswer(BaseModel):
    answer: str
    citations: list[str] = Field(default_factory=list, description="Chunk IDs the answer relies on.")
    abstained: bool = False


class ClaimVerdict(BaseModel):
    claim: str
    supported: bool
    evidence: str | None = Field(None, description="Chunk ID that supports the claim, if any.")


class FaithfulnessJudgment(BaseModel):
    reasoning: str
    claims: list[ClaimVerdict]


class RelevancyJudgment(BaseModel):
    reasoning: str
    score: int = Field(ge=1, le=5)


class ChunkJudgment(BaseModel):
    reasoning: str
    useful: bool


# ─────────────────────────────────────────────────────────────── agent trace
class StepKind(str, Enum):
    ROUTE = "route"
    RETRIEVE = "retrieve"
    GRADE = "grade"
    REWRITE = "rewrite"
    GENERATE = "generate"
    ABSTAIN = "abstain"


class AbstainReason(str, Enum):
    EMPTY_RETRIEVAL = "empty_retrieval"
    NO_RELEVANT_CONTEXT = "no_relevant_context"
    ROUTER_REFUSED = "router_refused"
    GENERATOR_ABSTAINED = "generator_abstained"
    GENERATION_ERROR = "generation_error"
    PROMPT_OVER_BUDGET = "prompt_over_budget"
    PIPELINE_ERROR = "pipeline_error"


class AgentStep(BaseModel):
    index: int
    kind: StepKind
    query: str
    latency_ms: float = 0.0
    detail: dict[str, Any] = Field(default_factory=dict)
    usage: list[LLMUsage] = Field(default_factory=list)
    error: str | None = None


class RetrievalOutcome(BaseModel):
    results: list[RetrievalResult] = Field(default_factory=list)
    retrieval_ms: float = 0.0
    rerank_ms: float = 0.0
    usage: list[LLMUsage] = Field(default_factory=list)


class AgentTrace(BaseModel):
    query_id: str
    question: str
    mode: str
    steps: list[AgentStep] = Field(default_factory=list)
    queries_tried: list[str] = Field(default_factory=list)
    retrieval_rounds: int = 0
    hit_max_steps: bool = False
    context: list[RetrievalResult] = Field(default_factory=list)  # what the generator saw
    dropped_chunk_ids: list[str] = Field(default_factory=list)  # removed by the token budget
    context_tokens: int = 0
    prompt_tokens_estimate: int = 0
    answer: str = ""
    citations: list[str] = Field(default_factory=list)
    invalid_citations: list[str] = Field(default_factory=list)
    abstained: bool = False
    abstain_reason: AbstainReason | None = None
    timings: StageTimings = Field(default_factory=StageTimings)
    usage: TokenUsage = Field(default_factory=TokenUsage)
    errors: list[str] = Field(default_factory=list)

    def context_doc_ids(self) -> list[str]:
        """Document IDs of the context in rank order, each once."""
        seen: dict[str, None] = {}
        for result in self.context:
            seen.setdefault(result.doc_id, None)
        return list(seen)


# ──────────────────────────────────────────────────────────────── evaluation
class EvalSample(BaseModel):
    """One line of the evaluation JSONL."""

    model_config = ConfigDict(extra="forbid")

    query_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    reference_answer: str | None = None
    reference_source: str | None = None  # e.g. "human" or "ragbench_response"
    gold_doc_ids: list[str] = Field(default_factory=list)
    answerable: bool = True
    filters: dict[str, Scalar] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class JudgeScore(BaseModel):
    metric: JudgeMetricName
    score: float | None = Field(None, ge=0.0, le=1.0)  # None = not applicable or judge failed
    reasoning: str = ""
    details: dict[str, Any] = Field(default_factory=dict)
    judge_model: str = ""
    error: str | None = None
    usage: list[LLMUsage] = Field(default_factory=list)


class QueryEvaluation(BaseModel):
    """One line of a run's ``traces.jsonl``."""

    sample: EvalSample
    trace: AgentTrace
    retrieval: dict[str, float | None] = Field(default_factory=dict)
    judges: list[JudgeScore] = Field(default_factory=list)
    abstention_correct: bool
    overall: float | None = None
    eval_usage: TokenUsage = Field(default_factory=TokenUsage)

    def judge(self, metric: str) -> JudgeScore | None:
        return next((j for j in self.judges if j.metric == metric), None)


class MetricSummary(BaseModel):
    mean: float | None
    ci_low: float | None = None
    ci_high: float | None = None
    n: int


class LatencySummary(BaseModel):
    p50_ms: float
    p95_ms: float
    mean_ms: float


class CostSummary(BaseModel):
    pipeline_cost_per_query_usd: float
    pipeline_cost_total_usd: float
    eval_cost_total_usd: float
    billed_cost_total_usd: float
    prompt_tokens_per_query: float
    completion_tokens_per_query: float
    llm_calls_per_query: float
    cache_hits: int
    unpriced_calls: int


class EvalReport(BaseModel):
    """A run's ``report.json``."""

    run_id: str
    mode: str
    provider: str
    created_at: str
    git_commit: str | None
    git_dirty: bool | None
    config_hash: str
    dataset: str
    dataset_hash: str
    n_queries: int
    n_errors: int
    models: dict[str, str]
    index: IndexManifest
    metrics: dict[str, MetricSummary]
    latency: dict[str, LatencySummary]
    cost: CostSummary
    warnings: list[str] = Field(default_factory=list)
