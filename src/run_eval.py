"""Benchmark CLI: run the pipeline on an evaluation set, judge every answer and
save a self-describing run folder.

    python -m src.run_eval --mode agentic                 # full pipeline (needs an API key)
    python -m src.run_eval --mode naive --top_k 5
    python -m src.run_eval --mode agentic --max_steps 3 --set retrieval.rrf_k=30
    python -m src.run_eval --provider offline             # keyless smoke test

Each run writes ``<results_dir>/<UTC timestamp>_<mode>[_<name>]/`` containing:

* ``config.resolved.yaml``: the exact configuration after presets and overrides;
* ``report.json``: aggregate metrics with 95% bootstrap CIs, latency percentiles,
  cost, git commit and dataset/index fingerprints (an ``EvalReport``);
* ``traces.jsonl``: one ``QueryEvaluation`` per question with retrieved chunk IDs,
  every agent step, the answer, judge rationales, tokens and latency;
* ``worst_cases.jsonl`` / ``worst_cases.md``: the lowest-scoring queries with
  their full traces, for error analysis;
* ``run.log``: the run's log as JSON lines.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from .agent import RAGAgent
from .evaluator import Judge, aggregate, evaluate_query, has_error, worst_cases
from .llm import LLMClient, MissingAPIKeyError
from .retriever import IndexMismatchError, Retriever
from .schemas import AppConfig, EvalReport, EvalSample, IndexManifest, QueryEvaluation
from .utils import (
    ConfigError,
    TokenCounter,
    config_hash,
    config_to_yaml,
    file_sha256,
    get_logger,
    git_info,
    load_config,
    load_env,
    log_event,
    parse_overrides,
    read_jsonl,
    run_timestamp,
    setup_logging,
    utc_now_iso,
    write_json,
    write_jsonl,
)

log = get_logger("run_eval")


def load_samples(cfg: AppConfig) -> list[EvalSample]:
    samples = read_jsonl(cfg.evaluation.dataset, EvalSample)
    ids = [s.query_id for s in samples]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise ValueError(f"{cfg.evaluation.dataset}: duplicate query_id {duplicates}")
    if cfg.evaluation.limit is not None:
        samples = samples[: cfg.evaluation.limit]
    if not samples:
        raise ValueError(f"{cfg.evaluation.dataset} contains no samples")
    return samples


def new_run_dir(cfg: AppConfig, name: str | None = None) -> Path:
    base = f"{run_timestamp()}_{cfg.run.mode}" + (f"_{name}" if name else "")
    root = Path(cfg.run.results_dir)
    candidate, suffix = root / base, 1
    while candidate.exists():
        suffix += 1
        candidate = root / f"{base}-{suffix}"
    candidate.mkdir(parents=True)
    return candidate


def build_report(
    cfg: AppConfig, run_id: str, evaluations: list[QueryEvaluation], manifest: IndexManifest
) -> EvalReport:
    metrics, latency, cost = aggregate(evaluations, cfg)
    commit, dirty = git_info()
    return EvalReport(
        run_id=run_id,
        mode=cfg.run.mode,
        provider=cfg.llm.provider,
        created_at=utc_now_iso(),
        git_commit=commit,
        git_dirty=dirty,
        config_hash=config_hash(cfg),
        dataset=cfg.evaluation.dataset,
        dataset_hash=file_sha256(cfg.evaluation.dataset)[:16],
        n_queries=len(evaluations),
        n_errors=sum(has_error(e) for e in evaluations),
        models={
            "embedding": cfg.models.embedding,
            "generator": cfg.models.generator,
            "agent": cfg.models.agent,
            "judge": cfg.models.judge,
            "reranker": cfg.models.reranker if cfg.retrieval.rerank else "none",
        },
        index=manifest,
        metrics=metrics,
        latency=latency,
        cost=cost,
        warnings=cfg.warnings(),
    )


def _fmt(value: float | None, digits: int = 3) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def format_summary(report: EvalReport, worst: list[QueryEvaluation], run_dir: Path) -> str:
    lines = [
        f"run {report.run_id}  (mode {report.mode}, provider {report.provider}, "
        f"{report.n_queries} queries, {report.n_errors} with errors)",
        "",
        f"{'metric':<22}{'mean':>8}  {'95% CI':<17}{'n':>4}",
    ]
    for name, summary in report.metrics.items():
        ci = f"[{_fmt(summary.ci_low)}, {_fmt(summary.ci_high)}]" if summary.ci_low is not None else ""
        lines.append(f"{name:<22}{_fmt(summary.mean):>8}  {ci:<17}{summary.n:>4}")
    lines += ["", f"{'latency (ms)':<22}{'p50':>8}{'p95':>10}"]
    for stage, summary in report.latency.items():
        lines.append(f"{stage:<22}{summary.p50_ms:>8.1f}{summary.p95_ms:>10.1f}")
    cost = report.cost
    lines += [
        "",
        f"cost: ${cost.pipeline_cost_per_query_usd:.5f}/query pipeline, "
        f"${cost.eval_cost_total_usd:.4f} judges in total, "
        f"${cost.billed_cost_total_usd:.4f} billed; {cost.llm_calls_per_query:.1f} LLM calls/query; "
        f"{cost.cache_hits} cache hits" + (f"; {cost.unpriced_calls} unpriced calls" if cost.unpriced_calls else ""),
    ]
    if report.provider == "offline":
        lines.append(
            "note: offline provider. Heuristic stand-ins answered and judged, so scores are not quality measurements."
        )
    for warning in report.warnings:
        lines.append(f"warning: {warning}")
    if worst:
        lines += ["", "worst cases:"]
        for e in worst:
            reason = f" ({e.trace.abstain_reason.value})" if e.trace.abstain_reason else ""
            lines.append(f"  {e.sample.query_id:<18} overall {_fmt(e.overall)}  abstained={e.trace.abstained}{reason}")
    lines += ["", f"saved to {run_dir}"]
    return "\n".join(lines)


def worst_cases_markdown(worst: list[QueryEvaluation]) -> str:
    parts = ["# Worst cases", ""]
    for e in worst:
        t = e.trace
        parts += [
            f"## {e.sample.query_id} (overall {_fmt(e.overall)})",
            "",
            f"**Question:** {e.sample.question}",
            "",
            f"**Reference:** {e.sample.reference_answer or '—'} ({e.sample.reference_source or 'n/a'})",
            "",
            f"**Answer:** {t.answer}",
            "",
            f"- abstained: {t.abstained}" + (f" ({t.abstain_reason.value})" if t.abstain_reason else ""),
            f"- gold docs: {', '.join(e.sample.gold_doc_ids) or '—'}",
            f"- context: {', '.join(r.chunk_id for r in t.context) or '—'}",
            f"- queries tried: {' → '.join(t.queries_tried) or '—'}",
            f"- steps: {' → '.join(s.kind.value for s in t.steps)}",
            "- retrieval: " + (", ".join(f"{k} {_fmt(v)}" for k, v in e.retrieval.items()) or "—"),
        ]
        for j in e.judges:
            parts.append(f"- {j.metric}: {_fmt(j.score)}: {j.reasoning or j.error or ''}")
        if t.errors:
            parts.append(f"- errors: {'; '.join(t.errors)}")
        parts.append("")
    return "\n".join(parts)


async def run_evaluation(
    cfg: AppConfig,
    *,
    run_name: str | None = None,
    llm: LLMClient | None = None,
    retriever: Retriever | None = None,
    log_to_file: bool = False,
) -> tuple[EvalReport, list[QueryEvaluation], Path]:
    """Run, judge and save one evaluation; returns the report, per-query records and run folder.

    The run folder is created only after the index, dataset and API client have
    loaded, so a run that cannot start leaves nothing behind in ``results/``."""
    own_llm = llm is None
    llm = llm or LLMClient.from_config(cfg)
    try:
        retriever = retriever or Retriever.load(cfg, llm)
        samples = load_samples(cfg)
        run_dir = new_run_dir(cfg, run_name)
        if log_to_file:
            setup_logging(cfg.run.log_level, cfg.run.log_format, log_file=run_dir / "run.log")
        counter = TokenCounter(cfg.tokens.encoding)
        agent = RAGAgent(cfg, retriever, llm, counter)
        judge = Judge(cfg, llm, counter)
        semaphore = asyncio.Semaphore(cfg.evaluation.max_concurrent_queries)
        finished = 0

        async def one(sample: EvalSample) -> QueryEvaluation:
            nonlocal finished
            async with semaphore:
                trace = await agent.run(sample)
                evaluation = await evaluate_query(sample, trace, judge, cfg)
            finished += 1
            log_event(
                log,
                "evaluated",
                query_id=sample.query_id,
                done=f"{finished}/{len(samples)}",
                overall=evaluation.overall,
            )
            return evaluation

        evaluations = list(await asyncio.gather(*(one(s) for s in samples)))
    finally:
        if own_llm:
            await llm.aclose()

    report = build_report(cfg, run_dir.name, evaluations, retriever.manifest)
    worst = worst_cases(evaluations, cfg.evaluation.worst_cases)
    (run_dir / "config.resolved.yaml").write_text(config_to_yaml(cfg), encoding="utf-8")
    write_json(run_dir / "report.json", report)
    write_jsonl(run_dir / "traces.jsonl", evaluations)
    write_jsonl(run_dir / "worst_cases.jsonl", worst)
    (run_dir / "worst_cases.md").write_text(worst_cases_markdown(worst), encoding="utf-8")
    return report, evaluations, run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run and judge the RAG pipeline on an evaluation set.")
    parser.add_argument("--config", default=None, help="YAML config (default: ./config.yaml)")
    parser.add_argument("--mode", help="preset from config.yaml `modes` (naive, hybrid, hybrid_rerank, agentic)")
    parser.add_argument("--top_k", type=int, help="retrieval.top_k")
    parser.add_argument("--max_steps", type=int, help="agent.max_steps")
    parser.add_argument("--provider", choices=["openai", "offline"], help="llm.provider")
    parser.add_argument("--dataset", help="evaluation.dataset")
    parser.add_argument("--limit", type=int, help="evaluation.limit: only the first N questions")
    parser.add_argument("--concurrency", type=int, help="evaluation.max_concurrent_queries")
    parser.add_argument("--results-dir", help="run.results_dir")
    parser.add_argument("--run-name", help="suffix for the run folder name")
    parser.add_argument("--no-cache", action="store_true", help="cache.enabled=false")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="any config override")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    load_env()
    try:
        overrides = parse_overrides(args.set)
        for key, value in (
            ("retrieval.top_k", args.top_k),
            ("agent.max_steps", args.max_steps),
            ("llm.provider", args.provider),
            ("evaluation.dataset", args.dataset),
            ("evaluation.limit", args.limit),
            ("evaluation.max_concurrent_queries", args.concurrency),
            ("run.results_dir", args.results_dir),
        ):
            if value is not None:
                overrides[key] = value
        if args.no_cache:
            overrides["cache.enabled"] = False
        cfg = load_config(args.config, mode=args.mode, overrides=overrides)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    setup_logging(cfg.run.log_level, cfg.run.log_format)
    for warning in cfg.warnings():
        log_event(log, f"config warning: {warning}", 30)

    try:
        report, evaluations, run_dir = asyncio.run(run_evaluation(cfg, run_name=args.run_name, log_to_file=True))
    except (MissingAPIKeyError, IndexMismatchError, FileNotFoundError, ValueError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        log_event(log, "run aborted", 40, error=str(exc))
        return 2
    print(format_summary(report, worst_cases(evaluations, cfg.evaluation.worst_cases), run_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
