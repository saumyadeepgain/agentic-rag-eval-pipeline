"""Compare two evaluation runs side by side.

    python -m src.compare_runs results/<baseline_run> results/<candidate_run> [--markdown]

Prints quality metrics (with each run's n), p50/p95 latency per stage and cost per
query for run A (baseline) and run B (candidate), with B − A and the relative
change. It warns when the runs are not comparable: different datasets, indexes,
providers or numbers of queries.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

from .schemas import EvalReport


@dataclass
class Row:
    name: str
    a: float | None
    b: float | None
    detail_a: str = ""
    detail_b: str = ""

    @property
    def delta(self) -> float | None:
        return None if self.a is None or self.b is None else self.b - self.a

    @property
    def relative(self) -> float | None:
        if self.delta is None or not self.a:
            return None
        return self.delta / abs(self.a)


def load_report(run_dir: str | Path) -> EvalReport:
    path = Path(run_dir) / "report.json"
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found; pass a run folder written by `python -m src.run_eval`")
    return EvalReport.model_validate_json(path.read_text(encoding="utf-8"))


def comparability_warnings(a: EvalReport, b: EvalReport) -> list[str]:
    warnings = []
    if a.dataset_hash != b.dataset_hash:
        warnings.append(f"different datasets ({a.dataset} vs {b.dataset}): quality deltas are not comparable")
    if a.n_queries != b.n_queries:
        warnings.append(f"different numbers of queries ({a.n_queries} vs {b.n_queries})")
    if (a.index.corpus_hash, a.index.embedding_model) != (b.index.corpus_hash, b.index.embedding_model):
        warnings.append("different indexes (corpus or embedding model)")
    if a.provider != b.provider:
        warnings.append(f"different providers ({a.provider} vs {b.provider})")
    if "offline" in (a.provider, b.provider):
        warnings.append("an offline run is involved: its scores come from heuristics, not models")
    if a.models.get("judge") != b.models.get("judge"):
        warnings.append("different judge models: judge scores are not directly comparable")
    return warnings


def build_rows(a: EvalReport, b: EvalReport) -> list[Row]:
    rows: list[Row] = []
    for name in list(dict.fromkeys([*a.metrics, *b.metrics])):
        ma, mb = a.metrics.get(name), b.metrics.get(name)
        rows.append(
            Row(
                name,
                ma.mean if ma else None,
                mb.mean if mb else None,
                f"n={ma.n}" if ma else "",
                f"n={mb.n}" if mb else "",
            )
        )
    for stage in dict.fromkeys([*a.latency, *b.latency]):
        la, lb = a.latency.get(stage), b.latency.get(stage)
        rows.append(Row(f"latency {stage} p50 ms", la.p50_ms if la else None, lb.p50_ms if lb else None))
        rows.append(Row(f"latency {stage} p95 ms", la.p95_ms if la else None, lb.p95_ms if lb else None))
    rows.append(Row("cost per query usd", a.cost.pipeline_cost_per_query_usd, b.cost.pipeline_cost_per_query_usd))
    rows.append(Row("tokens per query (prompt)", a.cost.prompt_tokens_per_query, b.cost.prompt_tokens_per_query))
    rows.append(Row("llm calls per query", a.cost.llm_calls_per_query, b.cost.llm_calls_per_query))
    return rows


def _num(value: float | None, name: str) -> str:
    if value is None:
        return "—"
    if name.startswith("cost"):
        return f"{value:.6f}"
    if name.startswith(("latency", "tokens", "llm calls")):
        return f"{value:.1f}"
    return f"{value:.3f}"


def _delta(row: Row) -> str:
    if row.delta is None:
        return "—"
    sign = "+" if row.delta >= 0 else "-"
    rel = f" ({row.relative:+.0%})" if row.relative is not None else ""
    return f"{sign}{_num(abs(row.delta), row.name)}{rel}"


def format_table(a: EvalReport, b: EvalReport, rows: list[Row], markdown: bool = False) -> str:
    header = ["metric", f"A: {a.mode}", f"B: {b.mode}", "B − A"]
    body = [
        [
            row.name,
            f"{_num(row.a, row.name)} {row.detail_a}".strip(),
            f"{_num(row.b, row.name)} {row.detail_b}".strip(),
            _delta(row),
        ]
        for row in rows
    ]
    if markdown:
        lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
        lines += ["| " + " | ".join(cells) + " |" for cells in body]
        return "\n".join(lines)
    widths = [max(len(r[i]) for r in [header, *body]) for i in range(len(header))]
    fmt = "  ".join(f"{{:<{w}}}" if i == 0 else f"{{:>{w}}}" for i, w in enumerate(widths))
    lines = [fmt.format(*header), fmt.format(*("-" * w for w in widths))]
    lines += [fmt.format(*cells) for cells in body]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare two run folders (A = baseline, B = candidate).")
    parser.add_argument("run_a")
    parser.add_argument("run_b")
    parser.add_argument("--markdown", action="store_true", help="print a Markdown table")
    args = parser.parse_args(argv)
    try:
        a, b = load_report(args.run_a), load_report(args.run_b)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"A = {a.run_id} ({a.mode}, {a.n_queries} queries)\nB = {b.run_id} ({b.mode}, {b.n_queries} queries)\n")
    print(format_table(a, b, build_rows(a, b), markdown=args.markdown))
    for warning in comparability_warnings(a, b):
        print(f"warning: {warning}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
