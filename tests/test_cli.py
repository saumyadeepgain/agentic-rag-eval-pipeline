"""End to end through the command-line entry points, offline: ingest → run_eval
(two modes) → compare_runs, plus configuration overrides and error paths."""

from __future__ import annotations

import json

import pytest
from conftest import CONFIG, ROOT

from src import compare_runs, ingestion, run_eval
from src.schemas import EvalReport, QueryEvaluation
from src.utils import ConfigError, load_config

SAMPLE_CORPUS = str(ROOT / "data" / "sample_corpus.jsonl")
SAMPLE_EVAL = str(ROOT / "data" / "sample_eval.jsonl")


def common(tmp_path) -> list[str]:
    return [
        "--config", str(CONFIG),
        "--provider", "offline",
        "--set", "tokens.encoding=approx",
        "--set", f"ingestion.index_dir={tmp_path / 'index'}",
        "--set", f"cache.path={tmp_path / 'cache.sqlite'}",
        "--set", "evaluation.bootstrap_samples=100",
    ]  # fmt: skip


def eval_args(tmp_path, *extra: str) -> list[str]:
    return [*common(tmp_path), "--dataset", SAMPLE_EVAL, "--results-dir", str(tmp_path / "results"), *extra]


@pytest.fixture
def ingested(tmp_path):
    assert ingestion.main([*common(tmp_path), "--corpus", SAMPLE_CORPUS]) == 0
    return tmp_path


def test_ingest_eval_and_compare_end_to_end(ingested, capsys):
    tmp_path = ingested
    assert run_eval.main(eval_args(tmp_path, "--mode", "naive", "--top_k", "3")) == 0
    assert run_eval.main(eval_args(tmp_path, "--mode", "agentic", "--max_steps", "2", "--run-name", "trial")) == 0
    by_mode = {r.name.split("_", 1)[1]: r for r in (tmp_path / "results").iterdir()}
    assert set(by_mode) == {"naive", "agentic_trial"}  # <UTC timestamp>_<mode>[_<name>]
    runs = [by_mode["naive"], by_mode["agentic_trial"]]

    for run_dir in runs:
        assert {p.name for p in run_dir.iterdir()} == {
            "config.resolved.yaml", "report.json", "traces.jsonl", "worst_cases.jsonl", "worst_cases.md", "run.log",
        }  # fmt: skip
        report = EvalReport.model_validate_json((run_dir / "report.json").read_text())
        traces = [
            QueryEvaluation.model_validate_json(line) for line in (run_dir / "traces.jsonl").read_text().splitlines()
        ]
        assert report.n_queries == len(traces) == 5 and report.n_errors == 0
        assert report.git_commit and len(report.dataset_hash) == 16
        assert all(0.0 <= s.mean <= 1.0 for s in report.metrics.values() if s.mean is not None)
        assert len((run_dir / "worst_cases.jsonl").read_text().splitlines()) == 5
        assert all(json.loads(line)["event"] for line in (run_dir / "run.log").read_text().splitlines())

    naive_cfg = load_config(runs[0] / "config.resolved.yaml")
    agentic_cfg = load_config(runs[1] / "config.resolved.yaml")
    assert (naive_cfg.retrieval.mode, naive_cfg.retrieval.top_k, naive_cfg.agent.enabled) == ("dense", 3, False)
    assert (agentic_cfg.retrieval.mode, agentic_cfg.agent.max_steps, agentic_cfg.agent.enabled) == ("hybrid", 2, True)
    agentic_traces = [json.loads(line) for line in (runs[1] / "traces.jsonl").read_text().splitlines()]
    assert all(t["trace"]["retrieval_rounds"] <= 2 for t in agentic_traces)
    unanswerable = next(t for t in agentic_traces if not t["sample"]["answerable"])
    assert unanswerable["trace"]["abstained"] and unanswerable["abstention_correct"]

    capsys.readouterr()
    assert compare_runs.main([str(runs[0]), str(runs[1])]) == 0
    out = capsys.readouterr().out
    assert "recall@5" in out and "latency total p95 ms" in out and "cost per query usd" in out
    assert "warning: an offline run is involved" in out
    assert compare_runs.main([str(runs[0]), str(runs[1]), "--markdown"]) == 0
    assert "| metric | A: naive | B: agentic | B − A |" in capsys.readouterr().out


def test_cached_rerun_makes_no_new_calls(ingested):
    tmp_path = ingested
    for _ in range(2):
        assert run_eval.main(eval_args(tmp_path, "--mode", "agentic", "--set", "cache.enabled=true")) == 0
    first, second = sorted((tmp_path / "results").iterdir())
    a = EvalReport.model_validate_json((first / "report.json").read_text())
    b = EvalReport.model_validate_json((second / "report.json").read_text())
    assert b.cost.cache_hits > a.cost.cache_hits
    assert {k: v.mean for k, v in a.metrics.items()} == {k: v.mean for k, v in b.metrics.items()}


def test_run_eval_without_an_index_fails_cleanly(tmp_path, capsys):
    assert run_eval.main(eval_args(tmp_path)) == 2
    assert "make ingest" in capsys.readouterr().err
    assert not (tmp_path / "results").exists()  # nothing half-written


@pytest.mark.parametrize(
    "bad",
    [["--set", "retrieval.nonexistent=1"], ["--mode", "turbo"], ["--set", "agent.max_steps=0"], ["--set", "oops"]],
)
def test_invalid_configuration_is_rejected(tmp_path, capsys, bad):
    assert run_eval.main(eval_args(tmp_path, *bad)) == 2
    assert "error:" in capsys.readouterr().err


def test_openai_provider_without_a_key_explains_how_to_proceed(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(run_eval, "load_env", lambda: None)  # ignore a developer's local .env
    args = [a for a in eval_args(tmp_path) if a not in ("--provider", "offline")]
    assert run_eval.main(args) == 2
    assert "OPENAI_API_KEY" in capsys.readouterr().err


def test_mode_presets_then_overrides_take_precedence():
    cfg = load_config(CONFIG, mode="hybrid_rerank", overrides={"retrieval.rerank": False, "retrieval.top_k": 7})
    assert cfg.run.mode == "hybrid_rerank" and cfg.retrieval.mode == "hybrid"
    assert cfg.retrieval.rerank is False and cfg.retrieval.top_k == 7
    with pytest.raises(ConfigError, match="unknown mode"):
        load_config(CONFIG, mode="turbo")
    with pytest.raises(ConfigError, match="chunk_overlap_tokens"):
        load_config(CONFIG, overrides={"ingestion.chunk_overlap_tokens": 500})


def test_compare_runs_rejects_a_folder_without_a_report(tmp_path, capsys):
    assert compare_runs.main([str(tmp_path), str(tmp_path)]) == 2
    assert "report.json" in capsys.readouterr().err
