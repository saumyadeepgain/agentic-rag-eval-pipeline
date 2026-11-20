"""Judges with mocked replies (valid JSON, malformed JSON that needs a repair,
persistent garbage, refusals, out-of-range ratings) and the aggregation: every
reported metric must stay finite and within [0, 1]."""

from __future__ import annotations

import math

import pytest
from conftest import ScriptedChat, client, make_cfg, result, run, sample

from src.evaluator import Judge, aggregate, bootstrap_ci, evaluate_query, summarize, worst_cases
from src.schemas import AbstainReason, AgentTrace, QueryEvaluation

CONTEXT = [
    result("acme-1::0000", "ACME revenue in 2020 was 5 million dollars.", rank=1),
    result("weather::0000", "It rained on Tuesday.", rank=2),
    result("acme-1::0001", "Revenue grew 10 percent.", rank=3),
]
REFUSAL = "I'm sorry, but I can't help with that request."


def faithfulness_reply(*supported: bool):
    return {
        "reasoning": "checked each claim",
        "claims": [{"claim": f"claim {i}", "supported": s, "evidence": None} for i, s in enumerate(supported)],
    }


def make_judge(tmp_path, replies=None, defaults=None):
    cfg = make_cfg(tmp_path)
    chat = ScriptedChat(replies=replies, defaults=defaults)
    return Judge(cfg, client(cfg, chat)), chat, cfg


def answered_trace(answer: str = "ACME revenue was 5 million dollars.", context=CONTEXT) -> AgentTrace:
    return AgentTrace(
        query_id="q1", question="What was ACME revenue in 2020?", mode="agentic", answer=answer, context=list(context)
    )


def abstained_trace() -> AgentTrace:
    return AgentTrace(
        query_id="q1",
        question="What was ACME revenue in 2020?",
        mode="agentic",
        answer="I don't know based on the available documents.",
        abstained=True,
        abstain_reason=AbstainReason.NO_RELEVANT_CONTEXT,
    )


# ──────────────────────────────────────────────────────────────── single judges
def test_valid_faithfulness_reply(tmp_path):
    judge, chat, _ = make_judge(tmp_path, replies={"judge_faithfulness": [faithfulness_reply(True, True, False)]})
    score = run(judge.faithfulness("q", "a", CONTEXT))
    assert score.score == pytest.approx(2 / 3) and score.error is None
    assert score.details["supported"] == 2 and score.reasoning == "checked each claim"
    assert chat.requests[0].temperature == 0.0
    # the schema instruction asks for reasoning before the verdict
    assert "reasoning first" in chat.requests[0].messages[0]["content"]


def test_malformed_judge_reply_is_repaired(tmp_path):
    judge, _, _ = make_judge(tmp_path, replies={"judge_faithfulness": ["{'claims': oops", faithfulness_reply(True)]})
    score = run(judge.faithfulness("q", "a", CONTEXT))
    assert score.score == 1.0 and score.usage[0].parse_retries == 1


@pytest.mark.parametrize("reply", ["{not json", REFUSAL, ""])
def test_unusable_judge_replies_give_none_not_a_number(tmp_path, reply):
    judge, _, _ = make_judge(tmp_path, defaults={"judge_faithfulness": lambda r: reply})
    score = run(judge.faithfulness("q", "a", CONTEXT))
    assert score.score is None and "StructuredOutputError" in score.error


def test_faithfulness_without_claims_is_not_scored(tmp_path):
    judge, _, _ = make_judge(tmp_path, replies={"judge_faithfulness": [faithfulness_reply()]})
    score = run(judge.faithfulness("q", "a", CONTEXT))
    assert score.score is None and "no claims" in score.error


@pytest.mark.parametrize(("rating", "expected"), [(1, 0.0), (3, 0.5), (5, 1.0)])
def test_relevancy_rating_maps_to_unit_interval(tmp_path, rating, expected):
    judge, _, _ = make_judge(tmp_path, replies={"judge_relevancy": [{"reasoning": "r", "score": rating}]})
    assert run(judge.answer_relevancy("q", "a")).score == expected


def test_out_of_range_rating_is_rejected(tmp_path):
    judge, _, _ = make_judge(tmp_path, defaults={"judge_relevancy": lambda r: {"reasoning": "r", "score": 7}})
    score = run(judge.answer_relevancy("q", "a"))
    assert score.score is None and "score" in score.error


def test_context_precision_is_average_precision_of_useful_chunks(tmp_path):
    judge, chat, _ = make_judge(
        tmp_path,
        defaults={
            "judge_context": lambda r: {
                "reasoning": "r",
                "useful": "ACME" in r.payload["chunk_text"] or "grew" in r.payload["chunk_text"],
            }
        },
    )
    score = run(judge.context_precision("q", "ACME made 5M", CONTEXT))
    assert score.score == pytest.approx((1 + 2 / 3) / 2)  # useful at ranks 1 and 3
    assert len(chat.calls("judge_context")) == 3  # one independent call per chunk
    assert [v["useful"] for v in score.details["verdicts"]] == [True, False, True]


def test_context_precision_with_a_failed_chunk_is_not_scored(tmp_path):
    judge, _, _ = make_judge(
        tmp_path,
        defaults={
            "judge_context": lambda r: (
                REFUSAL if "rained" in r.payload["chunk_text"] else {"reasoning": "r", "useful": True}
            )
        },
    )
    score = run(judge.context_precision("q", None, CONTEXT))
    assert score.score is None and "weather::0000" in score.error


# ──────────────────────────────────────────────────────────────── per-query edge cases
def test_unanswerable_question_correctly_abstained(tmp_path):
    judge, chat, cfg = make_judge(tmp_path)
    evaluation = run(evaluate_query(sample(answerable=False), abstained_trace(), judge, cfg))
    assert evaluation.abstention_correct and evaluation.overall == 1.0
    assert all(j.score is None for j in evaluation.judges) and evaluation.retrieval == {}
    assert chat.requests == []  # nothing to judge, no API calls


def test_unanswerable_question_answered_anyway(tmp_path):
    judge, _, cfg = make_judge(
        tmp_path,
        defaults={
            "judge_faithfulness": lambda r: faithfulness_reply(False),
            "judge_relevancy": lambda r: {"reasoning": "r", "score": 4},
        },
    )
    evaluation = run(evaluate_query(sample(answerable=False), answered_trace(), judge, cfg))
    assert not evaluation.abstention_correct and evaluation.overall == 0.0


def test_abstaining_on_an_answerable_question(tmp_path):
    judge, chat, cfg = make_judge(tmp_path)
    evaluation = run(evaluate_query(sample(gold_doc_ids=["acme-1"]), abstained_trace(), judge, cfg))
    assert not evaluation.abstention_correct and evaluation.overall == 0.0
    assert evaluation.judge("faithfulness").score is None
    assert evaluation.judge("answer_relevancy").score == 0.0
    assert evaluation.judge("context_precision").score == 0.0  # empty context
    assert evaluation.retrieval["recall@5"] == 0.0 and chat.requests == []


def test_full_evaluation_of_an_answer(tmp_path):
    judge, _, cfg = make_judge(
        tmp_path,
        defaults={
            "judge_faithfulness": lambda r: faithfulness_reply(True, True),
            "judge_relevancy": lambda r: {"reasoning": "r", "score": 5},
            "judge_context": lambda r: {"reasoning": "r", "useful": True},
        },
    )
    evaluation = run(evaluate_query(sample(gold_doc_ids=["acme-1"]), answered_trace(), judge, cfg))
    assert [j.metric for j in evaluation.judges] == ["faithfulness", "answer_relevancy", "context_precision"]
    assert [j.score for j in evaluation.judges] == [1.0, 1.0, 1.0]
    assert evaluation.retrieval["mrr"] == 1.0 and evaluation.overall == 1.0
    assert evaluation.eval_usage.llm_calls == 5 and evaluation.trace.timings.eval_ms > 0


# ──────────────────────────────────────────────────────────────── aggregation
def _evaluations(tmp_path) -> list[QueryEvaluation]:
    replies = iter([faithfulness_reply(True, False), "{bad", "{bad", faithfulness_reply(True)])
    judge, _, cfg = make_judge(
        tmp_path,
        defaults={
            "judge_faithfulness": lambda r: next(replies),
            "judge_relevancy": lambda r: REFUSAL,
            "judge_context": lambda r: {"reasoning": "r", "useful": "ACME" in r.payload["chunk_text"]},
        },
    )
    cases = [
        (sample(query_id="a", gold_doc_ids=["acme-1"]), answered_trace()),
        (sample(query_id="b", gold_doc_ids=["acme-1", "other"]), answered_trace(context=CONTEXT[1:])),
        (sample(query_id="c", gold_doc_ids=["acme-1"]), answered_trace()),
        (sample(query_id="d", answerable=False), abstained_trace()),
    ]
    return [run(evaluate_query(s, t, judge, cfg)) for s, t in cases], cfg


def test_aggregates_stay_finite_and_bounded_with_failures_and_refusals(tmp_path):
    evaluations, cfg = _evaluations(tmp_path)
    metrics, latency, cost = aggregate(evaluations, cfg)
    for name, summary in metrics.items():
        if summary.mean is None:
            assert summary.n == 0
            continue
        assert math.isfinite(summary.mean) and 0.0 <= summary.mean <= 1.0, name
        if summary.ci_low is not None:
            assert 0.0 <= summary.ci_low <= summary.mean <= summary.ci_high <= 1.0, name
    assert metrics["answer_relevancy"].n == 0  # every relevancy reply was a refusal
    assert metrics["faithfulness"].n == 2  # one judge failed, one query abstained
    assert metrics["abstention_accuracy"].mean == 1.0 and metrics["abstention_accuracy"].n == 4
    assert all(v.p50_ms >= 0 and v.p95_ms >= v.p50_ms for v in latency.values())
    assert cost.cache_hits == 0 and cost.pipeline_cost_per_query_usd == 0.0


def test_worst_cases_put_unscorable_queries_first(tmp_path):
    evaluations, _ = _evaluations(tmp_path)
    order = [e.sample.query_id for e in worst_cases(evaluations, 4)]
    assert order[-1] == "d"  # the correct abstention is the best case
    assert len(worst_cases(evaluations, 2)) == 2


def test_bootstrap_is_deterministic_and_bounded():
    values = [0.0, 0.5, 1.0, 1.0, 0.25]
    assert bootstrap_ci(values, 500, seed=1) == bootstrap_ci(values, 500, seed=1)
    low, high = bootstrap_ci(values, 500, seed=1)
    assert 0.0 <= low <= sum(values) / len(values) <= high <= 1.0
    assert bootstrap_ci([0.7], 500, seed=1) == (None, None)
    assert bootstrap_ci(values, 0, seed=1) == (None, None)


def test_summarize_ignores_missing_and_non_finite_values():
    summary = summarize([1.0, None, float("nan"), 0.0], 100, seed=0)
    assert summary.n == 2 and summary.mean == 0.5
    assert summarize([None, None], 100, seed=0).mean is None
