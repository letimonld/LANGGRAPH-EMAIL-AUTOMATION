from __future__ import annotations

import threading
from datetime import datetime, timedelta, timezone

from fakes import MemoryMailbox, ScriptedLLM, make_email, make_triage
from mailpilot.llm import LLMTransientError, Usage
from mailpilot.metrics import MetricsStore, percentile
from mailpilot.models import AgentResult, Category, Email, Outcome
from mailpilot.runner import Runner


class ByIdAgent:
    """Agent stub whose behaviour is chosen per email id."""

    def __init__(self, outcomes: dict[str, Outcome | Exception]) -> None:
        self.outcomes = outcomes
        self.seen: list[str] = []

    def process(self, email: Email) -> AgentResult:
        self.seen.append(email.id)
        outcome = self.outcomes[email.id]
        if isinstance(outcome, Exception):
            raise outcome
        reply = "reply" if outcome in (Outcome.DRAFTED, Outcome.SENT) else None
        return AgentResult(
            email_id=email.id,
            outcome=outcome,
            category=Category.TRACKING,
            confidence=0.9,
            reply=reply,
            attempts=1,
        )


def emails(*ids: str, **kw: object) -> list[Email]:
    return [make_email(id=i, thread_id=i, **kw) for i in ids]


def test_runner_applies_outcomes_and_is_idempotent(tmp_path):
    box = MemoryMailbox(emails("a", "b", "c", "d"))
    agent = ByIdAgent(
        {"a": Outcome.DRAFTED, "b": Outcome.SENT, "c": Outcome.ESCALATED, "d": Outcome.SKIPPED}
    )
    metrics = MetricsStore(tmp_path / "m.sqlite3")
    runner = Runner(agent, box, metrics=metrics)

    report = runner.run_once()
    assert report.fetched == 4 and not report.errors
    assert report.outcomes == {"drafted": 1, "sent": 1, "escalated": 1, "skipped": 1}
    assert box.drafts == {"a": "reply"} and box.sent == {"b": "reply"}
    assert box.marks["c"] is Outcome.ESCALATED

    again = runner.run_once()
    assert again.fetched == 0
    assert agent.seen == ["a", "b", "c", "d"]
    with metrics:
        assert metrics.summary().total == 4


def test_one_bad_email_does_not_stop_the_batch():
    box = MemoryMailbox(emails("a", "b"))
    runner = Runner(ByIdAgent({"a": RuntimeError("boom"), "b": Outcome.DRAFTED}), box)
    report = runner.run_once()

    assert report.outcomes == {"failed": 1, "drafted": 1}
    assert box.marks == {"a": Outcome.FAILED, "b": Outcome.DRAFTED}
    assert report.errors == ["a: RuntimeError: boom"]


def test_rate_limit_defers_the_rest_without_marking():
    box = MemoryMailbox(emails("a", "b", "c"))
    agent = ByIdAgent({"a": Outcome.DRAFTED, "b": LLMTransientError("429"), "c": Outcome.DRAFTED})
    report = Runner(agent, box).run_once()

    assert report.deferred == 2
    assert box.marks == {"a": Outcome.DRAFTED}
    assert agent.seen == ["a", "b"]
    assert box.fetch_unprocessed(10)[0].id == "b"  # retried on the next run


def test_auto_generated_mail_never_reaches_the_llm():
    box = MemoryMailbox(emails("bulk", auto_generated=True))
    agent = ByIdAgent({})
    report = Runner(agent, box).run_once()

    assert report.outcomes == {"skipped": 1}
    assert agent.seen == []


def test_label_failure_is_reported_not_raised():
    box = MemoryMailbox(emails("a"), fail_mark=True)
    report = Runner(ByIdAgent({"a": Outcome.DRAFTED}), box).run_once()

    assert box.drafts == {"a": "reply"}
    assert report.errors == ["a: label failed (label API down)"]


def test_time_budget_defers_remaining():
    box = MemoryMailbox(emails("a", "b"))
    report = Runner(ByIdAgent({"a": Outcome.DRAFTED}), box).run_once(time_budget_sec=-1)
    assert report.deferred == 2 and report.outcomes == {}


def test_watch_polls_until_stopped():
    box = MemoryMailbox(emails("a"))
    runner = Runner(ByIdAgent({"a": Outcome.DRAFTED}), box)
    assert runner.watch(0, max_runs=3) == 3
    stop = threading.Event()
    stop.set()
    assert runner.watch(0, stop=stop) == 0


def test_runner_with_real_graph_records_tokens(make_agent, tmp_path):
    llm = ScriptedLLM()
    metrics = MetricsStore(tmp_path / "m.sqlite3")
    box = MemoryMailbox([make_email()])
    Runner(make_agent(llm), box, metrics=metrics, usage=llm.usage).run_once()

    with metrics:
        summary = metrics.summary()
    assert summary.by_outcome == {"drafted": 1}
    assert summary.input_tokens == 300 and summary.output_tokens == 60  # 3 calls
    assert box.drafts["m1"].endswith("The Parcelo Support Team")


def test_runner_with_graph_skips_spam(make_agent):
    llm = ScriptedLLM(triage=make_triage(category=Category.UNRELATED))
    box = MemoryMailbox([make_email()])
    Runner(make_agent(llm), box).run_once()
    assert box.marks == {"m1": Outcome.SKIPPED} and not box.drafts


# ------------------------------------------------------------------ metrics


def result(outcome: Outcome, category: Category | None = Category.TRACKING) -> AgentResult:
    return AgentResult(email_id="x", outcome=outcome, category=category, attempts=1)


def test_summary_numbers():
    store = MetricsStore(":memory:")
    assert store.summary().total == 0
    for outcome, latency in [
        (Outcome.DRAFTED, 1000),
        (Outcome.DRAFTED, 3000),
        (Outcome.SENT, 2000),
        (Outcome.ESCALATED, 500),
        (Outcome.SKIPPED, 100),
    ]:
        store.record(result(outcome), latency_ms=latency, usage=Usage(3, 1000, 200))
    summary = store.summary(minutes_per_draft=5)

    assert summary.total == 5 and summary.handled == 4
    assert summary.automation_rate == 0.75
    assert summary.minutes_saved_estimate == 15
    assert summary.p95_latency_ms == 3000
    assert summary.input_tokens == 5000
    data = summary.to_dict()
    assert data["automation_rate"] == 0.75 and data["by_category"] == {"tracking_inquiry": 5}

    future = datetime.now(timezone.utc) + timedelta(days=1)
    assert store.summary(since=future).total == 0
    store.close()


def test_percentile_and_empty_rates():
    assert percentile([], 95) == 0.0
    assert percentile([1, 2, 3, 4], 50) in (2, 3)
    from mailpilot.metrics import Summary

    assert Summary().automation_rate == 0.0
