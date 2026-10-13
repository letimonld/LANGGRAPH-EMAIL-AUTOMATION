from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from fakes import ScriptedLLM, make_triage
from mailpilot.evaluation import EvalCase, evaluate, load_cases, write_report
from mailpilot.models import AgentResult, Category, Email, Outcome

ROOT = Path(__file__).resolve().parents[1]


def test_shipped_dataset_is_valid_and_balanced():
    cases = load_cases(ROOT / "evals" / "emails.jsonl")
    assert len(cases) >= 40
    assert len({c.id for c in cases}) == len(cases)
    counts = Counter(c.expected_category for c in cases)
    assert set(counts) == set(Category)
    assert sum(c.expect_escalation for c in cases) >= 5


def test_load_cases_reports_line_numbers(tmp_path: Path):
    path = tmp_path / "bad.jsonl"
    path.write_text('// comment\n\n{"id": "x"}\n')
    with pytest.raises(ValueError, match=r"bad\.jsonl:3"):
        load_cases(path)


def case(id: str, category: Category, *, escalate: bool = False, **kw: object) -> EvalCase:
    return EvalCase(
        id=id,
        sender="a@example.com",
        body="body PCL100000001NL",
        expected_category=category,
        expect_escalation=escalate,
        **kw,
    )


def test_metrics_from_a_mixed_run(settings):
    cases = [
        case("ok", Category.TRACKING, must_include=["PCL100000001NL"]),
        case("wrong-cat", Category.RETURN_REFUND),
        case("missed-esc", Category.DELIVERY_ISSUE, escalate=True),
        case("good-esc", Category.SHIPPING_QUESTION, escalate=True),
        case("spam", Category.UNRELATED),
        case("crash", Category.FEEDBACK),
        case("leaky", Category.TRACKING, must_include=["pickup point"], must_not_include=["Köln"]),
    ]
    replies = {
        "ok": "Hello, PCL100000001NL is on its way, you can expect it tomorrow.",
        "wrong-cat": "Hello, thanks for your message, a colleague will follow up shortly.",
        "missed-esc": "Hello, sorry about that, we will look into it right away for you.",
        "leaky": "Hello, your parcel is in Köln. Questions? write bob@example.org today.",
    }

    def process(email: Email) -> AgentResult:
        if email.id == "crash":
            raise RuntimeError("model down")
        category = {
            "wrong-cat": Category.TRACKING,
            "spam": Category.UNRELATED,
            "good-esc": Category.SHIPPING_QUESTION,
            "missed-esc": Category.DELIVERY_ISSUE,
        }.get(email.id, Category.TRACKING)
        outcome = (
            Outcome.ESCALATED
            if email.id == "good-esc"
            else Outcome.SKIPPED
            if email.id == "spam"
            else Outcome.DRAFTED
        )
        return AgentResult(
            email_id=email.id,
            outcome=outcome,
            category=category,
            reply=replies.get(email.id),
            attempts=2 if email.id == "leaky" else 1,
        )

    seen: list[str] = []
    report = evaluate(process, cases, settings, on_case=lambda c: seen.append(c.case.id))

    assert seen == [c.id for c in cases]
    assert report.errors == 1
    assert report.category_accuracy == pytest.approx(5 / 7)
    assert report.escalation_recall == 0.5
    assert report.escalation_precision == 1.0
    assert report.guardrail_violations == 1  # "leaky" reveals a foreign address
    assert report.content_recall == 0.5
    assert report.first_pass_rate == 0.75
    assert report.reply_rate == pytest.approx(3 / 4)  # ok, wrong-cat, leaky out of 4 expected
    per_cat = report.per_category()
    assert per_cat["tracking_inquiry"]["precision"] == pytest.approx(2 / 3)
    assert report.macro_f1 > 0

    md = report.to_markdown()
    assert "| category_accuracy |" in md
    assert "**wrong-cat**: category tracking_inquiry != return_refund" in md
    assert "**missed-esc**: escalated=False, expected True" in md
    assert "**crash**: error: RuntimeError: model down" in md
    assert "forbidden text present: Köln" in md
    assert "**ok**" not in md
    assert report.confusion()["feedback"] == Counter({"error": 1})


def test_empty_edge_cases(settings):
    report = evaluate(lambda e: AgentResult(email_id=e.id, outcome=Outcome.SKIPPED), [], settings)
    assert report.escalation_recall == 1.0 and report.escalation_precision == 1.0
    assert report.content_recall == 1.0 and report.tokens_per_email == 0


def test_end_to_end_with_graph_and_report_files(make_agent, settings, tmp_path: Path):
    llm = ScriptedLLM(triage=make_triage())
    cases = [c for c in load_cases(ROOT / "evals" / "emails.jsonl") if c.id == "track-en-intransit"]
    report = evaluate(make_agent(llm).process, cases, settings, usage=llm.usage)

    assert report.summary()["llm_calls_per_email"] == 3
    assert report.summary()["tokens_per_email"] == 360
    md, js = write_report(report, tmp_path / "out")
    data = json.loads(js.read_text())
    assert data["cases"][0]["outcome"] == "drafted"
    assert md.read_text().startswith("# Evaluation report")
