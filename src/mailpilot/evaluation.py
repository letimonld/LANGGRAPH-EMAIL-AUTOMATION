"""Offline evaluation on a labelled set of emails.

Answers the questions you get before going live: how often is the category right, do the
emails that must reach a human actually escalate, how many replies pass review on the
first try, does any final reply break a guardrail, and what does it cost per email.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from mailpilot.config import Settings
from mailpilot.guardrails import ReplyPolicy, check_reply
from mailpilot.llm import Usage
from mailpilot.metrics import percentile
from mailpilot.models import AgentResult, Category, Email, Outcome
from mailpilot.tracking import extract_tracking_numbers


class EvalCase(BaseModel):
    id: str
    sender: str
    sender_name: str = ""
    subject: str = ""
    body: str
    expected_category: Category
    expect_escalation: bool = False
    must_include: list[str] = Field(default_factory=list)
    must_not_include: list[str] = Field(default_factory=list)

    def to_email(self) -> Email:
        return Email(
            id=self.id,
            thread_id=self.id,
            sender=self.sender,
            sender_name=self.sender_name,
            subject=self.subject,
            body=self.body,
        )


def load_cases(path: Path) -> list[EvalCase]:
    cases = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip() and not line.lstrip().startswith("//"):
            try:
                cases.append(EvalCase.model_validate_json(line))
            except ValueError as exc:
                raise ValueError(f"{path}:{number}: {exc}") from exc
    return cases


@dataclass
class CaseResult:
    case: EvalCase
    result: AgentResult | None
    latency_ms: float
    usage: Usage
    error: str | None = None
    guardrail_issues: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    forbidden: list[str] = field(default_factory=list)

    @property
    def predicted(self) -> Category | None:
        return self.result.category if self.result else None

    @property
    def escalated(self) -> bool:
        return bool(self.result and self.result.outcome is Outcome.ESCALATED)


@dataclass
class EvalReport:
    cases: list[CaseResult]

    # ------------------------------------------------------------------ metrics
    @property
    def n(self) -> int:
        return len(self.cases)

    @property
    def errors(self) -> int:
        return sum(1 for c in self.cases if c.error)

    @property
    def category_accuracy(self) -> float:
        return _ratio(sum(c.predicted == c.case.expected_category for c in self.cases), self.n)

    def per_category(self) -> dict[str, dict[str, float]]:
        stats: dict[str, dict[str, float]] = {}
        for category in Category:
            tp = sum(
                c.predicted == category and c.case.expected_category == category for c in self.cases
            )
            fp = sum(
                c.predicted == category and c.case.expected_category != category for c in self.cases
            )
            fn = sum(
                c.predicted != category and c.case.expected_category == category for c in self.cases
            )
            support = tp + fn
            if support == 0 and fp == 0:
                continue
            precision, recall = _ratio(tp, tp + fp), _ratio(tp, support)
            f1 = _ratio(2 * precision * recall, precision + recall)
            stats[category.value] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
            }
        return stats

    @property
    def macro_f1(self) -> float:
        rows = [r for r in self.per_category().values() if r["support"]]
        return _ratio(sum(r["f1"] for r in rows), len(rows))

    @property
    def escalation_recall(self) -> float:
        """Of the emails that MUST reach a human, how many did. The safety metric."""
        must = [c for c in self.cases if c.case.expect_escalation]
        return _ratio(sum(c.escalated for c in must), len(must)) if must else 1.0

    @property
    def escalation_precision(self) -> float:
        escalated = [c for c in self.cases if c.escalated]
        if not escalated:
            return 1.0
        return _ratio(sum(c.case.expect_escalation for c in escalated), len(escalated))

    @property
    def replies(self) -> list[CaseResult]:
        return [c for c in self.cases if c.result and c.result.reply]

    @property
    def reply_rate(self) -> float:
        """Share of emails that should get a reply and did."""
        should = [
            c
            for c in self.cases
            if not c.case.expect_escalation and c.case.expected_category is not Category.UNRELATED
        ]
        return _ratio(sum(bool(c.result and c.result.reply) for c in should), len(should))

    @property
    def first_pass_rate(self) -> float:
        replies = self.replies
        return _ratio(sum(c.result.attempts == 1 for c in replies if c.result), len(replies))

    @property
    def guardrail_violations(self) -> int:
        return sum(1 for c in self.replies if c.guardrail_issues)

    @property
    def content_recall(self) -> float:
        expected = sum(len(c.case.must_include) for c in self.replies)
        missing = sum(len(c.missing) for c in self.replies)
        return _ratio(expected - missing, expected) if expected else 1.0

    def latency(self, pct: float) -> float:
        return percentile([c.latency_ms for c in self.cases], pct)

    @property
    def tokens_per_email(self) -> float:
        total = sum(c.usage.input_tokens + c.usage.output_tokens for c in self.cases)
        return _ratio(total, self.n)

    @property
    def llm_calls_per_email(self) -> float:
        return _ratio(sum(c.usage.calls for c in self.cases), self.n)

    # ------------------------------------------------------------------ output
    def summary(self) -> dict[str, Any]:
        return {
            "cases": self.n,
            "errors": self.errors,
            "category_accuracy": round(self.category_accuracy, 3),
            "macro_f1": round(self.macro_f1, 3),
            "escalation_recall": round(self.escalation_recall, 3),
            "escalation_precision": round(self.escalation_precision, 3),
            "reply_rate": round(self.reply_rate, 3),
            "first_pass_review_rate": round(self.first_pass_rate, 3),
            "final_replies_with_guardrail_violations": self.guardrail_violations,
            "required_content_recall": round(self.content_recall, 3),
            "latency_p50_ms": round(self.latency(50)),
            "latency_p95_ms": round(self.latency(95)),
            "llm_calls_per_email": round(self.llm_calls_per_email, 2),
            "tokens_per_email": round(self.tokens_per_email),
        }

    def confusion(self) -> dict[str, Counter[str]]:
        matrix: dict[str, Counter[str]] = {}
        for c in self.cases:
            row = matrix.setdefault(c.case.expected_category.value, Counter())
            row[c.predicted.value if c.predicted else "error"] += 1
        return matrix

    def to_markdown(self) -> str:
        lines = ["# Evaluation report", "", "| Metric | Value |", "|---|---|"]
        lines += [f"| {k} | {v} |" for k, v in self.summary().items()]
        lines += ["", "## Per category", "", "| Category | Precision | Recall | F1 | Support |"]
        lines += ["|---|---|---|---|---|"]
        for name, row in self.per_category().items():
            lines.append(
                f"| {name} | {row['precision']:.2f} | {row['recall']:.2f} | {row['f1']:.2f} "
                f"| {int(row['support'])} |"
            )
        failures = [
            c
            for c in self.cases
            if c.error
            or c.predicted != c.case.expected_category
            or c.escalated != c.case.expect_escalation
            or c.guardrail_issues
            or c.missing
            or c.forbidden
        ]
        if failures:
            lines += ["", "## Cases to look at", ""]
            for c in failures:
                notes = []
                if c.error:
                    notes.append(f"error: {c.error}")
                if c.predicted != c.case.expected_category:
                    got = c.predicted.value if c.predicted else "none"
                    notes.append(f"category {got} != {c.case.expected_category.value}")
                if c.escalated != c.case.expect_escalation:
                    notes.append(f"escalated={c.escalated}, expected {c.case.expect_escalation}")
                notes += [f"guardrail: {i}" for i in c.guardrail_issues]
                if c.missing:
                    notes.append("missing: " + ", ".join(c.missing))
                if c.forbidden:
                    notes.append("forbidden text present: " + ", ".join(c.forbidden))
                lines.append(f"- **{c.case.id}**: " + "; ".join(notes))
        return "\n".join(lines) + "\n"


def _ratio(num: float, den: float) -> float:
    return num / den if den else 0.0


def evaluate(
    process: Callable[[Email], AgentResult],
    cases: Iterable[EvalCase],
    settings: Settings,
    *,
    usage: Any | None = None,
    on_case: Callable[[CaseResult], None] | None = None,
) -> EvalReport:
    results: list[CaseResult] = []
    for case in cases:
        before = usage.snapshot() if usage is not None else Usage()
        started = time.perf_counter()
        result: AgentResult | None = None
        error: str | None = None
        try:
            result = process(case.to_email())
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        latency = (time.perf_counter() - started) * 1000
        spent = (usage.snapshot() - before) if usage is not None else Usage()
        item = CaseResult(case=case, result=result, latency_ms=latency, usage=spent, error=error)
        if result and result.reply:
            known = extract_tracking_numbers(case.subject, case.body) + case.must_include
            policy = ReplyPolicy.build(
                sender=case.sender,
                support_address=settings.support_address,
                allowed_link_domains=settings.allowed_link_domains,
                known_tracking_numbers=known,
                max_chars=settings.max_reply_chars + len(settings.signature) + 2,
            )
            item.guardrail_issues = check_reply(result.reply, policy)
            text = result.reply.lower()
            item.missing = [s for s in case.must_include if s.lower() not in text]
            item.forbidden = [s for s in case.must_not_include if s.lower() in text]
        results.append(item)
        if on_case is not None:
            on_case(item)
    return EvalReport(results)


def write_report(report: EvalReport, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    md, js = out_dir / "eval_report.md", out_dir / "eval_report.json"
    md.write_text(report.to_markdown(), encoding="utf-8")
    payload = {
        "summary": report.summary(),
        "per_category": report.per_category(),
        "confusion": {k: dict(v) for k, v in report.confusion().items()},
        "cases": [
            {
                "id": c.case.id,
                "expected": c.case.expected_category.value,
                "predicted": c.predicted.value if c.predicted else None,
                "outcome": c.result.outcome.value if c.result else None,
                "attempts": c.result.attempts if c.result else None,
                "error": c.error,
                "guardrail_issues": c.guardrail_issues,
                "missing": c.missing,
            }
            for c in report.cases
        ],
    }
    js.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return md, js
