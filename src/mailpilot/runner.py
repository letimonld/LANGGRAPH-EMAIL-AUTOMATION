"""Applies the agent's decisions to a mailbox, one email at a time, failure-isolated."""

from __future__ import annotations

import logging
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Protocol

from mailpilot.llm import LLMTransientError, Usage
from mailpilot.mailbox.base import Mailbox
from mailpilot.metrics import MetricsStore
from mailpilot.models import AgentResult, Email, Outcome

logger = logging.getLogger(__name__)


class Agent(Protocol):
    def process(self, email: Email) -> AgentResult: ...


@dataclass
class RunReport:
    fetched: int = 0
    outcomes: Counter[str] = field(default_factory=Counter)
    deferred: int = 0  # left untouched because the LLM was rate-limited; retried next run
    errors: list[str] = field(default_factory=list)
    duration_sec: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "fetched": self.fetched,
            "outcomes": dict(self.outcomes),
            "deferred": self.deferred,
            "errors": self.errors,
            "duration_sec": round(self.duration_sec, 2),
        }


class Runner:
    def __init__(
        self,
        agent: Agent,
        mailbox: Mailbox,
        *,
        metrics: MetricsStore | None = None,
        usage: Any | None = None,  # anything with .snapshot() -> Usage
    ) -> None:
        self.agent = agent
        self.mailbox = mailbox
        self.metrics = metrics
        self.usage = usage

    def _usage(self) -> Usage:
        return self.usage.snapshot() if self.usage is not None else Usage()

    def handle(self, email: Email) -> AgentResult:
        if email.auto_generated:  # never spend tokens on auto-replies and bulk mail
            return AgentResult(
                email_id=email.id, outcome=Outcome.SKIPPED, reason="auto-generated email"
            )
        result = self.agent.process(email)
        if result.outcome is Outcome.DRAFTED and result.reply:
            self.mailbox.create_draft(email, result.reply)
        elif result.outcome is Outcome.SENT and result.reply:
            self.mailbox.send_reply(email, result.reply)
        return result

    def run_once(self, limit: int = 25, *, time_budget_sec: float | None = None) -> RunReport:
        """Process up to ``limit`` new emails; stop early when the time budget is spent."""
        report = RunReport()
        started = time.perf_counter()
        emails = self.mailbox.fetch_unprocessed(limit)
        report.fetched = len(emails)
        for position, email in enumerate(emails):
            if time_budget_sec is not None and time.perf_counter() - started > time_budget_sec:
                report.deferred = len(emails) - position
                logger.info("time budget spent, deferring %d emails", report.deferred)
                break
            before, t0 = self._usage(), time.perf_counter()
            try:
                result = self.handle(email)
            except LLMTransientError as exc:
                # Quota/outage: stop here, leave this and the rest for the next run.
                report.deferred = len(emails) - position
                report.errors.append(f"{email.id}: deferred ({exc})")
                logger.warning("LLM unavailable, deferring %d emails: %s", report.deferred, exc)
                break
            except Exception as exc:
                logger.exception("Failed to process email %s", email.id)
                result = AgentResult(
                    email_id=email.id, outcome=Outcome.FAILED, reason=f"{type(exc).__name__}: {exc}"
                )
                report.errors.append(f"{email.id}: {result.reason}")
            try:
                self.mailbox.mark(email, result.outcome)
            except Exception as exc:  # the reply exists; surface but do not crash the batch
                logger.exception("Could not label email %s", email.id)
                report.errors.append(f"{email.id}: label failed ({exc})")
            latency_ms = int((time.perf_counter() - t0) * 1000)
            report.outcomes[result.outcome.value] += 1
            if self.metrics is not None:
                self.metrics.record(result, latency_ms=latency_ms, usage=self._usage() - before)
            logger.info(
                "email processed",
                extra={
                    "email_id": email.id,
                    "outcome": result.outcome.value,
                    "category": result.category.value if result.category else None,
                    "attempts": result.attempts,
                    "latency_ms": latency_ms,
                },
            )
        report.duration_sec = time.perf_counter() - started
        return report

    def watch(
        self,
        interval_sec: float,
        *,
        limit: int = 25,
        stop: threading.Event | None = None,
        max_runs: int | None = None,
    ) -> int:
        """Poll the mailbox until ``stop`` is set (Ctrl-C in the CLI). Returns runs made."""
        stop = stop or threading.Event()
        runs = 0
        while not stop.is_set() and (max_runs is None or runs < max_runs):
            report = self.run_once(limit)
            runs += 1
            logger.info("run finished", extra={"report": report.to_dict()})
            stop.wait(interval_sec)
        return runs
