"""Test doubles: a scripted LLM and an in-memory mailbox."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

from mailpilot.llm import UsageTracker
from mailpilot.models import (
    Category,
    DraftReply,
    Email,
    Outcome,
    Review,
    Sentiment,
    Triage,
    Urgency,
)


def make_triage(**overrides: Any) -> Triage:
    data: dict[str, Any] = {
        "category": Category.TRACKING,
        "confidence": 0.95,
        "urgency": Urgency.NORMAL,
        "sentiment": Sentiment.NEUTRAL,
        "language": "en",
        "tracking_numbers": [],
        "kb_queries": ["tracking status meaning"],
        "summary": "Customer asks where their parcel is.",
        "escalate_reason": None,
    }
    data.update(overrides)
    return Triage(**data)


def make_email(**overrides: Any) -> Email:
    data: dict[str, Any] = {
        "id": "m1",
        "thread_id": "t1",
        "sender": "emma.devries@example.nl",
        "sender_name": "Emma",
        "subject": "Where is my parcel?",
        "body": "Hi, where is PCL100000001NL?",
    }
    data.update(overrides)
    return Email(**data)


GOOD_REPLY = (
    "Hi Emma, thanks for reaching out. Your parcel PCL100000001NL is in transit with PostNL "
    "and is expected on 2026-10-15. You can follow it at https://track.parcelo.example."
)


@dataclass
class Call:
    schema: str
    system: str
    user: str


class ScriptedLLM:
    """Returns canned objects per schema. Lists are consumed in order; the last one repeats."""

    def __init__(
        self,
        triage: Triage | Callable[[str], Triage] | None = None,
        drafts: list[str] | None = None,
        reviews: list[Review] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.triage = triage or make_triage()
        self.drafts = list(drafts or [GOOD_REPLY])
        self.reviews = list(reviews or [Review(approved=True)])
        self.error = error
        self.calls: list[Call] = []
        self.usage = UsageTracker()

    def _next(self, items: list[Any]) -> Any:
        return items.pop(0) if len(items) > 1 else items[0]

    def structured(self, system: str, user: str, schema: type[BaseModel]) -> Any:
        self.calls.append(Call(schema.__name__, system, user))
        self.usage.record(100, 20)
        if self.error is not None:
            raise self.error
        if schema is Triage:
            return self.triage(user) if callable(self.triage) else self.triage
        if schema is DraftReply:
            return DraftReply(body=self._next(self.drafts), used_sources=["tracking#status"])
        if schema is Review:
            return self._next(self.reviews)
        raise AssertionError(f"unexpected schema {schema}")

    def count(self, schema: str) -> int:
        return sum(1 for c in self.calls if c.schema == schema)


@dataclass
class MemoryMailbox:
    emails: list[Email]
    drafts: dict[str, str] = field(default_factory=dict)
    sent: dict[str, str] = field(default_factory=dict)
    marks: dict[str, Outcome] = field(default_factory=dict)
    fail_mark: bool = False

    def fetch_unprocessed(self, limit: int) -> list[Email]:
        return [e for e in self.emails if e.id not in self.marks][:limit]

    def create_draft(self, email: Email, body: str) -> str:
        self.drafts[email.id] = body
        return f"d-{email.id}"

    def send_reply(self, email: Email, body: str) -> str:
        self.sent[email.id] = body
        return f"s-{email.id}"

    def mark(self, email: Email, outcome: Outcome) -> None:
        if self.fail_mark:
            raise RuntimeError("label API down")
        self.marks[email.id] = outcome
