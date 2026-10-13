from __future__ import annotations

from typing import Protocol

from mailpilot.models import Email, Outcome


class Mailbox(Protocol):
    """Where emails come from and where replies go.

    ``fetch_unprocessed`` must not return emails that were already ``mark``-ed, which is
    what makes the runner idempotent (safe to run on a schedule, or twice by accident).
    """

    def fetch_unprocessed(self, limit: int) -> list[Email]: ...

    def create_draft(self, email: Email, body: str) -> str: ...

    def send_reply(self, email: Email, body: str) -> str: ...

    def mark(self, email: Email, outcome: Outcome) -> None: ...


def label_for(prefix: str, outcome: Outcome) -> str:
    return f"{prefix}/{outcome.value}"
