"""A mailbox backed by a directory: ``*.json`` or ``*.eml`` files in, text files out.

Useful for demos without Gmail credentials, for replaying real (anonymised) emails, and
in tests. Processed state lives in ``<inbox>/.mailpilot-state.json``.
"""

from __future__ import annotations

import json
from email import policy
from email.message import EmailMessage
from email.parser import BytesParser
from pathlib import Path

from mailpilot.mailbox.parsing import (
    clean_body,
    html_to_text,
    is_auto_generated,
    parse_address,
    reply_subject,
)
from mailpilot.models import Email, Outcome


def _load_eml(path: Path) -> Email:
    message = BytesParser(policy=policy.default).parsebytes(path.read_bytes())
    assert isinstance(message, EmailMessage)
    part = message.get_body(preferencelist=("plain", "html"))
    content = part.get_content() if part is not None else ""
    if part is not None and part.get_content_type() == "text/html":
        content = html_to_text(content)
    name, address = parse_address(str(message.get("From", "")))
    return Email(
        id=path.stem,
        thread_id=path.stem,
        message_id=str(message.get("Message-ID", "")),
        references=str(message.get("References", "")),
        sender=address,
        sender_name=name,
        subject=str(message.get("Subject", "")),
        body=clean_body(content),
        auto_generated=is_auto_generated({k: str(v) for k, v in message.items()}),
    )


def _load_json(path: Path) -> Email:
    data = json.loads(path.read_text(encoding="utf-8"))
    data.setdefault("id", path.stem)
    data.setdefault("thread_id", data["id"])
    data["body"] = clean_body(data.get("body", ""))
    return Email.model_validate(data)


class FolderMailbox:
    def __init__(self, inbox: Path, outbox: Path | None = None) -> None:
        if not inbox.is_dir():
            raise FileNotFoundError(f"Inbox folder not found: {inbox}")
        self.inbox = inbox
        self.outbox = outbox or inbox / "outbox"
        self._state_file = inbox / ".mailpilot-state.json"

    def _state(self) -> dict[str, str]:
        if not self._state_file.exists():
            return {}
        data: dict[str, str] = json.loads(self._state_file.read_text(encoding="utf-8"))
        return data

    def fetch_unprocessed(self, limit: int) -> list[Email]:
        done = self._state()
        emails: list[Email] = []
        for path in sorted(self.inbox.iterdir()):
            if len(emails) >= limit:
                break
            if path.suffix not in (".json", ".eml") or path.name.startswith("."):
                continue
            email = _load_json(path) if path.suffix == ".json" else _load_eml(path)
            if email.id not in done:
                emails.append(email)
        return emails

    def _write(self, email: Email, body: str, kind: str) -> str:
        self.outbox.mkdir(parents=True, exist_ok=True)
        path = self.outbox / f"{email.id}.{kind}.txt"
        path.write_text(
            f"To: {email.sender}\nSubject: {reply_subject(email.subject)}\n\n{body}\n",
            encoding="utf-8",
        )
        return str(path)

    def create_draft(self, email: Email, body: str) -> str:
        return self._write(email, body, "draft")

    def send_reply(self, email: Email, body: str) -> str:
        return self._write(email, body, "sent")

    def mark(self, email: Email, outcome: Outcome) -> None:
        state = self._state()
        state[email.id] = outcome.value
        tmp = self._state_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(self._state_file)  # atomic, so a crash never corrupts the state
