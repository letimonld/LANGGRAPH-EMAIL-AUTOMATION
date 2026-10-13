"""Gmail mailbox.

Idempotency comes from Gmail labels (``mailpilot/drafted``, ``mailpilot/escalated``...):
the fetch query excludes every message that already carries one, so the state lives in the
mailbox itself and the job can run statelessly (e.g. on AWS Lambda) on a schedule.
"""

from __future__ import annotations

import base64
import logging
from email.message import EmailMessage
from pathlib import Path
from typing import Any

from mailpilot.mailbox.base import label_for
from mailpilot.mailbox.parsing import (
    clean_body,
    html_to_text,
    is_auto_generated,
    parse_address,
    reply_subject,
)
from mailpilot.models import Email, Outcome

logger = logging.getLogger(__name__)

SCOPES = ["https://www.googleapis.com/auth/gmail.modify"]


def build_service(
    credentials_file: Path, token_file: Path, *, interactive: bool = False
) -> Any:  # pragma: no cover - talks to Google
    """Load (and refresh) OAuth credentials. ``interactive`` opens the browser consent flow."""
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build

    creds = None
    if token_file.exists():
        creds = Credentials.from_authorized_user_file(str(token_file), SCOPES)
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
        token_file.write_text(creds.to_json(), encoding="utf-8")
    if not creds or not creds.valid:
        if not interactive:
            raise RuntimeError(
                f"No valid Gmail token at {token_file}. Run `mailpilot gmail-auth` once."
            )
        flow = InstalledAppFlow.from_client_secrets_file(str(credentials_file), SCOPES)
        creds = flow.run_local_server(port=0)
        token_file.write_text(creds.to_json(), encoding="utf-8")
    return build("gmail", "v1", credentials=creds, cache_discovery=False)


def _decode(data: str) -> str:
    return base64.urlsafe_b64decode(data + "=" * (-len(data) % 4)).decode("utf-8", "replace")


def _extract_text(payload: dict[str, Any]) -> str:
    """Prefer text/plain anywhere in the MIME tree; fall back to text/html."""
    plain: list[str] = []
    html: list[str] = []

    def walk(part: dict[str, Any]) -> None:
        mime = part.get("mimeType", "")
        data = part.get("body", {}).get("data")
        if data and mime == "text/plain":
            plain.append(_decode(data))
        elif data and mime == "text/html":
            html.append(_decode(data))
        for child in part.get("parts", []) or []:
            walk(child)

    walk(payload)
    if plain:
        return "\n".join(plain)
    return "\n".join(html_to_text(h) for h in html)


class GmailMailbox:
    def __init__(
        self,
        service: Any,
        *,
        support_address: str,
        query: str,
        label_prefix: str = "mailpilot",
    ) -> None:
        self.service = service
        self.support_address = support_address.lower()
        self.query = query
        self.label_prefix = label_prefix
        self._label_ids: dict[str, str] | None = None

    # ------------------------------------------------------------------ reading
    def _full_query(self) -> str:
        exclusions = " ".join(
            f"-label:{label_for(self.label_prefix, o).replace('/', '-')}" for o in Outcome
        )
        return f"{self.query} {exclusions}".strip()

    def fetch_unprocessed(self, limit: int) -> list[Email]:
        users = self.service.users()
        response = (
            users.messages().list(userId="me", q=self._full_query(), maxResults=limit).execute()
        )
        emails: list[Email] = []
        seen_threads: set[str] = set()
        for ref in response.get("messages", []):  # newest first
            if ref["threadId"] in seen_threads:
                continue  # only answer the latest message of a thread
            seen_threads.add(ref["threadId"])
            message = users.messages().get(userId="me", id=ref["id"], format="full").execute()
            labels = set(message.get("labelIds", []))
            if labels & {"DRAFT", "SENT", "SPAM", "TRASH"}:
                continue
            email = self._to_email(message)
            if email.sender in ("", self.support_address):
                continue
            emails.append(email)
        return emails

    def _to_email(self, message: dict[str, Any]) -> Email:
        payload = message.get("payload", {})
        headers = {h["name"]: h["value"] for h in payload.get("headers", [])}
        lookup = {k.lower(): v for k, v in headers.items()}
        name, address = parse_address(lookup.get("from", ""))
        return Email(
            id=message["id"],
            thread_id=message.get("threadId", message["id"]),
            message_id=lookup.get("message-id", ""),
            references=lookup.get("references", ""),
            sender=address,
            sender_name=name,
            subject=lookup.get("subject", ""),
            body=clean_body(_extract_text(payload) or message.get("snippet", "")),
            auto_generated=is_auto_generated(headers),
        )

    # ------------------------------------------------------------------ writing
    def _mime_reply(self, email: Email, body: str) -> dict[str, Any]:
        msg = EmailMessage()
        msg["To"] = email.sender
        msg["From"] = self.support_address
        msg["Subject"] = reply_subject(email.subject)
        if email.message_id:
            msg["In-Reply-To"] = email.message_id
            msg["References"] = f"{email.references} {email.message_id}".strip()
        msg.set_content(body)
        raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
        return {"raw": raw, "threadId": email.thread_id or email.id}

    def create_draft(self, email: Email, body: str) -> str:
        draft = (
            self.service.users()
            .drafts()
            .create(userId="me", body={"message": self._mime_reply(email, body)})
            .execute()
        )
        return str(draft.get("id", ""))

    def send_reply(self, email: Email, body: str) -> str:
        sent = (
            self.service.users()
            .messages()
            .send(userId="me", body=self._mime_reply(email, body))
            .execute()
        )
        return str(sent.get("id", ""))

    # ------------------------------------------------------------------ labels
    def _label_id(self, name: str) -> str:
        if self._label_ids is None:
            existing = self.service.users().labels().list(userId="me").execute()
            self._label_ids = {lbl["name"]: lbl["id"] for lbl in existing.get("labels", [])}
        if name not in self._label_ids:
            created = (
                self.service.users()
                .labels()
                .create(
                    userId="me",
                    body={
                        "name": name,
                        "labelListVisibility": "labelShow",
                        "messageListVisibility": "show",
                    },
                )
                .execute()
            )
            self._label_ids[name] = created["id"]
        return self._label_ids[name]

    def mark(self, email: Email, outcome: Outcome) -> None:
        label_id = self._label_id(label_for(self.label_prefix, outcome))
        self.service.users().messages().modify(
            userId="me", id=email.id, body={"addLabelIds": [label_id]}
        ).execute()
