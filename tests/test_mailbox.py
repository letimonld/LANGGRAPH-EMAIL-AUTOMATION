from __future__ import annotations

import base64
import json
from email import message_from_bytes
from pathlib import Path
from typing import Any

import pytest

from fakes import make_email
from mailpilot.mailbox import FolderMailbox, clean_body, label_for, parse_address, reply_subject
from mailpilot.mailbox.gmail import GmailMailbox
from mailpilot.mailbox.parsing import html_to_text, is_auto_generated
from mailpilot.models import Outcome

ROOT = Path(__file__).resolve().parents[1]

# ------------------------------------------------------------------ parsing


def test_clean_body_drops_quoted_history():
    text = "Thanks!\r\n\r\n\r\n\r\nStill waiting.\n> old line\nOn Mon, Bob wrote:\n> quoted"
    assert clean_body(text) == "Thanks!\n\nStill waiting."
    assert clean_body("Bedankt\nOp ma 1 okt. 2026 schreef Parcelo:\nxx") == "Bedankt"
    assert len(clean_body("x" * 20_000)) == 8_000


def test_html_to_text():
    html = "<html><head><style>p{}</style></head><body><p>Hello</p><div>World<br>!</div>"
    html += "<script>alert(1)</script></body></html>"
    text = html_to_text(html)
    assert "Hello" in text and "World" in text
    assert "alert" not in text and "p{}" not in text


def test_address_and_subject_helpers():
    assert parse_address('"Emma de Vries" <Emma@Example.NL>') == (
        "Emma de Vries",
        "emma@example.nl",
    )
    assert reply_subject("Where is it") == "Re: Where is it"
    assert reply_subject("RE: x") == "RE: x"
    assert reply_subject("AW: Paket") == "AW: Paket"
    assert label_for("mailpilot", Outcome.DRAFTED) == "mailpilot/drafted"


@pytest.mark.parametrize(
    ("headers", "expected"),
    [
        ({"Auto-Submitted": "auto-replied"}, True),
        ({"Precedence": "bulk"}, True),
        ({"List-Unsubscribe": "<mailto:x>"}, True),
        ({"X-Autoreply": "yes"}, True),
        ({"Auto-Submitted": "no", "Subject": "hi"}, False),
    ],
)
def test_auto_generated_detection(headers: dict[str, str], expected: bool):
    assert is_auto_generated(headers) is expected


# ------------------------------------------------------------------ folder mailbox


def test_folder_mailbox_round_trip(tmp_path: Path):
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    for path in (ROOT / "samples" / "inbox").iterdir():
        if path.suffix in (".json", ".eml"):
            (inbox / path.name).write_bytes(path.read_bytes())
    (inbox / "notes.txt").write_text("ignored")

    box = FolderMailbox(inbox)
    emails = box.fetch_unprocessed(limit=50)
    assert len(emails) == 8
    newsletter = next(e for e in emails if e.id == "08-newsletter")
    assert newsletter.auto_generated and newsletter.sender == "news@retailweekly.example"
    assert newsletter.message_id == "<abc123@retailweekly.example>"

    first = emails[0]
    path = box.create_draft(first, "Hello")
    assert Path(path).read_text().startswith(f"To: {first.sender}\nSubject: Re: ")
    box.send_reply(first, "Hi")
    box.mark(first, Outcome.DRAFTED)
    assert first.id not in {e.id for e in box.fetch_unprocessed(limit=50)}
    assert json.loads((inbox / ".mailpilot-state.json").read_text()) == {first.id: "drafted"}
    assert len(box.fetch_unprocessed(limit=2)) == 2


def test_folder_eml_html_body(tmp_path: Path):
    (tmp_path / "a.eml").write_bytes(
        b"From: Jo <jo@example.com>\r\nSubject: Hi\r\nContent-Type: text/html\r\n\r\n"
        b"<p>Where is <b>PCL100000001NL</b>?</p>"
    )
    email = FolderMailbox(tmp_path).fetch_unprocessed(5)[0]
    assert "PCL100000001NL" in email.body and "<b>" not in email.body
    assert not email.auto_generated


def test_folder_missing_inbox(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        FolderMailbox(tmp_path / "missing")


# ------------------------------------------------------------------ gmail mailbox


def b64(text: str) -> str:
    return base64.urlsafe_b64encode(text.encode()).decode().rstrip("=")


class Exec:
    def __init__(self, value: Any) -> None:
        self.value = value

    def execute(self) -> Any:
        return self.value


class FakeGmail:
    """Mimics the chained googleapiclient interface used by GmailMailbox."""

    def __init__(self, messages: dict[str, dict[str, Any]], listing: list[dict[str, str]]):
        self.messages_by_id = messages
        self.listing = listing
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.labels_store = [{"id": "L1", "name": "mailpilot/drafted"}]

    # users() -> self; then .messages()/.drafts()/.labels() -> self
    def users(self) -> FakeGmail:
        return self

    def messages(self) -> FakeGmail:
        self._resource = "messages"
        return self

    def drafts(self) -> FakeGmail:
        self._resource = "drafts"
        return self

    def labels(self) -> FakeGmail:
        self._resource = "labels"
        return self

    def list(self, **kw: Any) -> Exec:
        self.calls.append((f"{self._resource}.list", kw))
        if self._resource == "labels":
            return Exec({"labels": self.labels_store})
        return Exec({"messages": self.listing})

    def get(self, **kw: Any) -> Exec:
        self.calls.append(("messages.get", kw))
        return Exec(self.messages_by_id[kw["id"]])

    def create(self, **kw: Any) -> Exec:
        self.calls.append((f"{self._resource}.create", kw))
        if self._resource == "labels":
            label = {"id": f"L{len(self.labels_store) + 1}", "name": kw["body"]["name"]}
            self.labels_store.append(label)
            return Exec(label)
        return Exec({"id": "draft-1"})

    def send(self, **kw: Any) -> Exec:
        self.calls.append(("messages.send", kw))
        return Exec({"id": "sent-1"})

    def modify(self, **kw: Any) -> Exec:
        self.calls.append(("messages.modify", kw))
        return Exec({})


def gmail_message(mid: str, thread: str, sender: str, body: Any, **extra: Any) -> dict[str, Any]:
    headers = [
        {"name": "From", "value": sender},
        {"name": "Subject", "value": "Parcel"},
        {"name": "Message-ID", "value": f"<{mid}@mail>"},
        {"name": "References", "value": "<root@mail>"},
        *extra.pop("headers", []),
    ]
    payload = body if isinstance(body, dict) else {"mimeType": "text/plain", "body": {"data": body}}
    payload["headers"] = headers
    return {"id": mid, "threadId": thread, "payload": payload, **extra}


@pytest.fixture
def gmail() -> tuple[GmailMailbox, FakeGmail]:
    multipart = {
        "mimeType": "multipart/alternative",
        "parts": [
            {"mimeType": "text/html", "body": {"data": b64("<p>html</p>")}},
            {"mimeType": "text/plain", "body": {"data": b64("Where is PCL100000001NL?\n> old")}},
        ],
    }
    html_only = {"mimeType": "text/html", "body": {"data": b64("<p>Only <b>html</b></p>")}}
    messages = {
        "m1": gmail_message("m1", "t1", "Emma <emma@example.nl>", multipart),
        "m1-older": gmail_message("m1-older", "t1", "emma@example.nl", b64("older")),
        "m2": gmail_message("m2", "t2", "support@parcelo.example", b64("our own reply")),
        "m3": gmail_message("m3", "t3", "x@example.com", b64("draft"), labelIds=["DRAFT"]),
        "m4": gmail_message(
            "m4",
            "t4",
            "list@news.example",
            html_only,
            headers=[{"name": "List-Unsubscribe", "value": "<mailto:u>"}],
        ),
    }
    listing = [{"id": k, "threadId": v["threadId"]} for k, v in messages.items()]
    service = FakeGmail(messages, listing)
    box = GmailMailbox(service, support_address="Support@Parcelo.example", query="in:inbox")
    return box, service


def test_gmail_fetch_filters_and_parses(gmail: tuple[GmailMailbox, FakeGmail]):
    box, service = gmail
    emails = box.fetch_unprocessed(limit=10)

    assert [e.id for e in emails] == ["m1", "m4"]
    first = emails[0]
    assert first.body == "Where is PCL100000001NL?"
    assert (first.sender, first.sender_name) == ("emma@example.nl", "Emma")
    assert emails[1].auto_generated and emails[1].body == "Only html"
    query = service.calls[0][1]["q"]
    assert query.startswith("in:inbox ") and "-label:mailpilot-drafted" in query
    assert "-label:mailpilot-escalated" in query


def test_gmail_draft_send_and_label(gmail: tuple[GmailMailbox, FakeGmail]):
    box, service = gmail
    email = make_email(id="m1", thread_id="t1", message_id="<m1@mail>", references="<root@mail>")

    assert box.create_draft(email, "Hello Emma") == "draft-1"
    message = service.calls[-1][1]["body"]["message"]
    assert message["threadId"] == "t1"
    mime = message_from_bytes(base64.urlsafe_b64decode(message["raw"]))
    assert mime["In-Reply-To"] == "<m1@mail>"
    assert mime["References"] == "<root@mail> <m1@mail>"
    assert mime["Subject"] == "Re: Where is my parcel?"
    assert mime["From"] == "support@parcelo.example"

    assert box.send_reply(email, "Hi") == "sent-1"

    box.mark(email, Outcome.DRAFTED)  # existing label
    box.mark(email, Outcome.ESCALATED)  # created on first use, then cached
    box.mark(email, Outcome.ESCALATED)
    creates = [c for c in service.calls if c[0] == "labels.create"]
    modifies = [c[1]["body"]["addLabelIds"] for c in service.calls if c[0] == "messages.modify"]
    assert len(creates) == 1 and creates[0][1]["body"]["name"] == "mailpilot/escalated"
    assert modifies == [["L1"], ["L2"], ["L2"]]
    assert sum(c[0] == "labels.list" for c in service.calls) == 1


def test_gmail_reply_without_message_id(gmail: tuple[GmailMailbox, FakeGmail]):
    box, _ = gmail
    reply = box._mime_reply(make_email(message_id="", thread_id=""), "x")
    assert reply["threadId"] == "m1"
    mime = message_from_bytes(base64.urlsafe_b64decode(reply["raw"]))
    assert mime["In-Reply-To"] is None
