"""Email parsing helpers shared by the Gmail and folder mailboxes."""

from __future__ import annotations

import re
from collections.abc import Mapping
from email.utils import parseaddr
from html.parser import HTMLParser

_QUOTE_HEADER_RE = re.compile(
    r"^(?:On .+ wrote:|Op .+ schreef .+:|Am .+ schrieb .+:|Le .+ a écrit\s*:"
    r"|-{2,}\s*Original Message\s*-{2,}|From: .+)$",
    re.IGNORECASE,
)


class _TextExtractor(HTMLParser):
    _BLOCK = frozenset({"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "table"})

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in ("script", "style", "head"):
            self._skip += 1
        elif tag in self._BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style", "head") and self._skip:
            self._skip -= 1
        elif tag in self._BLOCK:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip:
            self.parts.append(data)


def html_to_text(html: str) -> str:
    parser = _TextExtractor()
    parser.feed(html)
    parser.close()
    return "".join(parser.parts)


def clean_body(text: str, *, max_chars: int = 8_000) -> str:
    """Drop quoted history and signatures' noise, normalise whitespace, cap length.

    Quoted previous messages are the biggest source of wasted tokens in support threads.
    """
    kept: list[str] = []
    for line in text.replace("\r\n", "\n").split("\n"):
        stripped = line.strip()
        if _QUOTE_HEADER_RE.match(stripped):
            break
        if stripped.startswith(">"):
            continue
        kept.append(line.rstrip())
    body = re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip()
    return body[:max_chars]


def parse_address(value: str) -> tuple[str, str]:
    name, address = parseaddr(value or "")
    return name.strip().strip('"'), address.strip().lower()


def reply_subject(subject: str) -> str:
    subject = (subject or "").strip()
    return subject if re.match(r"^(re|aw|antw)\s*:", subject, re.IGNORECASE) else f"Re: {subject}"


def is_auto_generated(headers: Mapping[str, str]) -> bool:
    """RFC 3834 auto-replies, bulk mail and mailing lists: never worth an LLM call."""
    lowered = {k.lower(): v.lower() for k, v in headers.items()}
    auto = lowered.get("auto-submitted", "no")
    return (
        auto not in ("", "no")
        or lowered.get("precedence", "") in ("bulk", "list", "junk")
        or "list-unsubscribe" in lowered
        or "list-id" in lowered
        or lowered.get("x-autoreply", "") == "yes"
    )
