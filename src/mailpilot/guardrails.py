"""Deterministic checks every reply must pass before it can reach a customer.

They run before the (paid, probabilistic) LLM reviewer, catch the failures that matter most
for a support team - invented tracking numbers, foreign links, promised refunds, leaked
addresses, template leftovers - and their messages are fed back to the writer on retry.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from urllib.parse import urlparse

from mailpilot.tracking import extract_tracking_numbers, normalise

_URL_RE = re.compile(r"\b(?:https?://|www\.)[^\s<>()\"']+", re.IGNORECASE)
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_PLACEHOLDER_RE = re.compile(
    r"\[(?:your|customer|name|insert|tracking|order|date|company)[^\]]*\]|\{\{?[a-z_ ]+\}?\}"
    r"|<(?:name|insert)[^>]*>|\b(?:TODO|TBD|XXX|lorem ipsum)\b",
    re.IGNORECASE,
)
# Commitments only a human agent may make (EN / NL / DE / FR).
_COMMITMENT_RE = re.compile(
    r"\b(?:we\s*(?:will|'ll|shall)\s+(?:refund|reimburse|compensate|pay)"
    r"|(?:full\s+)?refund\s+(?:has\s+been|is)\s+(?:approved|issued|processed)"
    r"|we\s+guarantee|guaranteed\s+(?:delivery|refund|compensation)"
    r"|(?:wij|we)\s+(?:zullen|gaan)\s+(?:u\s+|je\s+)?(?:het\s+bedrag\s+)?"
    r"(?:terugbetalen|vergoeden|terugstorten)"
    r"|wir\s+(?:werden\s+)?(?:ihnen\s+)?(?:den\s+betrag\s+)?"
    r"(?:erstatten|zurückzahlen|zurückerstatten)"
    r"|nous\s+(?:allons\s+)?vous\s+rembourser)",
    re.IGNORECASE,
)
_LEAK_RE = re.compile(
    r"\b(?:as an ai|language model|system prompt|knowledge base|snippet|my instructions"
    r"|i (?:do not|don't) have access)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ReplyPolicy:
    sender: str
    support_address: str
    allowed_link_domains: tuple[str, ...]
    known_tracking_numbers: frozenset[str] = field(default_factory=frozenset)
    public_addresses: frozenset[str] = field(default_factory=frozenset)
    max_chars: int = 2_500

    @classmethod
    def build(
        cls,
        *,
        sender: str,
        support_address: str,
        allowed_link_domains: Iterable[str],
        known_tracking_numbers: Iterable[str] = (),
        public_addresses: Iterable[str] = (),
        max_chars: int = 2_500,
    ) -> ReplyPolicy:
        return cls(
            sender=sender.lower(),
            support_address=support_address.lower(),
            allowed_link_domains=tuple(d.lower() for d in allowed_link_domains),
            known_tracking_numbers=frozenset(normalise(n) for n in known_tracking_numbers),
            public_addresses=frozenset(a.lower() for a in public_addresses),
            max_chars=max_chars,
        )


def find_addresses(text: str) -> set[str]:
    return {a.lower() for a in _EMAIL_RE.findall(text)}


def _domain_allowed(url: str, allowed: tuple[str, ...]) -> bool:
    host = urlparse(url if "://" in url else f"https://{url}").hostname or ""
    return any(host == d or host.endswith("." + d) for d in allowed)


def check_reply(text: str, policy: ReplyPolicy) -> list[str]:
    """Return a list of human-readable problems; empty means the reply passed."""
    issues: list[str] = []
    stripped = text.strip()
    if len(stripped) < 20:
        issues.append("The reply is empty or too short to be useful.")
    if len(stripped) > policy.max_chars:
        issues.append(f"The reply is {len(stripped)} characters; keep it under {policy.max_chars}.")

    unknown = [n for n in extract_tracking_numbers(text) if n not in policy.known_tracking_numbers]
    if unknown:
        issues.append(
            "Mentions tracking numbers that are not in the verified facts: "
            + ", ".join(unknown)
            + ". Only use tracking numbers given to you."
        )

    urls = [u.rstrip(".,;:!?") for u in _URL_RE.findall(text)]
    bad_links = [u for u in urls if not _domain_allowed(u, policy.allowed_link_domains)]
    if bad_links:
        issues.append(
            "Contains links outside the allowed domains ("
            + ", ".join(policy.allowed_link_domains)
            + "): "
            + ", ".join(bad_links)
        )

    allowed_addresses = {policy.sender, policy.support_address, *policy.public_addresses}
    foreign = sorted(
        a
        for a in find_addresses(text) - allowed_addresses
        if a.rsplit("@", 1)[1] not in policy.allowed_link_domains  # company addresses are fine
    )
    if foreign:
        issues.append("Reveals email addresses other than the customer's: " + ", ".join(foreign))

    if match := _COMMITMENT_RE.search(text):
        issues.append(
            f'Makes a commitment only a human may make ("{match.group(0)}"). Explain the '
            "process or that a colleague will review it, without promising refunds or outcomes."
        )
    if match := _PLACEHOLDER_RE.search(text):
        issues.append(f'Contains a template placeholder: "{match.group(0)}".')
    if match := _LEAK_RE.search(text):
        issues.append(f'Mentions internal tooling or being an AI ("{match.group(0)}").')
    return issues
