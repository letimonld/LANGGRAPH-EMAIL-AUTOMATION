"""Parcel lookup tool with a privacy check.

A parcel's details are only shared with the person it is addressed to: if the sender's
address does not match the recipient on file, the agent is told the parcel exists but
must ask the customer to verify instead of revealing status or location.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel

from mailpilot.models import Parcel

# Formats we recognise: Parcelo (PCL + 9 digits + country), UPS-style 1Z..., DHL-style JJD...
TRACKING_RE = re.compile(r"\b(PCL\d{9}[A-Z]{2}|1Z[0-9A-Z]{16}|JJD\d{12,18})\b")
# Same formats, but tolerant of the spaces/dashes and lower case people type.
_LOOSE_RE = re.compile(
    r"\b(PCL(?:[\s-]?\d){9}[\s-]?[A-Z]{2}|1Z(?:[\s-]?[0-9A-Z]){16}|JJD(?:[\s-]?\d){12,18})\b",
    re.IGNORECASE,
)


def normalise(number: str) -> str:
    return re.sub(r"[\s-]", "", number).upper()


def extract_tracking_numbers(*texts: str) -> list[str]:
    """Find tracking numbers in free text, in order of appearance, without duplicates."""
    found: list[str] = []
    for text in texts:
        for match in _LOOSE_RE.finditer(text):
            number = normalise(match.group(0))
            if TRACKING_RE.fullmatch(number) and number not in found:
                found.append(number)
    return found


class TrackingService(Protocol):
    def lookup(self, tracking_number: str) -> Parcel | None: ...


class JsonTrackingService:
    """Reads parcels from a JSON file. Stand-in for the carrier/tracking API."""

    def __init__(self, parcels: Iterable[Parcel]) -> None:
        self._parcels = {normalise(p.tracking_number): p for p in parcels}

    @classmethod
    def from_file(cls, path: Path) -> JsonTrackingService:
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(Parcel.model_validate(item) for item in data)

    def lookup(self, tracking_number: str) -> Parcel | None:
        return self._parcels.get(normalise(tracking_number))


class ParcelFact(BaseModel):
    tracking_number: str
    found: bool
    verified: bool
    parcel: Parcel | None = None


def lookup_for_sender(
    service: TrackingService, numbers: Sequence[str], sender: str, *, limit: int = 5
) -> list[ParcelFact]:
    facts: list[ParcelFact] = []
    for number in list(dict.fromkeys(normalise(n) for n in numbers if n.strip()))[:limit]:
        parcel = service.lookup(number)
        if parcel is None:
            facts.append(ParcelFact(tracking_number=number, found=False, verified=False))
            continue
        verified = parcel.recipient_email.strip().lower() == sender.strip().lower()
        facts.append(
            ParcelFact(
                tracking_number=number,
                found=True,
                verified=verified,
                parcel=parcel if verified else None,
            )
        )
    return facts


def render_facts(facts: Sequence[ParcelFact]) -> str:
    """Plain-text block for the prompt. Unverified parcels carry no details at all."""
    if not facts:
        return "No tracking numbers were provided or found."
    lines: list[str] = []
    for fact in facts:
        if not fact.found:
            lines.append(
                f"- {fact.tracking_number}: NOT FOUND in our system. Ask the customer to "
                "double-check the number; do not guess a status."
            )
        elif not fact.verified or fact.parcel is None:
            lines.append(
                f"- {fact.tracking_number}: exists, but the sender is NOT the recipient on file. "
                "Do NOT reveal any status, location or dates. Ask them to write from the email "
                "address used for the order, or to contact the shop that sent it."
            )
        else:
            p = fact.parcel
            lines.append(
                f"- {p.tracking_number} via {p.carrier} to {p.destination_country}: "
                f"status={p.status}; estimated delivery={p.estimated_delivery or 'unknown'}"
            )
            for event in p.events[-3:]:
                lines.append(
                    f"    {event.timestamp:%Y-%m-%d %H:%M} {event.location}: {event.description}"
                )
    return "\n".join(lines)
