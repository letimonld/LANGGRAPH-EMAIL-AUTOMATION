"""Per-email outcomes in SQLite, summarised into the numbers a support lead cares about."""

from __future__ import annotations

import sqlite3
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from mailpilot.llm import Usage
from mailpilot.models import AgentResult, Outcome

_SCHEMA = """
CREATE TABLE IF NOT EXISTS outcomes (
    email_id      TEXT NOT NULL,
    processed_at  TEXT NOT NULL,
    outcome       TEXT NOT NULL,
    category      TEXT,
    confidence    REAL,
    attempts      INTEGER NOT NULL,
    latency_ms    INTEGER NOT NULL,
    llm_calls     INTEGER NOT NULL,
    input_tokens  INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    reason        TEXT
);
CREATE INDEX IF NOT EXISTS idx_outcomes_time ON outcomes(processed_at);
"""


@dataclass
class Summary:
    total: int = 0
    by_outcome: dict[str, int] = field(default_factory=dict)
    by_category: dict[str, int] = field(default_factory=dict)
    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    avg_attempts: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    minutes_saved_estimate: float = 0.0

    @property
    def handled(self) -> int:
        """Emails that needed a response (everything except skipped noise)."""
        return self.total - self.by_outcome.get(Outcome.SKIPPED.value, 0)

    @property
    def automation_rate(self) -> float:
        """Share of emails needing a response that got a ready reply (drafted or sent)."""
        replied = self.by_outcome.get(Outcome.DRAFTED.value, 0) + self.by_outcome.get(
            Outcome.SENT.value, 0
        )
        return replied / self.handled if self.handled else 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "total": self.total,
            "handled": self.handled,
            "automation_rate": round(self.automation_rate, 3),
            "by_outcome": self.by_outcome,
            "by_category": self.by_category,
            "avg_latency_ms": round(self.avg_latency_ms),
            "p95_latency_ms": round(self.p95_latency_ms),
            "avg_attempts": round(self.avg_attempts, 2),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "minutes_saved_estimate": round(self.minutes_saved_estimate, 1),
        }


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(pct / 100 * (len(ordered) - 1))))
    return ordered[index]


class MetricsStore:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path) if str(path) != ":memory:" else path
        if isinstance(self.path, Path):
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.executescript(_SCHEMA)

    def record(self, result: AgentResult, *, latency_ms: int, usage: Usage) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT INTO outcomes VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    result.email_id,
                    datetime.now(timezone.utc).isoformat(),
                    result.outcome.value,
                    result.category.value if result.category else None,
                    result.confidence,
                    result.attempts,
                    latency_ms,
                    usage.calls,
                    usage.input_tokens,
                    usage.output_tokens,
                    result.reason,
                ),
            )

    def summary(self, *, since: datetime | None = None, minutes_per_draft: float = 4.0) -> Summary:
        query = (
            "SELECT outcome, category, latency_ms, attempts, input_tokens, output_tokens"
            " FROM outcomes"
        )
        params: tuple[str, ...] = ()
        if since is not None:
            query += " WHERE processed_at >= ?"
            params = (since.astimezone(timezone.utc).isoformat(),)
        rows = self._conn.execute(query, params).fetchall()
        if not rows:
            return Summary()
        outcomes = Counter(r[0] for r in rows)
        latencies = [float(r[2]) for r in rows]
        replied = outcomes[Outcome.DRAFTED.value] + outcomes[Outcome.SENT.value]
        return Summary(
            total=len(rows),
            by_outcome=dict(outcomes),
            by_category=dict(Counter(r[1] for r in rows if r[1])),
            avg_latency_ms=sum(latencies) / len(latencies),
            p95_latency_ms=percentile(latencies, 95),
            avg_attempts=sum(r[3] for r in rows) / len(rows),
            input_tokens=sum(r[4] for r in rows),
            output_tokens=sum(r[5] for r in rows),
            minutes_saved_estimate=replied * minutes_per_draft,
        )

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> MetricsStore:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
