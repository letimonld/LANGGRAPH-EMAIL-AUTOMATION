"""AWS Lambda entry point, triggered by an EventBridge schedule (see ``deploy/template.yaml``).

Stateless by design: idempotency lives in Gmail labels, secrets come from environment
variables populated from AWS Secrets Manager, logs are JSON lines for CloudWatch.
Run with reserved concurrency 1 so two invocations never work on the same inbox.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from mailpilot.config import Settings
from mailpilot.observability import configure_logging
from mailpilot.runner import Runner

logger = logging.getLogger(__name__)
_runner: Runner | None = None  # reused across warm invocations


def _materialise_secret(env_var: str, path: Path) -> Path:
    """Lambda has no browser: the Gmail OAuth token is injected as a JSON env var."""
    value = os.environ.get(env_var)
    if value and not path.exists():
        path.write_text(value, encoding="utf-8")
    return path


def _build_runner() -> Runner:  # pragma: no cover - wires real Gmail and LLM clients
    from mailpilot.agent import create_agent
    from mailpilot.mailbox.gmail import GmailMailbox, build_service

    settings = Settings.from_env(env_file=None).with_overrides(metrics_db=None)
    token = _materialise_secret("GMAIL_TOKEN_JSON", Path("/tmp/gmail-token.json"))  # noqa: S108
    service = build_service(settings.gmail_credentials_file, token)
    mailbox = GmailMailbox(
        service,
        support_address=settings.support_address,
        query=settings.gmail_query,
        label_prefix=settings.label_prefix,
    )
    agent = create_agent(settings)
    return Runner(agent, mailbox, usage=agent.deps.llm.usage)  # type: ignore[attr-defined]


def handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    global _runner  # noqa: PLW0603
    configure_logging(os.environ.get("LOG_LEVEL", "INFO"), json_logs=True)
    if _runner is None:
        _runner = _build_runner()
    limit = int(event.get("limit", os.environ.get("BATCH_LIMIT", "20")))
    # Leave a safety margin before Lambda's hard timeout; unfinished emails wait for next run.
    remaining = context.get_remaining_time_in_millis() / 1000 if context is not None else 600
    report = _runner.run_once(limit, time_budget_sec=max(remaining - 90, 10))
    summary = report.to_dict()
    logger.info("lambda run", extra={"report": json.dumps(summary)})
    return summary
