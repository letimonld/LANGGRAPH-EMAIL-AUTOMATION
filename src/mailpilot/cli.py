"""Command line: ``mailpilot run | draft | eval | stats | serve | graph | gmail-auth``."""

from __future__ import annotations

import argparse
import json
import signal
import sys
import threading
from collections.abc import Callable, Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from mailpilot import __version__
from mailpilot.config import Settings
from mailpilot.mailbox.base import Mailbox
from mailpilot.metrics import MetricsStore
from mailpilot.models import Email
from mailpilot.observability import configure_logging


class Factories:
    """Construction seams. Tests replace these to run the CLI without network access."""

    @staticmethod
    def agent(settings: Settings) -> Any:  # pragma: no cover - needs an LLM provider
        from mailpilot.agent import create_agent

        return create_agent(settings)

    @staticmethod
    def gmail(settings: Settings, interactive: bool = False) -> Mailbox:  # pragma: no cover
        from mailpilot.mailbox.gmail import GmailMailbox, build_service

        service = build_service(
            settings.gmail_credentials_file, settings.gmail_token_file, interactive=interactive
        )
        return GmailMailbox(
            service,
            support_address=settings.support_address,
            query=settings.gmail_query,
            label_prefix=settings.label_prefix,
        )


def _usage_of(agent: Any) -> Any:
    return getattr(getattr(getattr(agent, "deps", None), "llm", None), "usage", None)


def _metrics(settings: Settings) -> MetricsStore | None:
    return MetricsStore(settings.metrics_db) if settings.metrics_db else None


def _print(data: Any) -> None:
    print(json.dumps(data, indent=2, ensure_ascii=False, default=str))


# ---------------------------------------------------------------------- commands
def cmd_run(args: argparse.Namespace, settings: Settings) -> int:
    from mailpilot.mailbox.folder import FolderMailbox
    from mailpilot.runner import Runner

    if args.mode:
        settings = settings.with_overrides(delivery_mode=args.mode)
    mailbox: Mailbox
    if args.mailbox == "gmail":
        mailbox = Factories.gmail(settings)
    else:
        mailbox = FolderMailbox(Path(args.inbox), Path(args.outbox) if args.outbox else None)
    agent = Factories.agent(settings)
    metrics = _metrics(settings)
    runner = Runner(agent, mailbox, metrics=metrics, usage=_usage_of(agent))
    try:
        if not args.watch:
            report = runner.run_once(args.limit)
            _print(report.to_dict())
            return 1 if report.errors and not report.outcomes else 0
        stop = threading.Event()
        signal.signal(signal.SIGINT, lambda *_: stop.set())
        signal.signal(signal.SIGTERM, lambda *_: stop.set())
        runner.watch(args.interval, limit=args.limit, stop=stop, max_runs=args.max_runs)
        return 0
    finally:
        if metrics is not None:
            metrics.close()


def cmd_draft(args: argparse.Namespace, settings: Settings) -> int:
    if args.file:
        data = json.loads(Path(args.file).read_text(encoding="utf-8"))
        data.setdefault("id", Path(args.file).stem)
    else:
        if not (args.sender and args.body):
            print("Provide --file, or --sender and --body", file=sys.stderr)
            return 2
        data = {"id": "cli", "sender": args.sender, "subject": args.subject, "body": args.body}
    email = Email.model_validate(data)
    result = Factories.agent(settings).process(email)
    _print(result.model_dump(mode="json"))
    return 0


def cmd_eval(args: argparse.Namespace, settings: Settings) -> int:
    from mailpilot.evaluation import evaluate, load_cases, write_report

    cases = load_cases(Path(args.dataset))
    if args.limit:
        cases = cases[: args.limit]
    agent = Factories.agent(settings)

    def progress(item: Any) -> None:
        got = item.predicted.value if item.predicted else "error"
        mark = "ok " if item.predicted == item.case.expected_category else "XX "
        print(f"{mark}{item.case.id:<28} {got}", file=sys.stderr)

    report = evaluate(agent.process, cases, settings, usage=_usage_of(agent), on_case=progress)
    md, js = write_report(report, Path(args.out))
    _print(report.summary())
    print(f"\nReport: {md}\nDetails: {js}", file=sys.stderr)
    passed = report.escalation_recall >= args.min_escalation_recall and (
        report.guardrail_violations == 0
    )
    return 0 if passed else 1


def cmd_stats(args: argparse.Namespace, settings: Settings) -> int:
    store = _metrics(settings)
    if store is None:
        print("Metrics are disabled (METRICS_DB=none).", file=sys.stderr)
        return 1
    since = datetime.now(timezone.utc) - timedelta(days=args.days) if args.days else None
    with store:
        summary = store.summary(since=since, minutes_per_draft=settings.minutes_saved_per_draft)
    _print(summary.to_dict())
    return 0


def cmd_serve(args: argparse.Namespace, settings: Settings) -> int:  # pragma: no cover
    import uvicorn

    from mailpilot.api import create_app

    app = create_app(
        Factories.agent(settings), api_key=settings.api_key or "", metrics=_metrics(settings)
    )
    uvicorn.run(app, host=args.host, port=args.port, log_config=None)
    return 0


def cmd_graph(_args: argparse.Namespace, settings: Settings) -> int:
    from mailpilot.agent import AgentDeps, SupportAgent
    from mailpilot.knowledge import BM25Retriever
    from mailpilot.tracking import JsonTrackingService

    deps = AgentDeps(
        llm=None,  # type: ignore[arg-type]  # drawing the graph never calls the model
        retriever=BM25Retriever([]),
        tracking=JsonTrackingService([]),
        settings=settings,
    )
    print(SupportAgent(deps).mermaid())
    return 0


def cmd_gmail_auth(_args: argparse.Namespace, settings: Settings) -> int:  # pragma: no cover
    Factories.gmail(settings, interactive=True)
    print(f"Gmail token saved to {settings.gmail_token_file}")
    return 0


# ---------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mailpilot", description=__doc__)
    parser.add_argument("--version", action="version", version=f"mailpilot {__version__}")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--json-logs", action="store_true", help="JSON lines (for log shipping)")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Process new emails from a mailbox")
    run.add_argument("--mailbox", choices=["folder", "gmail"], default="folder")
    run.add_argument("--inbox", default="samples/inbox", help="Folder mailbox: input directory")
    run.add_argument("--outbox", default=None, help="Folder mailbox: where replies are written")
    run.add_argument("--limit", type=int, default=25)
    run.add_argument("--mode", choices=["draft", "send"], default=None)
    run.add_argument("--watch", action="store_true", help="Keep polling")
    run.add_argument("--interval", type=float, default=60.0)
    run.add_argument("--max-runs", type=int, default=None)
    run.set_defaults(func=cmd_run)

    draft = sub.add_parser("draft", help="Dry-run the agent on one email and print the result")
    draft.add_argument("--file")
    draft.add_argument("--sender")
    draft.add_argument("--subject", default="")
    draft.add_argument("--body")
    draft.set_defaults(func=cmd_draft)

    ev = sub.add_parser("eval", help="Evaluate on a labelled dataset")
    ev.add_argument("--dataset", default="evals/emails.jsonl")
    ev.add_argument("--out", default="reports")
    ev.add_argument("--limit", type=int, default=None)
    ev.add_argument("--min-escalation-recall", type=float, default=1.0)
    ev.set_defaults(func=cmd_eval)

    stats = sub.add_parser("stats", help="Show processing metrics")
    stats.add_argument("--days", type=float, default=None)
    stats.set_defaults(func=cmd_stats)

    serve = sub.add_parser("serve", help="Start the HTTP API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.set_defaults(func=cmd_serve)

    graph = sub.add_parser("graph", help="Print the agent graph as Mermaid")
    graph.set_defaults(func=cmd_graph)

    auth = sub.add_parser("gmail-auth", help="Run the Gmail OAuth consent flow once")
    auth.set_defaults(func=cmd_gmail_auth)
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    settings_loader: Callable[[str], Settings] = Settings.from_env,
) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(args.log_level, json_logs=args.json_logs)
    try:
        settings = settings_loader(args.env_file)
    except ValueError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    func: Callable[[argparse.Namespace, Settings], int] = args.func
    return func(args, settings)
