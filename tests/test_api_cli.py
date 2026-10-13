from __future__ import annotations

import json
import logging
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from fakes import ScriptedLLM, make_triage
from mailpilot import aws_lambda, cli
from mailpilot.api import create_app
from mailpilot.config import Settings
from mailpilot.llm import LLMTransientError
from mailpilot.metrics import MetricsStore
from mailpilot.models import Category, Outcome
from mailpilot.observability import JsonFormatter, configure_logging, redact
from mailpilot.runner import RunReport

KEY = "test-key-0123456789"
GENERIC = "Hi, thanks for your message. A colleague will look into it and follow up shortly."
ROOT = Path(__file__).resolve().parents[1]

# ------------------------------------------------------------------ API


@pytest.fixture
def client(make_agent) -> Iterator[TestClient]:
    with MetricsStore(":memory:") as metrics:
        yield TestClient(create_app(make_agent(ScriptedLLM()), api_key=KEY, metrics=metrics))


def test_health_is_public(client: TestClient):
    assert client.get("/healthz").json()["status"] == "ok"


def test_draft_requires_the_api_key(client: TestClient):
    payload = {"sender": "emma.devries@example.nl", "body": "Where is PCL100000001NL?"}
    assert client.post("/v1/draft", json=payload).status_code == 401
    bad = client.post("/v1/draft", json=payload, headers={"X-API-Key": "wrong"})
    assert bad.status_code == 401

    response = client.post("/v1/draft", json=payload, headers={"X-API-Key": KEY})
    assert response.status_code == 200
    body = response.json()
    assert body["result"]["outcome"] == "drafted"
    assert body["result"]["category"] == "tracking_inquiry"
    assert body["result"]["email_id"] == f"api-{body['request_id']}"


@pytest.mark.parametrize(
    "payload",
    [
        {"sender": "not-an-email", "body": "hi"},
        {"sender": "a@b.example", "body": ""},
        {"sender": "a@b.example", "body": "x" * 20_001},
    ],
)
def test_draft_validates_input(client: TestClient, payload: dict[str, Any]):
    response = client.post("/v1/draft", json=payload, headers={"X-API-Key": KEY})
    assert response.status_code == 422


def test_stats_endpoint(client: TestClient, make_agent):
    assert client.get("/v1/stats", headers={"X-API-Key": KEY}).json()["total"] == 0
    no_metrics = TestClient(create_app(make_agent(ScriptedLLM()), api_key=KEY))
    assert no_metrics.get("/v1/stats", headers={"X-API-Key": KEY}).status_code == 404


def test_llm_outage_returns_503(make_agent):
    app = create_app(make_agent(ScriptedLLM(error=LLMTransientError("quota"))), api_key=KEY)
    response = TestClient(app).post(
        "/v1/draft", json={"sender": "a@b.example", "body": "hi"}, headers={"X-API-Key": KEY}
    )
    assert response.status_code == 503
    assert response.headers["Retry-After"] == "30"


def test_weak_api_key_is_refused(make_agent):
    with pytest.raises(ValueError, match="16 characters"):
        create_app(make_agent(ScriptedLLM()), api_key="short")


# ------------------------------------------------------------------ CLI


@pytest.fixture
def run_cli(monkeypatch: pytest.MonkeyPatch, make_agent, settings: Settings):
    def runner(argv: list[str], llm: ScriptedLLM | None = None, **overrides: Any) -> int:
        agent = make_agent(llm or ScriptedLLM())
        monkeypatch.setattr(cli.Factories, "agent", staticmethod(lambda _s: agent))
        merged = settings.with_overrides(**overrides) if overrides else settings
        return cli.main(argv, settings_loader=lambda _env: merged)

    return runner


def test_cli_graph(run_cli, capsys: pytest.CaptureFixture[str]):
    assert run_cli(["graph"]) == 0
    assert "write_draft" in capsys.readouterr().out


def test_cli_draft_from_args_and_file(run_cli, capsys, tmp_path: Path):
    assert (
        run_cli(["draft", "--sender", "emma.devries@example.nl", "--body", "PCL100000001NL?"]) == 0
    )
    assert json.loads(capsys.readouterr().out)["outcome"] == "drafted"

    path = tmp_path / "mail.json"
    path.write_text(json.dumps({"sender": "x@y.example", "body": "Rank #1 on Google"}))
    spam = ScriptedLLM(triage=make_triage(category=Category.UNRELATED))
    assert run_cli(["draft", "--file", str(path)], llm=spam) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["outcome"] == "skipped" and out["email_id"] == "mail"

    assert run_cli(["draft", "--sender", "a@b.example"]) == 2


def test_cli_run_folder_then_stats(run_cli, capsys, tmp_path: Path):
    inbox = tmp_path / "inbox"
    shutil.copytree(ROOT / "samples" / "inbox", inbox)

    assert (
        run_cli(["run", "--inbox", str(inbox), "--mode", "draft"], ScriptedLLM(drafts=[GENERIC]))
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert report["fetched"] == 8
    assert report["outcomes"] == {"drafted": 7, "skipped": 1}  # newsletter skipped by headers
    assert len(list((inbox / "outbox").glob("*.draft.txt"))) == 7

    assert run_cli(["run", "--inbox", str(inbox)]) == 0
    assert json.loads(capsys.readouterr().out)["fetched"] == 0

    assert run_cli(["stats", "--days", "1"]) == 0
    stats = json.loads(capsys.readouterr().out)
    assert stats["total"] == 8 and stats["automation_rate"] == 1.0
    assert run_cli(["stats"], metrics_db=None) == 1


def test_cli_run_watch(run_cli, tmp_path: Path):
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    assert (
        run_cli(["run", "--inbox", str(inbox), "--watch", "--interval", "0", "--max-runs", "2"])
        == 0
    )


def test_cli_eval(run_cli, capsys, tmp_path: Path):
    dataset = tmp_path / "e.jsonl"
    rows = [
        {
            "id": "a",
            "sender": "emma.devries@example.nl",
            "body": "PCL100000001NL?",
            "expected_category": "tracking_inquiry",
        },
        {
            "id": "b",
            "sender": "x@lawfirm.example",
            "body": "We will sue.",
            "expected_category": "delivery_issue",
            "expect_escalation": True,
        },
    ]
    dataset.write_text("\n".join(json.dumps(r) for r in rows))
    out = tmp_path / "reports"
    generic = ScriptedLLM(drafts=[GENERIC])
    code = run_cli(["eval", "--dataset", str(dataset), "--out", str(out), "--limit", "2"], generic)
    assert code == 1  # the fake never escalates, so the safety gate fails
    summary = json.loads(capsys.readouterr().out)
    assert summary["cases"] == 2 and summary["escalation_recall"] == 0.0
    assert (out / "eval_report.md").exists()

    code = run_cli(
        ["eval", "--dataset", str(dataset), "--out", str(out), "--min-escalation-recall", "0"]
    )
    assert code == 0


def test_cli_config_error(capsys):
    def broken(_env: str) -> Settings:
        raise ValueError("DELIVERY_MODE must be 'draft' or 'send'")

    assert cli.main(["graph"], settings_loader=broken) == 2
    assert "Configuration error" in capsys.readouterr().err


# ------------------------------------------------------------------ config


def test_settings_from_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    env = tmp_path / ".env"
    env.write_text("COMPANY_NAME=Acme\n")
    monkeypatch.setenv("DELIVERY_MODE", "SEND")
    monkeypatch.setenv("AUTO_SEND_CATEGORIES", "tracking_inquiry, address_change")
    monkeypatch.setenv("ALLOWED_LINK_DOMAINS", "acme.example,help.acme.example")
    monkeypatch.setenv("REPLY_SIGNATURE", "Bye\\nAcme")
    monkeypatch.setenv("METRICS_DB", "none")
    monkeypatch.setenv("MIN_CONFIDENCE", "0.7")
    monkeypatch.setenv("SUPPORT_ADDRESS", "Help@Acme.example")
    monkeypatch.delenv("COMPANY_NAME", raising=False)

    s = Settings.from_env(env)
    assert s.company_name == "Acme"
    assert s.delivery_mode == "send"
    assert s.auto_send_categories == (Category.TRACKING, Category.ADDRESS_CHANGE)
    assert s.allowed_link_domains == ("acme.example", "help.acme.example")
    assert s.signature == "Bye\nAcme"
    assert s.metrics_db is None
    assert s.min_confidence == 0.7
    assert s.support_address == "help@acme.example"


@pytest.mark.parametrize(("name", "value"), [("DELIVERY_MODE", "yolo"), ("MIN_CONFIDENCE", "high")])
def test_settings_reject_bad_values(monkeypatch: pytest.MonkeyPatch, name: str, value: str):
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=name):
        Settings.from_env(None)


# ------------------------------------------------------------------ observability & lambda


def test_redaction():
    assert redact("from emma.devries@example.nl") == "from e***@example.nl"
    assert redact("call +31 6 1234 5678 now") == "call [phone] now"
    assert redact("parcel PCL100000001NL") == "parcel PCL100000001NL"


def test_json_logs_are_redacted(capsys):
    configure_logging("INFO", json_logs=True)
    logging.getLogger("t").info(
        "mail from bob@example.com", extra={"who": "bob@example.com", "n": 3}
    )
    line = json.loads(capsys.readouterr().err.strip().splitlines()[-1])
    assert line["msg"] == "mail from b***@example.com"
    assert line["who"] == "b***@example.com" and line["n"] == 3
    try:
        raise ValueError("bad bob@example.com")
    except ValueError:
        record = logging.LogRecord("t", logging.ERROR, "", 0, "oops", (), None)
        import sys

        record.exc_info = sys.exc_info()
    assert "b***@example.com" in JsonFormatter().format(record)
    configure_logging("WARNING")


class FakeRunner:
    def __init__(self) -> None:
        self.calls: list[tuple[int, float | None]] = []

    def run_once(self, limit: int, *, time_budget_sec: float | None = None) -> RunReport:
        self.calls.append((limit, time_budget_sec))
        report = RunReport(fetched=1)
        report.outcomes[Outcome.DRAFTED.value] += 1
        return report


class Ctx:
    def get_remaining_time_in_millis(self) -> int:
        return 300_000


def test_lambda_handler(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    fake = FakeRunner()
    monkeypatch.setattr(aws_lambda, "_runner", None)
    monkeypatch.setattr(aws_lambda, "_build_runner", lambda: fake)
    assert aws_lambda.handler({"limit": 5}, Ctx())["outcomes"] == {"drafted": 1}
    assert fake.calls == [(5, 210.0)]
    aws_lambda.handler({}, None)
    assert fake.calls[-1] == (20, 510)

    monkeypatch.setenv("GMAIL_TOKEN_JSON", '{"token": "x"}')
    path = aws_lambda._materialise_secret("GMAIL_TOKEN_JSON", tmp_path / "t.json")
    assert path.read_text() == '{"token": "x"}'
    configure_logging("WARNING")
