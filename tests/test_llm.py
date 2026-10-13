from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from mailpilot.llm import (
    LangChainLLM,
    LLMError,
    LLMTransientError,
    RateLimiter,
    Usage,
    is_transient,
)
from mailpilot.models import Review


class StubRunnable:
    def __init__(self, outputs: list[Any]) -> None:
        self.outputs = outputs
        self.calls = 0

    def invoke(self, messages: list[Any]) -> Any:
        self.calls += 1
        item = self.outputs.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class StubChatModel:
    def __init__(self, outputs: list[Any]) -> None:
        self.runnable = StubRunnable(outputs)
        self.schemas: list[Any] = []

    def with_structured_output(self, schema: Any, include_raw: bool = False) -> StubRunnable:
        assert include_raw
        self.schemas.append(schema)
        return self.runnable


def raw(input_tokens: int = 10, output_tokens: int = 5) -> Any:
    return SimpleNamespace(
        usage_metadata={"input_tokens": input_tokens, "output_tokens": output_tokens}
    )


class RateLimited(Exception):
    status_code = 429


def make(outputs: list[Any], **kwargs: Any) -> tuple[LangChainLLM, StubChatModel, list[float]]:
    model = StubChatModel(outputs)
    sleeps: list[float] = []
    llm = LangChainLLM(model, sleep=sleeps.append, backoff_sec=1.0, **kwargs)
    return llm, model, sleeps


def test_parsed_object_and_usage_are_returned():
    ok = {"raw": raw(), "parsed": Review(approved=True)}
    llm, model, _ = make([ok, ok])
    assert llm.structured("sys", "user", Review).approved is True
    llm.structured("sys", "again", Review)
    assert llm.usage.snapshot() == Usage(2, 20, 10)
    assert model.schemas == [Review]  # structured runnable is built once per schema


def test_dict_output_is_validated():
    llm, _, _ = make([{"raw": raw(), "parsed": {"approved": False, "issues": ["x"]}}])
    assert llm.structured("s", "u", Review).issues == ["x"]


def test_transient_errors_are_retried_with_backoff():
    ok = {"raw": raw(), "parsed": Review(approved=True)}
    llm, model, sleeps = make([RateLimited("slow down"), RateLimited("again"), ok])
    assert llm.structured("s", "u", Review).approved
    assert sleeps == [1.0, 2.0]
    assert model.runnable.calls == 3


def test_transient_errors_exhaust_into_transient_error():
    llm, _, _ = make([RateLimited("x")] * 3, max_retries=2)
    with pytest.raises(LLMTransientError):
        llm.structured("s", "u", Review)


def test_permanent_errors_fail_fast():
    llm, model, _ = make([ValueError("invalid api key")])
    with pytest.raises(LLMError, match="invalid api key"):
        llm.structured("s", "u", Review)
    assert model.runnable.calls == 1


def test_unparseable_output_is_retried_then_fails():
    bad = {"raw": raw(), "parsed": None, "parsing_error": ValueError("bad json")}
    invalid = {"raw": raw(), "parsed": {"approved": "maybe?"}}
    llm, _, _ = make([bad, invalid], max_retries=1)
    with pytest.raises(LLMError, match="Review not produced"):
        llm.structured("s", "u", Review)
    assert llm.usage.snapshot().calls == 2


def test_rate_limiter_waits_for_the_window():
    now = [0.0]
    slept: list[float] = []

    def sleep(seconds: float) -> None:
        slept.append(seconds)
        now[0] += seconds

    limiter = RateLimiter(2, period=10, clock=lambda: now[0], sleep=sleep)
    assert limiter.acquire() == 0
    now[0] = 1
    assert limiter.acquire() == 0
    assert limiter.acquire() == pytest.approx(9)
    assert slept == [pytest.approx(9)]


def test_rate_limiter_and_llm_integration():
    limiter = RateLimiter(5)
    llm, _, _ = make([{"raw": None, "parsed": Review(approved=True)}], rate_limiter=limiter)
    llm.structured("s", "u", Review)
    assert llm.usage.snapshot() == Usage(1, 0, 0)
    with pytest.raises(ValueError):
        RateLimiter(0)


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (RateLimited(), True),
        (SimpleNamespace(code=503), True),
        (Exception("Resource exhausted: quota"), True),
        (TimeoutError("timed out"), True),
        (Exception("400 bad request"), False),
        (KeyError("x"), False),
    ],
)
def test_is_transient(exc: Any, expected: bool):
    assert is_transient(exc) is expected
