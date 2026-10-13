"""Language-model access behind a one-method protocol.

The agent only ever asks for *structured* output (a Pydantic object), so swapping
providers is a config change (``LLM_MODEL=groq:llama-3.3-70b-versatile``) and tests can
use a scripted fake. The LangChain adapter adds client-side rate limiting, bounded
exponential backoff on transient errors (429/5xx/timeouts/unparseable output) and token
accounting.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

logger = logging.getLogger(__name__)

ModelT = TypeVar("ModelT", bound=BaseModel)


class LLMError(Exception):
    """The model could not produce a usable answer."""


class LLMTransientError(LLMError):
    """Rate limit, quota or provider outage. Safe to retry the whole email later."""


class LLM(Protocol):
    def structured(self, system: str, user: str, schema: type[ModelT]) -> ModelT: ...


@dataclass
class Usage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    def __sub__(self, other: Usage) -> Usage:
        return Usage(
            self.calls - other.calls,
            self.input_tokens - other.input_tokens,
            self.output_tokens - other.output_tokens,
        )


class UsageTracker:
    def __init__(self) -> None:
        self._usage = Usage()
        self._lock = threading.Lock()

    def record(self, input_tokens: int, output_tokens: int) -> None:
        with self._lock:
            self._usage.calls += 1
            self._usage.input_tokens += input_tokens
            self._usage.output_tokens += output_tokens

    def snapshot(self) -> Usage:
        with self._lock:
            return Usage(self._usage.calls, self._usage.input_tokens, self._usage.output_tokens)


class RateLimiter:
    """Sliding-window limiter: at most ``max_calls`` per ``period`` seconds."""

    def __init__(
        self,
        max_calls: int,
        period: float = 60.0,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if max_calls < 1:
            raise ValueError("max_calls must be >= 1")
        self.max_calls = max_calls
        self.period = period
        self._clock = clock
        self._sleep = sleep
        self._calls: deque[float] = deque()
        self._lock = threading.Lock()

    def acquire(self) -> float:
        """Block until a call is allowed; return how long we waited."""
        waited = 0.0
        with self._lock:
            while True:
                now = self._clock()
                while self._calls and now - self._calls[0] >= self.period:
                    self._calls.popleft()
                if len(self._calls) < self.max_calls:
                    self._calls.append(now)
                    return waited
                delay = self.period - (now - self._calls[0])
                self._sleep(delay)
                waited += delay


_TRANSIENT_MARKERS = (
    "429",
    "rate limit",
    "ratelimit",
    "resource exhausted",
    "resource_exhausted",
    "quota",
    "timeout",
    "timed out",
    "temporarily unavailable",
    "service unavailable",
    "overloaded",
    "503",
    "502",
    "500 internal",
)


def is_transient(exc: BaseException) -> bool:
    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if isinstance(status, int) and (status == 429 or status >= 500):
        return True
    text = f"{type(exc).__name__} {exc}".lower()
    return any(marker in text for marker in _TRANSIENT_MARKERS)


class LangChainLLM:
    """Adapter over any LangChain chat model that supports ``with_structured_output``."""

    def __init__(
        self,
        chat_model: Any,
        *,
        rate_limiter: RateLimiter | None = None,
        usage: UsageTracker | None = None,
        max_retries: int = 3,
        backoff_sec: float = 2.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._model = chat_model
        self._runnables: dict[type[BaseModel], Any] = {}
        self.rate_limiter = rate_limiter
        self.usage = usage or UsageTracker()
        self._max_retries = max_retries
        self._backoff = backoff_sec
        self._sleep = sleep

    @classmethod
    def from_settings(cls, settings: Any) -> LangChainLLM:  # pragma: no cover - needs provider
        from langchain.chat_models import init_chat_model

        model = init_chat_model(settings.model, temperature=settings.temperature)
        return cls(
            model,
            rate_limiter=RateLimiter(settings.llm_calls_per_minute),
            max_retries=settings.llm_max_retries,
        )

    def _runnable(self, schema: type[BaseModel]) -> Any:
        if schema not in self._runnables:
            self._runnables[schema] = self._model.with_structured_output(schema, include_raw=True)
        return self._runnables[schema]

    def structured(self, system: str, user: str, schema: type[ModelT]) -> ModelT:
        runnable = self._runnable(schema)
        messages = [SystemMessage(system), HumanMessage(user)]
        last_error: BaseException | None = None
        for attempt in range(self._max_retries + 1):
            if attempt:
                delay = self._backoff * 2 ** (attempt - 1)
                logger.warning("LLM retry %d for %s in %.1fs", attempt, schema.__name__, delay)
                self._sleep(delay)
            if self.rate_limiter is not None:
                self.rate_limiter.acquire()
            try:
                out = runnable.invoke(messages)
            except Exception as exc:
                if not is_transient(exc):
                    raise LLMError(f"{type(exc).__name__}: {exc}") from exc
                last_error = exc
                continue
            self._record_usage(out.get("raw"))
            parsed = out.get("parsed")
            if isinstance(parsed, schema):
                return parsed
            if isinstance(parsed, dict):
                try:
                    return schema.model_validate(parsed)
                except Exception as exc:
                    last_error = exc
                    continue
            last_error = out.get("parsing_error") or LLMError("model returned no parsed output")
        if last_error is not None and is_transient(last_error):
            raise LLMTransientError(f"LLM unavailable after retries: {last_error}") from last_error
        raise LLMError(f"{schema.__name__} not produced after retries: {last_error}")

    def _record_usage(self, raw: Any) -> None:
        meta = getattr(raw, "usage_metadata", None) or {}
        self.usage.record(int(meta.get("input_tokens", 0)), int(meta.get("output_tokens", 0)))
