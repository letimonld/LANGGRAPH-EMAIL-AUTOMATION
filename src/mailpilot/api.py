"""HTTP API (FastAPI). Dry-run only: it returns the decision, never touches a mailbox.

Security defaults, unlike a typical demo: an API key is required (``X-API-Key``), no CORS,
request size is bounded, and the CLI binds to 127.0.0.1 unless told otherwise.
"""

from __future__ import annotations

import hmac
import time
import uuid
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from mailpilot import __version__
from mailpilot.llm import LLMTransientError
from mailpilot.metrics import MetricsStore
from mailpilot.models import AgentResult, Email
from mailpilot.runner import Agent


class DraftRequest(BaseModel):
    sender: str = Field(pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$", max_length=320)
    sender_name: str = Field(default="", max_length=200)
    subject: str = Field(default="", max_length=500)
    body: str = Field(min_length=1, max_length=20_000)


class DraftResponse(BaseModel):
    request_id: str
    latency_ms: int
    result: AgentResult


def create_app(agent: Agent, *, api_key: str, metrics: MetricsStore | None = None) -> FastAPI:
    if not api_key or len(api_key) < 16:
        raise ValueError("MAILPILOT_API_KEY must be set to a secret of at least 16 characters")

    app = FastAPI(title="mailpilot", version=__version__, docs_url="/docs", redoc_url=None)

    @app.exception_handler(LLMTransientError)
    def llm_unavailable(_request: Request, exc: LLMTransientError) -> JSONResponse:
        return JSONResponse(
            {"detail": "Language model temporarily unavailable, retry later."},
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            headers={"Retry-After": "30"},
        )

    def require_key(x_api_key: Annotated[str | None, Header()] = None) -> None:
        if x_api_key is None or not hmac.compare_digest(x_api_key, api_key):
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or missing API key")

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @app.post("/v1/draft", response_model=DraftResponse, dependencies=[Depends(require_key)])
    def draft(request: DraftRequest) -> DraftResponse:
        request_id = uuid.uuid4().hex
        email = Email(
            id=f"api-{request_id}",
            thread_id=f"api-{request_id}",
            sender=request.sender,
            sender_name=request.sender_name,
            subject=request.subject,
            body=request.body,
        )
        started = time.perf_counter()
        result = agent.process(email)
        return DraftResponse(
            request_id=request_id,
            latency_ms=int((time.perf_counter() - started) * 1000),
            result=result,
        )

    @app.get("/v1/stats", dependencies=[Depends(require_key)])
    def stats() -> dict[str, Any]:
        if metrics is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Metrics are disabled")
        return metrics.summary().to_dict()

    return app
