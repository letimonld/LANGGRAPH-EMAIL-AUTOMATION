# Changelog

## 2.0.0: rewrite

The domain moved from a generic AI agency to parcel-shipping customer support. Everything
below was rebuilt.

### Fixed (problems in 1.x)
- **Crash:** after a draft failed review three times, the graph routed back to
  `categorize_email`, which read `state["emails"][-1]` on an empty list (`IndexError`).
  `trials` was not reset either, so the next email started with no attempts left.
- **State mutated inside routers:** `must_rewrite` and `skip_unrelated_email` changed state
  directly (`pop()`, assignments). This only worked because lists are shared by reference.
- **Unsafe API:** `deploy_api.py` exposed the workflow, which had Gmail `modify` scope, on
  `0.0.0.0` with `CORS *` and no authentication.
- **Duplicate replies:** "already answered" was inferred from drafts on the thread. Any
  email whose draft was sent or deleted got answered again.
- The Gmail fetch window was hard-coded to the last 8 hours, the support account's own
  replies were not excluded, and multipart/HTML bodies were parsed unreliably.
- Dependencies were unpinned, it used the retired `gemini-1.5-flash`, and the binary Chroma
  database was committed.
- There were no tests.

### Added
- A side-effect-free LangGraph with a `Router`, and a `Runner` with per-email failure
  isolation, deferral on rate limits and a time budget.
- A tracking tool with recipient verification, and multilingual tracking-number extraction.
- Deterministic guardrails before the LLM reviewer.
- BM25 / embedding / hybrid retrieval over a markdown help center.
- A provider-agnostic structured LLM adapter with retries, rate limiting and token
  accounting.
- Gmail labels for idempotency, threading headers, bulk-mail detection, and a folder mailbox.
- An evaluation harness and a labelled set of 47 emails; the eval exits non-zero when
  escalation recall is below target.
- SQLite metrics: automation rate, latency, tokens, estimated time saved.
- An authenticated FastAPI dry-run API, a CLI, JSON logs with PII redaction.
- An AWS SAM deployment (Lambda + EventBridge + Secrets Manager + alarm), a Dockerfile, and
  CI (ruff, mypy, pytest ≥ 90% coverage, Python 3.10–3.13).
