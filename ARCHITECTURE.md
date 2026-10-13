# Architecture and design decisions

## One graph per email, side-effect free

The graph handles **one** email and returns an `AgentResult`
(`drafted | sent | escalated | skipped | failed` + reply + reason). It reads from the LLM,
the help center and the tracking API, but never writes anywhere.

Why:
- The original design looped over the whole inbox inside the graph. Routers mutated state
  (`emails.pop()`), and one bad email could crash the whole batch: an `IndexError` on an
  empty list after a rejected draft. With one email per graph run, the graph always
  terminates and errors stay contained.
- `Runner` owns side effects. It isolates failures per email, defers the rest of the batch
  on rate limits, and labels the email *after* acting on it.
- The same graph serves the CLI, the scheduled job, the HTTP dry-run API and the
  evaluation harness.

## Routing policy lives in one class

`Router.after_triage` and `Router.after_review` hold every branching rule:

1. `escalate_reason` set, or `confidence < MIN_CONFIDENCE` → **escalate**. An uncertain
   "unrelated" email also goes to a human, not to the bin.
2. `unrelated` → **skip**.
3. Otherwise → gather context → write → review. If the draft is rejected and attempts are
   left (`MAX_DRAFT_ATTEMPTS`), rewrite with the review issues as feedback. If attempts run
   out → **escalate**.

`finalize` decides between a draft and auto-send. Auto-send needs all of these: `send`
mode, an allow-listed category, confidence ≥ `AUTO_SEND_MIN_CONFIDENCE`, a customer who is
not angry, and every parcel verified.

## Why two review layers

| Layer | Catches | Cost |
|---|---|---|
| `guardrails.check_reply` (regex/rules) | invented tracking numbers, foreign links, leaked addresses, refund promises, placeholders, "as an AI" | free, deterministic, unit-tested |
| LLM reviewer | ungrounded claims, wrong language, not answering the question, tone | 1 call; only runs if the rules pass |

The rule messages are specific ("Mentions tracking numbers that are not in the verified
facts: PCL123..."), so the writer fixes the actual problem on the retry.

## Privacy by construction

`lookup_for_sender` compares the sender's address with the parcel's recipient. For a
mismatch, the agent gets a fact with **no parcel data at all**, just an instruction to ask
for verification. The model cannot leak what it never sees. The guardrail also rejects any
tracking number the customer did not provide that is not a verified fact.

## Retrieval

The help center is markdown, split by `##` section, and each section has an id like
`returns#how-to-return-a-parcel`. BM25 is the default: support questions are short and
keyword-heavy, BM25 needs no API key, and it is deterministic in tests. `EmbeddingRetriever`
(any LangChain `Embeddings`) and `HybridRetriever` (reciprocal rank fusion) are included
for larger or more paraphrase-heavy knowledge bases. The triage call also writes up to
three English search queries, which saves the separate "query writer" LLM call the
original design used.

## LLM calls per email

| Path | Calls |
|---|---|
| bulk/auto-reply (header-detected) | 0 |
| spam / escalation | 1 (triage) |
| normal reply, approved first time | 3 (triage, write, review) |
| each rejected draft | +1 (write) and +1 (review, only if the rules pass) |

The original used 1 + up to 3 (RAG) + 2×(up to 3) calls and re-ran categorisation after
failures.

## Failure handling

- `LLMTransientError` (429, quota, 5xx, timeouts, after bounded backoff): the email is
  **not** labelled and the rest of the batch is deferred to the next run.
- Any other exception: the email is labelled `mailpilot/failed` so a human sees it, and the
  batch continues.
- Small schema drift (confidence 1.2, four queries, the string "None") is repaired by
  validators instead of a re-ask.
- On Lambda, `run_once(time_budget_sec=...)` stops before the hard timeout.

## Extending

- **Real tracking API:** implement `TrackingService.lookup()`; nothing else changes.
- **Another inbox** (Zendesk, Freshdesk, Outlook): implement the four-method `Mailbox`
  protocol.
- **Another model:** set `LLM_MODEL` and install the provider extra.
- **New category:** add it to `Category` and the triage prompt, and add labelled cases to
  `evals/emails.jsonl`.
