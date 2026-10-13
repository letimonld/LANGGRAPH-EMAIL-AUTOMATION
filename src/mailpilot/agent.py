"""The LangGraph workflow that decides what to do with ONE email.

The graph is side-effect free: it reads (LLM, knowledge base, tracking API) and returns an
``AgentResult``. Touching the mailbox (drafting, sending, labelling) is the runner's job,
which keeps the graph trivially testable and lets the HTTP API reuse it for dry runs.

Flow::

    triage ─┬─ unrelated ──────────────────────────────> skip ──────> END
            ├─ low confidence / needs a human ─────────> escalate ──> END
            └─> gather_context ─> write_draft ─> review ─┬─ approved ─> finalize ─> END
                                       ^                  ├─ rejected, attempts left ┘
                                       └──────────────────┘
                                                          └─ out of attempts ─> escalate
"""

from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph

from mailpilot import prompts
from mailpilot.config import Settings
from mailpilot.guardrails import ReplyPolicy, check_reply, find_addresses
from mailpilot.knowledge import Retriever
from mailpilot.llm import LLM
from mailpilot.models import (
    AgentResult,
    Category,
    DraftReply,
    Email,
    Outcome,
    Review,
    Sentiment,
    Snippet,
    Triage,
)
from mailpilot.tracking import (
    TRACKING_RE,
    ParcelFact,
    TrackingService,
    extract_tracking_numbers,
    lookup_for_sender,
    normalise,
    render_facts,
)

RECURSION_LIMIT = 40
_NO_KB_CATEGORIES = frozenset({Category.FEEDBACK, Category.UNRELATED})


class AgentState(TypedDict, total=False):
    email: Email
    triage: Triage
    parcel_facts: list[ParcelFact]
    snippets: list[Snippet]
    draft: DraftReply
    approved: bool
    feedback: list[str]  # issues from the latest review, shown to the writer
    issue_log: Annotated[list[str], operator.add]  # every issue ever raised, for reporting
    attempts: int
    result: AgentResult


@dataclass
class AgentDeps:
    llm: LLM
    retriever: Retriever
    tracking: TrackingService
    settings: Settings


class Nodes:
    def __init__(self, deps: AgentDeps) -> None:
        self.deps = deps
        self.s = deps.settings

    # ------------------------------------------------------------------ triage
    def triage(self, state: AgentState) -> dict[str, Any]:
        email = state["email"]
        result = self.deps.llm.structured(
            prompts.triage_system(self.s.company_name), prompts.triage_user(email), Triage
        )
        # Trust the regex over the model for identifiers, but keep valid ones the model saw.
        from_model = [normalise(n) for n in result.tracking_numbers]
        numbers = extract_tracking_numbers(email.subject, email.body)
        numbers += [n for n in from_model if TRACKING_RE.fullmatch(n) and n not in numbers]
        return {
            "triage": result.model_copy(
                update={"tracking_numbers": numbers, "language": result.language.lower()[:5]}
            )
        }

    # ------------------------------------------------------------------ terminal nodes
    def skip(self, state: AgentState) -> dict[str, Any]:
        triage = state["triage"]
        return {"result": self._result(state, Outcome.SKIPPED, reason=triage.summary)}

    def escalate(self, state: AgentState) -> dict[str, Any]:
        triage = state["triage"]
        if state.get("attempts", 0) >= self.s.max_draft_attempts and not state.get("approved"):
            reason = f"Draft rejected {state['attempts']} times: " + "; ".join(
                state.get("feedback", [])
            )
        elif triage.escalate_reason:
            reason = triage.escalate_reason
        else:
            reason = f"Low triage confidence ({triage.confidence:.2f})"
        return {"result": self._result(state, Outcome.ESCALATED, reason=reason)}

    # ------------------------------------------------------------------ drafting loop
    def gather_context(self, state: AgentState) -> dict[str, Any]:
        email, triage = state["email"], state["triage"]
        facts = lookup_for_sender(self.deps.tracking, triage.tracking_numbers, email.sender)
        snippets: list[Snippet] = []
        if triage.category not in _NO_KB_CATEGORIES:
            queries = triage.kb_queries or [triage.summary]
            snippets = self.deps.retriever.search(queries, self.s.kb_top_k)
        return {"parcel_facts": facts, "snippets": snippets}

    def write_draft(self, state: AgentState) -> dict[str, Any]:
        draft = self.deps.llm.structured(
            prompts.draft_system(self.s.company_name, self.s.support_address),
            prompts.draft_user(
                state["email"],
                state["triage"],
                render_facts(state.get("parcel_facts", [])),
                state.get("snippets", []),
                state.get("feedback", []),
            ),
            DraftReply,
        )
        return {"draft": draft, "attempts": state.get("attempts", 0) + 1}

    def review(self, state: AgentState) -> dict[str, Any]:
        email, draft = state["email"], state["draft"]
        policy = ReplyPolicy.build(
            sender=email.sender,
            support_address=self.s.support_address,
            allowed_link_domains=self.s.allowed_link_domains,
            known_tracking_numbers=self._shareable_numbers(state),
            public_addresses=find_addresses(" ".join(s.text for s in state.get("snippets", []))),
            max_chars=self.s.max_reply_chars,
        )
        issues = check_reply(draft.body, policy)
        if not issues:  # only pay for the LLM reviewer once the cheap checks pass
            verdict = self.deps.llm.structured(
                prompts.review_system(self.s.company_name),
                prompts.review_user(
                    email,
                    state["triage"],
                    render_facts(state.get("parcel_facts", [])),
                    state.get("snippets", []),
                    draft.body,
                ),
                Review,
            )
            if not verdict.approved:
                issues = verdict.issues or ["Reviewer rejected the draft without details."]
        return {"approved": not issues, "feedback": issues, "issue_log": issues}

    def finalize(self, state: AgentState) -> dict[str, Any]:
        triage = state["triage"]
        reply = f"{state['draft'].body.strip()}\n\n{self.s.signature}"
        auto_send = (
            self.s.delivery_mode == "send"
            and triage.category in self.s.auto_send_categories
            and triage.confidence >= self.s.auto_send_min_confidence
            and triage.sentiment is not Sentiment.ANGRY
            and all(f.verified for f in state.get("parcel_facts", []))
        )
        outcome = Outcome.SENT if auto_send else Outcome.DRAFTED
        return {"result": self._result(state, outcome, reply=reply)}

    # ------------------------------------------------------------------ helpers
    def _shareable_numbers(self, state: AgentState) -> set[str]:
        """Tracking numbers the reply may mention: the customer's own, or verified ones."""
        allowed = set(extract_tracking_numbers(state["email"].subject, state["email"].body))
        allowed |= {f.tracking_number for f in state.get("parcel_facts", []) if f.verified}
        return allowed

    def _result(
        self,
        state: AgentState,
        outcome: Outcome,
        *,
        reply: str | None = None,
        reason: str | None = None,
    ) -> AgentResult:
        triage = state.get("triage")
        draft = state.get("draft")
        return AgentResult(
            email_id=state["email"].id,
            outcome=outcome,
            category=triage.category if triage else None,
            confidence=triage.confidence if triage else None,
            reply=reply,
            reason=reason,
            attempts=state.get("attempts", 0),
            sources=list(draft.used_sources) if draft else [],
            review_issues=list(state.get("issue_log", [])),
        )


class Router:
    """Every branching decision lives here, so the policy is readable in one place."""

    def __init__(self, settings: Settings) -> None:
        self.s = settings

    def after_triage(self, state: AgentState) -> str:
        triage = state["triage"]
        if triage.escalate_reason or triage.confidence < self.s.min_confidence:
            return "escalate"
        if triage.category is Category.UNRELATED:
            return "skip"
        return "gather_context"

    def after_review(self, state: AgentState) -> str:
        if state.get("approved"):
            return "finalize"
        if state.get("attempts", 0) < self.s.max_draft_attempts:
            return "write_draft"
        return "escalate"


def build_graph(deps: AgentDeps) -> Any:
    nodes, router = Nodes(deps), Router(deps.settings)
    graph = StateGraph(AgentState)
    for name in (
        "triage",
        "skip",
        "escalate",
        "gather_context",
        "write_draft",
        "review",
        "finalize",
    ):
        graph.add_node(name, getattr(nodes, name))
    graph.add_edge(START, "triage")
    graph.add_conditional_edges(
        "triage", router.after_triage, ["escalate", "skip", "gather_context"]
    )
    graph.add_edge("gather_context", "write_draft")
    graph.add_edge("write_draft", "review")
    graph.add_conditional_edges(
        "review", router.after_review, ["finalize", "write_draft", "escalate"]
    )
    for terminal in ("skip", "escalate", "finalize"):
        graph.add_edge(terminal, END)
    return graph.compile()


class SupportAgent:
    def __init__(self, deps: AgentDeps) -> None:
        self.deps = deps
        self.graph = build_graph(deps)

    def process(self, email: Email) -> AgentResult:
        state = self.graph.invoke(
            {"email": email, "attempts": 0}, {"recursion_limit": RECURSION_LIMIT}
        )
        result: AgentResult = state["result"]
        return result

    def mermaid(self) -> str:
        return str(self.graph.get_graph().draw_mermaid())


def create_agent(settings: Settings | None = None) -> SupportAgent:  # pragma: no cover
    """Production wiring: LangChain chat model + BM25 help center + JSON tracking stub."""
    from mailpilot.knowledge import BM25Retriever
    from mailpilot.llm import LangChainLLM
    from mailpilot.tracking import JsonTrackingService

    settings = settings or Settings.from_env()
    return SupportAgent(
        AgentDeps(
            llm=LangChainLLM.from_settings(settings),
            retriever=BM25Retriever.from_directory(settings.knowledge_dir),
            tracking=JsonTrackingService.from_file(settings.parcels_file),
            settings=settings,
        )
    )
