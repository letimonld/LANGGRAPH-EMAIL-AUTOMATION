"""Domain models shared by the agent, the mailbox adapters, the API and the evaluation."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Category(str, Enum):
    """What the customer wants. The router and the metrics are keyed on this."""

    TRACKING = "tracking_inquiry"  # "where is my parcel?"
    DELIVERY_ISSUE = "delivery_issue"  # damaged, lost, "delivered" but not received, late
    RETURN_REFUND = "return_refund"  # how to return, refund status
    ADDRESS_CHANGE = "address_change"  # change address / redirect / reschedule
    SHIPPING_QUESTION = "shipping_question"  # prices, carriers, customs, packaging, account
    FEEDBACK = "feedback"  # compliments or general feedback, no action needed
    UNRELATED = "unrelated"  # spam, newsletters, sales pitches, auto-replies


class Urgency(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"


class Sentiment(str, Enum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    ANGRY = "angry"


class Outcome(str, Enum):
    """What happened to an email. ``DRAFTED`` leaves a reply for a human to send."""

    DRAFTED = "drafted"
    SENT = "sent"
    ESCALATED = "escalated"
    SKIPPED = "skipped"
    FAILED = "failed"


class Email(BaseModel):
    """An inbound customer email. Everything in it is untrusted input."""

    model_config = ConfigDict(frozen=True)

    id: str
    thread_id: str = ""
    message_id: str = ""
    references: str = ""
    sender: str = Field(description="Bare email address of the sender, lower-cased.")
    sender_name: str = ""
    subject: str = ""
    body: str = Field(max_length=50_000)
    received_at: datetime | None = None
    auto_generated: bool = Field(
        default=False,
        description="Bulk mail, mailing list or auto-reply (detected from headers).",
    )

    @field_validator("sender")
    @classmethod
    def _normalise_sender(cls, value: str) -> str:
        return value.strip().lower()


# ---------------------------------------------------------------- LLM outputs
# These are the JSON schemas the model must fill. Descriptions are part of the prompt.


class Triage(BaseModel):
    category: Category = Field(description="The single best category for the email.")
    confidence: float = Field(
        ge=0.0, le=1.0, description="How sure you are about the category, 0 to 1."
    )
    urgency: Urgency = Field(description="high: parcel lost/damaged, perishable, deadline today.")
    sentiment: Sentiment
    language: str = Field(
        description="ISO 639-1 code of the language the customer wrote in, e.g. en, nl, de, fr."
    )
    tracking_numbers: list[str] = Field(
        default_factory=list,
        description="Tracking numbers exactly as written in the email. Empty if none.",
    )
    kb_queries: list[str] = Field(
        default_factory=list,
        max_length=3,
        description="Up to 3 short English search queries for the help-center knowledge base "
        "that would help answer the email. Empty for unrelated or feedback emails.",
    )
    summary: str = Field(description="One English sentence describing what the customer wants.")
    escalate_reason: str | None = Field(
        default=None,
        description="Set ONLY if a human must handle this: legal threat, fraud or account "
        "takeover, GDPR/data request, press, a request to speak to a person, a threat of harm, "
        "or a business/contract negotiation. Otherwise null.",
    )

    # Models drift slightly outside the schema now and then; repair instead of re-asking.
    @field_validator("confidence", mode="before")
    @classmethod
    def _clamp_confidence(cls, value: object) -> object:
        if isinstance(value, (int, float)):
            return min(1.0, max(0.0, float(value)))
        return value

    @field_validator("kb_queries", mode="before")
    @classmethod
    def _cap_queries(cls, value: object) -> object:
        return value[:3] if isinstance(value, list) else value

    @field_validator("escalate_reason", mode="before")
    @classmethod
    def _blank_is_none(cls, value: object) -> object:
        if isinstance(value, str) and value.strip().lower() in ("", "none", "null", "n/a"):
            return None
        return value


class DraftReply(BaseModel):
    body: str = Field(
        description="The reply body in the customer's language, without greeting signature "
        "block (it is appended automatically). Plain text, no markdown."
    )
    used_sources: list[str] = Field(
        default_factory=list,
        description="IDs of the knowledge snippets and tracking numbers the reply relies on.",
    )


class Review(BaseModel):
    approved: bool = Field(description="True only if the reply can go to the customer as is.")
    issues: list[str] = Field(
        default_factory=list,
        description="Concrete problems to fix. Empty when approved.",
    )


# ---------------------------------------------------------------- tool results


class TrackingEvent(BaseModel):
    timestamp: datetime
    location: str
    description: str


class Parcel(BaseModel):
    tracking_number: str
    carrier: str
    status: str
    recipient_email: str
    destination_country: str
    estimated_delivery: str | None = None
    events: list[TrackingEvent] = Field(default_factory=list)


class Snippet(BaseModel):
    """A passage of the help-center knowledge base."""

    id: str
    title: str
    text: str
    score: float = 0.0


class AgentResult(BaseModel):
    """The decision the agent made for one email. Side effects are applied by the runner."""

    email_id: str
    outcome: Outcome
    category: Category | None = None
    confidence: float | None = None
    reply: str | None = None
    reason: str | None = None
    attempts: int = 0
    sources: list[str] = Field(default_factory=list)
    review_issues: list[str] = Field(default_factory=list)
