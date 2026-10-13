"""Settings, read once from the environment (and an optional ``.env`` file)."""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

from mailpilot.models import Category

DeliveryMode = Literal["draft", "send"]

DEFAULT_MODEL = "google_genai:gemini-2.5-flash"


def _env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    return value.strip() if value and value.strip() else default


def _env_float(name: str, default: float) -> float:
    raw = _env(name)
    try:
        return float(raw) if raw is not None else default
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc


def _env_int(name: str, default: int) -> int:
    return int(_env_float(name, default))


def _env_list(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw = _env(name)
    if raw is None:
        return default
    return tuple(part.strip() for part in raw.split(",") if part.strip())


@dataclass(frozen=True)
class Settings:
    # Language model (any LangChain provider string, e.g. "groq:llama-3.3-70b-versatile")
    model: str = DEFAULT_MODEL
    temperature: float = 0.2
    llm_max_retries: int = 3
    llm_calls_per_minute: int = 30

    # Company / reply style
    company_name: str = "Parcelo"
    support_address: str = "support@parcelo.example"
    signature: str = "Kind regards,\nThe Parcelo Support Team"
    allowed_link_domains: tuple[str, ...] = ("parcelo.example",)

    # Agent policy
    min_confidence: float = 0.6  # below this the email goes to a human
    max_draft_attempts: int = 3  # writer -> reviewer loops before escalating
    max_reply_chars: int = 2_500
    kb_top_k: int = 4

    # Delivery: "draft" leaves Gmail drafts for a human; "send" auto-sends the safe subset
    delivery_mode: DeliveryMode = "draft"
    auto_send_categories: tuple[Category, ...] = (Category.TRACKING,)
    auto_send_min_confidence: float = 0.9

    # Files
    knowledge_dir: Path = Path("knowledge")
    parcels_file: Path = Path("data/parcels.json")
    metrics_db: Path | None = Path(".mailpilot/metrics.sqlite3")

    # Mailbox
    gmail_credentials_file: Path = Path("credentials.json")
    gmail_token_file: Path = Path("token.json")
    gmail_query: str = "in:inbox newer_than:2d -from:me -category:promotions"
    label_prefix: str = "mailpilot"

    # API
    api_key: str | None = None

    # Business reporting
    minutes_saved_per_draft: float = 4.0

    @classmethod
    def from_env(cls, env_file: str | Path | None = ".env") -> Settings:
        if env_file is not None and Path(env_file).exists():
            from dotenv import load_dotenv

            load_dotenv(env_file, override=False)
        mode = (_env("DELIVERY_MODE", "draft") or "draft").lower()
        if mode not in ("draft", "send"):
            raise ValueError(f"DELIVERY_MODE must be 'draft' or 'send', got {mode!r}")
        metrics = _env("METRICS_DB", str(cls.metrics_db))
        return cls(
            model=_env("LLM_MODEL", DEFAULT_MODEL) or DEFAULT_MODEL,
            temperature=_env_float("LLM_TEMPERATURE", cls.temperature),
            llm_max_retries=_env_int("LLM_MAX_RETRIES", cls.llm_max_retries),
            llm_calls_per_minute=_env_int("LLM_CALLS_PER_MINUTE", cls.llm_calls_per_minute),
            company_name=_env("COMPANY_NAME", cls.company_name) or cls.company_name,
            support_address=(_env("SUPPORT_ADDRESS", cls.support_address) or "").lower(),
            signature=(_env("REPLY_SIGNATURE", cls.signature) or cls.signature).replace(
                "\\n", "\n"
            ),
            allowed_link_domains=_env_list("ALLOWED_LINK_DOMAINS", cls.allowed_link_domains),
            min_confidence=_env_float("MIN_CONFIDENCE", cls.min_confidence),
            max_draft_attempts=_env_int("MAX_DRAFT_ATTEMPTS", cls.max_draft_attempts),
            max_reply_chars=_env_int("MAX_REPLY_CHARS", cls.max_reply_chars),
            kb_top_k=_env_int("KB_TOP_K", cls.kb_top_k),
            delivery_mode=mode,  # type: ignore[arg-type]
            auto_send_categories=tuple(
                Category(c) for c in _env_list("AUTO_SEND_CATEGORIES", ("tracking_inquiry",))
            ),
            auto_send_min_confidence=_env_float(
                "AUTO_SEND_MIN_CONFIDENCE", cls.auto_send_min_confidence
            ),
            knowledge_dir=Path(_env("KNOWLEDGE_DIR", str(cls.knowledge_dir)) or "knowledge"),
            parcels_file=Path(_env("PARCELS_FILE", str(cls.parcels_file)) or "data/parcels.json"),
            metrics_db=Path(metrics) if metrics and metrics.lower() != "none" else None,
            gmail_credentials_file=Path(
                _env("GMAIL_CREDENTIALS_FILE", str(cls.gmail_credentials_file)) or ""
            ),
            gmail_token_file=Path(_env("GMAIL_TOKEN_FILE", str(cls.gmail_token_file)) or ""),
            gmail_query=_env("GMAIL_QUERY", cls.gmail_query) or cls.gmail_query,
            label_prefix=_env("LABEL_PREFIX", cls.label_prefix) or cls.label_prefix,
            api_key=_env("MAILPILOT_API_KEY"),
            minutes_saved_per_draft=_env_float(
                "MINUTES_SAVED_PER_DRAFT", cls.minutes_saved_per_draft
            ),
        )

    def with_overrides(self, **changes: object) -> Settings:
        return replace(self, **changes)  # type: ignore[arg-type]
