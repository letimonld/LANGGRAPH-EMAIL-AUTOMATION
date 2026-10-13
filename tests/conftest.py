from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from fakes import ScriptedLLM
from mailpilot.agent import AgentDeps, SupportAgent
from mailpilot.config import Settings
from mailpilot.knowledge import BM25Retriever
from mailpilot.tracking import JsonTrackingService

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        knowledge_dir=ROOT / "knowledge",
        parcels_file=ROOT / "data" / "parcels.json",
        metrics_db=tmp_path / "metrics.sqlite3",
    )


@pytest.fixture
def retriever(settings: Settings) -> BM25Retriever:
    return BM25Retriever.from_directory(settings.knowledge_dir)


@pytest.fixture
def tracking(settings: Settings) -> JsonTrackingService:
    return JsonTrackingService.from_file(settings.parcels_file)


@pytest.fixture
def make_agent(settings: Settings, retriever: BM25Retriever, tracking: JsonTrackingService):
    def factory(llm: ScriptedLLM, **overrides: object) -> SupportAgent:
        return SupportAgent(
            AgentDeps(
                llm=llm,
                retriever=retriever,
                tracking=tracking,
                settings=settings.with_overrides(**overrides) if overrides else settings,
            )
        )

    return factory
