from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from mailpilot.knowledge import (
    BM25Retriever,
    EmbeddingRetriever,
    HybridRetriever,
    load_markdown,
    tokenize,
)
from mailpilot.models import Snippet
from mailpilot.tracking import (
    JsonTrackingService,
    extract_tracking_numbers,
    lookup_for_sender,
    render_facts,
)

# ------------------------------------------------------------------ knowledge


def test_markdown_is_split_into_sections(settings):
    snippets = load_markdown(settings.knowledge_dir)
    ids = {s.id for s in snippets}
    assert "returns#how-to-return-a-parcel" in ids
    assert "customs#shipping-outside-the-eu" in ids
    assert all(s.text.strip() for s in snippets)
    assert any(s.title.startswith("Customs and duties ›") for s in snippets)


def test_markdown_without_sections_or_title(tmp_path: Path):
    (tmp_path / "faq_page.md").write_text("intro only\n## Q1\nAnswer one\n## Empty\n")
    snippets = load_markdown(tmp_path)
    assert [s.id for s in snippets] == ["faq_page#q1"]
    assert snippets[0].title == "Faq Page › Q1"


@pytest.mark.parametrize(
    ("query", "expected_doc"),
    [
        ("customs form parcels to the UK", "customs#"),
        ("how do I get a return label", "returns#"),
        ("parcel marked delivered but not received", "delivery_issues#"),
        ("can I ship power banks lithium batteries", "shipping#prohibited-items"),
        ("redirect parcel to pickup point", "address_change#"),
        ("label printer prints tiny labels", "shipping#printing-labels"),
    ],
)
def test_bm25_finds_the_right_article(retriever: BM25Retriever, query: str, expected_doc: str):
    top = retriever.search([query], k=2)
    assert top and top[0].id.startswith(expected_doc), [s.id for s in top]
    assert top[0].score > 0


def test_bm25_merges_queries_and_ignores_noise(retriever: BM25Retriever):
    assert retriever.search(["the a of"], k=3) == []
    merged = retriever.search(["customs duties", "return label"], k=4)
    docs = {s.id.split("#")[0] for s in merged}
    assert {"customs", "returns"} <= docs


def test_missing_directory_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        BM25Retriever.from_directory(tmp_path / "nope")


def test_tokenize_drops_stopwords():
    assert tokenize("Where is MY parcel, PCL123?") == ["parcel", "pcl123"]


class HashEmbeddings:
    """Bag-of-words hashed into 64 dims: similar words -> similar vectors, no network."""

    def _vec(self, text: str) -> list[float]:
        vec = [0.0] * 64
        for token in tokenize(text):
            vec[int(hashlib.md5(token.encode()).hexdigest(), 16) % 64] += 1.0  # noqa: S324
        return vec

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


def test_embedding_and_hybrid_retrievers(retriever: BM25Retriever):
    dense = EmbeddingRetriever(retriever.snippets, HashEmbeddings())
    assert dense.search(["customs declaration UK"], k=1)[0].id.startswith("customs#")
    strict = EmbeddingRetriever(retriever.snippets, HashEmbeddings(), min_score=0.99)
    assert strict.search(["zzzz qqqq"], k=3) == []

    hybrid = HybridRetriever(retriever, dense)
    top = hybrid.search(["return label refund"], k=3)
    assert top[0].id.startswith("returns#")
    assert len(top) == 3


def test_hybrid_requires_the_same_corpus(retriever: BM25Retriever):
    other = EmbeddingRetriever([Snippet(id="x", title="x", text="x")], HashEmbeddings())
    with pytest.raises(ValueError, match="same snippets"):
        HybridRetriever(retriever, other)


# ------------------------------------------------------------------ tracking


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("my parcel PCL100000001NL", ["PCL100000001NL"]),
        ("pcl 100 000 001 nl please", ["PCL100000001NL"]),
        ("PCL-100000001-NL and again PCL100000001NL", ["PCL100000001NL"]),
        ("ups 1Z999AA10123456784, dhl JJD000390007788123456", None),
        ("order 12345, phone +31 6 1234 5678", []),
        ("PCL12345NL is too short", []),
    ],
)
def test_extract_tracking_numbers(text: str, expected: list[str] | None):
    found = extract_tracking_numbers(text)
    if expected is None:
        assert found == ["1Z999AA10123456784", "JJD000390007788123456"]
    else:
        assert found == expected


def test_lookup_respects_recipient_privacy(tracking: JsonTrackingService):
    facts = lookup_for_sender(
        tracking,
        ["PCL100000001NL", "pcl100000002nl", "PCL999999999NL", "PCL100000001NL", " "],
        "Emma.DeVries@example.nl",
    )
    assert [(f.tracking_number, f.found, f.verified) for f in facts] == [
        ("PCL100000001NL", True, True),
        ("PCL100000002NL", True, False),
        ("PCL999999999NL", False, False),
    ]
    assert facts[1].parcel is None  # unverified parcels carry no data at all

    text = render_facts(facts)
    assert "PostNL" in text and "Nieuwegein" in text
    assert "NOT the recipient" in text and "DHL" not in text
    assert "NOT FOUND" in text


def test_render_without_numbers():
    assert render_facts([]) == "No tracking numbers were provided or found."


def test_lookup_limit(tracking: JsonTrackingService):
    numbers = [f"PCL10000000{i}NL" for i in range(1, 10)]
    assert len(lookup_for_sender(tracking, numbers, "x@y.z", limit=3)) == 3


# ------------------------------------------------------------------ model repair


def test_triage_repairs_small_schema_drift():
    from fakes import make_triage

    triage = make_triage(confidence=1.3, kb_queries=["a", "b", "c", "d"], escalate_reason="None")
    assert triage.confidence == 1.0
    assert triage.kb_queries == ["a", "b", "c"]
    assert triage.escalate_reason is None
    assert make_triage(escalate_reason="legal threat").escalate_reason == "legal threat"
