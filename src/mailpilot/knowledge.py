"""Help-center knowledge base: markdown files split into sections, searched with BM25.

BM25 needs no API key, is deterministic and is strong on the short, keyword-heavy
questions support emails contain ("customs", "return label", "Germany"). An embedding
retriever and a hybrid (reciprocal-rank-fusion) retriever are provided for larger or more
paraphrase-heavy knowledge bases; they work with any LangChain ``Embeddings``.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

from mailpilot.models import Snippet

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    (
        "a an and are as at be by can do does for from has have how i if in is it its my of "
        "on or our the this to was we what when where which will with you your"
    ).split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if len(t) > 1 and t not in _STOPWORDS]


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _sections(path: Path) -> list[Snippet]:
    doc_title = path.stem.replace("_", " ").title()
    sections: list[tuple[str, list[str]]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("# "):
            doc_title = line[2:].strip()
        elif line.startswith("## "):
            sections.append((line[3:].strip(), []))
        elif sections:
            sections[-1][1].append(line)
    return [
        Snippet(id=f"{path.stem}#{_slug(heading)}", title=f"{doc_title} › {heading}", text=body)
        for heading, lines in sections
        if (body := "\n".join(lines).strip())
    ]


def load_markdown(directory: Path) -> list[Snippet]:
    """One snippet per ``##`` section; the file's ``#`` title is kept as context."""
    return [snippet for path in sorted(directory.glob("*.md")) for snippet in _sections(path)]


class Retriever(Protocol):
    def search(self, queries: Sequence[str], k: int) -> list[Snippet]: ...


class BM25Retriever:
    def __init__(self, snippets: Sequence[Snippet], *, k1: float = 1.5, b: float = 0.75) -> None:
        self.snippets = list(snippets)
        self._k1 = k1
        self._b = b
        self._docs = [Counter(tokenize(f"{s.title} {s.text}")) for s in self.snippets]
        self._lengths = [sum(d.values()) for d in self._docs]
        self._avg_len = (sum(self._lengths) / len(self._lengths)) if self._lengths else 0.0
        df: Counter[str] = Counter()
        for doc in self._docs:
            df.update(doc.keys())
        n = len(self._docs)
        self._idf = {term: math.log(1 + (n - f + 0.5) / (f + 0.5)) for term, f in df.items()}

    @classmethod
    def from_directory(cls, directory: Path) -> BM25Retriever:
        if not directory.is_dir():
            raise FileNotFoundError(f"Knowledge directory not found: {directory}")
        return cls(load_markdown(directory))

    def _score(self, query_terms: list[str], index: int) -> float:
        doc, length = self._docs[index], self._lengths[index]
        score = 0.0
        for term in query_terms:
            freq = doc.get(term, 0)
            if not freq:
                continue
            norm = self._k1 * (1 - self._b + self._b * length / (self._avg_len or 1))
            score += self._idf[term] * freq * (self._k1 + 1) / (freq + norm)
        return score

    def ranked(self, query: str) -> list[tuple[int, float]]:
        terms = tokenize(query)
        scores = [(i, self._score(terms, i)) for i in range(len(self._docs))]
        return sorted(((i, s) for i, s in scores if s > 0), key=lambda x: -x[1])

    def search(self, queries: Sequence[str], k: int) -> list[Snippet]:
        best: dict[int, float] = {}
        for query in queries:
            for index, score in self.ranked(query):
                best[index] = max(best.get(index, 0.0), score)
        top = sorted(best.items(), key=lambda x: -x[1])[:k]
        return [self.snippets[i].model_copy(update={"score": round(s, 3)}) for i, s in top]


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


class EmbeddingRetriever:
    """Dense retrieval with any LangChain ``Embeddings`` (Gemini, OpenAI, Bedrock...)."""

    def __init__(self, snippets: Sequence[Snippet], embeddings: Any, *, min_score: float = 0.0):
        self.snippets = list(snippets)
        self._embeddings = embeddings
        self._vectors = embeddings.embed_documents([f"{s.title}\n{s.text}" for s in self.snippets])
        self._min_score = min_score

    def ranked(self, query: str) -> list[tuple[int, float]]:
        vector = self._embeddings.embed_query(query)
        scores = [(i, _cosine(vector, v)) for i, v in enumerate(self._vectors)]
        return sorted(((i, s) for i, s in scores if s > self._min_score), key=lambda x: -x[1])

    def search(self, queries: Sequence[str], k: int) -> list[Snippet]:
        best: dict[int, float] = {}
        for query in queries:
            for index, score in self.ranked(query):
                best[index] = max(best.get(index, 0.0), score)
        top = sorted(best.items(), key=lambda x: -x[1])[:k]
        return [self.snippets[i].model_copy(update={"score": round(s, 3)}) for i, s in top]


class HybridRetriever:
    """Reciprocal rank fusion of a lexical and a dense retriever over the same snippets."""

    def __init__(
        self, lexical: BM25Retriever, dense: EmbeddingRetriever, *, rrf_k: int = 60
    ) -> None:
        if [s.id for s in lexical.snippets] != [s.id for s in dense.snippets]:
            raise ValueError("Both retrievers must index the same snippets in the same order")
        self.snippets = lexical.snippets
        self._lexical = lexical
        self._dense = dense
        self._rrf_k = rrf_k

    def search(self, queries: Sequence[str], k: int) -> list[Snippet]:
        fused: dict[int, float] = {}
        for query in queries:
            for ranking in (self._lexical.ranked(query), self._dense.ranked(query)):
                for rank, (index, _score) in enumerate(ranking):
                    fused[index] = fused.get(index, 0.0) + 1 / (self._rrf_k + rank + 1)
        top = sorted(fused.items(), key=lambda x: -x[1])[:k]
        return [self.snippets[i].model_copy(update={"score": round(s, 4)}) for i, s in top]
