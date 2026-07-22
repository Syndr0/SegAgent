from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from .schemas import ProtocolMatch, RetrievalHit


TOKEN_RE = re.compile(r"[A-Za-z0-9]+|[\u4e00-\u9fff]")


def normalize(text: str) -> str:
    return " ".join(TOKEN_RE.findall(text.casefold()))


def tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text.casefold())


class BM25Index:
    def __init__(self, documents: Iterable[str], k1: float = 1.5, b: float = 0.75):
        self.documents = list(documents)
        self.tokens = [tokens(document) for document in self.documents]
        self.term_counts = [Counter(items) for items in self.tokens]
        self.lengths = [len(items) for items in self.tokens]
        self.avg_len = sum(self.lengths) / max(len(self.lengths), 1)
        self.k1 = k1
        self.b = b
        document_frequency: Counter[str] = Counter()
        for items in self.tokens:
            document_frequency.update(set(items))
        total = max(len(self.tokens), 1)
        self.idf = {
            term: math.log(1.0 + (total - freq + 0.5) / (freq + 0.5))
            for term, freq in document_frequency.items()
        }

    def scores(self, query: str) -> np.ndarray:
        query_terms = tokens(query)
        values = np.zeros(len(self.documents), dtype=np.float32)
        for index, counts in enumerate(self.term_counts):
            length = self.lengths[index]
            for term in query_terms:
                frequency = counts.get(term, 0)
                if frequency == 0:
                    continue
                denominator = frequency + self.k1 * (
                    1.0 - self.b + self.b * length / max(self.avg_len, 1.0)
                )
                values[index] += self.idf.get(term, 0.0) * (
                    frequency * (self.k1 + 1.0) / denominator
                )
        maximum = float(values.max()) if values.size else 0.0
        return values / maximum if maximum > 0 else values


class HybridKnowledgeBase:
    """Auditable protocol lookup plus hybrid guideline retrieval."""

    def __init__(
        self,
        root: Path,
        embed_fn: Callable[[list[str]], np.ndarray] | None = None,
        semantic_weight: float = 0.45,
    ):
        self.root = Path(root)
        self.embed_fn = embed_fn
        self.semantic_weight = semantic_weight
        protocols_path = self.root / "protocols.json"
        payload = json.loads(protocols_path.read_text(encoding="utf-8"))
        self.protocols = payload.get("protocols", [])
        self.guidelines = self._load_guidelines(self.root / "guidelines")
        protocol_docs = [self._protocol_text(item) for item in self.protocols]
        self.protocol_bm25 = BM25Index(protocol_docs)
        self.guideline_bm25 = BM25Index(item["text"] for item in self.guidelines)
        self._protocol_embeddings: np.ndarray | None = None
        self._guideline_embeddings: np.ndarray | None = None

    @staticmethod
    def _protocol_text(protocol: dict) -> str:
        return " ".join(
            [
                str(protocol.get("site", "")),
                " ".join(protocol.get("aliases", [])),
                str(protocol.get("description", "")),
                " ".join(protocol.get("oars", [])),
            ]
        )

    @staticmethod
    def _load_guidelines(folder: Path) -> list[dict]:
        chunks: list[dict] = []
        if not folder.exists():
            return chunks
        for path in sorted(folder.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in {".md", ".txt"}:
                continue
            if path.name.casefold() == "readme.md":
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            section = "Document"
            buffer: list[str] = []
            for block in re.split(r"\n\s*\n", text):
                block = block.strip()
                if not block:
                    continue
                if block.startswith("#"):
                    section = block.lstrip("# ").strip() or section
                    continue
                buffer.append(block)
                joined = "\n\n".join(buffer)
                if len(joined) >= 900:
                    chunks.append(
                        {"text": joined, "source": path.name, "section": section}
                    )
                    buffer = []
            if buffer:
                chunks.append(
                    {"text": "\n\n".join(buffer), "source": path.name, "section": section}
                )
        return chunks

    @staticmethod
    def _normalize_embeddings(values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float32)
        norms = np.linalg.norm(array, axis=-1, keepdims=True)
        return array / np.clip(norms, 1e-8, None)

    def _hybrid_scores(
        self,
        query: str,
        documents: list[str],
        lexical: np.ndarray,
        cache_name: str,
    ) -> np.ndarray:
        if self.embed_fn is None or not documents:
            return lexical
        cached = getattr(self, cache_name)
        if cached is None:
            cached = self._normalize_embeddings(self.embed_fn(documents))
            setattr(self, cache_name, cached)
        query_embedding = self._normalize_embeddings(self.embed_fn([query]))[0]
        semantic = np.clip(cached @ query_embedding, 0.0, 1.0)
        weight = self.semantic_weight
        return (1.0 - weight) * lexical + weight * semantic

    def lookup_protocol(self, query: str) -> ProtocolMatch | None:
        normalized = normalize(query)
        if not normalized or not self.protocols:
            return None
        exact: tuple[int, dict] | None = None
        for protocol in self.protocols:
            for alias in [protocol.get("site", ""), *protocol.get("aliases", [])]:
                key = normalize(alias)
                if key and (key in normalized or normalized in key):
                    if exact is None or len(key) > exact[0]:
                        exact = (len(key), protocol)
        if exact is not None:
            protocol = exact[1]
            return self._match(protocol, 1.0, "keyword")
        documents = [self._protocol_text(item) for item in self.protocols]
        scores = self._hybrid_scores(
            query,
            documents,
            self.protocol_bm25.scores(query),
            "_protocol_embeddings",
        )
        index = int(np.argmax(scores))
        score = float(scores[index])
        return self._match(self.protocols[index], score, "hybrid") if score >= 0.18 else None

    @staticmethod
    def _match(protocol: dict, score: float, matched_by: str) -> ProtocolMatch:
        return ProtocolMatch(
            protocol_id=str(protocol["id"]),
            site=str(protocol["site"]),
            oars=list(protocol.get("oars", [])),
            score=round(score, 4),
            matched_by=matched_by,
            source=str(protocol.get("source", "local protocol registry")),
            source_version=protocol.get("source_version"),
            citations=list(protocol.get("citations", [])),
        )

    def retrieve_guidelines(self, query: str, k: int = 4) -> list[RetrievalHit]:
        if not query.strip() or not self.guidelines:
            return []
        documents = [item["text"] for item in self.guidelines]
        scores = self._hybrid_scores(
            query,
            documents,
            self.guideline_bm25.scores(query),
            "_guideline_embeddings",
        )
        order = np.argsort(scores)[::-1][:k]
        hits: list[RetrievalHit] = []
        for index in order:
            score = float(scores[int(index)])
            if score < 0.12:
                continue
            item = self.guidelines[int(index)]
            citation = f'[{item["source"]} § {item["section"]}]'
            hits.append(
                RetrievalHit(
                    text=item["text"],
                    source=item["source"],
                    section=item["section"],
                    score=round(score, 4),
                    citation=citation,
                )
            )
        return hits
