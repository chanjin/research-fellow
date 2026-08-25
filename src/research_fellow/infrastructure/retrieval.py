from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen


INDEX_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class RetrievalResult:
    card: dict[str, Any]
    score: float
    method: str
    reason: str


def _tokens(text: str) -> set[str]:
    return {token.lower() for token in re.findall(r"[\w가-힣]{2,}", text)}


def _card_text(card: dict[str, Any]) -> str:
    return " ".join([
        card.get("title", ""), card.get("claim", ""), card.get("evidence_excerpt", ""),
        " ".join(card.get("labels", [])), card.get("conditions", ""), card.get("limits", ""),
    ])


class OllamaEmbeddingClient:
    """Optional local embedding adapter; failures leave lexical retrieval intact."""

    def __init__(self, model: str, base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url.rstrip("/")

    def embed(self, texts: list[str]) -> list[list[float]] | None:
        try:
            body = json.dumps({"model": self.model, "input": texts}).encode("utf-8")
            request = Request(f"{self.base_url}/api/embed", data=body, headers={"Content-Type": "application/json"})
            with urlopen(request, timeout=45) as response:  # nosec B310: local Ollama endpoint configured by user
                payload = json.loads(response.read().decode("utf-8"))
            embeddings = payload.get("embeddings")
            if isinstance(embeddings, list) and len(embeddings) == len(texts):
                return embeddings
        except (URLError, TimeoutError, ValueError, OSError):
            return None
        return None


class LlamaIndexAdapter:
    """Optional P3 hand-off point for LlamaIndex ingestion/chunking when installed."""

    @staticmethod
    def available() -> bool:
        try:
            import llama_index.core  # noqa: F401
        except ImportError:
            return False
        return True

    @staticmethod
    def documents(cards: list[dict[str, Any]]) -> list[Any]:
        """Expose cards as page/provenance-aware LlamaIndex Documents without changing JSONL truth."""
        from llama_index.core import Document
        return [Document(text=_card_text(card), metadata={"card_id": card["card_id"], **card.get("provenance", {})}) for card in cards]


class KnowledgeRetriever:
    """JSONL-derived lexical baseline plus regenerable optional Ollama embedding index."""

    def __init__(self, index_path: str | Path):
        self.index_path = Path(index_path)
        self.index_path.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _fingerprint(cards: list[dict[str, Any]], embedding_model: str) -> str:
        canonical = [{"card_id": card["card_id"], "text": _card_text(card)} for card in cards]
        payload = json.dumps({"schema": INDEX_SCHEMA_VERSION, "model": embedding_model, "cards": canonical}, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _load_index(self) -> dict[str, Any] | None:
        if not self.index_path.exists():
            return None
        try:
            return json.loads(self.index_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

    def _semantic_vectors(self, cards: list[dict[str, Any]], query: str, model: str) -> tuple[list[list[float]], list[float]] | None:
        fingerprint = self._fingerprint(cards, model)
        index = self._load_index()
        vectors: list[list[float]] | None = None
        if index and index.get("fingerprint") == fingerprint:
            vectors = index.get("vectors")
        client = OllamaEmbeddingClient(model)
        if vectors is None:
            vectors = client.embed([_card_text(card) for card in cards])
            if vectors is None:
                return None
            self.index_path.write_text(json.dumps({"schema_version": INDEX_SCHEMA_VERSION, "embedding_model": model, "fingerprint": fingerprint, "vectors": vectors}, ensure_ascii=False), encoding="utf-8")
        query_vector = client.embed([query])
        if not query_vector:
            return None
        return vectors, query_vector[0]

    @staticmethod
    def _cosine(left: list[float], right: list[float]) -> float:
        denominator = math.sqrt(sum(value * value for value in left)) * math.sqrt(sum(value * value for value in right))
        return sum(a * b for a, b in zip(left, right)) / denominator if denominator else 0.0

    def search(self, cards: list[dict[str, Any]], query: str, *, limit: int = 6, semantic: bool = False, embedding_model: str = "nomic-embed-text") -> list[RetrievalResult]:
        query_tokens = _tokens(query)
        ranked: dict[str, RetrievalResult] = {}
        for card in cards:
            matched = sorted(query_tokens & _tokens(_card_text(card)))
            if matched:
                score = len(matched) / max(len(query_tokens), 1)
                ranked[card["card_id"]] = RetrievalResult(card, score, "lexical", f"키워드 일치: {', '.join(matched[:5])}")
        if semantic and cards:
            embedded = self._semantic_vectors(cards, query, embedding_model)
            if embedded:
                vectors, query_vector = embedded
                for card, vector in zip(cards, vectors):
                    score = self._cosine(vector, query_vector)
                    existing = ranked.get(card["card_id"])
                    if existing:
                        ranked[card["card_id"]] = RetrievalResult(card, max(existing.score, score), "lexical+semantic", f"{existing.reason}; 임베딩 유사도 {score:.2f}")
                    elif score > 0:
                        ranked[card["card_id"]] = RetrievalResult(card, score, "semantic", f"Ollama 임베딩 유사도 {score:.2f}")
        return sorted(ranked.values(), key=lambda item: item.score, reverse=True)[:limit]
