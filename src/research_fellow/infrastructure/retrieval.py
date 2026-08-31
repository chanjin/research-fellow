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
    lexical_score: float | None = None
    semantic_score: float | None = None


@dataclass(frozen=True)
class ClusterMember:
    """One card selected for a local, relation-bounded evidence cluster."""

    result: RetrievalResult
    distance: int
    seed_card_id: str
    relation_ids: tuple[str, ...] = ()
    relation_types: tuple[str, ...] = ()


@dataclass(frozen=True)
class KnowledgeCluster:
    """A deterministic evidence bundle around direct-match seed cards.

    It deliberately stores the route by which an adjacent card entered the
    bundle.  The LLM can therefore use the cluster for interpretation but
    cannot silently invent the graph traversal.
    """

    query: str
    members: tuple[ClusterMember, ...]

    @property
    def card_ids(self) -> list[str]:
        return [member.result.card["card_id"] for member in self.members]

    @property
    def relation_ids(self) -> list[str]:
        return [relation_id for member in self.members for relation_id in member.relation_ids]


def _tokens(text: str) -> set[str]:
    return {token.lower() for token in re.findall(r"[\w가-힣]{2,}", text)}


def _card_text(card: dict[str, Any]) -> str:
    return " ".join([
        card.get("title", ""), card.get("claim", ""), card.get("evidence_excerpt", ""),
        " ".join(card.get("labels", [])), " ".join(card.get("concepts", [])),
        " ".join(card.get("applies_to", [])), " ".join(card.get("excludes", [])),
        " ".join(card.get("supports_question_types", [])), card.get("conditions", ""), card.get("limits", ""),
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
                ranked[card["card_id"]] = RetrievalResult(card, score, "lexical", f"키워드 일치: {', '.join(matched[:5])}", lexical_score=score)
        if semantic and cards:
            embedded = self._semantic_vectors(cards, query, embedding_model)
            if embedded:
                vectors, query_vector = embedded
                for card, vector in zip(cards, vectors):
                    semantic_score = self._cosine(vector, query_vector)
                    existing = ranked.get(card["card_id"])
                    if existing:
                        # For semantic seed search the embedding match is the
                        # primary signal. Lexical overlap is only a small,
                        # reproducible preference for an exact expression.
                        score = (semantic_score * 0.85) + (existing.score * 0.15)
                        ranked[card["card_id"]] = RetrievalResult(
                            card, score, "semantic_seed+lexical",
                            f"임베딩 유사도 {semantic_score:.2f}; {existing.reason}",
                            lexical_score=existing.lexical_score, semantic_score=semantic_score,
                        )
                    elif semantic_score > 0:
                        ranked[card["card_id"]] = RetrievalResult(
                            card, semantic_score, "semantic_seed", f"임베딩 유사도 {semantic_score:.2f}",
                            semantic_score=semantic_score,
                        )
        return sorted(ranked.values(), key=lambda item: item.score, reverse=True)[:limit]

    @staticmethod
    def _seed_eligible(card: dict[str, Any], query: str) -> bool:
        """Apply non-generative safety gates before graph expansion."""
        if card.get("status") == "obsolete":
            return False
        query_tokens = _tokens(query)
        # `excludes` contains short contexts where the card must not be used.
        # An overlap is only a conservative veto; absence of an overlap never
        # claims that the card is applicable.
        for excluded in card.get("excludes", []):
            if query_tokens & _tokens(str(excluded)):
                return False
        return True

    def cluster(
        self, cards: list[dict[str, Any]], relations: list[dict[str, Any]], query: str, *,
        seed_limit: int = 3, max_hops: int = 2, limit: int = 10,
        semantic: bool = True, embedding_model: str = "nomic-embed-text",
    ) -> KnowledgeCluster:
        """Build a small approved-card neighbourhood without a generative call.

        Hop 1 permits every approved relation because it can supply direct
        support, contradiction, or a limitation. Hop 2 is restricted to
        method, qualification, extension, gap, and argument relations so a
        broadly connected card cannot pull in the whole memory.
        """
        # Embedding search is the normal seed finder because a question and a
        # card rarely use the same wording. If local embeddings are unavailable
        # `search` deliberately falls back to its lexical baseline.
        candidates = self.search(cards, query, limit=max(seed_limit * 3, seed_limit), semantic=semantic, embedding_model=embedding_model)
        seeds = [candidate for candidate in candidates if self._seed_eligible(candidate.card, query)][:seed_limit]
        if not seeds:
            return KnowledgeCluster(query=query, members=())
        card_by_id = {str(card["card_id"]): card for card in cards}
        active_relations = sorted(
            [relation for relation in relations if str(relation.get("source_card_id")) in card_by_id and str(relation.get("target_card_id")) in card_by_id],
            key=lambda relation: str(relation.get("relation_id", "")),
        )
        confidence = {"high": 1.0, "medium": 0.8, "low": 0.6}
        allowed_second_hop = {"supports", "contradicts", "qualifies", "uses_method", "extends", "addresses_gap"}
        selected: dict[str, ClusterMember] = {}
        frontier: list[ClusterMember] = []
        for seed in seeds:
            member = ClusterMember(seed, 0, seed.card["card_id"])
            selected[seed.card["card_id"]] = member
            frontier.append(member)
        for hop in range(1, max(0, max_hops) + 1):
            next_frontier: list[ClusterMember] = []
            for parent in frontier:
                parent_id = parent.result.card["card_id"]
                for relation in active_relations:
                    relation_type = str(relation.get("relation_type", ""))
                    if hop == 2 and relation_type not in allowed_second_hop:
                        continue
                    source, target = str(relation["source_card_id"]), str(relation["target_card_id"])
                    if parent_id == source:
                        neighbour_id = target
                    elif parent_id == target:
                        neighbour_id = source
                    else:
                        continue
                    if neighbour_id in selected or neighbour_id not in card_by_id:
                        continue
                    score = parent.result.score * (0.72 if hop == 1 else 0.48) * confidence.get(str(relation.get("confidence")), 0.5)
                    result = RetrievalResult(
                        card_by_id[neighbour_id], score, "relation_cluster",
                        f"[{parent_id}]에서 {relation_type} 관계 [{relation.get('relation_id')}]로 {hop}-hop 확장",
                    )
                    member = ClusterMember(
                        result, hop, parent.seed_card_id,
                        (*parent.relation_ids, str(relation.get("relation_id"))),
                        (*parent.relation_types, relation_type),
                    )
                    selected[neighbour_id] = member
                    next_frontier.append(member)
                    if len(selected) >= limit:
                        break
                if len(selected) >= limit:
                    break
            frontier = next_frontier
            if not frontier or len(selected) >= limit:
                break
        ordered = sorted(selected.values(), key=lambda member: (member.distance, -member.result.score, member.result.card["card_id"]))[:limit]
        return KnowledgeCluster(query=query, members=tuple(ordered))
