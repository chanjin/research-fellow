"""Regenerable local embedding index for case-based episodic recall."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from research_fellow.domain.episodes import EpisodicMemory
from research_fellow.infrastructure.retrieval import OllamaEmbeddingClient


EPISODIC_INDEX_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class EpisodeRecall:
    episode: EpisodicMemory
    score: float
    method: str
    reason: str


def _tokens(text: str) -> set[str]:
    return {token.lower() for token in re.findall(r"[\w가-힣]{2,}", text)}


class EpisodicRetriever:
    """Recall similar situations, not previously asserted domain facts."""

    def __init__(self, index_path: str | Path):
        self.index_path = Path(index_path)
        self.index_path.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _fingerprint(episodes: list[EpisodicMemory], model: str) -> str:
        payload = [{"episode_id": episode.episode_id, "text": episode.retrieval_text()} for episode in episodes]
        encoded = json.dumps({"schema": EPISODIC_INDEX_SCHEMA_VERSION, "model": model, "episodes": payload}, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def _load_index(self) -> dict[str, Any] | None:
        try:
            return json.loads(self.index_path.read_text(encoding="utf-8")) if self.index_path.exists() else None
        except (OSError, json.JSONDecodeError):
            return None

    @staticmethod
    def _cosine(left: list[float], right: list[float]) -> float:
        denominator = math.sqrt(sum(item * item for item in left)) * math.sqrt(sum(item * item for item in right))
        return sum(a * b for a, b in zip(left, right)) / denominator if denominator else 0.0

    def recall(
        self, episodes: list[EpisodicMemory], situation: str, *, limit: int = 3,
        semantic: bool = True, embedding_model: str = "nomic-embed-text",
    ) -> list[EpisodeRecall]:
        eligible = [episode for episode in episodes if episode.lifecycle_status != "superseded"]
        if not eligible:
            return []
        lexical_scores = {
            episode.episode_id: len(_tokens(situation) & _tokens(episode.retrieval_text())) / max(len(_tokens(situation)), 1)
            for episode in eligible
        }
        if semantic:
            fingerprint = self._fingerprint(eligible, embedding_model)
            index = self._load_index()
            vectors = index.get("vectors") if index and index.get("fingerprint") == fingerprint else None
            client = OllamaEmbeddingClient(embedding_model)
            if vectors is None:
                vectors = client.embed([episode.retrieval_text() for episode in eligible])
                if vectors is not None:
                    self.index_path.write_text(json.dumps({"schema_version": EPISODIC_INDEX_SCHEMA_VERSION, "fingerprint": fingerprint, "embedding_model": embedding_model, "vectors": vectors}, ensure_ascii=False), encoding="utf-8")
            query_vectors = client.embed([situation]) if vectors is not None else None
            if vectors is not None and query_vectors:
                recalls = []
                for episode, vector in zip(eligible, vectors):
                    semantic_score = self._cosine(vector, query_vectors[0])
                    lexical_score = lexical_scores[episode.episode_id]
                    score = (semantic_score * 0.9) + (lexical_score * 0.1)
                    recalls.append(EpisodeRecall(episode, score, "semantic_episode", f"상황 임베딩 유사도 {semantic_score:.2f}; 어휘 보조 점수 {lexical_score:.2f}"))
                return sorted(recalls, key=lambda item: item.score, reverse=True)[:limit]
        recalls = [
            EpisodeRecall(episode, score, "lexical_fallback", f"임베딩을 사용할 수 없어 어휘 겹침 {score:.2f}로 대체")
            for episode_id, score in lexical_scores.items()
            for episode in eligible if episode.episode_id == episode_id and score > 0
        ]
        return sorted(recalls, key=lambda item: item.score, reverse=True)[:limit]
