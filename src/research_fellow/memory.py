from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .domain.knowledge import KnowledgeCard, KnowledgeRelation


class KnowledgeMemory:
    """Append-only approved knowledge cards; JSONL is the canonical semantic-memory record."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)
        self.tombstone_path = self.path.with_name(f"{self.path.stem}_tombstones.jsonl")
        self.tombstone_path.touch(exist_ok=True)

    def add(self, card: dict[str, Any]) -> dict[str, Any]:
        canonical_card = KnowledgeCard.model_validate({
            "card_id": card.get("card_id", f"kc-{uuid.uuid4().hex[:12]}"),
            **card,
        }).model_dump(mode="json")
        stored = {
            "approved_at": datetime.now(UTC).isoformat(timespec="seconds"),
            **canonical_card,
        }
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(stored, ensure_ascii=False) + "\n")
        return stored

    def _deleted_ids(self) -> set[str]:
        with self.tombstone_path.open(encoding="utf-8") as stream:
            return {json.loads(line)["subject_id"] for line in stream if line.strip()}

    def remove(self, card_id: str, note: str = "") -> bool:
        if card_id in self._deleted_ids() or not any(card["card_id"] == card_id for card in self.all(include_deleted=True)):
            return False
        with self.tombstone_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"subject_id": card_id, "deleted_at": datetime.now(UTC).isoformat(timespec="seconds"), "note": note}, ensure_ascii=False) + "\n")
        return True

    def all(self, *, include_deleted: bool = False) -> list[dict[str, Any]]:
        cards: list[dict[str, Any]] = []
        with self.path.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    cards.append(json.loads(line))
        cards = list(reversed(cards))
        if include_deleted:
            return cards
        deleted = self._deleted_ids()
        return [card for card in cards if card["card_id"] not in deleted]

    def search(self, query: str, limit: int = 6) -> list[dict[str, Any]]:
        words = {word.lower() for word in query.split() if len(word) > 1}
        scored = []
        for card in self.all():
            haystack = json.dumps(card, ensure_ascii=False).lower()
            score = sum(word in haystack for word in words)
            if score:
                scored.append((score, card))
        return [card for _, card in sorted(scored, key=lambda item: item[0], reverse=True)[:limit]]


class RelationMemory:
    """Append-only canonical store for researcher-approved knowledge relations."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)
        self.tombstone_path = self.path.with_name(f"{self.path.stem}_tombstones.jsonl")
        self.tombstone_path.touch(exist_ok=True)

    def add(self, relation: dict[str, Any]) -> dict[str, Any]:
        stored = {
            "approved_at": datetime.now(UTC).isoformat(timespec="seconds"),
            **KnowledgeRelation.model_validate(relation).model_dump(mode="json"),
        }
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(stored, ensure_ascii=False) + "\n")
        return stored

    def _deleted_ids(self) -> set[str]:
        with self.tombstone_path.open(encoding="utf-8") as stream:
            return {json.loads(line)["subject_id"] for line in stream if line.strip()}

    def remove(self, relation_id: str, note: str = "") -> bool:
        if relation_id in self._deleted_ids() or not any(item["relation_id"] == relation_id for item in self.all(include_deleted=True)):
            return False
        with self.tombstone_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"subject_id": relation_id, "deleted_at": datetime.now(UTC).isoformat(timespec="seconds"), "note": note}, ensure_ascii=False) + "\n")
        return True

    def all(self, *, include_deleted: bool = False) -> list[dict[str, Any]]:
        relations: list[dict[str, Any]] = []
        with self.path.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    relations.append(json.loads(line))
        relations = list(reversed(relations))
        if include_deleted:
            return relations
        deleted = self._deleted_ids()
        return [relation for relation in relations if relation["relation_id"] not in deleted]

    def active_for_cards(self, card_ids: set[str]) -> list[dict[str, Any]]:
        return [relation for relation in self.all() if relation["source_card_id"] in card_ids and relation["target_card_id"] in card_ids]
