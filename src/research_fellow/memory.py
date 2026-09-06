from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .domain.knowledge import KnowledgeCard, KnowledgeRelation


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class KnowledgeMemory:
    """Approved knowledge memory.

    A .db path uses SQLite as the canonical durable store. JSONL remains supported
    for tests and legacy scripts. When ``legacy_path`` is supplied, existing JSONL
    cards/tombstones are imported once into an empty SQLite store.
    """

    def __init__(self, path: str | Path, legacy_path: str | Path | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.sqlite = self.path.suffix.lower() in {".db", ".sqlite", ".sqlite3"}
        self.legacy_path = Path(legacy_path) if legacy_path else None
        if self.sqlite:
            self._ensure_sqlite()
            if self.legacy_path:
                self._migrate_legacy_if_empty(self.legacy_path)
        else:
            self.path.touch(exist_ok=True)
            self.tombstone_path = self.path.with_name(f"{self.path.stem}_tombstones.jsonl")
            self.tombstone_path.touch(exist_ok=True)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_sqlite(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS knowledge_cards (
                    card_id TEXT PRIMARY KEY,
                    card_json TEXT NOT NULL,
                    approved_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    deleted_at TEXT,
                    deleted_note TEXT NOT NULL DEFAULT ''
                );
                CREATE INDEX IF NOT EXISTS idx_knowledge_cards_active
                    ON knowledge_cards(deleted_at, updated_at DESC);
                """
            )

    def _migrate_legacy_if_empty(self, legacy_path: Path) -> None:
        if not legacy_path.exists():
            return
        with self._connect() as conn:
            count = int(conn.execute("SELECT COUNT(*) FROM knowledge_cards").fetchone()[0])
        if count:
            return
        legacy = KnowledgeMemory(legacy_path)
        deleted_ids = legacy._deleted_ids()
        for card in legacy.all(include_deleted=True):
            card_id = str(card["card_id"])
            approved_at = str(card.get("approved_at") or _now())
            updated_at = str(card.get("updated_at") or approved_at)
            payload = {key: value for key, value in card.items() if key != "approved_at"}
            with self._connect() as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO knowledge_cards VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        card_id,
                        json.dumps(payload, ensure_ascii=False),
                        approved_at,
                        updated_at,
                        _now() if card_id in deleted_ids else None,
                        "legacy tombstone" if card_id in deleted_ids else "",
                    ),
                )

    def add(self, card: dict[str, Any]) -> dict[str, Any]:
        canonical_card = KnowledgeCard.model_validate({
            "card_id": card.get("card_id", f"kc-{uuid.uuid4().hex[:12]}"),
            **card,
        }).model_dump(mode="json")
        approved_at = str(card.get("approved_at") or _now())
        stored = {"approved_at": approved_at, **canonical_card}
        if self.sqlite:
            updated_at = str(canonical_card.get("updated_at") or approved_at)
            with self._connect() as conn:
                conn.execute(
                    """INSERT INTO knowledge_cards(card_id, card_json, approved_at, updated_at, deleted_at, deleted_note)
                       VALUES (?, ?, ?, ?, NULL, '')
                       ON CONFLICT(card_id) DO UPDATE SET
                         card_json=excluded.card_json,
                         approved_at=excluded.approved_at,
                         updated_at=excluded.updated_at,
                         deleted_at=NULL,
                         deleted_note=''""",
                    (canonical_card["card_id"], json.dumps(canonical_card, ensure_ascii=False), approved_at, updated_at),
                )
            return stored
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(stored, ensure_ascii=False) + "\n")
        return stored

    def _deleted_ids(self) -> set[str]:
        if self.sqlite:
            with self._connect() as conn:
                rows = conn.execute("SELECT card_id FROM knowledge_cards WHERE deleted_at IS NOT NULL").fetchall()
            return {str(row[0]) for row in rows}
        with self.tombstone_path.open(encoding="utf-8") as stream:
            return {json.loads(line)["subject_id"] for line in stream if line.strip()}

    def remove(self, card_id: str, note: str = "") -> bool:
        if self.sqlite:
            with self._connect() as conn:
                result = conn.execute(
                    "UPDATE knowledge_cards SET deleted_at=?, deleted_note=?, updated_at=? WHERE card_id=? AND deleted_at IS NULL",
                    (_now(), note, _now(), card_id),
                )
            return result.rowcount > 0
        if card_id in self._deleted_ids() or not any(card["card_id"] == card_id for card in self.all(include_deleted=True)):
            return False
        with self.tombstone_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"subject_id": card_id, "deleted_at": _now(), "note": note}, ensure_ascii=False) + "\n")
        return True

    def all(self, *, include_deleted: bool = False) -> list[dict[str, Any]]:
        if self.sqlite:
            query = "SELECT * FROM knowledge_cards"
            if not include_deleted:
                query += " WHERE deleted_at IS NULL"
            query += " ORDER BY updated_at DESC, approved_at DESC"
            with self._connect() as conn:
                rows = conn.execute(query).fetchall()
            result = []
            for row in rows:
                payload = json.loads(row["card_json"])
                result.append({"approved_at": row["approved_at"], **payload})
            return result
        cards: list[dict[str, Any]] = []
        with self.path.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    cards.append(json.loads(line))
        latest_by_id: dict[str, dict[str, Any]] = {}
        for card in cards:
            latest_by_id[str(card["card_id"])] = card
        cards = list(reversed(list(latest_by_id.values())))
        if include_deleted:
            return cards
        deleted = self._deleted_ids()
        return [card for card in cards if card["card_id"] not in deleted]

    def add_supporting_evidence(self, card_id: str, evidence: dict[str, Any]) -> dict[str, Any]:
        existing = next((card for card in self.all() if card["card_id"] == card_id), None)
        if not existing:
            raise ValueError("근거를 보강할 기존 지식카드를 찾을 수 없습니다.")
        prior = list(existing.get("supporting_evidence", []))
        source_key = (str(evidence.get("source_name", "")), str(evidence.get("evidence_excerpt", "")))
        if not any((str(item.get("source_name", "")), str(item.get("evidence_excerpt", ""))) == source_key for item in prior):
            prior.append(evidence)
        updated = {**existing, "supporting_evidence": prior, "updated_at": _now()}
        if self.sqlite:
            approved_at = str(existing.get("approved_at") or _now())
            payload = {key: value for key, value in updated.items() if key != "approved_at"}
            with self._connect() as conn:
                conn.execute(
                    "UPDATE knowledge_cards SET card_json=?, updated_at=? WHERE card_id=? AND deleted_at IS NULL",
                    (json.dumps(payload, ensure_ascii=False), updated["updated_at"], card_id),
                )
            return updated
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(updated, ensure_ascii=False) + "\n")
        return updated

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
    """Approved knowledge relations, backed by SQLite or legacy JSONL."""

    def __init__(self, path: str | Path, legacy_path: str | Path | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.sqlite = self.path.suffix.lower() in {".db", ".sqlite", ".sqlite3"}
        self.legacy_path = Path(legacy_path) if legacy_path else None
        if self.sqlite:
            self._ensure_sqlite()
            if self.legacy_path:
                self._migrate_legacy_if_empty(self.legacy_path)
        else:
            self.path.touch(exist_ok=True)
            self.tombstone_path = self.path.with_name(f"{self.path.stem}_tombstones.jsonl")
            self.tombstone_path.touch(exist_ok=True)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_sqlite(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS knowledge_relations (
                    relation_id TEXT PRIMARY KEY,
                    source_card_id TEXT NOT NULL,
                    target_card_id TEXT NOT NULL,
                    relation_type TEXT NOT NULL,
                    evidence TEXT NOT NULL,
                    conditions TEXT NOT NULL,
                    confidence TEXT NOT NULL,
                    approved_at TEXT NOT NULL,
                    deleted_at TEXT,
                    deleted_note TEXT NOT NULL DEFAULT ''
                );
                """
            )

    def _migrate_legacy_if_empty(self, legacy_path: Path) -> None:
        if not legacy_path.exists():
            return
        with self._connect() as conn:
            count = int(conn.execute("SELECT COUNT(*) FROM knowledge_relations").fetchone()[0])
        if count:
            return
        legacy = RelationMemory(legacy_path)
        deleted_ids = legacy._deleted_ids()
        for relation in legacy.all(include_deleted=True):
            stored = KnowledgeRelation.model_validate(relation).model_dump(mode="json")
            approved_at = str(relation.get("approved_at") or _now())
            with self._connect() as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO knowledge_relations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        stored["relation_id"], stored["source_card_id"], stored["target_card_id"], stored["relation_type"],
                        stored.get("evidence", ""), stored.get("conditions", ""), stored.get("confidence", ""), approved_at,
                        _now() if stored["relation_id"] in deleted_ids else None,
                        "legacy tombstone" if stored["relation_id"] in deleted_ids else "",
                    ),
                )

    def add(self, relation: dict[str, Any]) -> dict[str, Any]:
        stored = {"approved_at": _now(), **KnowledgeRelation.model_validate(relation).model_dump(mode="json")}
        if self.sqlite:
            with self._connect() as conn:
                conn.execute(
                    """INSERT OR REPLACE INTO knowledge_relations
                       (relation_id, source_card_id, target_card_id, relation_type, evidence, conditions, confidence, approved_at, deleted_at, deleted_note)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, '')""",
                    (
                        stored["relation_id"], stored["source_card_id"], stored["target_card_id"], stored["relation_type"],
                        stored.get("evidence", ""), stored.get("conditions", ""), stored.get("confidence", ""), stored["approved_at"],
                    ),
                )
            return stored
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(stored, ensure_ascii=False) + "\n")
        return stored

    def _deleted_ids(self) -> set[str]:
        if self.sqlite:
            with self._connect() as conn:
                rows = conn.execute("SELECT relation_id FROM knowledge_relations WHERE deleted_at IS NOT NULL").fetchall()
            return {str(row[0]) for row in rows}
        with self.tombstone_path.open(encoding="utf-8") as stream:
            return {json.loads(line)["subject_id"] for line in stream if line.strip()}

    def remove(self, relation_id: str, note: str = "") -> bool:
        if self.sqlite:
            with self._connect() as conn:
                result = conn.execute(
                    "UPDATE knowledge_relations SET deleted_at=?, deleted_note=? WHERE relation_id=? AND deleted_at IS NULL",
                    (_now(), note, relation_id),
                )
            return result.rowcount > 0
        if relation_id in self._deleted_ids() or not any(item["relation_id"] == relation_id for item in self.all(include_deleted=True)):
            return False
        with self.tombstone_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"subject_id": relation_id, "deleted_at": _now(), "note": note}, ensure_ascii=False) + "\n")
        return True

    def all(self, *, include_deleted: bool = False) -> list[dict[str, Any]]:
        if self.sqlite:
            query = "SELECT * FROM knowledge_relations"
            if not include_deleted:
                query += " WHERE deleted_at IS NULL"
            query += " ORDER BY approved_at DESC"
            with self._connect() as conn:
                return [dict(row) for row in conn.execute(query).fetchall()]
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
