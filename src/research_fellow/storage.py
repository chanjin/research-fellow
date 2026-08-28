from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .domain.phenomena import PhenomenonDraft, validate_payload


PHENOMENON_TYPES = {
    "research_update",
    "advice_report",
    "decision_request",
    "decision",
    "curation_intent",
    "knowledge_update",
    "advisory_exchange",
    "activity_summary",
}


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class Ledger:
    """SQLite record of cross-domain phenomena, not a replacement for knowledge memory."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def initialize(self) -> None:
        with self.connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS schema_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS cases (
                    case_id TEXT PRIMARY KEY,
                    case_type TEXT NOT NULL,
                    title TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS phenomena (
                    phenomenon_id TEXT PRIMARY KEY,
                    case_id TEXT NOT NULL,
                    phenomenon_type TEXT NOT NULL,
                    producer TEXT NOT NULL,
                    recipients_json TEXT NOT NULL,
                    subject_type TEXT NOT NULL,
                    subject_id TEXT,
                    payload_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(case_id) REFERENCES cases(case_id)
                );
                CREATE TABLE IF NOT EXISTS decisions (
                    decision_id TEXT PRIMARY KEY,
                    phenomenon_id TEXT NOT NULL,
                    decision TEXT NOT NULL,
                    note TEXT NOT NULL,
                    decided_by TEXT NOT NULL,
                    decided_at TEXT NOT NULL,
                    FOREIGN KEY(phenomenon_id) REFERENCES phenomena(phenomenon_id)
                );
                CREATE TABLE IF NOT EXISTS search_profiles (
                    profile_id TEXT PRIMARY KEY,
                    intent_id TEXT UNIQUE NOT NULL,
                    title TEXT NOT NULL,
                    question TEXT NOT NULL,
                    context TEXT NOT NULL,
                    keywords_json TEXT NOT NULL,
                    core_terms_json TEXT NOT NULL DEFAULT '[]',
                    cadence TEXT NOT NULL,
                    is_active INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    last_run_at TEXT,
                    deleted_at TEXT,
                    deleted_note TEXT NOT NULL DEFAULT ''
                );
                CREATE TABLE IF NOT EXISTS search_runs (
                    run_id TEXT PRIMARY KEY,
                    profile_id TEXT NOT NULL,
                    trigger TEXT NOT NULL,
                    query TEXT NOT NULL,
                    candidates_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    error TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(profile_id) REFERENCES search_profiles(profile_id)
                );
                CREATE TABLE IF NOT EXISTS llm_calls (
                    call_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    profile_name TEXT NOT NULL,
                    model TEXT NOT NULL,
                    prompt TEXT NOT NULL,
                    settings_json TEXT NOT NULL,
                    response TEXT NOT NULL,
                    error TEXT NOT NULL,
                    diagnostics_json TEXT NOT NULL
                );
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
                CREATE INDEX IF NOT EXISTS idx_knowledge_relations_source ON knowledge_relations(source_card_id);
                CREATE INDEX IF NOT EXISTS idx_knowledge_relations_target ON knowledge_relations(target_card_id);
                CREATE INDEX IF NOT EXISTS idx_knowledge_relations_active ON knowledge_relations(deleted_at, relation_type);
                CREATE TABLE IF NOT EXISTS paper_shelf (
                    paper_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    authors_json TEXT NOT NULL DEFAULT '[]',
                    publication_year TEXT NOT NULL DEFAULT '',
                    source_url TEXT NOT NULL DEFAULT '',
                    source_id TEXT NOT NULL DEFAULT '',
                    pdf_path TEXT NOT NULL DEFAULT '',
                    labels_json TEXT NOT NULL DEFAULT '[]',
                    shelf_status TEXT NOT NULL DEFAULT 'reference',
                    reading_status TEXT NOT NULL DEFAULT 'unread',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE UNIQUE INDEX IF NOT EXISTS idx_paper_shelf_source_id
                    ON paper_shelf(source_id) WHERE source_id <> '';
                CREATE INDEX IF NOT EXISTS idx_paper_shelf_status
                    ON paper_shelf(shelf_status, reading_status, updated_at DESC);
                CREATE TABLE IF NOT EXISTS paper_analyses (
                    paper_id TEXT PRIMARY KEY,
                    research_question TEXT NOT NULL DEFAULT '',
                    summary TEXT NOT NULL DEFAULT '',
                    researcher_note TEXT NOT NULL DEFAULT '',
                    generated_at TEXT NOT NULL DEFAULT '',
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(paper_id) REFERENCES paper_shelf(paper_id)
                );
                CREATE TABLE IF NOT EXISTS paper_card_links (
                    paper_id TEXT NOT NULL,
                    card_id TEXT NOT NULL,
                    linked_at TEXT NOT NULL,
                    PRIMARY KEY(paper_id, card_id),
                    FOREIGN KEY(paper_id) REFERENCES paper_shelf(paper_id)
                );
                """
            )
            # Safe for v0.1 databases: SQLite preserves all existing rows.
            columns = {row[1] for row in conn.execute("PRAGMA table_info(search_profiles)").fetchall()}
            if "core_terms_json" not in columns:
                conn.execute("ALTER TABLE search_profiles ADD COLUMN core_terms_json TEXT NOT NULL DEFAULT '[]'")
            if "deleted_at" not in columns:
                conn.execute("ALTER TABLE search_profiles ADD COLUMN deleted_at TEXT")
            if "deleted_note" not in columns:
                conn.execute("ALTER TABLE search_profiles ADD COLUMN deleted_note TEXT NOT NULL DEFAULT ''")
            paper_columns = {row[1] for row in conn.execute("PRAGMA table_info(paper_shelf)").fetchall()}
            if "labels_json" not in paper_columns:
                conn.execute("ALTER TABLE paper_shelf ADD COLUMN labels_json TEXT NOT NULL DEFAULT '[]'")
            conn.execute("INSERT OR REPLACE INTO schema_meta VALUES (?, ?)", ("schema_version", "3"))
            duplicates = conn.execute(
                "SELECT phenomenon_id FROM decisions GROUP BY phenomenon_id HAVING COUNT(*) > 1"
            ).fetchone()
            # A legacy database may already contain duplicate decisions. Preserve it
            # untouched; guarded transitions still prevent any new duplication.
            if duplicates is None:
                conn.execute(
                    "CREATE UNIQUE INDEX IF NOT EXISTS decisions_one_per_request "
                    "ON decisions(phenomenon_id)"
                )

    def record_llm_call(self, *, profile_name: str, model: str, prompt: str, settings: dict[str, Any], response: str | None, error: str | None, diagnostics: dict[str, Any] | None) -> None:
        """Append-only diagnostic record. It never changes research knowledge."""
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO llm_calls VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (f"llm-{uuid.uuid4().hex[:12]}", now(), profile_name, model, prompt,
                 json.dumps(settings, ensure_ascii=False), response or "", error or "", json.dumps(diagnostics or {}, ensure_ascii=False)),
            )

    def llm_calls(self, limit: int = 50) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute("SELECT * FROM llm_calls ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["settings"] = json.loads(item.pop("settings_json"))
            item["diagnostics"] = json.loads(item.pop("diagnostics_json"))
            result.append(item)
        return result

    def clear_llm_calls(self) -> int:
        """Remove only diagnostic LLM records; research knowledge and decisions remain intact."""
        with self.connect() as conn:
            result = conn.execute("DELETE FROM llm_calls")
        return result.rowcount

    def create_case(self, case_type: str, title: str, status: str = "open") -> str:
        case_id = f"case-{uuid.uuid4().hex[:12]}"
        timestamp = now()
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO cases VALUES (?, ?, ?, ?, ?, ?)",
                (case_id, case_type, title, status, timestamp, timestamp),
            )
        return case_id

    def record(
        self,
        case_id: str,
        phenomenon_type: str,
        producer: str,
        recipients: list[str],
        subject_type: str,
        payload: dict[str, Any],
        subject_id: str | None = None,
        status: str = "proposed",
    ) -> str:
        if phenomenon_type not in PHENOMENON_TYPES:
            raise ValueError(f"Unknown phenomenon type: {phenomenon_type}")
        normalized_payload = validate_payload(phenomenon_type, payload)
        PhenomenonDraft(
            phenomenon_type=phenomenon_type, producer=producer, recipients=recipients,
            subject_type=subject_type, subject_id=subject_id, payload=normalized_payload, status=status,
        )
        phenomenon_id = f"ph-{uuid.uuid4().hex[:12]}"
        timestamp = now()
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO phenomena VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    phenomenon_id,
                    case_id,
                    phenomenon_type,
                    producer,
                    json.dumps(recipients, ensure_ascii=False),
                    subject_type,
                    subject_id,
                    json.dumps(normalized_payload, ensure_ascii=False),
                    status,
                    timestamp,
                ),
            )
            conn.execute("UPDATE cases SET updated_at=? WHERE case_id=?", (timestamp, case_id))
        return phenomenon_id

    @staticmethod
    def _row(row: sqlite3.Row) -> dict[str, Any]:
        result = dict(row)
        result["recipients"] = json.loads(result.pop("recipients_json"))
        result["payload"] = json.loads(result.pop("payload_json"))
        return result

    def phenomena(self, *, recipient: str | None = None, type_: str | None = None, status: str | None = None) -> list[dict[str, Any]]:
        clauses, values = [], []
        if type_:
            clauses.append("phenomenon_type=?")
            values.append(type_)
        if status:
            clauses.append("status=?")
            values.append(status)
        query = "SELECT * FROM phenomena"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY created_at DESC"
        with self.connect() as conn:
            records = [self._row(row) for row in conn.execute(query, values).fetchall()]
        return [row for row in records if not recipient or recipient in row["recipients"]]

    def case(self, case_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM cases WHERE case_id=?", (case_id,)).fetchone()
        return dict(row) if row else None

    def phenomenon(self, phenomenon_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM phenomena WHERE phenomenon_id=?", (phenomenon_id,)).fetchone()
        return self._row(row) if row else None

    def timeline(self, case_id: str) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM phenomena WHERE case_id=? ORDER BY created_at ASC", (case_id,)
            ).fetchall()
        return [self._row(row) for row in rows]

    def cases(self) -> list[dict[str, Any]]:
        with self.connect() as conn:
            return [dict(row) for row in conn.execute("SELECT * FROM cases ORDER BY updated_at DESC").fetchall()]

    def set_status(self, phenomenon_id: str, status: str) -> None:
        with self.connect() as conn:
            conn.execute("UPDATE phenomena SET status=? WHERE phenomenon_id=?", (status, phenomenon_id))

    def transition(self, phenomenon_id: str, expected_status: str, next_status: str) -> bool:
        """Compare-and-set transition used by application services for idempotency."""
        allowed = {
            "proposed": {"approved", "deferred", "rejected"},
            "ready": {"completed", "failed"},
        }
        if next_status not in allowed.get(expected_status, set()):
            raise ValueError(f"Invalid status transition: {expected_status} → {next_status}")
        with self.connect() as conn:
            result = conn.execute(
                "UPDATE phenomena SET status=? WHERE phenomenon_id=? AND status=?",
                (next_status, phenomenon_id, expected_status),
            )
        return result.rowcount == 1

    def add_decision(self, phenomenon_id: str, decision: str, note: str = "") -> None:
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO decisions VALUES (?, ?, ?, ?, ?, ?)",
                (f"dec-{uuid.uuid4().hex[:12]}", phenomenon_id, decision, note, "researcher", now()),
            )

    def decisions(self) -> list[dict[str, Any]]:
        with self.connect() as conn:
            return [dict(row) for row in conn.execute("SELECT * FROM decisions ORDER BY decided_at DESC").fetchall()]

    def create_search_profile(self, intent: dict[str, Any]) -> dict[str, Any]:
        """Create one editable M1 search profile when a researcher approves an Intent."""
        intent_id = str(intent["intent_id"])
        timestamp = now()
        profile = {
            "profile_id": f"sp-{uuid.uuid4().hex[:12]}", "intent_id": intent_id,
            "title": str(intent["title"]), "question": str(intent["question"]),
            "context": "\n".join(filter(None, [
                f"Research context: {intent.get('research_context') or intent['question']}",
                f"탐색 목적: {intent.get('purpose', '')}", f"연구 질문: {intent['question']}",
                f"기대 근거: {intent.get('expected_evidence', '')}", f"완료 조건: {intent.get('completion_condition', '')}",
            ])),
            # arXiv receives only English search terms. Korean labels remain in
            # the Intent context and M2 translates them on the profile screen.
            "keywords": [str(item) for item in intent.get("labels", []) if _is_english_search_term(str(item))],
            "core_terms": [],
            "cadence": "daily", "is_active": True, "created_at": timestamp, "updated_at": timestamp, "last_run_at": None,
        }
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM search_profiles WHERE intent_id=?", (intent_id,)).fetchone()
            if row:
                return self._search_profile_row(row)
            conn.execute(
                "INSERT INTO search_profiles (profile_id, intent_id, title, question, context, keywords_json, core_terms_json, cadence, is_active, created_at, updated_at, last_run_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (profile["profile_id"], profile["intent_id"], profile["title"], profile["question"], profile["context"],
                 json.dumps(profile["keywords"], ensure_ascii=False), json.dumps(profile["core_terms"], ensure_ascii=False), profile["cadence"], int(profile["is_active"]),
                 timestamp, timestamp, None),
            )
        return profile

    @staticmethod
    def _search_profile_row(row: sqlite3.Row) -> dict[str, Any]:
        result = dict(row)
        result["keywords"] = json.loads(result.pop("keywords_json"))
        result["core_terms"] = json.loads(result.pop("core_terms_json", "[]"))
        result["is_active"] = bool(result["is_active"])
        return result

    def search_profiles(self, *, active_only: bool = False, include_deleted: bool = False) -> list[dict[str, Any]]:
        conditions = [] if include_deleted else ["deleted_at IS NULL"]
        if active_only:
            conditions.append("is_active=1")
        query = "SELECT * FROM search_profiles" + (" WHERE " + " AND ".join(conditions) if conditions else "") + " ORDER BY updated_at DESC"
        with self.connect() as conn:
            return [self._search_profile_row(row) for row in conn.execute(query).fetchall()]

    def update_search_profile(self, profile_id: str, *, context: str, keywords: list[str], cadence: str, is_active: bool, core_terms: list[str] | None = None) -> None:
        if cadence not in {"daily", "weekly", "manual"}:
            raise ValueError("지원하지 않는 탐색 주기입니다.")
        cleaned = [item.strip() for item in keywords if item.strip()][:12]
        if not cleaned:
            raise ValueError("탐색 키워드를 한 개 이상 지정하세요.")
        if invalid := [item for item in cleaned if not _is_english_search_term(item)]:
            raise ValueError(f"arXiv 탐색 키워드는 영어로만 입력하세요: {', '.join(invalid[:3])}")
        core_terms = [item.strip() for item in (core_terms or []) if item.strip()][:5]
        if invalid := [item for item in core_terms if not _is_english_search_term(item)]:
            raise ValueError(f"핵심 검색어는 영어로만 입력하세요: {', '.join(invalid[:3])}")
        with self.connect() as conn:
            conn.execute(
                "UPDATE search_profiles SET context=?, keywords_json=?, core_terms_json=?, cadence=?, is_active=?, updated_at=? WHERE profile_id=?",
                (context.strip(), json.dumps(cleaned, ensure_ascii=False), json.dumps(core_terms, ensure_ascii=False), cadence, int(is_active), now(), profile_id),
            )

    def complete_search_profile(self, profile_id: str) -> None:
        """Remove an executed Intent from the runnable queue but keep its history."""
        with self.connect() as conn:
            conn.execute(
                "UPDATE search_profiles SET is_active=0, updated_at=? WHERE profile_id=?",
                (now(), profile_id),
            )

    def delete_search_profile(self, profile_id: str, note: str = "연구자가 탐색 큐에서 삭제") -> bool:
        """Hide an Intent from queue/history while retaining it and its runs for audit."""
        with self.connect() as conn:
            result = conn.execute(
                "UPDATE search_profiles SET is_active=0, deleted_at=?, deleted_note=?, updated_at=? WHERE profile_id=? AND deleted_at IS NULL",
                (now(), note, now(), profile_id),
            )
        return result.rowcount == 1

    def record_search_run(self, profile_id: str, trigger: str, query: str, candidates: list[dict[str, Any]], status: str, error: str = "") -> str:
        run_id, timestamp = f"sr-{uuid.uuid4().hex[:12]}", now()
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO search_runs VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (run_id, profile_id, trigger, query, json.dumps(candidates, ensure_ascii=False), status, error, timestamp),
            )
            conn.execute("UPDATE search_profiles SET last_run_at=?, updated_at=? WHERE profile_id=?", (timestamp, timestamp, profile_id))
        return run_id

    def search_runs(self, profile_id: str, limit: int = 10) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute("SELECT * FROM search_runs WHERE profile_id=? ORDER BY created_at DESC LIMIT ?", (profile_id, limit)).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["candidates"] = json.loads(item.pop("candidates_json"))
            result.append(item)
        return result

    def update_search_run_candidates(self, run_id: str, candidates: list[dict[str, Any]]) -> None:
        with self.connect() as conn:
            conn.execute("UPDATE search_runs SET candidates_json=? WHERE run_id=?", (json.dumps(candidates, ensure_ascii=False), run_id))

    def upsert_knowledge_relation(self, relation: dict[str, Any]) -> None:
        """Persist the approved relation projection used by graph traversal.

        The append-only JSONL relation record remains the audit artifact; this
        table is the indexed, mutable projection for local graph queries.
        """
        with self.connect() as conn:
            conn.execute(
                """INSERT INTO knowledge_relations
                   (relation_id, source_card_id, target_card_id, relation_type, evidence, conditions, confidence, approved_at, deleted_at, deleted_note)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, '')
                   ON CONFLICT(relation_id) DO NOTHING""",
                (str(relation["relation_id"]), str(relation["source_card_id"]), str(relation["target_card_id"]),
                 str(relation["relation_type"]), str(relation["evidence"]), str(relation["conditions"]),
                 str(relation["confidence"]), str(relation.get("approved_at") or now())),
            )

    def sync_knowledge_relations(self, relations: list[dict[str, Any]]) -> None:
        for relation in relations:
            self.upsert_knowledge_relation(relation)

    def active_knowledge_relations(self, card_ids: set[str] | None = None) -> list[dict[str, Any]]:
        query = "SELECT * FROM knowledge_relations WHERE deleted_at IS NULL"
        values: list[object] = []
        if card_ids is not None:
            if not card_ids:
                return []
            placeholders = ", ".join("?" for _ in card_ids)
            query += f" AND source_card_id IN ({placeholders}) AND target_card_id IN ({placeholders})"
            values.extend(sorted(card_ids))
            values.extend(sorted(card_ids))
        query += " ORDER BY approved_at DESC"
        with self.connect() as conn:
            return [dict(row) for row in conn.execute(query, values).fetchall()]

    def delete_knowledge_relation(self, relation_id: str, note: str = "") -> bool:
        with self.connect() as conn:
            result = conn.execute(
                "UPDATE knowledge_relations SET deleted_at=?, deleted_note=? WHERE relation_id=? AND deleted_at IS NULL",
                (now(), note, relation_id),
            )
        return result.rowcount == 1

    def upsert_shelf_paper(self, paper: dict[str, Any]) -> dict[str, Any]:
        """Store a paper as a research asset, independently of knowledge approval."""
        title = str(paper.get("title", "")).strip()
        if not title:
            raise ValueError("논문 제목은 비어 있을 수 없습니다.")
        source_id = str(paper.get("source_id", "")).strip()
        timestamp = now()
        with self.connect() as conn:
            existing = None
            if source_id:
                existing = conn.execute("SELECT paper_id FROM paper_shelf WHERE source_id=?", (source_id,)).fetchone()
            paper_id = str(existing["paper_id"]) if existing else str(paper.get("paper_id") or f"paper-{uuid.uuid4().hex[:12]}")
            conn.execute(
                """INSERT INTO paper_shelf
                   (paper_id, title, authors_json, publication_year, source_url, source_id, pdf_path, labels_json, shelf_status, reading_status, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(paper_id) DO UPDATE SET
                     title=excluded.title, authors_json=excluded.authors_json, publication_year=excluded.publication_year,
                     source_url=excluded.source_url, source_id=excluded.source_id,
                     pdf_path=CASE WHEN excluded.pdf_path <> '' THEN excluded.pdf_path ELSE paper_shelf.pdf_path END,
                     labels_json=CASE WHEN excluded.labels_json <> '[]' THEN excluded.labels_json ELSE paper_shelf.labels_json END,
                     shelf_status=excluded.shelf_status, reading_status=excluded.reading_status, updated_at=excluded.updated_at""",
                (paper_id, title, json.dumps(paper.get("authors", []), ensure_ascii=False), str(paper.get("publication_year", "")),
                 str(paper.get("source_url", "")), source_id, str(paper.get("pdf_path", "")),
                 json.dumps(_clean_paper_labels(paper.get("labels", [])), ensure_ascii=False),
                 str(paper.get("shelf_status", "reference")), str(paper.get("reading_status", "unread")), timestamp, timestamp),
            )
        return self.shelf_paper(paper_id) or {}

    def shelf_papers(self) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute("SELECT * FROM paper_shelf ORDER BY updated_at DESC").fetchall()
        return [self._paper_shelf_row(row) for row in rows]

    def shelf_paper(self, paper_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM paper_shelf WHERE paper_id=?", (paper_id,)).fetchone()
        return self._paper_shelf_row(row) if row else None

    def _paper_shelf_row(self, row: sqlite3.Row) -> dict[str, Any]:
        item = dict(row)
        item["authors"] = json.loads(item.pop("authors_json"))
        item["labels"] = json.loads(item.pop("labels_json"))
        return item

    def update_shelf_paper(self, paper_id: str, *, shelf_status: str, reading_status: str, labels: list[str] | None = None) -> None:
        if shelf_status not in {"core", "reference", "held", "excluded"}:
            raise ValueError("지원하지 않는 서재 상태입니다.")
        if reading_status not in {"unread", "reading", "read"}:
            raise ValueError("지원하지 않는 읽기 상태입니다.")
        with self.connect() as conn:
            if labels is None:
                conn.execute("UPDATE paper_shelf SET shelf_status=?, reading_status=?, updated_at=? WHERE paper_id=?", (shelf_status, reading_status, now(), paper_id))
            else:
                conn.execute("UPDATE paper_shelf SET shelf_status=?, reading_status=?, labels_json=?, updated_at=? WHERE paper_id=?", (shelf_status, reading_status, json.dumps(_clean_paper_labels(labels), ensure_ascii=False), now(), paper_id))

    def paper_analysis(self, paper_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM paper_analyses WHERE paper_id=?", (paper_id,)).fetchone()
        return dict(row) if row else None

    def save_paper_analysis(self, paper_id: str, *, research_question: str = "", summary: str = "", researcher_note: str = "", generated: bool = False) -> None:
        timestamp = now()
        previous = self.paper_analysis(paper_id) or {}
        with self.connect() as conn:
            conn.execute(
                """INSERT INTO paper_analyses (paper_id, research_question, summary, researcher_note, generated_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?)
                   ON CONFLICT(paper_id) DO UPDATE SET research_question=excluded.research_question,
                     summary=excluded.summary, researcher_note=excluded.researcher_note,
                     generated_at=CASE WHEN excluded.generated_at <> '' THEN excluded.generated_at ELSE paper_analyses.generated_at END,
                     updated_at=excluded.updated_at""",
                (paper_id, research_question or previous.get("research_question", ""), summary or previous.get("summary", ""),
                 researcher_note, timestamp if generated else "", timestamp),
            )

    def paper_card_ids(self, paper_id: str) -> list[str]:
        with self.connect() as conn:
            rows = conn.execute("SELECT card_id FROM paper_card_links WHERE paper_id=? ORDER BY linked_at DESC", (paper_id,)).fetchall()
        return [str(row["card_id"]) for row in rows]

    def set_paper_card_links(self, paper_id: str, card_ids: list[str]) -> None:
        with self.connect() as conn:
            conn.execute("DELETE FROM paper_card_links WHERE paper_id=?", (paper_id,))
            conn.executemany("INSERT INTO paper_card_links VALUES (?, ?, ?)", [(paper_id, card_id, now()) for card_id in dict.fromkeys(card_ids)])


def _is_english_search_term(value: str) -> bool:
    """Permit searchable ASCII terms; Korean UI/context never enter an arXiv query."""
    return bool(value.strip()) and not any("가" <= character <= "힣" for character in value) and all(
        character.isascii() and (character.isalnum() or character in " -_()/+.#") for character in value
    )


def _clean_paper_labels(labels: Any) -> list[str]:
    """Keep shelf labels compact, deterministic and appropriate for filtering."""
    values = labels.split(",") if isinstance(labels, str) else labels
    cleaned: list[str] = []
    for value in values if isinstance(values, list) else []:
        label = " ".join(str(value).strip().split())
        key = label.casefold()
        if 1 < len(label) <= 48 and key not in {item.casefold() for item in cleaned}:
            cleaned.append(label)
        if len(cleaned) == 5:
            break
    return cleaned
