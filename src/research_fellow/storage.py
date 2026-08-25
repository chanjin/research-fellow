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
                """
            )
            # Safe for v0.1 databases: SQLite preserves all existing rows.
            conn.execute("INSERT OR REPLACE INTO schema_meta VALUES (?, ?)", ("schema_version", "2"))
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
