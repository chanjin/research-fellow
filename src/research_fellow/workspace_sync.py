from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .storage import Ledger


SYNC_TABLES = (
    "cases",
    "phenomena",
    "decisions",
    "search_profiles",
    "knowledge_cards",
    "knowledge_relations",
    "paper_shelf",
    "paper_analyses",
    "paper_card_links",
    "paper_reading_questions",
    "paper_reading_reviews",
    "paper_asset_events",
    "ontology_facets",
    "ontology_types",
    "ontology_card_assignments",
    "ontology_type_relations",
    "episode_memories",
)

# These values describe a machine-local artifact. They are deliberately excluded
# from row comparison and never overwrite the value on another machine.
LOCAL_ONLY_COLUMNS: dict[str, set[str]] = {
    "paper_shelf": {"pdf_path"},
}


def _utcnow() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _row_hash(table: str, row: dict[str, Any] | None) -> str | None:
    if row is None:
        return None
    ignored = LOCAL_ONLY_COLUMNS.get(table, set())
    comparable = {key: row[key] for key in sorted(row) if key not in ignored}
    return hashlib.sha256(_canonical_json(comparable).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SyncChange:
    table: str
    key: str
    action: str
    direction: str
    detail: str = ""


@dataclass
class SyncPreview:
    changes: list[SyncChange]

    @property
    def conflicts(self) -> list[SyncChange]:
        return [item for item in self.changes if item.action == "conflict"]

    @property
    def actionable(self) -> list[SyncChange]:
        return [item for item in self.changes if item.action != "conflict"]

    def counts(self) -> dict[str, int]:
        result: dict[str, int] = {}
        for item in self.changes:
            label = f"{item.direction}:{item.action}"
            result[label] = result.get(label, 0) + 1
        return result


class WorkspaceSync:
    """Explicit record-level merge between a local working DB and an integrated server DB.

    The local database stores a per-record baseline from the last successful sync.
    A missing local row is therefore considered a deletion only when that row existed
    in the baseline. This prevents stale replicas from deleting records created on
    another machine.
    """

    def __init__(self, local_db: str | Path, server_workspace: str | Path):
        self.local_db = Path(local_db).expanduser()
        requested = Path(server_workspace).expanduser()
        # Backward compatibility: callers may still pass an explicit .db path.
        # The UI now asks for a workspace directory and appends the canonical name.
        self.server_dir = requested.parent if requested.suffix.lower() == ".db" else requested
        self.server_db = requested if requested.suffix.lower() == ".db" else requested / "research_fellow.db"
        if self.local_db.resolve() == self.server_db.resolve():
            raise ValueError("로컬 DB와 서버 DB는 서로 다른 파일이어야 합니다.")
        self.local_db.parent.mkdir(parents=True, exist_ok=True)
        Ledger(self.local_db)
        self._ensure_sync_tables()
        # Do not create an empty server DB here. Missing server DB means
        # 'first initialization' and should be created from the current local DB.
        if self.server_db.exists():
            Ledger(self.server_db)


    @property
    def server_exists(self) -> bool:
        return self.server_db.is_file()

    def initialize_server_from_local(self) -> None:
        """Create the integrated server DB as a consistent snapshot of local state.

        This is intentionally different from a normal merge. On the first run there
        is no server history to compare with, so the current local workspace becomes
        the initial integrated state and the same state is recorded as the baseline.
        """
        if self.server_exists:
            raise FileExistsError(f"서버 DB가 이미 존재합니다: {self.server_db}")
        self.server_dir.mkdir(parents=True, exist_ok=True)
        with self._connect(self.local_db) as source, self._connect(self.server_db) as target:
            source.backup(target)
            # Sync metadata is replica-local. Keep the schema but not this machine's history.
            target.execute("DELETE FROM sync_baseline")
            target.execute("DELETE FROM sync_runs")
        Ledger(self.server_db)

        # Every durable row now has the same state on both sides. Record that as
        # the initial baseline so later absence can be interpreted as a real delete.
        for table in SYNC_TABLES:
            local_rows, _ = self._rows(self.local_db, table)
            if local_rows:
                self._refresh_baseline(table, set(local_rows))

        timestamp = _utcnow()
        summary = {"initialized": True, "applied": 0, "conflicts": 0, "counts": {}}
        with self._connect(self.local_db) as conn:
            conn.execute(
                "INSERT INTO sync_runs VALUES (?, ?, ?, ?, ?, ?, ?)",
                (f"sync-{uuid.uuid4().hex[:12]}", self.server_id, timestamp, timestamp, 0, 0, _canonical_json(summary)),
            )

    def initialization_counts(self) -> dict[str, int]:
        """Count local durable rows that will seed a new server workspace."""
        counts: dict[str, int] = {}
        for table in SYNC_TABLES:
            rows, _ = self._rows(self.local_db, table)
            if rows:
                counts[table] = len(rows)
        return counts

    @staticmethod
    def _connect(path: Path) -> sqlite3.Connection:
        conn = sqlite3.connect(path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_sync_tables(self) -> None:
        with self._connect(self.local_db) as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS sync_baseline (
                    server_id TEXT NOT NULL,
                    table_name TEXT NOT NULL,
                    record_key TEXT NOT NULL,
                    row_hash TEXT,
                    synced_at TEXT NOT NULL,
                    PRIMARY KEY(server_id, table_name, record_key)
                );
                CREATE TABLE IF NOT EXISTS sync_runs (
                    sync_id TEXT PRIMARY KEY,
                    server_id TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    completed_at TEXT NOT NULL,
                    applied_count INTEGER NOT NULL,
                    conflict_count INTEGER NOT NULL,
                    summary_json TEXT NOT NULL
                );
                """
            )

    @property
    def server_id(self) -> str:
        return hashlib.sha256(str(self.server_db.resolve()).encode("utf-8")).hexdigest()[:20]

    @staticmethod
    def _pk_columns(conn: sqlite3.Connection, table: str) -> list[str]:
        rows = conn.execute(f'PRAGMA table_info("{table}")').fetchall()
        keys = sorted(((int(row[5]), str(row[1])) for row in rows if int(row[5]) > 0))
        return [name for _, name in keys]

    @staticmethod
    def _record_key(row: dict[str, Any], pk_columns: list[str]) -> str:
        if not pk_columns:
            raise ValueError("동기화 테이블에는 PRIMARY KEY가 필요합니다.")
        return _canonical_json([row.get(column) for column in pk_columns])

    def _rows(self, path: Path, table: str) -> tuple[dict[str, dict[str, Any]], list[str]]:
        with self._connect(path) as conn:
            exists = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
            ).fetchone()
            if not exists:
                return {}, []
            pk_columns = self._pk_columns(conn, table)
            rows = [dict(row) for row in conn.execute(f'SELECT * FROM "{table}"').fetchall()]
        return {self._record_key(row, pk_columns): row for row in rows}, pk_columns

    def _baseline(self, table: str) -> dict[str, str | None]:
        with self._connect(self.local_db) as conn:
            rows = conn.execute(
                "SELECT record_key, row_hash FROM sync_baseline WHERE server_id=? AND table_name=?",
                (self.server_id, table),
            ).fetchall()
        return {str(row[0]): row[1] for row in rows}

    def preview(self) -> SyncPreview:
        if not self.server_exists:
            raise FileNotFoundError("서버 DB가 아직 없습니다. 현재 로컬 DB로 서버를 먼저 초기화하세요.")
        changes: list[SyncChange] = []
        for table in SYNC_TABLES:
            local_rows, _ = self._rows(self.local_db, table)
            server_rows, _ = self._rows(self.server_db, table)
            baseline = self._baseline(table)
            all_keys = set(local_rows) | set(server_rows) | set(baseline)
            for key in sorted(all_keys):
                local_hash = _row_hash(table, local_rows.get(key))
                server_hash = _row_hash(table, server_rows.get(key))
                base_hash = baseline.get(key)

                if local_hash == server_hash:
                    continue

                local_changed = local_hash != base_hash
                server_changed = server_hash != base_hash

                if local_changed and server_changed:
                    changes.append(SyncChange(table, key, "conflict", "both", "양쪽에서 같은 레코드가 변경됨"))
                    continue

                if local_changed:
                    action = "delete" if local_hash is None else ("add" if base_hash is None else "update")
                    changes.append(SyncChange(table, key, action, "local→server"))
                elif server_changed:
                    action = "delete" if server_hash is None else ("add" if base_hash is None else "update")
                    changes.append(SyncChange(table, key, action, "server→local"))
        return SyncPreview(changes)

    @staticmethod
    def _copy_row(
        source_path: Path,
        target_path: Path,
        table: str,
        key: str,
    ) -> None:
        source_rows, source_pk = WorkspaceSync._static_rows(source_path, table)
        target_rows, target_pk = WorkspaceSync._static_rows(target_path, table)
        pk_columns = source_pk or target_pk
        source = source_rows.get(key)
        target = target_rows.get(key)
        with WorkspaceSync._connect(target_path) as conn:
            if source is None:
                values = json.loads(key)
                where = " AND ".join(f'"{column}"=?' for column in pk_columns)
                conn.execute(f'DELETE FROM "{table}" WHERE {where}', values)
                return
            merged = dict(source)
            for column in LOCAL_ONLY_COLUMNS.get(table, set()):
                if target is not None and column in target:
                    merged[column] = target[column]
                elif target is None:
                    # Do not leak a source machine's local path into a new replica.
                    merged[column] = ""
            columns = list(merged)
            placeholders = ",".join("?" for _ in columns)
            column_sql = ",".join(f'"{column}"' for column in columns)
            conn.execute(
                f'INSERT OR REPLACE INTO "{table}" ({column_sql}) VALUES ({placeholders})',
                [merged[column] for column in columns],
            )

    @staticmethod
    def _static_rows(path: Path, table: str) -> tuple[dict[str, dict[str, Any]], list[str]]:
        with WorkspaceSync._connect(path) as conn:
            exists = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
            ).fetchone()
            if not exists:
                return {}, []
            pk_columns = WorkspaceSync._pk_columns(conn, table)
            rows = [dict(row) for row in conn.execute(f'SELECT * FROM "{table}"').fetchall()]
        return {WorkspaceSync._record_key(row, pk_columns): row for row in rows}, pk_columns

    def _refresh_baseline(self, table: str, keys: set[str]) -> None:
        local_rows, _ = self._rows(self.local_db, table)
        timestamp = _utcnow()
        with self._connect(self.local_db) as conn:
            for key in keys:
                row_hash = _row_hash(table, local_rows.get(key))
                if row_hash is None:
                    conn.execute(
                        "DELETE FROM sync_baseline WHERE server_id=? AND table_name=? AND record_key=?",
                        (self.server_id, table, key),
                    )
                else:
                    conn.execute(
                        """INSERT INTO sync_baseline(server_id, table_name, record_key, row_hash, synced_at)
                           VALUES (?, ?, ?, ?, ?)
                           ON CONFLICT(server_id, table_name, record_key)
                           DO UPDATE SET row_hash=excluded.row_hash, synced_at=excluded.synced_at""",
                        (self.server_id, table, key, row_hash, timestamp),
                    )

    def apply(self) -> SyncPreview:
        if not self.server_exists:
            self.initialize_server_from_local()
            return SyncPreview([])
        preview = self.preview()
        applied: list[SyncChange] = []
        touched: dict[str, set[str]] = {}
        for change in preview.actionable:
            if change.direction == "local→server":
                self._copy_row(self.local_db, self.server_db, change.table, change.key)
            elif change.direction == "server→local":
                self._copy_row(self.server_db, self.local_db, change.table, change.key)
            applied.append(change)
            touched.setdefault(change.table, set()).add(change.key)

        # Records that were already identical but have no baseline yet need a
        # baseline too; otherwise a later deletion could be mistaken for a new state.
        for table in SYNC_TABLES:
            local_rows, _ = self._rows(self.local_db, table)
            server_rows, _ = self._rows(self.server_db, table)
            baseline = self._baseline(table)
            for key in set(local_rows) & set(server_rows):
                if key not in baseline and _row_hash(table, local_rows[key]) == _row_hash(table, server_rows[key]):
                    touched.setdefault(table, set()).add(key)

        for table, keys in touched.items():
            self._refresh_baseline(table, keys)

        summary = {
            "applied": len(applied),
            "conflicts": len(preview.conflicts),
            "counts": preview.counts(),
        }
        sync_id = f"sync-{uuid.uuid4().hex[:12]}"
        timestamp = _utcnow()
        with self._connect(self.local_db) as conn:
            conn.execute(
                "INSERT INTO sync_runs VALUES (?, ?, ?, ?, ?, ?, ?)",
                (sync_id, self.server_id, timestamp, timestamp, len(applied), len(preview.conflicts), _canonical_json(summary)),
            )
        return preview

    def last_run(self) -> dict[str, Any] | None:
        with self._connect(self.local_db) as conn:
            row = conn.execute(
                "SELECT * FROM sync_runs WHERE server_id=? ORDER BY completed_at DESC LIMIT 1",
                (self.server_id,),
            ).fetchone()
        if not row:
            return None
        result = dict(row)
        result["summary"] = json.loads(result.pop("summary_json"))
        return result
