from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import shutil
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
    "search_runs",
    "literature_discovery_sessions",
    "literature_discovery_workspace",
    "literature_references",
    "paper_projects",
    "paper_project_events",
    "knowledge_cards",
    "knowledge_relations",
    "paper_shelf",
    "paper_abstracts",
    "paper_analyses",
    "paper_card_links",
    "paper_reading_questions",
    "paper_reading_reviews",
    "paper_asset_events",
    "paper_ontology_candidates",
    "ontology_facets",
    "ontology_types",
    "ontology_card_assignments",
    "ontology_type_relations",
    "ontology_versions",
    "ontology_change_reviews",
    "episode_memories",
    "research_questions",
    "research_question_intents",
    "research_state_reviews",
    "research_state_review_cards",
    "research_state_review_questions",
    "research_question_sources",
    "research_question_changes",
    "research_question_threads",
    "research_question_versions",
    "m2_reports",
    "sensemaking_threads",
    "sensemaking_turns",
    "thread_current_states",
    "thread_report_snapshots",
    "auto_research_runs",
    "auto_research_failures",
    "manual_recovery_attempts",
)

# Every durable table must be classified explicitly. User-visible research
# assets belong in SYNC_TABLES; only machine/runtime metadata belongs here.
# This allowlist prevents a newly added feature from silently becoming local-only.
LOCAL_ONLY_TABLES = frozenset({
    "schema_meta",
    "llm_calls",
    "sync_baseline",
    "sync_runs",
})

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


@dataclass(frozen=True)
class AssetSyncResult:
    uploaded: int = 0
    downloaded: int = 0
    unchanged: int = 0
    missing: int = 0


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sync_paper_assets(local_db: str | Path, server_db: str | Path, local_asset_dir: str | Path, server_asset_dir: str | Path) -> AssetSyncResult:
    """Safely exchange paper files using versioned, content-addressed names."""
    local_db, server_db = Path(local_db), Path(server_db)
    local_asset_dir, server_asset_dir = Path(local_asset_dir), Path(server_asset_dir)
    local_asset_dir.mkdir(parents=True, exist_ok=True)
    server_asset_dir.mkdir(parents=True, exist_ok=True)
    uploaded = downloaded = unchanged = missing = 0
    with WorkspaceSync._connect(local_db) as conn:
        exists = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='paper_shelf'").fetchone()
        local_rows = conn.execute("SELECT paper_id, pdf_path FROM paper_shelf").fetchall() if exists else []
    for row in local_rows:
        paper_id = str(row["paper_id"])
        safe_paper_id = re.sub(r"[^A-Za-z0-9._-]+", "_", paper_id)[:100] or "paper"
        local_path = Path(str(row["pdf_path"] or "")).expanduser() if str(row["pdf_path"] or "").strip() else None
        if local_path and local_path.is_file():
            digest = _file_sha256(local_path)
            suffix = local_path.suffix.lower() if local_path.suffix else ".bin"
            server_path = server_asset_dir / f"{safe_paper_id}--{digest[:16]}{suffix}"
            if not server_path.exists():
                shutil.copy2(local_path, server_path)
                uploaded += 1
            else:
                unchanged += 1
            with WorkspaceSync._connect(server_db) as server_conn:
                server_conn.execute("UPDATE paper_shelf SET pdf_path=? WHERE paper_id=?", (str(server_path), paper_id))
            continue
        candidates = sorted(server_asset_dir.glob(f"{safe_paper_id}--*"), key=lambda path: path.stat().st_mtime, reverse=True)
        if not candidates:
            missing += 1
            continue
        source = candidates[0]
        destination = local_asset_dir / source.name
        if not destination.exists():
            shutil.copy2(source, destination)
            downloaded += 1
        else:
            unchanged += 1
        with WorkspaceSync._connect(local_db) as local_conn:
            local_conn.execute("UPDATE paper_shelf SET pdf_path=? WHERE paper_id=?", (str(destination), paper_id))
    return AssetSyncResult(uploaded=uploaded, downloaded=downloaded, unchanged=unchanged, missing=missing)


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
        self._validate_table_policy(self.local_db)
        # Do not create an empty server DB here. Missing server DB means
        # 'first initialization' and should be created from the current local DB.
        if self.server_db.exists():
            Ledger(self.server_db)
            self._validate_table_policy(self.server_db)


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

    @classmethod
    def _database_tables(cls, path: Path) -> set[str]:
        with cls._connect(path) as conn:
            rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            ).fetchall()
        return {str(row[0]) for row in rows}

    @classmethod
    def unclassified_tables(cls, path: str | Path) -> set[str]:
        """Return durable tables that have no explicit sync/local-only policy."""
        tables = cls._database_tables(Path(path))
        return tables - set(SYNC_TABLES) - set(LOCAL_ONLY_TABLES)

    @classmethod
    def _validate_table_policy(cls, path: Path) -> None:
        missing = sorted(cls.unclassified_tables(path))
        if missing:
            raise ValueError(
                "동기화 정책이 지정되지 않은 테이블이 있습니다: " + ", ".join(missing)
            )

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


@dataclass
class WorkspaceBatchStatus:
    key: str
    label: str
    local_db: Path
    server_db: Path
    server_exists: bool
    upload_changes: int
    download_changes: int
    conflicts: int
    initialization_rows: int


@dataclass
class WorkspaceBatchResult:
    key: str
    initialized: bool
    applied: int
    conflicts: int
    assets: AssetSyncResult


class AllWorkspacesSync:
    """Coordinate database and paper-asset sync for every active workspace."""

    def __init__(self, profiles: dict[str, Any], data_dir: str | Path, server_root: str | Path):
        self.profiles = profiles
        self.data_dir = Path(data_dir).expanduser()
        self.server_root = Path(server_root).expanduser()

    def _paths(self, key: str, profile: Any) -> tuple[Path, Path, Path, Path]:
        local_db = self.data_dir / str(profile.db_filename)
        server_workspace = self.server_root / "workspaces" / key
        server_db = server_workspace / str(profile.server_db_filename)
        local_assets = self.data_dir / "workspaces" / key / "paper_shelf"
        server_assets = server_workspace / "assets" / "paper_shelf"
        return local_db, server_db, local_assets, server_assets

    def preview(self) -> list[WorkspaceBatchStatus]:
        statuses: list[WorkspaceBatchStatus] = []
        for key, profile in self.profiles.items():
            local_db, server_db, _, _ = self._paths(key, profile)
            sync = WorkspaceSync(local_db, server_db)
            if sync.server_exists:
                preview = sync.preview()
                counts = preview.counts()
                upload = sum(value for label, value in counts.items() if label.startswith("local→server:"))
                download = sum(value for label, value in counts.items() if label.startswith("server→local:"))
                initialization_rows = 0
                conflicts = len(preview.conflicts)
            else:
                upload = download = conflicts = 0
                initialization_rows = sum(sync.initialization_counts().values())
            statuses.append(WorkspaceBatchStatus(
                key=key, label=str(profile.label), local_db=local_db, server_db=server_db,
                server_exists=sync.server_exists, upload_changes=upload,
                download_changes=download, conflicts=conflicts,
                initialization_rows=initialization_rows,
            ))
        return statuses

    def apply(self) -> list[WorkspaceBatchResult]:
        results: list[WorkspaceBatchResult] = []
        for key, profile in self.profiles.items():
            local_db, server_db, local_assets, server_assets = self._paths(key, profile)
            sync = WorkspaceSync(local_db, server_db)
            initialized = not sync.server_exists
            if initialized:
                sync.initialize_server_from_local()
                applied = 0
                conflicts = 0
            else:
                result = sync.apply()
                applied = len(result.actionable)
                conflicts = len(result.conflicts)
            assets = sync_paper_assets(local_db, server_db, local_assets, server_assets)
            results.append(WorkspaceBatchResult(
                key=key, initialized=initialized, applied=applied,
                conflicts=conflicts, assets=assets,
            ))
        return results
