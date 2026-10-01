from __future__ import annotations

"""Machine-local application settings.

These settings are intentionally kept outside workspace databases because values such
as a server mount/root path are machine-specific and must not be propagated by
Workspace Sync.
"""

import sqlite3
from pathlib import Path


SETTINGS_DB_FILENAME = "local_settings.db"
WORKSPACE_SYNC_SERVER_ROOT_KEY = "workspace_sync_server_root"


def _db_path(data_dir: str | Path) -> Path:
    return Path(data_dir).expanduser() / SETTINGS_DB_FILENAME


def _ensure(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS local_settings (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )


def get_local_setting(data_dir: str | Path, key: str, default: str = "") -> str:
    path = _db_path(data_dir)
    if not path.exists():
        return default
    try:
        with sqlite3.connect(path) as conn:
            _ensure(conn)
            row = conn.execute("SELECT value FROM local_settings WHERE key=?", (key,)).fetchone()
        return str(row[0]) if row else default
    except sqlite3.DatabaseError:
        return default


def set_local_setting(data_dir: str | Path, key: str, value: str) -> None:
    path = _db_path(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        _ensure(conn)
        conn.execute(
            """
            INSERT INTO local_settings(key, value, updated_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(key) DO UPDATE SET
                value=excluded.value,
                updated_at=CURRENT_TIMESTAMP
            """,
            (key, str(value)),
        )


def get_workspace_sync_server_root(data_dir: str | Path) -> str:
    return get_local_setting(data_dir, WORKSPACE_SYNC_SERVER_ROOT_KEY, "")


def set_workspace_sync_server_root(data_dir: str | Path, value: str) -> None:
    normalized = str(value or "").strip()
    if not normalized:
        return
    set_local_setting(data_dir, WORKSPACE_SYNC_SERVER_ROOT_KEY, normalized)
