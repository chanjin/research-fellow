from __future__ import annotations

"""Portable workspace snapshots: workspace.json + content-addressed PDF assets.

SQLite remains the fast working store.  A portable snapshot is deliberately
self-contained and can rebuild the working DB when that DB is missing.
"""

from dataclasses import dataclass, asdict
from datetime import UTC, datetime
from pathlib import Path
import base64
import hashlib
import json
import re
import shutil
import sqlite3
from typing import Any, Iterable

from .storage import Ledger
from .workspace_profiles import WORKSPACE_PROFILE_META_KEY
from .workspace_sync import LOCAL_ONLY_TABLES, LOCAL_ONLY_COLUMNS


PORTABLE_FORMAT = "research-fellow-portable-v1"
PORTABLE_JSON = "workspace.json"
PORTABLE_PDF_DIR = "pdf"
# schema_meta is represented explicitly in the snapshot metadata rather than as
# ordinary application data.  Sync history and LLM audit logs are machine-local.
PORTABLE_EXCLUDED_TABLES = frozenset(set(LOCAL_ONLY_TABLES) | {"schema_meta"})


def _utcnow() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _workspace_key_from_db(db_path: str | Path) -> str:
    name = Path(db_path).name
    if name == "research_fellow.db":
        return "general"
    match = re.fullmatch(r"research_fellow_(.+)\.db", name)
    if match:
        return match.group(1)
    return re.sub(r"[^a-z0-9]+", "_", Path(db_path).stem.lower()).strip("_") or "workspace"


def workspace_bundle_dir(data_dir: str | Path, workspace_key: str) -> Path:
    return Path(data_dir).expanduser() / "workspaces" / workspace_key


def workspace_json_path(data_dir: str | Path, workspace_key: str) -> Path:
    return workspace_bundle_dir(data_dir, workspace_key) / PORTABLE_JSON


def _quote_identifier(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def _tables(conn: sqlite3.Connection) -> list[str]:
    return [
        str(row[0])
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
    ]


def _columns(conn: sqlite3.Connection, table: str) -> list[str]:
    return [str(row[1]) for row in conn.execute(f"PRAGMA table_info({_quote_identifier(table)})").fetchall()]


def _pk_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    info = conn.execute(f"PRAGMA table_info({_quote_identifier(table)})").fetchall()
    ordered = sorted((int(row[5]), str(row[1])) for row in info if int(row[5] or 0) > 0)
    return [name for _, name in ordered]


def _jsonable(value: Any) -> Any:
    if isinstance(value, bytes):
        return {"__bytes_b64__": base64.b64encode(value).decode("ascii")}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _from_jsonable(value: Any) -> Any:
    if isinstance(value, dict) and set(value) == {"__bytes_b64__"}:
        return base64.b64decode(str(value["__bytes_b64__"]).encode("ascii"))
    return value


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _profile_payload(conn: sqlite3.Connection, workspace_key: str) -> dict[str, Any]:
    try:
        row = conn.execute("SELECT value FROM schema_meta WHERE key=?", (WORKSPACE_PROFILE_META_KEY,)).fetchone()
    except sqlite3.DatabaseError:
        row = None
    if row:
        try:
            parsed = json.loads(str(row[0] or "{}"))
            if isinstance(parsed, dict):
                return {**parsed, "key": str(parsed.get("key") or workspace_key)}
        except json.JSONDecodeError:
            pass
    return {"key": workspace_key, "label": workspace_key.replace("_", " ").title()}


def _resolve_pdf_path(raw: Any, *, db_path: Path, bundle: Path | None = None) -> Path | None:
    value = str(raw or "").strip()
    if not value or value.lower().startswith(("http://", "https://")):
        return None
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    if bundle is not None:
        candidate = bundle / path
        if candidate.exists():
            return candidate
    return db_path.parent / path


def _portable_pdf_ref(path: Path) -> tuple[str, str, int]:
    digest = _sha256(path)
    return f"{PORTABLE_PDF_DIR}/{digest}.pdf", digest, int(path.stat().st_size)


def _portable_row(
    table: str,
    row: dict[str, Any],
    *,
    db_path: Path,
    bundle: Path,
    copy_assets: bool,
    assets: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    ignored = LOCAL_ONLY_COLUMNS.get(table, set())
    result = {key: _jsonable(value) for key, value in row.items() if key not in ignored}
    if table != "paper_shelf":
        return result

    # pdf_path is machine-local in normal DB sync, but portable snapshots replace
    # it with a content-addressed relative path.
    raw_pdf = row.get("pdf_path")
    paper_id = str(row.get("paper_id") or "")
    source = _resolve_pdf_path(raw_pdf, db_path=db_path, bundle=bundle)
    portable_path = ""
    asset: dict[str, Any] = {"paper_id": paper_id, "path": "", "sha256": "", "size": 0, "status": "none"}
    if source and source.is_file():
        portable_path, digest, size = _portable_pdf_ref(source)
        asset.update({"path": portable_path, "sha256": digest, "size": size, "status": "available"})
        if copy_assets:
            destination = bundle / portable_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists() or destination.stat().st_size != size:
                shutil.copy2(source, destination)
    elif str(raw_pdf or "").strip():
        asset.update({"status": "missing_source", "source_path": str(raw_pdf)})
    result["pdf_path"] = portable_path
    if paper_id:
        assets[paper_id] = asset
    return result


def _record_key(row: dict[str, Any], pk_columns: list[str]) -> str:
    if pk_columns:
        return "|".join(str(row.get(column, "")) for column in pk_columns)
    return hashlib.sha256(_canonical(row).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class PortableExportResult:
    workspace_key: str
    bundle_dir: str
    json_path: str
    table_count: int
    row_count: int
    pdf_count: int
    pdf_bytes: int
    missing_pdf_sources: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PortableRestoreResult:
    restored: bool
    db_path: str
    json_path: str
    inserted_rows: int = 0
    skipped_tables: tuple[str, ...] = ()
    skipped_columns: int = 0

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["skipped_tables"] = list(self.skipped_tables)
        return data


def export_workspace_snapshot(
    db_path: str | Path,
    *,
    data_dir: str | Path | None = None,
    workspace_key: str | None = None,
) -> PortableExportResult:
    """Write a self-contained workspace.json and content-addressed PDF folder."""
    db = Path(db_path).expanduser()
    if not db.is_file():
        raise FileNotFoundError(f"Workspace DB not found: {db}")
    root = Path(data_dir).expanduser() if data_dir is not None else db.parent
    key = workspace_key or _workspace_key_from_db(db)
    bundle = workspace_bundle_dir(root, key)
    bundle.mkdir(parents=True, exist_ok=True)

    assets: dict[str, dict[str, Any]] = {}
    tables_payload: dict[str, Any] = {}
    row_count = 0
    with sqlite3.connect(db) as conn:
        conn.row_factory = sqlite3.Row
        profile = _profile_payload(conn, key)
        for table in _tables(conn):
            if table in PORTABLE_EXCLUDED_TABLES:
                continue
            columns = _columns(conn, table)
            pk_columns = _pk_columns(conn, table)
            rows = []
            for raw in conn.execute(f"SELECT * FROM {_quote_identifier(table)}").fetchall():
                row = {column: raw[column] for column in columns}
                rows.append(_portable_row(table, row, db_path=db, bundle=bundle, copy_assets=True, assets=assets))
            rows.sort(key=lambda item: _record_key(item, pk_columns))
            tables_payload[table] = {
                "columns": columns,
                "primary_key": pk_columns,
                "rows": rows,
            }
            row_count += len(rows)

    payload = {
        "format": PORTABLE_FORMAT,
        "format_version": 1,
        "exported_at": _utcnow(),
        "workspace": profile,
        "source": {"db_filename": db.name},
        "tables": tables_payload,
        "assets": {"papers": assets},
    }
    json_path = bundle / PORTABLE_JSON
    temporary = json_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(json_path)

    available = [item for item in assets.values() if item.get("status") == "available"]
    return PortableExportResult(
        workspace_key=key,
        bundle_dir=str(bundle),
        json_path=str(json_path),
        table_count=len(tables_payload),
        row_count=row_count,
        pdf_count=len({str(item.get("sha256")) for item in available if item.get("sha256")}),
        pdf_bytes=sum(int(item.get("size") or 0) for item in {str(x.get('sha256')): x for x in available if x.get('sha256')}.values()),
        missing_pdf_sources=sum(1 for item in assets.values() if item.get("status") == "missing_source"),
    )


def _load_snapshot(json_path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read portable workspace JSON: {json_path}") from exc
    if not isinstance(payload, dict) or payload.get("format") != PORTABLE_FORMAT:
        raise ValueError("Unsupported portable workspace format.")
    if not isinstance(payload.get("tables"), dict):
        raise ValueError("Portable workspace has no tables payload.")
    return payload


def restore_workspace_db_from_snapshot(
    json_path: str | Path,
    db_path: str | Path,
    *,
    overwrite: bool = False,
) -> PortableRestoreResult:
    """Rebuild a missing working DB from workspace.json + pdf/.

    The current application schema is created first.  Removed legacy tables are
    skipped and columns unknown to the current schema are ignored, which makes an
    older portable snapshot usable after application upgrades.
    """
    source_json = Path(json_path).expanduser()
    target = Path(db_path).expanduser()
    if target.exists() and not overwrite:
        return PortableRestoreResult(False, str(target), str(source_json))
    payload = _load_snapshot(source_json)
    bundle = source_json.parent
    if target.exists():
        target.unlink()
    target.parent.mkdir(parents=True, exist_ok=True)
    Ledger(target)

    inserted = 0
    skipped_tables: list[str] = []
    skipped_columns = 0
    with sqlite3.connect(target) as conn:
        conn.row_factory = sqlite3.Row
        current_tables = set(_tables(conn))
        for table, spec in payload.get("tables", {}).items():
            if table not in current_tables or not isinstance(spec, dict):
                skipped_tables.append(str(table))
                continue
            current_columns = _columns(conn, table)
            rows = spec.get("rows") if isinstance(spec.get("rows"), list) else []
            for raw_row in rows:
                if not isinstance(raw_row, dict):
                    continue
                row = dict(raw_row)
                if table == "paper_shelf" and "pdf_path" in row:
                    rel = str(row.get("pdf_path") or "").strip()
                    row["pdf_path"] = str((bundle / rel).resolve()) if rel else ""
                usable = {key: _from_jsonable(value) for key, value in row.items() if key in current_columns}
                skipped_columns += max(0, len(row) - len(usable))
                if not usable:
                    continue
                names = list(usable)
                placeholders = ",".join("?" for _ in names)
                sql = (
                    f"INSERT OR REPLACE INTO {_quote_identifier(table)} "
                    f"({','.join(_quote_identifier(name) for name in names)}) VALUES ({placeholders})"
                )
                conn.execute(sql, tuple(usable[name] for name in names))
                inserted += 1
        profile = payload.get("workspace") if isinstance(payload.get("workspace"), dict) else {}
        if profile:
            conn.execute(
                "INSERT OR REPLACE INTO schema_meta(key,value) VALUES (?,?)",
                (WORKSPACE_PROFILE_META_KEY, json.dumps(profile, ensure_ascii=False)),
            )
        conn.commit()
    return PortableRestoreResult(True, str(target), str(source_json), inserted, tuple(skipped_tables), skipped_columns)


def restore_workspace_db_if_missing(
    db_path: str | Path,
    *,
    data_dir: str | Path | None = None,
    workspace_key: str | None = None,
) -> PortableRestoreResult:
    db = Path(db_path).expanduser()
    root = Path(data_dir).expanduser() if data_dir is not None else db.parent
    key = workspace_key or _workspace_key_from_db(db)
    json_path = workspace_json_path(root, key)
    if db.exists() or not json_path.is_file():
        return PortableRestoreResult(False, str(db), str(json_path))
    return restore_workspace_db_from_snapshot(json_path, db)


def check_workspace_consistency(
    db_path: str | Path,
    *,
    data_dir: str | Path | None = None,
    workspace_key: str | None = None,
) -> dict[str, Any]:
    """Compare the working DB with its portable JSON/PDF snapshot without modifying either."""
    db = Path(db_path).expanduser()
    root = Path(data_dir).expanduser() if data_dir is not None else db.parent
    key = workspace_key or _workspace_key_from_db(db)
    bundle = workspace_bundle_dir(root, key)
    json_path = bundle / PORTABLE_JSON
    report: dict[str, Any] = {
        "workspace_key": key,
        "bundle_dir": str(bundle),
        "json_path": str(json_path),
        "db_path": str(db),
        "db_exists": db.is_file(),
        "json_exists": json_path.is_file(),
        "consistent": False,
        "tables": [],
        "db_only": 0,
        "json_only": 0,
        "changed": 0,
        "pdf_missing": 0,
        "pdf_hash_mismatch": 0,
        "pdf_extra": 0,
    }
    if not db.is_file() or not json_path.is_file():
        report["status"] = "missing_db" if not db.is_file() else "missing_snapshot"
        return report

    payload = _load_snapshot(json_path)
    json_tables = payload.get("tables", {})
    assets_for_compare: dict[str, dict[str, Any]] = {}
    with sqlite3.connect(db) as conn:
        conn.row_factory = sqlite3.Row
        db_tables = set(_tables(conn)) - set(PORTABLE_EXCLUDED_TABLES)
        all_tables = sorted(db_tables | set(json_tables))
        for table in all_tables:
            db_rows: dict[str, dict[str, Any]] = {}
            json_rows: dict[str, dict[str, Any]] = {}
            spec = json_tables.get(table) if isinstance(json_tables.get(table), dict) else {}
            pk = [str(x) for x in (spec.get("primary_key") or [])]
            if table in db_tables:
                if not pk:
                    pk = _pk_columns(conn, table)
                columns = _columns(conn, table)
                for raw in conn.execute(f"SELECT * FROM {_quote_identifier(table)}").fetchall():
                    row = {column: raw[column] for column in columns}
                    portable = _portable_row(table, row, db_path=db, bundle=bundle, copy_assets=False, assets=assets_for_compare)
                    db_rows[_record_key(portable, pk)] = portable
            for raw in (spec.get("rows") or []):
                if isinstance(raw, dict):
                    json_rows[_record_key(raw, pk)] = raw
            db_keys, json_keys = set(db_rows), set(json_rows)
            common = db_keys & json_keys
            changed = sum(1 for item in common if _canonical(db_rows[item]) != _canonical(json_rows[item]))
            row_report = {
                "table": table,
                "db_rows": len(db_rows),
                "json_rows": len(json_rows),
                "db_only": len(db_keys - json_keys),
                "json_only": len(json_keys - db_keys),
                "changed": changed,
            }
            report["tables"].append(row_report)
            report["db_only"] += row_report["db_only"]
            report["json_only"] += row_report["json_only"]
            report["changed"] += changed

    expected_assets = payload.get("assets", {}).get("papers", {}) if isinstance(payload.get("assets"), dict) else {}
    referenced_paths: set[Path] = set()
    for item in expected_assets.values() if isinstance(expected_assets, dict) else []:
        if not isinstance(item, dict) or item.get("status") != "available":
            continue
        rel = str(item.get("path") or "").strip()
        if not rel:
            continue
        path = bundle / rel
        referenced_paths.add(path.resolve())
        if not path.is_file():
            report["pdf_missing"] += 1
            continue
        expected_hash = str(item.get("sha256") or "")
        if expected_hash and _sha256(path) != expected_hash:
            report["pdf_hash_mismatch"] += 1
    pdf_dir = bundle / PORTABLE_PDF_DIR
    existing = {path.resolve() for path in pdf_dir.glob("*.pdf") if path.is_file()} if pdf_dir.exists() else set()
    report["pdf_extra"] = len(existing - referenced_paths)
    report["consistent"] = not any(
        int(report[key] or 0)
        for key in ("db_only", "json_only", "changed", "pdf_missing", "pdf_hash_mismatch")
    )
    report["status"] = "consistent" if report["consistent"] else "different"
    return report
