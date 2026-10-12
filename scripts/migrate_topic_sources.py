#!/usr/bin/env python3
"""One-time, idempotent migration of legacy paper assets into Topic directories.

Run from the project root. The default mode is a read-only plan. Pass ``--apply``
to copy files, write Topic source manifests, and repoint local SQLite PDF paths.
This script is intended to be removed after the migration has been verified.
"""
from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from research_fellow.topic_store import TOPIC_FORMAT, ensure_topic_directory, topic_dir  # noqa: E402
from research_fellow.workspace_profiles import load_workspace_profiles  # noqa: E402


def safe_part(value: str, limit: int = 48) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value or "")).strip("-._")
    return cleaned[:limit] or "paper"


def paper_filename(paper_id: str, source: Path) -> str:
    identity = hashlib.sha256(paper_id.encode("utf-8")).hexdigest()[:12]
    suffix = source.suffix.lower() or ".bin"
    return f"{safe_part(source.stem)}-{identity}{suffix}"


def text_filename(paper_id: str, content: str = "") -> str:
    identity = hashlib.sha256(paper_id.encode("utf-8")).hexdigest()[:12]
    content_part = f"-{hashlib.sha256(content.encode('utf-8')).hexdigest()[:12]}" if content else ""
    return f"{safe_part(paper_id, 48)}-{identity}{content_part}.txt"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _candidate_source_paths(
    raw_path: str, *, data_dir: Path, workspace_key: str, paper_id: str,
    server_root: Path | None,
) -> list[Path]:
    candidates: list[Path] = []
    raw = str(raw_path or "").strip()
    if raw and not raw.lower().startswith(("http://", "https://")):
        path = Path(raw).expanduser()
        candidates.append(path)
        candidates.extend((
            data_dir / "paper_shelf" / path.name,
            data_dir / "paper_pdfs" / workspace_key / path.name,
        ))
    safe_id = safe_part(paper_id, 100)
    candidates.extend(sorted((data_dir / "paper_pdfs" / workspace_key).glob(f"{safe_id}--*")))
    if server_root:
        server_assets = server_root / "workspaces" / workspace_key / "assets" / "paper_shelf"
        candidates.extend(sorted(server_assets.glob(f"{safe_id}--*")))
        if raw:
            candidates.append(server_assets / Path(raw).name)
    seen: set[str] = set()
    result: list[Path] = []
    for candidate in candidates:
        try:
            key = str(candidate.expanduser().resolve())
        except OSError:
            key = str(candidate)
        if key not in seen:
            seen.add(key)
            result.append(candidate.expanduser())
    return result


def resolve_source(
    raw_path: str, *, data_dir: Path, workspace_key: str, paper_id: str,
    server_root: Path | None,
) -> Path | None:
    for candidate in _candidate_source_paths(
        raw_path, data_dir=data_dir, workspace_key=workspace_key,
        paper_id=paper_id, server_root=server_root,
    ):
        if candidate.is_file():
            return candidate
    return None


def extract_pdf_text(path: Path, *, max_chars: int = 150_000) -> tuple[str, str]:
    """Extract text with the project's declared pypdf dependency when available."""
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    chunks = [page.extract_text() or "" for page in reader.pages]
    text = "\n\n".join(chunk for chunk in chunks if chunk.strip()).strip()
    if not text:
        raise ValueError("PDF에서 텍스트를 추출하지 못했습니다.")
    truncated = len(text) > max_chars
    text = text[:max_chars]
    note = f"기존 PDF {len(reader.pages)}페이지에서 텍스트 {len(text):,}자를 추출했습니다."
    if truncated:
        note += f" 저장 상한 {max_chars:,}자에 따라 뒤쪽 텍스트를 생략했습니다."
    return text, note


def _read_sources_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"format": TOPIC_FORMAT, "sources": {}}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Topic source manifest를 읽을 수 없습니다: {path}: {error}") from error
    if not isinstance(value, dict) or not isinstance(value.get("sources"), dict):
        raise ValueError(f"Topic source manifest 구조가 올바르지 않습니다: {path}")
    return value


def _preserving_target(preferred: Path, source_hash: str) -> Path:
    """Return a deterministic destination without replacing different content."""
    if not preferred.exists() or _sha256(preferred) == source_hash:
        return preferred
    alternate = preferred.with_name(
        f"{preferred.stem}-{source_hash[:12]}{preferred.suffix}"
    )
    if not alternate.exists() or _sha256(alternate) == source_hash:
        return alternate
    # A longer digest also avoids turning an extraordinarily unlikely short
    # digest collision into an overwrite.
    alternate = preferred.with_name(
        f"{preferred.stem}-{source_hash}{preferred.suffix}"
    )
    if not alternate.exists() or _sha256(alternate) == source_hash:
        return alternate
    raise FileExistsError(f"내용이 다른 마이그레이션 대상 파일이 이미 있습니다: {alternate}")


def _paper_rows(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if "paper_shelf" not in tables:
        return []
    columns = {row[1] for row in conn.execute("PRAGMA table_info(paper_shelf)")}
    selected = [name for name in (
        "paper_id", "title", "authors_json", "publication_year", "source_id",
        "source_url", "full_text_url", "pdf_url", "pdf_path", "asset_type",
    ) if name in columns]
    return [dict(row) for row in conn.execute(f"SELECT {', '.join(selected)} FROM paper_shelf ORDER BY paper_id")]


def _full_text_rows(conn: sqlite3.Connection) -> dict[str, dict[str, Any]]:
    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if "paper_full_texts" not in tables:
        return {}
    return {str(row["paper_id"]): dict(row) for row in conn.execute("SELECT * FROM paper_full_texts")}


def _db_backup(db_path: Path) -> Path:
    backup_dir = db_path.parent / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    backup_path = backup_dir / f"{db_path.stem}.pre-topic-migration-{stamp}.db"
    with sqlite3.connect(db_path) as source, sqlite3.connect(backup_path) as target:
        source.backup(target)
    return backup_path


def migrate_workspace(
    *, key: str, profile: Any, db_path: Path, data_dir: Path,
    server_root: Path | None, apply: bool, update_db_paths: bool,
) -> dict[str, Any]:
    if not db_path.is_file():
        return {"workspace": key, "status": "no_database", "papers": 0}
    root = topic_dir(data_dir, key)
    manifest_path = root / "sources" / "sources.json"
    manifest = _read_sources_manifest(manifest_path) if apply else {"format": TOPIC_FORMAT, "sources": {}}
    if apply:
        root = ensure_topic_directory(
            data_dir, topic_key=key, label=str(profile.label), purpose=str(profile.purpose),
        )
    plans: list[dict[str, Any]] = []
    path_updates: dict[str, str] = {}
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        papers = _paper_rows(conn)
        full_texts = _full_text_rows(conn)
        manifest_path = root / "sources" / "sources.json"
        entries = manifest["sources"]
        for paper in papers:
            paper_id = str(paper.get("paper_id") or "").strip()
            if not paper_id:
                continue
            full_text_row = full_texts.get(paper_id, {})
            source = resolve_source(
                str(paper.get("pdf_path") or ""), data_dir=data_dir,
                workspace_key=key, paper_id=paper_id, server_root=server_root,
            )
            text = str(full_text_row.get("content") or "").strip()
            extraction_note = str(full_text_row.get("extraction_note") or "")
            if not text and source and source.suffix.lower() == ".pdf":
                try:
                    text, extraction_note = extract_pdf_text(source)
                except Exception as error:
                    extraction_note = f"자동 추출 실패: {error}"

            pdf_ref = ""
            copied_source: Path | None = None
            if source:
                try:
                    relative_target = source.resolve().relative_to(root.resolve())
                    if not relative_target.as_posix().startswith("sources/"):
                        raise ValueError("Existing Topic source is outside sources/")
                except (OSError, ValueError):
                    source_dir = "html" if (
                        str(paper.get("asset_type") or "") == "web_page"
                        or source.suffix.lower() in {".html", ".htm"}
                    ) else "papers"
                    relative_target = Path("sources") / source_dir / paper_filename(paper_id, source)
                target = root / relative_target
                target = _preserving_target(target, _sha256(source))
                relative_target = target.relative_to(root)
                pdf_ref = relative_target.as_posix()
                if apply:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if not target.exists():
                        shutil.copy2(source, target)
                    copied_source = target
                    try:
                        same_path = Path(str(paper.get("pdf_path") or "")).expanduser().resolve() == target.resolve()
                    except OSError:
                        same_path = False
                    if update_db_paths and not same_path:
                        path_updates[paper_id] = str(target)
            else:
                old_path = str(paper.get("pdf_path") or "").strip()
                if old_path.lower().startswith(("http://", "https://")):
                    pdf_ref = old_path

            full_text_ref = str(entries.get(paper_id, {}).get("full_text") or "")
            if text:
                preferred_text_path = root / "sources" / "text" / text_filename(paper_id)
                text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
                text_path = _preserving_target(preferred_text_path, text_hash)
                full_text_ref = text_path.relative_to(root).as_posix()
                if apply:
                    text_path.parent.mkdir(parents=True, exist_ok=True)
                    if not text_path.exists():
                        text_path.write_text(text, encoding="utf-8")

            authors: list[str] = []
            try:
                decoded = json.loads(str(paper.get("authors_json") or "[]"))
                authors = decoded if isinstance(decoded, list) else []
            except json.JSONDecodeError:
                pass
            previous = dict(entries.get(paper_id) or {})
            entry = {
                **previous,
                "paper_id": paper_id,
                "title": str(paper.get("title") or previous.get("title") or ""),
                "authors": authors or previous.get("authors") or [],
                "publication_year": str(paper.get("publication_year") or previous.get("publication_year") or ""),
                "doi_or_source_id": str(paper.get("source_id") or previous.get("doi_or_source_id") or ""),
                "source_url": str(full_text_row.get("source_url") or paper.get("full_text_url") or paper.get("source_url") or paper.get("pdf_url") or previous.get("source_url") or ""),
                "source_type": str(full_text_row.get("source_type") or previous.get("source_type") or ("local_file" if source else "")),
                "source_name": str(full_text_row.get("source_name") or (source.name if source else previous.get("source_name") or "")),
                "pdf": pdf_ref or str(previous.get("pdf") or ""),
                "full_text": full_text_ref,
                "extraction_note": extraction_note or str(previous.get("extraction_note") or ""),
                "updated_at": datetime.now(UTC).isoformat(timespec="seconds"),
            }
            plans.append({
                "paper_id": paper_id,
                "title": entry["title"],
                "source": str(source) if source else "",
                "topic_pdf": str(copied_source or (root / pdf_ref if pdf_ref.startswith("sources/") else "")),
                "text_chars": len(text),
                "text_path": str(root / full_text_ref) if full_text_ref else "",
                "status": "ready" if source or text or entry["source_url"] else "no_source_found",
            })
            if apply:
                entries[paper_id] = entry

        if apply:
            manifest["format"] = TOPIC_FORMAT
            manifest["updated_at"] = datetime.now(UTC).isoformat(timespec="seconds")
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
            if path_updates:
                backup = _db_backup(db_path)
                with conn:
                    conn.executemany(
                        "UPDATE paper_shelf SET pdf_path=? WHERE paper_id=?",
                        [(path, paper_id) for paper_id, path in path_updates.items()],
                    )
            else:
                backup = None
        else:
            backup = None
        return {
            "workspace": key,
            "topic_dir": str(root),
            "database": str(db_path),
            "status": "applied" if apply else "dry_run",
            "paper_count": len(papers),
            "source_files_found": sum(bool(item["source"]) for item in plans),
            "full_texts_written_or_available": sum(bool(item["text_path"]) for item in plans),
            "missing_sources": [item for item in plans if item["status"] == "no_source_found"],
            "database_backup": str(backup) if backup else "",
            "items": plans,
        }
    finally:
        conn.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(os.environ.get("RESEARCH_FELLOW_DATA_DIR", PROJECT_ROOT / "data")))
    parser.add_argument("--server-root", type=Path, default=Path(os.environ["RESEARCH_FELLOW_SERVER_DIR"]).expanduser() if os.environ.get("RESEARCH_FELLOW_SERVER_DIR", "").strip() else None)
    parser.add_argument("--workspace", action="append", help="Workspace key; may be repeated. Defaults to every configured workspace.")
    parser.add_argument("--apply", action="store_true", help="Copy sources, write Topic manifests, and update local DB PDF paths.")
    parser.add_argument("--no-update-db-paths", action="store_true", help="Copy files while keeping existing SQLite PDF paths.")
    args = parser.parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    server_root = args.server_root.expanduser().resolve() if args.server_root else None
    profiles = load_workspace_profiles(data_dir / "workspace_profiles.json", data_dir=data_dir, server_root=server_root)
    keys = args.workspace or sorted(profiles)
    results = []
    for key in keys:
        profile = profiles.get(key)
        if profile is None:
            results.append({"workspace": key, "status": "unknown_workspace"})
            continue
        db_path = data_dir / profile.db_filename
        results.append(migrate_workspace(
            key=key, profile=profile, db_path=db_path, data_dir=data_dir,
            server_root=server_root, apply=args.apply,
            update_db_paths=not args.no_update_db_paths,
        ))
    print(json.dumps({"mode": "apply" if args.apply else "dry_run", "results": results}, ensure_ascii=False, indent=2))
    return 1 if any(result.get("status") == "unknown_workspace" for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
