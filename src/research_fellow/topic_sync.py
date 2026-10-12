"""Portable synchronization for Topic Directory source assets.

Topic files are synchronized independently from workspace SQLite replicas. The
SQLite ``paper_shelf.pdf_path`` value is refreshed locally from the Topic source
manifest after synchronization and is never used as the portable identifier.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import sqlite3
from typing import Any

import yaml


@dataclass(frozen=True)
class TopicSyncResult:
    uploaded: int = 0
    downloaded: int = 0
    unchanged: int = 0
    conflicts: int = 0


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.sync-tmp")
    temp_path.write_bytes(content)
    temp_path.replace(path)


def _atomic_copy(source: Path, target: Path) -> None:
    if source.resolve() == target.resolve():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target.with_name(f".{target.name}.sync-tmp")
    shutil.copy2(source, temp_path)
    temp_path.replace(target)


def _conflict_path(root: Path, relative: str, side: str, content: bytes) -> Path:
    rel_label = re.sub(r"[^A-Za-z0-9._-]+", "_", relative).strip("._")[:100] or "topic-file"
    return root / "index" / "conflicts" / f"{rel_label}.{_digest(content)[:16]}.{side}.conflict"


def _preserve_conflict(root: Path, relative: str, side: str, content: bytes) -> None:
    path = _conflict_path(root, relative, side, content)
    if not path.exists():
        _atomic_write(path, content)


def _merge_sources(
    local_value: dict[str, Any] | None,
    server_value: dict[str, Any] | None,
    *, direction: str,
    local_root: Path,
    server_root: Path,
    apply: bool,
) -> tuple[dict[str, Any], int]:
    local = local_value or {"sources": {}}
    server = server_value or {"sources": {}}
    local_sources = local.get("sources") if isinstance(local.get("sources"), dict) else {}
    server_sources = server.get("sources") if isinstance(server.get("sources"), dict) else {}
    merged: dict[str, Any] = {}
    conflicts = 0
    for key in sorted(set(local_sources) | set(server_sources)):
        left, right = local_sources.get(key), server_sources.get(key)
        if left is None:
            merged[key] = right
        elif right is None or left == right:
            merged[key] = left
        else:
            left_data, right_data = _json_bytes(left), _json_bytes(right)
            if left_data != right_data:
                conflicts += 1
                if apply:
                    for root, data, side in ((local_root, left_data, "local"), (server_root, right_data, "server")):
                        _preserve_conflict(root, f"sources/{key}.json", side, data)
            if direction == "pull":
                winner = right
            else:
                winner = left
            merged[key] = winner

    # Keep extra top-level metadata while letting the selected side own conflicts.
    primary, secondary = (server, local) if direction == "pull" else (local, server)
    payload = {**secondary, **primary, "sources": merged}
    payload.setdefault("format", "research-fellow-topic-v1")
    return payload, conflicts


def _merge_topic_yaml(
    local_bytes: bytes | None,
    server_bytes: bytes | None,
    *, direction: str,
    local_root: Path,
    server_root: Path,
    apply: bool,
) -> tuple[bytes | None, int]:
    if local_bytes is None:
        return server_bytes, 0
    if server_bytes is None or local_bytes == server_bytes:
        return local_bytes, 0
    try:
        local = yaml.safe_load(local_bytes) or {}
        server = yaml.safe_load(server_bytes) or {}
    except yaml.YAMLError:
        if apply:
            _preserve_conflict(local_root, "topic.yaml", "local", local_bytes)
            _preserve_conflict(server_root, "topic.yaml", "server", server_bytes)
        return (server_bytes if direction == "pull" else local_bytes), 1
    if not isinstance(local, dict):
        if apply:
            _preserve_conflict(local_root, "topic.yaml", "local", local_bytes)
            _preserve_conflict(server_root, "topic.yaml", "server", server_bytes)
        return (server_bytes if direction == "pull" else local_bytes), 1
    if not isinstance(server, dict):
        if apply:
            _preserve_conflict(local_root, "topic.yaml", "local", local_bytes)
            _preserve_conflict(server_root, "topic.yaml", "server", server_bytes)
        return (server_bytes if direction == "pull" else local_bytes), 1
    primary, secondary = (server, local) if direction == "pull" else (local, server)
    conflicts = int(any(
        key != "updated_at"
        and key in primary
        and key in secondary
        and primary[key] != secondary[key]
        for key in set(primary) | set(secondary)
    ))
    if conflicts and apply:
        _preserve_conflict(local_root, "topic.yaml", "local", local_bytes)
        _preserve_conflict(server_root, "topic.yaml", "server", server_bytes)
    merged = {**secondary, **primary}
    return yaml.safe_dump(merged, allow_unicode=True, sort_keys=False).encode("utf-8"), conflicts


def _file_map(root: Path) -> dict[str, Path]:
    if not root.is_dir():
        return {}
    files: dict[str, Path] = {}
    resolved_root = root.resolve()
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        if relative in {"topic.yaml", "sources/sources.json"} or path.is_symlink() or not path.is_file():
            continue
        try:
            path.resolve().relative_to(resolved_root)
        except ValueError:
            continue
        files[relative] = path
    return files


def sync_topic_directory(
    local_topic_dir: str | Path,
    server_topic_dir: str | Path,
    *,
    direction: str = "merge",
    apply: bool = True,
) -> TopicSyncResult:
    """Merge or pull a Topic tree, retaining both versions of conflicting files.

    ``merge`` uploads local-only files and downloads server-only files. For a
    conflicting path, both byte versions are retained in ``index/conflicts``;
    local remains canonical. ``pull`` only downloads server-only content and
    makes the server version canonical on conflict. Preview mode is read-only.
    """
    if direction not in {"merge", "pull"}:
        raise ValueError("Topic sync direction must be 'merge' or 'pull'.")
    local_root, server_root = Path(local_topic_dir), Path(server_topic_dir)
    local_files, server_files = _file_map(local_root), _file_map(server_root)
    uploaded = downloaded = unchanged = conflicts = 0
    for relative in sorted(set(local_files) | set(server_files)):
        local_path, server_path = local_files.get(relative), server_files.get(relative)
        if local_path is None:
            downloaded += 1
            if apply and direction in {"merge", "pull"}:
                _atomic_copy(server_path, local_root / PurePosixPath(relative))
        elif server_path is None:
            if direction == "merge":
                uploaded += 1
                if apply:
                    _atomic_copy(local_path, server_root / PurePosixPath(relative))
        else:
            local_bytes, server_bytes = local_path.read_bytes(), server_path.read_bytes()
            if local_bytes == server_bytes:
                unchanged += 1
            else:
                conflicts += 1
                if apply:
                    _preserve_conflict(local_root, relative, "local", local_bytes)
                    _preserve_conflict(local_root, relative, "server", server_bytes)
                    _preserve_conflict(server_root, relative, "local", local_bytes)
                    _preserve_conflict(server_root, relative, "server", server_bytes)
                    winner = server_bytes if direction == "pull" else local_bytes
                    _atomic_write(local_path, winner)
                    _atomic_write(server_path, winner)

    local_manifest_path = local_root / "sources" / "sources.json"
    server_manifest_path = server_root / "sources" / "sources.json"
    local_manifest = _read_json(local_manifest_path) if local_manifest_path.exists() else None
    server_manifest = _read_json(server_manifest_path) if server_manifest_path.exists() else None
    if local_manifest is not None or server_manifest is not None:
        merged, manifest_conflicts = _merge_sources(
            local_manifest, server_manifest, direction=direction,
            local_root=local_root, server_root=server_root,
            apply=apply,
        )
        conflicts += manifest_conflicts
        merged_bytes = _json_bytes(merged)
        local_differs = local_manifest is None or _json_bytes(local_manifest) != merged_bytes
        server_differs = server_manifest is None or _json_bytes(server_manifest) != merged_bytes
        if local_differs:
            if server_manifest is not None:
                downloaded += 1
            if apply:
                _atomic_write(local_manifest_path, merged_bytes)
        if server_differs and direction == "merge":
            uploaded += 1
            if apply:
                _atomic_write(server_manifest_path, merged_bytes)
        elif server_manifest is None and direction == "pull":
            # Startup pull leaves a local-only Topic as-is on the server.
            pass

    local_yaml_path, server_yaml_path = local_root / "topic.yaml", server_root / "topic.yaml"
    local_yaml = local_yaml_path.read_bytes() if local_yaml_path.is_file() else None
    server_yaml = server_yaml_path.read_bytes() if server_yaml_path.is_file() else None
    if local_yaml is not None or server_yaml is not None:
        merged_yaml, yaml_conflicts = _merge_topic_yaml(
            local_yaml, server_yaml, direction=direction,
            local_root=local_root, server_root=server_root,
            apply=apply,
        )
        conflicts += yaml_conflicts
        if merged_yaml is not None:
            if local_yaml != merged_yaml:
                if server_yaml is not None:
                    downloaded += 1
                if apply:
                    _atomic_write(local_yaml_path, merged_yaml)
            if direction == "merge" and server_yaml != merged_yaml:
                uploaded += 1
                if apply:
                    _atomic_write(server_yaml_path, merged_yaml)

    if apply:
        local_root.mkdir(parents=True, exist_ok=True)
        server_root.mkdir(parents=True, exist_ok=True)
    return TopicSyncResult(uploaded, downloaded, unchanged, conflicts)


def reconcile_paper_paths_from_topic(db_path: str | Path, topic_directory: str | Path) -> int:
    """Refresh machine-local PDF paths from portable Topic manifest references."""
    root = Path(topic_directory).expanduser().resolve()
    manifest = _read_json(root / "sources" / "sources.json")
    if not manifest:
        return 0
    sources = manifest.get("sources")
    if not isinstance(sources, dict):
        return 0
    updates: list[tuple[str, str]] = []
    for paper_id, item in sources.items():
        if not isinstance(item, dict):
            continue
        reference = str(item.get("pdf") or "").strip()
        if not reference or reference.lower().startswith(("http://", "https://")):
            continue
        candidate = (root / PurePosixPath(reference)).resolve()
        if root not in candidate.parents or not candidate.is_file():
            continue
        updates.append((str(candidate), str(paper_id)))
    if not updates:
        return 0
    with sqlite3.connect(db_path) as conn:
        exists = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='paper_shelf'").fetchone()
        if not exists:
            return 0
        known = {str(row[0]) for row in conn.execute("SELECT paper_id FROM paper_shelf")}
        valid = [(path, paper_id) for path, paper_id in updates if paper_id in known]
        conn.executemany("UPDATE paper_shelf SET pdf_path=? WHERE paper_id=?", valid)
        return len(valid)
