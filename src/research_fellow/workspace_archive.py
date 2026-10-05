from __future__ import annotations

import io
import json
import re
import shutil
import tempfile
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .workspace_portable import (
    PORTABLE_JSON,
    export_workspace_snapshot,
    restore_workspace_db_from_snapshot,
)
from .workspace_profiles import (
    BUILTIN_WORKSPACE_KEYS,
    ResearchWorkspaceProfile,
    load_workspace_profiles,
    persist_workspace_profile_metadata,
    save_custom_workspace,
)


ARCHIVE_FORMAT = "research-fellow-workspace-v2"
LEGACY_ARCHIVE_FORMAT = "research-fellow-workspace-v1"
TRANSFER_MANIFEST = "transfer_manifest.json"
PORTABLE_PREFIX = "portable"


def _normalized_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")


def _unique_import_key(base_key: str, existing: set[str]) -> str:
    base = _normalized_key(base_key) or "workspace"
    if base not in existing and base not in BUILTIN_WORKSPACE_KEYS:
        return base
    candidate = f"{base}_imported"
    if candidate not in existing and candidate not in BUILTIN_WORKSPACE_KEYS:
        return candidate
    index = 2
    while True:
        candidate = f"{base}_imported_{index}"
        if candidate not in existing and candidate not in BUILTIN_WORKSPACE_KEYS:
            return candidate
        index += 1


def build_workspace_archive(profile: ResearchWorkspaceProfile, data_dir: str | Path) -> bytes:
    """Create a portable workspace-transfer ZIP.

    The archive contains a schema-tolerant portable JSON snapshot plus referenced
    PDF assets. Extracted full text and all other durable workspace records are
    already part of the portable JSON because they live in SQLite tables.
    """
    root = Path(data_dir)
    database = root / profile.db_filename
    if not database.is_file():
        raise ValueError(f"워크스페이스 DB를 찾을 수 없습니다: {database}")

    with tempfile.TemporaryDirectory(prefix="research-fellow-transfer-") as tmp:
        temp_root = Path(tmp)
        result = export_workspace_snapshot(
            database,
            data_dir=temp_root,
            workspace_key=profile.key,
        )
        bundle = Path(result.bundle_dir)
        manifest = {
            "format": ARCHIVE_FORMAT,
            "format_version": 2,
            "exported_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "workspace": {
                "key": profile.key,
                "label": profile.label,
                "purpose": profile.purpose,
                "expertise_instruction": profile.expertise_instruction,
            },
            "contents": {
                "tables": result.table_count,
                "rows": result.row_count,
                "pdf_files": result.pdf_count,
                "pdf_bytes": result.pdf_bytes,
                "missing_pdf_sources": result.missing_pdf_sources,
            },
        }
        output = io.BytesIO()
        with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(TRANSFER_MANIFEST, json.dumps(manifest, ensure_ascii=False, indent=2))
            for path in sorted(bundle.rglob("*")):
                if path.is_file():
                    relative = path.relative_to(bundle).as_posix()
                    archive.write(path, arcname=f"{PORTABLE_PREFIX}/{relative}")
        return output.getvalue()


def _restore_legacy_archive(
    archive: zipfile.ZipFile,
    manifest: dict[str, Any],
    *,
    config_path: str | Path,
    data_dir: str | Path,
) -> tuple[ResearchWorkspaceProfile, bool]:
    payload: dict[str, Any] = manifest["workspace"]
    source_key = _normalized_key(str(payload.get("key", "")))
    if not source_key:
        raise ValueError("아카이브에 유효한 워크스페이스 키가 없습니다.")
    existing = load_workspace_profiles(config_path, data_dir=data_dir)
    key = _unique_import_key(source_key, set(existing))
    label = str(payload.get("label") or source_key)
    if key != source_key:
        label = f"{label} (Imported)"
    profile = save_custom_workspace(
        config_path,
        key=key,
        label=label,
        purpose=str(payload.get("purpose") or ""),
        expertise_instruction=str(payload.get("expertise_instruction") or ""),
    )
    target = Path(data_dir) / profile.db_filename
    restored = False
    if "data/workspace.db" in set(archive.namelist()):
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(archive.read("data/workspace.db"))
        persist_workspace_profile_metadata(target, profile)
        restored = True
    return profile, restored


def restore_workspace_archive(
    archive_bytes: bytes, *, config_path: str | Path, data_dir: str | Path
) -> tuple[ResearchWorkspaceProfile, bool]:
    """Import a transfer ZIP as a new local workspace.

    Import never overwrites an existing workspace. If the original key already
    exists (including built-ins), an ``_imported`` suffix is assigned. The
    portable bundle is kept under ``data/workspaces/<new-key>`` so restored PDF
    paths remain valid after import.
    """
    try:
        archive = zipfile.ZipFile(io.BytesIO(archive_bytes), "r")
    except zipfile.BadZipFile as error:
        raise ValueError("유효한 워크스페이스 ZIP 파일이 아닙니다.") from error

    with archive:
        names = set(archive.namelist())
        # Backward compatibility with the former DB-only archive.
        if "workspace.json" in names and TRANSFER_MANIFEST not in names:
            try:
                legacy_manifest = json.loads(archive.read("workspace.json").decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError("워크스페이스 manifest를 읽을 수 없습니다.") from error
            if legacy_manifest.get("format") != LEGACY_ARCHIVE_FORMAT or not isinstance(legacy_manifest.get("workspace"), dict):
                raise ValueError("지원하지 않는 워크스페이스 아카이브 형식입니다.")
            return _restore_legacy_archive(
                archive,
                legacy_manifest,
                config_path=config_path,
                data_dir=data_dir,
            )

        if TRANSFER_MANIFEST not in names:
            raise ValueError("transfer_manifest.json이 없는 Workspace Transfer ZIP입니다.")
        try:
            manifest = json.loads(archive.read(TRANSFER_MANIFEST).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("워크스페이스 transfer manifest를 읽을 수 없습니다.") from error
        if manifest.get("format") != ARCHIVE_FORMAT or not isinstance(manifest.get("workspace"), dict):
            raise ValueError("지원하지 않는 Workspace Transfer ZIP 형식입니다.")
        portable_json_name = f"{PORTABLE_PREFIX}/{PORTABLE_JSON}"
        if portable_json_name not in names:
            raise ValueError("portable/workspace.json이 없는 Workspace Transfer ZIP입니다.")

        workspace = manifest["workspace"]
        source_key = _normalized_key(str(workspace.get("key") or ""))
        if not source_key:
            raise ValueError("아카이브에 유효한 워크스페이스 키가 없습니다.")
        existing = load_workspace_profiles(config_path, data_dir=data_dir)
        target_key = _unique_import_key(source_key, set(existing))
        label = str(workspace.get("label") or source_key)
        if target_key != source_key:
            label = f"{label} (Imported)"
        profile = save_custom_workspace(
            config_path,
            key=target_key,
            label=label,
            purpose=str(workspace.get("purpose") or ""),
            expertise_instruction=str(workspace.get("expertise_instruction") or ""),
        )

        bundle = Path(data_dir) / "workspaces" / profile.key
        if bundle.exists():
            shutil.rmtree(bundle)
        bundle.mkdir(parents=True, exist_ok=True)
        prefix = f"{PORTABLE_PREFIX}/"
        for name in sorted(names):
            if not name.startswith(prefix) or name.endswith("/"):
                continue
            relative = name[len(prefix):]
            destination = (bundle / relative).resolve()
            if bundle.resolve() not in destination.parents and destination != bundle.resolve():
                raise ValueError("ZIP 내부에 허용되지 않는 경로가 있습니다.")
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(archive.read(name))

        snapshot_json = bundle / PORTABLE_JSON
        try:
            snapshot_payload = json.loads(snapshot_json.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("portable/workspace.json을 읽을 수 없습니다.") from error
        snapshot_payload["workspace"] = {
            "key": profile.key,
            "label": profile.label,
            "short_label": profile.short_label,
            "purpose": profile.purpose,
            "topic_ko": profile.topic_ko,
            "topic_en": profile.topic_en,
            "browser_title": profile.browser_title,
            "expertise_instruction": profile.expertise_instruction,
        }
        snapshot_payload["source"] = {"db_filename": profile.db_filename}
        snapshot_json.write_text(
            json.dumps(snapshot_payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        target_db = Path(data_dir) / profile.db_filename
        restore_result = restore_workspace_db_from_snapshot(snapshot_json, target_db, overwrite=True)
        persist_workspace_profile_metadata(target_db, profile)
        return profile, bool(restore_result.restored)
