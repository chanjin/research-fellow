from __future__ import annotations

import io
import json
import re
import zipfile
from pathlib import Path
from typing import Any

from .workspace_profiles import ResearchWorkspaceProfile, save_custom_workspace


ARCHIVE_FORMAT = "research-fellow-workspace-v1"


def build_workspace_archive(profile: ResearchWorkspaceProfile, data_dir: str | Path) -> bytes:
    """Create a portable archive. The local database is never deleted here."""
    root = Path(data_dir)
    database = root / profile.db_filename
    manifest = {
        "format": ARCHIVE_FORMAT,
        "workspace": {
            "key": profile.key,
            "label": profile.label,
            "purpose": profile.purpose,
            "expertise_instruction": profile.expertise_instruction,
            "db_filename": profile.db_filename,
        },
    }
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("workspace.json", json.dumps(manifest, ensure_ascii=False, indent=2))
        if database.exists():
            archive.write(database, arcname="data/workspace.db")
    return output.getvalue()


def restore_workspace_archive(
    archive_bytes: bytes, *, config_path: str | Path, data_dir: str | Path
) -> tuple[ResearchWorkspaceProfile, bool]:
    """Register an archived workspace and restore its DB when no local DB exists.

    Returns ``(profile, database_restored)``. Existing local databases are kept
    intact so reloading a locally archived workspace cannot overwrite newer data.
    """
    try:
        archive = zipfile.ZipFile(io.BytesIO(archive_bytes), "r")
    except zipfile.BadZipFile as error:
        raise ValueError("유효한 워크스페이스 ZIP 파일이 아닙니다.") from error
    with archive:
        names = set(archive.namelist())
        if "workspace.json" not in names:
            raise ValueError("workspace.json manifest가 없는 ZIP입니다.")
        try:
            manifest = json.loads(archive.read("workspace.json").decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("워크스페이스 manifest를 읽을 수 없습니다.") from error
        if manifest.get("format") != ARCHIVE_FORMAT or not isinstance(manifest.get("workspace"), dict):
            raise ValueError("지원하지 않는 워크스페이스 아카이브 형식입니다.")
        payload: dict[str, Any] = manifest["workspace"]
        key = re.sub(r"[^a-z0-9]+", "_", str(payload.get("key", "")).lower()).strip("_")
        if not key:
            raise ValueError("아카이브에 유효한 워크스페이스 키가 없습니다.")
        profile = save_custom_workspace(
            config_path,
            key=key,
            label=str(payload.get("label") or key),
            purpose=str(payload.get("purpose") or ""),
            expertise_instruction=str(payload.get("expertise_instruction") or ""),
        )
        target = Path(data_dir) / profile.db_filename
        database_restored = False
        if "data/workspace.db" in names and not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read("data/workspace.db"))
            database_restored = True
    return profile, database_restored
