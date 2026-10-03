from __future__ import annotations

"""Research workspace profiles for one shared Research Fellow codebase.

A workspace profile separates durable research memory and domain specialization while
reusing exactly the same application, workflows, prompt templates, and UI code.
"""

from dataclasses import dataclass
import json
import os
import re
import sqlite3
import shutil
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ResearchWorkspaceProfile:
    key: str
    label: str
    short_label: str
    db_filename: str
    cache_name: str
    purpose: str
    topic_ko: str
    topic_en: str
    browser_title: str
    expertise_instruction: str
    server_db_filename: str


WORKSPACE_PROFILES: dict[str, ResearchWorkspaceProfile] = {
    "general": ResearchWorkspaceProfile(
        key="general",
        label="General Research Fellow",
        short_label="전체 관심사",
        db_filename="research_fellow.db",
        cache_name="research-fellow",
        purpose="개발·연구 전반에서 축적하는 전체 관심사와 장기 연구 메모리",
        topic_ko="전체 연구 관심사 · 교차 도메인 연구와 장기 지식 축적",
        topic_en="General Research · Cross-domain inquiry and long-term research memory",
        browser_title="General Research Fellow",
        expertise_instruction=(
            "Maintain a broad, cross-domain research perspective. Connect the current question to the researcher's "
            "accumulated knowledge without forcing it into a single specialty. Preserve useful cross-domain analogies "
            "and explicitly mark when a claim falls outside well-supported accumulated expertise."
        ),
        server_db_filename="research_fellow.db",
    ),
    "agent_development": ResearchWorkspaceProfile(
        key="agent_development",
        label="Agent Development Research Fellow",
        short_label="에이전트 개발 전문",
        db_filename="research_fellow_agent_development.db",
        cache_name="research-fellow-agent-development",
        purpose="현재 논문과 연결된 AI 에이전트 개발·명세·워크플로우·메모리·평가 전문 연구공간",
        topic_ko="AI 에이전트 개발 · 명세 · 워크플로우 · 메모리 · 평가",
        topic_en="AI Agent Development · Specification · Workflows · Memory · Evaluation",
        browser_title="Agent Development Research Fellow",
        expertise_instruction=(
            "Specialize in AI agent engineering and research, especially specification-based agent development, "
            "requirements and world-machine boundaries, APF/AJD-style modeling, agent workflows, memory and knowledge "
            "representation, harness/loop engineering, evaluation, long-running agent behavior, and the translation of "
            "implementation lessons into publishable research claims. Keep the scope disciplined: connect evidence to "
            "agent-development research questions and clearly separate software implementation observations from "
            "generalizable research findings."
        ),
        server_db_filename="research_fellow_agent_development.db",
    ),
    "vision_ai": ResearchWorkspaceProfile(
        key="vision_ai",
        label="Vision AI Research Fellow",
        short_label="Vision AI 전문",
        db_filename="research_fellow_vision_ai.db",
        cache_name="research-fellow-vision-ai",
        purpose="Computer Vision·멀티모달 AI·산업 비전 검사·표현학습을 중심으로 축적하는 전문 연구공간",
        topic_ko="Vision AI · Computer Vision · Multimodal AI · 산업 비전 검사",
        topic_en="Vision AI · Computer Vision · Multimodal AI · Industrial Vision",
        browser_title="Vision AI Research Fellow",
        expertise_instruction=(
            "Specialize in computer vision and multimodal AI research, including representation learning, vision "
            "foundation models, image/video understanding, industrial visual inspection, defect detection, segmentation, "
            "classification, anomaly detection, vision-language models, multimodal learning, synthetic data, simulation, "
            "domain adaptation, transfer learning, explainability, uncertainty, and physical-world deployment. When "
            "interpreting evidence, explicitly distinguish image-level, object-level, pixel-level, lot-level, and "
            "process-level modeling; separate representation learning from downstream task design; distinguish feature-based, "
            "embedding-based, and end-to-end multimodal approaches; and consider data granularity, label quality, domain shift, "
            "class imbalance, deployment constraints, and evaluation validity. Connect findings to reusable research questions "
            "and clearly separate benchmark performance from evidence of industrial usefulness or generalization."
        ),
        server_db_filename="research_fellow_vision_ai.db",
    ),
}

BUILTIN_WORKSPACE_KEYS = frozenset(WORKSPACE_PROFILES)


WORKSPACE_PROFILE_META_KEY = "workspace_profile_json"


def _profile_payload(profile: ResearchWorkspaceProfile) -> dict[str, str]:
    return {
        "key": profile.key, "label": profile.label, "short_label": profile.short_label,
        "purpose": profile.purpose, "topic_ko": profile.topic_ko, "topic_en": profile.topic_en,
        "browser_title": profile.browser_title, "expertise_instruction": profile.expertise_instruction,
    }


def persist_workspace_profile_metadata(db_path: str | Path, profile: ResearchWorkspaceProfile) -> None:
    """Persist minimal workspace identity with the durable DB for version-safe recovery."""
    path = Path(db_path)
    if not path.exists():
        return
    try:
        with sqlite3.connect(path) as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS schema_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            conn.execute(
                "INSERT OR REPLACE INTO schema_meta(key,value) VALUES (?,?)",
                (WORKSPACE_PROFILE_META_KEY, json.dumps(_profile_payload(profile), ensure_ascii=False)),
            )
    except sqlite3.DatabaseError:
        return


def _discovered_profile(db_path: Path, *, key_hint: str = "") -> ResearchWorkspaceProfile | None:
    name = db_path.name
    if name == "research_fellow.db":
        return None
    match = re.fullmatch(r"research_fellow_(.+)\.db", name)
    if not match:
        return None
    fallback_key = _workspace_key(key_hint or match.group(1))
    payload: dict[str, Any] = {}
    try:
        with sqlite3.connect(db_path) as conn:
            row = conn.execute("SELECT value FROM schema_meta WHERE key=?", (WORKSPACE_PROFILE_META_KEY,)).fetchone()
        if row:
            parsed = json.loads(str(row[0] or "{}"))
            if isinstance(parsed, dict):
                payload = parsed
    except (sqlite3.DatabaseError, json.JSONDecodeError, OSError):
        payload = {}
    payload = {**payload, "key": str(payload.get("key") or fallback_key)}
    if not str(payload.get("label") or "").strip():
        payload["label"] = fallback_key.replace("_", " ").title()
    try:
        profile = _custom_profile(payload)
    except ValueError:
        return None
    if profile.db_filename != db_path.name:
        profile = ResearchWorkspaceProfile(
            key=profile.key, label=profile.label, short_label=profile.short_label,
            db_filename=db_path.name, cache_name=profile.cache_name, purpose=profile.purpose,
            topic_ko=profile.topic_ko, topic_en=profile.topic_en, browser_title=profile.browser_title,
            expertise_instruction=profile.expertise_instruction, server_db_filename=db_path.name,
        )
    return profile



def bootstrap_local_workspace_from_server(
    profile: ResearchWorkspaceProfile, *, data_dir: str | Path, server_root: str | Path | None
) -> Path | None:
    """Seed a missing local workspace DB from the configured server workspace.

    This is a one-time bootstrap only. Existing local databases are never overwritten;
    subsequent reconciliation remains the responsibility of Workspace Sync.
    """
    if not server_root:
        return None
    local_db = Path(data_dir).expanduser() / profile.db_filename
    if local_db.exists():
        return local_db
    server_db = (
        Path(server_root).expanduser() / "workspaces" / profile.key / profile.server_db_filename
    )
    if not server_db.exists():
        return None
    local_db.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(server_db, local_db)
    return local_db

def _workspace_key(value: str) -> str:
    key = re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")
    if not key:
        raise ValueError("워크스페이스 영문 키를 입력하세요.")
    return key[:48]


def _custom_profile(payload: dict[str, Any]) -> ResearchWorkspaceProfile:
    key = _workspace_key(str(payload.get("key") or payload.get("label") or ""))
    label = str(payload.get("label") or key.replace("_", " ").title()).strip()
    purpose = str(payload.get("purpose") or "별도로 축적·관리하는 연구 작업공간").strip()
    expertise = str(payload.get("expertise_instruction") or f"Maintain research knowledge and evidence for {label}.").strip()
    return ResearchWorkspaceProfile(
        key=key,
        label=label,
        short_label=str(payload.get("short_label") or label).strip(),
        db_filename=f"research_fellow_{key}.db",
        cache_name=f"research-fellow-{key.replace('_', '-')}",
        purpose=purpose,
        topic_ko=str(payload.get("topic_ko") or label).strip(),
        topic_en=str(payload.get("topic_en") or label).strip(),
        browser_title=str(payload.get("browser_title") or f"{label} · Research Fellow").strip(),
        expertise_instruction=expertise,
        server_db_filename=f"research_fellow_{key}.db",
    )



def _discovered_portable_profile(manifest_path: Path) -> ResearchWorkspaceProfile | None:
    """Recover a workspace profile from workspaces/<key>/workspace.json."""
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or payload.get("format") != "research-fellow-portable-v1":
        return None
    workspace = payload.get("workspace") if isinstance(payload.get("workspace"), dict) else {}
    fallback_key = manifest_path.parent.name
    workspace = {**workspace, "key": str(workspace.get("key") or fallback_key)}
    try:
        return _custom_profile(workspace)
    except ValueError:
        return None

def load_workspace_profiles(
    config_path: str | Path, *, data_dir: str | Path | None = None, server_root: str | Path | None = None
) -> dict[str, ResearchWorkspaceProfile]:
    """Load built-ins/configured profiles and recover orphan workspace DBs when possible."""
    profiles = dict(WORKSPACE_PROFILES)
    path = Path(config_path)
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = {}
        entries = payload.get("workspaces", []) if isinstance(payload, dict) else []
        for item in entries:
            if not isinstance(item, dict):
                continue
            try:
                profile = _custom_profile(item)
            except ValueError:
                continue
            if profile.key not in BUILTIN_WORKSPACE_KEYS:
                profiles[profile.key] = profile
    known_files = {profile.db_filename for profile in profiles.values()}
    if data_dir is not None:
        root = Path(data_dir)
        if root.exists():
            # Portable workspace folders are sufficient for discovery even when the
            # working SQLite DB has intentionally been omitted.
            portable_root = root / "workspaces"
            if portable_root.exists():
                for manifest_path in sorted(portable_root.glob("*/workspace.json")):
                    profile = _discovered_portable_profile(manifest_path)
                    if profile and profile.key not in profiles:
                        profiles[profile.key] = profile
                        known_files.add(profile.db_filename)
            for db_path in sorted(root.glob("research_fellow_*.db")):
                if db_path.name in known_files:
                    continue
                profile = _discovered_profile(db_path)
                if profile and profile.key not in profiles:
                    profiles[profile.key] = profile
                    known_files.add(profile.db_filename)
    if server_root:
        workspaces_root = Path(server_root).expanduser() / "workspaces"
        if workspaces_root.exists():
            for workspace_dir in sorted(path for path in workspaces_root.iterdir() if path.is_dir()):
                candidates = sorted(workspace_dir.glob("research_fellow*.db"))
                for db_path in candidates:
                    profile = _discovered_profile(db_path, key_hint=workspace_dir.name)
                    if profile and profile.key not in profiles:
                        profiles[profile.key] = profile
                        known_files.add(profile.db_filename)
                        break
    return profiles


def save_custom_workspace(
    config_path: str | Path, *, key: str, label: str, purpose: str = "", expertise_instruction: str = ""
) -> ResearchWorkspaceProfile:
    profile = _custom_profile({
        "key": key, "label": label, "purpose": purpose,
        "expertise_instruction": expertise_instruction,
    })
    if profile.key in BUILTIN_WORKSPACE_KEYS:
        raise ValueError("기본 워크스페이스 키는 사용할 수 없습니다.")
    path = Path(config_path)
    profiles = load_workspace_profiles(path)
    if profile.key in profiles:
        raise ValueError("이미 존재하는 워크스페이스 키입니다.")
    custom = [item for item in profiles.values() if item.key not in BUILTIN_WORKSPACE_KEYS]
    custom.append(profile)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"workspaces": [
        {
            "key": item.key, "label": item.label, "short_label": item.short_label,
            "purpose": item.purpose, "topic_ko": item.topic_ko, "topic_en": item.topic_en,
            "browser_title": item.browser_title, "expertise_instruction": item.expertise_instruction,
        } for item in custom
    ]}, ensure_ascii=False, indent=2), encoding="utf-8")
    return profile


def delete_custom_workspace(config_path: str | Path, key: str) -> bool:
    normalized = _workspace_key(key)
    if normalized in BUILTIN_WORKSPACE_KEYS:
        raise ValueError("기본 워크스페이스는 삭제할 수 없습니다.")
    path = Path(config_path)
    profiles = load_workspace_profiles(path)
    if normalized not in profiles:
        return False
    remaining = [item for item in profiles.values() if item.key not in BUILTIN_WORKSPACE_KEYS and item.key != normalized]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"workspaces": [
        {
            "key": item.key, "label": item.label, "short_label": item.short_label,
            "purpose": item.purpose, "topic_ko": item.topic_ko, "topic_en": item.topic_en,
            "browser_title": item.browser_title, "expertise_instruction": item.expertise_instruction,
        } for item in remaining
    ]}, ensure_ascii=False, indent=2), encoding="utf-8")
    return True


def get_workspace_profile(key: str | None, profiles: dict[str, ResearchWorkspaceProfile] | None = None) -> ResearchWorkspaceProfile:
    normalized = (key or "general").strip().lower()
    if profiles is None:
        config_path = os.environ.get("RESEARCH_FELLOW_WORKSPACE_CONFIG", "").strip()
        profiles = load_workspace_profiles(config_path) if config_path else WORKSPACE_PROFILES
    available = profiles
    return available.get(normalized, available["general"])
