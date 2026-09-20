from __future__ import annotations

"""Research workspace profiles for one shared Research Fellow codebase.

A workspace profile separates durable research memory and domain specialization while
reusing exactly the same application, workflows, prompt templates, and UI code.
"""

from dataclasses import dataclass
import json
import os
import re
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


def load_workspace_profiles(config_path: str | Path) -> dict[str, ResearchWorkspaceProfile]:
    """Load built-ins plus user-created profiles; malformed entries are ignored."""
    profiles = dict(WORKSPACE_PROFILES)
    path = Path(config_path)
    if not path.exists():
        return profiles
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return profiles
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
