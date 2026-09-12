from __future__ import annotations

"""Research workspace profiles for one shared Research Fellow codebase.

A workspace profile separates durable research memory and domain specialization while
reusing exactly the same application, workflows, prompt templates, and UI code.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ResearchWorkspaceProfile:
    key: str
    label: str
    short_label: str
    db_filename: str
    cache_name: str
    purpose: str
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
}


def get_workspace_profile(key: str | None) -> ResearchWorkspaceProfile:
    normalized = (key or "general").strip().lower()
    return WORKSPACE_PROFILES.get(normalized, WORKSPACE_PROFILES["general"])
