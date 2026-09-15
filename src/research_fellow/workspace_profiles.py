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


def get_workspace_profile(key: str | None) -> ResearchWorkspaceProfile:
    normalized = (key or "general").strip().lower()
    return WORKSPACE_PROFILES.get(normalized, WORKSPACE_PROFILES["general"])
