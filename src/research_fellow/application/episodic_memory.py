"""Create, recall, and safely apply episodic precedent to an AdvisoryActSpec."""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from research_fellow.domain.episodes import EpisodicMemory
from research_fellow.infrastructure.episodic_retrieval import EpisodeRecall, EpisodicRetriever
from research_fellow.storage import Ledger


@dataclass(frozen=True)
class AdvisoryActSpec:
    current_situation: str
    recalled_episodes: tuple[EpisodeRecall, ...]
    response_strategy: str
    revalidation_card_ids: tuple[str, ...]


def store_advisory_episode(
    ledger: Ledger, *, case_id: str, episode_type: str, situation_summary: str,
    decision_question: str, advisory_plan: list[str], answer: str,
    evidence_card_ids: list[str], evidence_relation_ids: list[str], unresolved_items: list[str] | None = None,
    outcome: str = "", lifecycle_status: str = "provisional",
) -> EpisodicMemory:
    episode = EpisodicMemory(
        episode_id=f"ep-{uuid.uuid4().hex[:12]}", case_id=case_id, episode_type=episode_type,
        lifecycle_status=lifecycle_status, situation_summary=situation_summary, decision_question=decision_question,
        advisory_plan=advisory_plan, answer_summary=answer[:2400],
        unresolved_items=unresolved_items or [], evidence_card_ids=evidence_card_ids,
        evidence_relation_ids=evidence_relation_ids, outcome=outcome,
    )
    ledger.upsert_episode_memory(episode.model_dump(mode="json"))
    return episode


def store_researcher_curation_episode(
    ledger: Ledger, *, case_id: str, episode_type: str, paper_title: str,
    situation: str, decision: str, action_summary: str, action_steps: list[str],
    evidence_card_ids: list[str], unresolved_items: list[str] | None = None,
) -> EpisodicMemory:
    """Store a researcher action precedent, distinct from an advisory answer."""
    episode = EpisodicMemory(
        episode_id=f"ep-{uuid.uuid4().hex[:12]}", case_id=case_id, episode_type=episode_type,
        lifecycle_status="confirmed",
        situation_summary=f"논문: {paper_title}\n작업 상황: {situation}",
        decision_question=decision, advisory_plan=action_steps,
        answer_summary=action_summary[:2400], unresolved_items=unresolved_items or [],
        evidence_card_ids=evidence_card_ids, evidence_relation_ids=[],
        outcome="연구자가 수행·확정한 지식화 작업 선례",
    )
    ledger.upsert_episode_memory(episode.model_dump(mode="json"))
    return episode


def recall_act_spec(
    ledger: Ledger, retriever: EpisodicRetriever, *, situation: str,
    active_card_ids: set[str], semantic: bool = True, embedding_model: str = "nomic-embed-text",
) -> AdvisoryActSpec:
    recalls = retriever.recall([EpisodicMemory.model_validate(item) for item in ledger.episode_memories()], situation, semantic=semantic, embedding_model=embedding_model)
    top = recalls[0] if recalls else None
    valid_cards = tuple(card_id for card_id in (top.episode.evidence_card_ids if top else []) if card_id in active_card_ids)
    # A confirmed, very similar precedent can accelerate planning, but it never
    # bypasses revalidation of the cards from which its answer was built.
    fast = bool(top and top.episode.lifecycle_status == "confirmed" and top.score >= 0.82 and valid_cards)
    return AdvisoryActSpec(situation, tuple(recalls), "precedent_adaptation" if fast else "planned_investigation", valid_cards)


def recall_context(spec: AdvisoryActSpec) -> str:
    if not spec.recalled_episodes:
        return "관련 과거 사례가 리콜되지 않았습니다."
    lines = [f"응답 전략: {spec.response_strategy}"]
    for recall in spec.recalled_episodes:
        episode = recall.episode
        lines.extend([
            f"[과거 사례 {episode.episode_id}] {episode.situation_summary}",
            f"과거 판단: {episode.decision_question}",
            f"과거 답변 요약(선례이며 현재 근거가 아님): {episode.answer_summary[:700]}",
            f"미결 사항: {'; '.join(episode.unresolved_items) or '없음'}",
        ])
    if spec.revalidation_card_ids:
        lines.append("재검증할 활성 근거 카드: " + ", ".join(spec.revalidation_card_ids))
    return "\n".join(lines)
