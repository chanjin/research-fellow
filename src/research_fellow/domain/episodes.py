"""Searchable projections of completed research and advisory episodes."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


EpisodeType = Literal[
    "research_question", "external_advisory", "research_review",
    "paper_card_registration", "ontology_curation",
]
EpisodeStatus = Literal["provisional", "confirmed", "superseded"]


class EpisodicMemory(BaseModel):
    """A compact recall record; the ledger remains the event-level authority."""

    episode_id: str = Field(min_length=1)
    case_id: str = Field(min_length=1)
    episode_type: EpisodeType
    lifecycle_status: EpisodeStatus = "provisional"
    situation_summary: str = Field(min_length=3)
    decision_question: str = Field(min_length=3)
    advisory_plan: list[str] = Field(default_factory=list)
    answer_summary: str = ""
    conditions_and_limits: str = ""
    unresolved_items: list[str] = Field(default_factory=list)
    follow_up_intent_ids: list[str] = Field(default_factory=list)
    evidence_card_ids: list[str] = Field(default_factory=list)
    evidence_relation_ids: list[str] = Field(default_factory=list)
    outcome: str = ""

    def retrieval_text(self) -> str:
        """Situation dominates retrieval; answer text is secondary case context."""
        return "\n".join(filter(None, [
            f"Episode type: {self.episode_type}",
            f"Situation: {self.situation_summary}",
            f"Decision question: {self.decision_question}",
            "Plan: " + "; ".join(self.advisory_plan),
            f"Answer summary: {self.answer_summary}",
            f"Conditions and limits: {self.conditions_and_limits}",
            "Unresolved: " + "; ".join(self.unresolved_items),
            f"Outcome: {self.outcome}",
        ]))
