"""Validated research-state and M2 curation-intent shapes for P4."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


ConfidenceLevel = Literal["low", "medium", "high"]
Priority = Literal["높음", "보통", "낮음"]
ResearchQuestionStatus = Literal["candidate", "interested", "exploring", "hold", "rejected"]
ExecutionMode = Literal["manual", "auto"]
IntentCreator = Literal["m2", "m2_auto"]


class ResearchState(BaseModel):
    """A researcher-owned snapshot interpreted by M2, not hidden agent memory."""

    question: str = Field(min_length=3)
    current_hypothesis: str = "아직 명시되지 않았습니다."
    constraints: list[str] = Field(default_factory=list)
    unresolved_issues: list[str] = Field(default_factory=list)
    recent_evidence_changes: list[str] = Field(default_factory=list)
    researcher_note: str = ""
    confidence: ConfidenceLevel = "medium"

    @field_validator("constraints", "unresolved_issues", "recent_evidence_changes")
    @classmethod
    def compact_text_items(cls, value: list[str]) -> list[str]:
        return [item.strip() for item in value if item.strip()][:8]


class ResearchQuestionCandidate(BaseModel):
    """An evidence-grounded question that can mature in the researcher backlog."""

    rq_id: str = ""
    question: str = Field(min_length=3)
    rationale: str = Field(min_length=3)
    gap_or_tension: str = ""
    research_context: str = ""
    exploration_need: str = ""
    source_card_ids: list[str] = Field(default_factory=list)
    source_update_ids: list[str] = Field(default_factory=list)
    status: ResearchQuestionStatus = "candidate"

    @field_validator("source_card_ids", "source_update_ids")
    @classmethod
    def compact_ids(cls, value: list[str]) -> list[str]:
        return list(dict.fromkeys(item.strip() for item in value if item.strip()))[:12]


class CurationIntent(BaseModel):
    """A proposed M2→M1 request. Researcher approval is still required."""

    intent_id: str = Field(min_length=1)
    title: str = Field(min_length=3)
    purpose: str = Field(min_length=3)
    question: str = Field(min_length=3)
    research_context: str = Field(min_length=3)
    labels: list[str] = Field(default_factory=list)
    priority: Priority = "보통"
    expected_evidence: str = Field(min_length=3)
    completion_condition: str = Field(min_length=3)
    execution_mode: ExecutionMode = "manual"
    created_by: IntentCreator = "m2"

    @field_validator("labels")
    @classmethod
    def compact_labels(cls, value: list[str]) -> list[str]:
        return [item.strip() for item in value if item.strip()][:6]
