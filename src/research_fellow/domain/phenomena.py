from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from research_fellow.domain.research import CurationIntent, ResearchState


PhenomenonType = Literal[
    "research_update", "advice_report", "decision_request", "decision",
    "curation_intent", "knowledge_update", "advisory_exchange",
    "activity_summary",
]


class Payload(BaseModel):
    """Payload base. Unknown fields are retained for forward-compatible projections."""

    model_config = ConfigDict(extra="allow")


class DecisionRequestPayload(Payload):
    title: str = Field(min_length=1)
    next_action: str = Field(min_length=1)
    card: dict[str, Any] | None = None
    intent: dict[str, Any] | None = None
    relation: dict[str, Any] | None = None

    @model_validator(mode="after")
    def has_exactly_one_decision_subject(self) -> "DecisionRequestPayload":
        if sum(item is not None for item in (self.card, self.intent, self.relation)) != 1:
            raise ValueError("decision_request에는 card, intent, relation 중 하나가 필요합니다.")
        return self


class DecisionPayload(Payload):
    request_id: str = Field(min_length=1)
    decision: Literal["approved", "deferred", "rejected"]
    note: str = ""


class CurationIntentPayload(Payload):
    title: str = Field(min_length=1)
    question: str = Field(min_length=1)
    research_context: str = "The approved research question and its stated constraints define this search context."
    labels: list[str] = Field(default_factory=list)
    priority: str = Field(min_length=1)
    purpose: str = "후속 탐색의 목적을 연구자가 검토해야 합니다."
    expected_evidence: str = "관련 근거 카드와 출처·조건·한계"
    completion_condition: str = "출처·조건·한계가 연결된 탐색 결과를 보고합니다."


class ResearchUpdatePayload(Payload):
    state: ResearchState


class AdviceReportPayload(Payload):
    title: str = Field(min_length=1)
    report: str = Field(min_length=1)
    state: ResearchState | None = None
    evidence_card_ids: list[str] = Field(default_factory=list)
    knowledge_update_ids: list[str] = Field(default_factory=list)


class KnowledgeUpdatePayload(Payload):
    title: str = Field(min_length=1)


class ActivitySummaryPayload(Payload):
    title: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    baseline: dict[str, list[str]] = Field(default_factory=dict)
    delta: dict[str, list[str]] = Field(default_factory=dict)
    is_initial_baseline: bool = False


class PhenomenonDraft(BaseModel):
    """The validated ledger representation of one observable shared phenomenon."""

    phenomenon_type: PhenomenonType
    producer: str = Field(min_length=1)
    recipients: list[str] = Field(min_length=1)
    subject_type: str = Field(min_length=1)
    subject_id: str | None = None
    payload: Payload | dict[str, Any]
    status: str = Field(min_length=1)

    @model_validator(mode="after")
    def decision_requests_identify_their_subject(self) -> "PhenomenonDraft":
        if self.phenomenon_type == "decision_request" and not self.subject_id:
            raise ValueError("decision_request에는 안정적인 subject_id가 필요합니다.")
        return self


def validate_payload(phenomenon_type: str, payload: dict[str, Any]) -> dict[str, Any]:
    models: dict[str, type[Payload]] = {
        "decision_request": DecisionRequestPayload,
        "decision": DecisionPayload,
        "curation_intent": CurationIntentPayload,
        "research_update": ResearchUpdatePayload,
        "advice_report": AdviceReportPayload,
        "knowledge_update": KnowledgeUpdatePayload,
        "activity_summary": ActivitySummaryPayload,
    }
    model = models.get(phenomenon_type, Payload)
    return model.model_validate(payload).model_dump(mode="json")
