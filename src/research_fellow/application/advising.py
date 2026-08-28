"""P4 M2 research-state interpretation and researcher-approved curation intents."""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from typing import Any

from research_fellow.domain.research import CurationIntent, ResearchState
from research_fellow.infrastructure.retrieval import RetrievalResult
from research_fellow.storage import Ledger


@dataclass(frozen=True)
class M2DirectionDraft:
    report: str
    intents: list[CurationIntent]


def latest_research_state(ledger: Ledger) -> ResearchState | None:
    """Read the latest researcher-visible state from the shared-phenomena ledger."""
    for item in ledger.phenomena(type_="research_update"):
        state = item["payload"].get("state")
        if state:
            try:
                return ResearchState.model_validate(state)
            except ValueError:
                continue
    return None


def recent_research_questions(ledger: Ledger, limit: int = 8) -> list[str]:
    """Researcher-owned question history, newest first and without duplicates."""
    questions: list[str] = []
    for item in ledger.phenomena(type_="research_update"):
        state = item["payload"].get("state", {})
        question = state.get("question") if isinstance(state, dict) else None
        if isinstance(question, str) and question.strip() and question not in questions:
            questions.append(question)
        if len(questions) >= limit:
            break
    return questions


def recent_knowledge_updates(ledger: Ledger, limit: int = 8) -> list[dict[str, Any]]:
    """M2's observable input inbox; updates themselves remain owned by M1."""
    return ledger.phenomena(recipient="m2", type_="knowledge_update")[:limit]


def direction_prompt(state: ResearchState, evidence: list[RetrievalResult], updates: list[dict[str, Any]]) -> str:
    from research_fellow.infrastructure.prompt_renderer import render_prompt

    return render_prompt("m2_research_direction.j2", state=state, evidence=evidence, recent_updates=updates, max_intents=3)


def research_context_mapping_prompt(question: str, note: str) -> str:
    from research_fellow.infrastructure.prompt_renderer import render_prompt

    return render_prompt("m2_research_context_mapping.j2", question=question, researcher_note=note)


def parse_research_context_mapping(text: str) -> dict[str, object]:
    """Accept a small readable draft; researcher confirmation remains the state gate."""
    fields: dict[str, object] = {"hypothesis": "", "constraints": [], "unresolved": [], "changes": []}
    current: str | None = None
    aliases = {
        "current hypothesis": "hypothesis", "현재 가설": "hypothesis",
        "constraints": "constraints", "제약": "constraints",
        "unresolved issues": "unresolved", "미결 사항": "unresolved",
        "recent evidence changes": "changes", "최근 근거 변화": "changes",
    }
    for line in text.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            current = aliases.get(key.strip().lower())
            if current == "hypothesis":
                fields[current] = value.strip()
            elif current and value.strip():
                fields[current] = [value.strip()]
        elif current in {"constraints", "unresolved", "changes"} and line.strip().lstrip("-• "):
            fields[current] = [*fields[current], line.strip().lstrip("-• ")]
    return fields


def draft_research_direction(
    state: ResearchState, evidence: list[RetrievalResult], updates: list[dict[str, Any]], llm_text: str | None,
) -> M2DirectionDraft:
    report, raw_intents = _split_direction_draft(llm_text or "")
    intents = _normalize_intents(raw_intents, state)
    if not report:
        report = _deterministic_report(state, evidence, updates)
    if not intents:
        intents = _fallback_intents(state, updates)
    return M2DirectionDraft(report=report, intents=intents[:3])


def record_research_direction(
    ledger: Ledger, state: ResearchState, evidence: list[RetrievalResult], updates: list[dict[str, Any]], draft: M2DirectionDraft,
) -> tuple[str, list[str]]:
    """Persist a P4 vertical slice without allowing LLM text to control transitions."""
    case_id = ledger.create_case("research", state.question[:80])
    ledger.record(
        case_id, "research_update", "researcher", ["m2"], "research_state",
        {"state": state.model_dump(mode="json")}, status="completed",
    )
    ledger.record(
        case_id, "advice_report", "m2", ["researcher"], "research_review",
        {
            "title": "M2 연구 상태·방향 검토", "report": draft.report,
            "state": state.model_dump(mode="json"),
            "evidence_card_ids": [item.card["card_id"] for item in evidence],
            "knowledge_update_ids": [item["phenomenon_id"] for item in updates],
        },
        status="completed",
    )
    request_ids = []
    for intent in draft.intents[:3]:
        request_ids.append(ledger.record(
            case_id, "decision_request", "m2", ["researcher"], "curation_intent",
            {
                "title": f"M1 탐색 Intent 승인: {intent.title}", "intent": intent.model_dump(mode="json"),
                "next_action": "승인 시 M1 실행함으로 전달",
            },
            subject_id=intent.intent_id,
        ))
    return case_id, request_ids


def record_update_report(ledger: Ledger, state: ResearchState | None, update_report: str, updates: list[dict[str, Any]]) -> str:
    """M2's read-only synthesis of M1 updates, visible to the researcher."""
    case_id = ledger.create_case("research", "M1 지식 업데이트 검토")
    return ledger.record(
        case_id, "advice_report", "m2", ["researcher"], "knowledge_update_review",
        {
            "title": "M2 · M1 새 정보 요약", "report": update_report,
            "state": state.model_dump(mode="json") if state else None,
            "knowledge_update_ids": [item["phenomenon_id"] for item in updates],
        },
        status="completed",
    )


def _split_direction_draft(text: str) -> tuple[str, list[str]]:
    if not text.strip():
        return "", []
    marker = re.search(r"(?im)^##?\s*(?:탐색\s*Intent\s*후보|curation\s*intent(?:s)?(?:\s+candidates)?)\s*$", text)
    if not marker:
        return text.strip(), []
    report = text[:marker.start()].strip()
    blocks = [part.strip() for part in re.split(r"(?m)^---+\s*$", text[marker.end():]) if part.strip()]
    return report, blocks


def _fields(block: str) -> dict[str, str]:
    aliases = {
        "목적": "purpose", "purpose": "purpose", "제목": "title", "title": "title",
        "질문": "question", "question": "question", "레이블": "labels", "labels": "labels",
        "연구 맥락": "research_context", "research context": "research_context",
        "우선순위": "priority", "priority": "priority", "기대 근거": "expected_evidence",
        "expected evidence": "expected_evidence", "완료 조건": "completion_condition", "completion condition": "completion_condition",
    }
    result: dict[str, str] = {}
    for line in block.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        normalized = aliases.get(key.strip().lower())
        if normalized and value.strip():
            result[normalized] = value.strip()
    return result


def _normalize_intents(blocks: list[str], state: ResearchState) -> list[CurationIntent]:
    intents = []
    for block in blocks[:3]:
        fields = _fields(block)
        if not all(fields.get(field) for field in ("purpose", "title", "question", "expected_evidence", "completion_condition")):
            continue
        priority = fields.get("priority", "보통")
        priority = {"high": "높음", "medium": "보통", "low": "낮음"}.get(priority.lower(), priority)
        try:
            intents.append(CurationIntent(
                intent_id=f"intent-{uuid.uuid4().hex[:12]}", title=fields["title"], purpose=fields["purpose"],
                question=fields["question"], research_context=fields.get("research_context") or _intent_context_from_state(state), labels=[item.strip() for item in fields.get("labels", "").split(",")],
                priority=priority, expected_evidence=fields["expected_evidence"], completion_condition=fields["completion_condition"],
            ))
        except ValueError:
            continue
    return intents


def _fallback_intents(state: ResearchState, updates: list[dict[str, Any]]) -> list[CurationIntent]:
    topics = state.unresolved_issues[:3] or [state.question]
    intents = []
    for index, topic in enumerate(topics, start=1):
        intent = CurationIntent(
            intent_id=f"intent-{uuid.uuid4().hex[:12]}", title=f"Evidence search: {topic[:32]}",
            purpose="Narrow an unresolved research issue with source-grounded evidence in the stated research domain.", question=topic,
            research_context=_intent_context_from_state(state),
            labels=[], priority="높음" if index == 1 else "보통",
            expected_evidence="Contrary evidence, conditions of application, and source-grounded knowledge cards.",
            completion_condition="Report at least one source-grounded candidate card or an explicit knowledge gap to M2 and the researcher.",
        )
        intents.append(intent)
    return intents


def _intent_context_from_state(state: ResearchState) -> str:
    """Carry the whole research situation into every search, not only its topic label."""
    parts = [f"Research question: {state.question}", f"Working hypothesis: {state.current_hypothesis}"]
    if state.constraints:
        parts.append("Constraints: " + "; ".join(state.constraints))
    if state.unresolved_issues:
        parts.append("Unresolved issues: " + "; ".join(state.unresolved_issues))
    if state.researcher_note:
        parts.append("Researcher note: " + state.researcher_note)
    return "\n".join(parts)


def _deterministic_report(state: ResearchState, evidence: list[RetrievalResult], updates: list[dict[str, Any]]) -> str:
    evidence_lines = "\n".join(f"- [{item.card['card_id']}] {item.card['claim']}" for item in evidence) or "- 관련 승인 지식카드가 아직 없습니다."
    update_lines = "\n".join(f"- {item['payload'].get('title', 'M1 새 정보')}" for item in updates) or "- 최근 M1 새 정보가 없습니다."
    issues = "\n".join(f"- {item}" for item in state.unresolved_issues) or "- 연구자가 명시한 미결 사항이 없습니다."
    return (
        f"## 현재 해석\n{state.question}\n\n## 지지 근거\n{evidence_lines}\n\n"
        f"## 반대 근거 또는 숨은 가정\n- 현재 가설과 근거의 적용 조건을 분리해 비교할 필요가 있습니다.\n\n"
        f"## 조건과 한계\n- 연구 제약: {', '.join(state.constraints) or '명시되지 않았습니다.'}\n"
        f"- 현재 확신 수준: {state.confidence}\n\n## 최근 근거 변화\n{update_lines}\n\n"
        f"## 공백과 다음 판단\n{issues}"
    )
