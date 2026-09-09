"""P4 M2 research-state interpretation and researcher-approved curation intents."""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from typing import Any

from research_fellow.domain.research import CurationIntent, ResearchQuestionCandidate, ResearchState
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


def recent_knowledge_updates(ledger: Ledger, limit: int = 100) -> list[dict[str, Any]]:
    """Unreviewed M1 knowledge-card updates that have not yet been consumed by an M2 state review.

    Multiple pending events for the same card are collapsed into one inbox item while
    retaining every pending update id so a completed review can consume them together.
    """
    reviewed = ledger.reviewed_knowledge_update_ids()
    pending_by_card: dict[str, dict[str, Any]] = {}
    pending_ids_by_card: dict[str, list[str]] = {}
    for item in ledger.phenomena(recipient="m2", type_="knowledge_update"):
        if item.get("phenomenon_id") in reviewed:
            continue
        payload = item.get("payload") or {}
        card_id = str(payload.get("card_id", "")).strip()
        if not card_id or item.get("subject_type") != "knowledge_card" or payload.get("operation") == "deleted":
            continue
        pending_ids_by_card.setdefault(card_id, []).append(str(item.get("phenomenon_id", "")))
        if card_id not in pending_by_card:
            pending_by_card[card_id] = dict(item)
    result: list[dict[str, Any]] = []
    for card_id, item in pending_by_card.items():
        item["pending_update_ids"] = [value for value in pending_ids_by_card.get(card_id, []) if value]
        result.append(item)
    return result[: max(1, min(int(limit), 500))]


def parse_research_question_suggestions(
    text: str, *, valid_card_ids: set[str] | None = None, limit: int = 10,
) -> list[ResearchQuestionCandidate]:
    """Parse evidence-grounded RQ blocks while tolerating the legacy numbered-list format."""
    valid_card_ids = valid_card_ids or set()
    candidates: list[ResearchQuestionCandidate] = []
    blocks = [part.strip() for part in re.split(r"(?im)^##\s*RQ\s*\d+\s*$", text) if part.strip()]
    for block in blocks:
        fields: dict[str, str] = {}
        aliases = {
            "question": "question", "질문": "question",
            "why now": "rationale", "왜 지금": "rationale", "도출 이유": "rationale",
            "gap/tension": "gap_or_tension", "gap or tension": "gap_or_tension", "공백/긴장": "gap_or_tension",
            "research context": "research_context", "연구 맥락": "research_context",
            "source card ids": "source_card_ids", "근거 카드 ids": "source_card_ids", "근거 카드": "source_card_ids",
            "exploration need": "exploration_need", "추가 탐색 필요": "exploration_need",
        }
        current: str | None = None
        for line in block.splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                current = aliases.get(key.strip().lower())
                if current:
                    fields[current] = value.strip()
            elif current and line.strip():
                fields[current] = (fields.get(current, "") + " " + line.strip()).strip()
        question = fields.get("question", "").strip()
        rationale = fields.get("rationale", "").strip()
        if not question or not rationale:
            continue
        raw_ids = [item.strip() for item in re.split(r"[,;\s]+", fields.get("source_card_ids", "")) if item.strip()]
        source_ids = [item for item in raw_ids if not valid_card_ids or item in valid_card_ids]
        try:
            candidates.append(ResearchQuestionCandidate(
                question=question, rationale=rationale, gap_or_tension=fields.get("gap_or_tension", ""),
                research_context=fields.get("research_context", ""), exploration_need=fields.get("exploration_need", ""),
                source_card_ids=source_ids,
            ))
        except ValueError:
            continue
        if len(candidates) >= limit:
            return candidates

    # Backward compatibility for pasted legacy output: keep it as a candidate,
    # but make the missing rationale explicit rather than silently inventing one.
    if not candidates:
        for line in text.splitlines():
            matched = re.match(r"^\s*\d{1,2}[.)]\s+(.+?)\s*$", line)
            if not matched:
                continue
            question = matched.group(1).strip()
            if question:
                candidates.append(ResearchQuestionCandidate(
                    question=question,
                    rationale="외부 응답이 질문만 제공하여 도출 이유가 기록되지 않았습니다. 저장 후 연구자가 맥락을 보완해야 합니다.",
                ))
            if len(candidates) >= limit:
                break
    return candidates


def store_research_question_candidates(
    ledger: Ledger, candidates: list[ResearchQuestionCandidate], updates: list[dict[str, Any]], *, review_id: str | None = None,
) -> list[dict[str, Any]]:
    """Persist candidate questions with cumulative M1 provenance and optional review-batch links."""
    update_by_card: dict[str, str] = {}
    for update in updates:
        payload = update.get("payload", {}) if isinstance(update.get("payload"), dict) else {}
        card_id = str(payload.get("card_id", ""))
        if card_id:
            update_by_card[card_id] = str(update.get("phenomenon_id", ""))
    saved: list[dict[str, Any]] = []
    for candidate in candidates:
        existing = ledger.research_question_by_text(candidate.question)
        payload = candidate.model_dump(mode="json")
        payload["source_update_ids"] = [update_by_card[card_id] for card_id in candidate.source_card_ids if card_id in update_by_card]
        stored = ledger.upsert_research_question(payload)
        stored["_change_kind"] = "strengthened" if existing else "new"
        if review_id:
            ledger.link_research_question_review(review_id, str(stored["rq_id"]), stored["_change_kind"])
            for card_id in candidate.source_card_ids:
                ledger.add_research_question_source(
                    str(stored["rq_id"]), review_id, card_id, update_by_card.get(card_id, ""), candidate.rationale,
                )
            if stored["_change_kind"] == "new":
                summary = f"새 지식카드 {len(candidate.source_card_ids)}건을 근거로 새 연구질문이 생성되었습니다."
                change_type = "created"
            else:
                summary = f"새 지식카드 {len(candidate.source_card_ids)}건이 추가 근거로 연결되어 연구질문이 보강되었습니다."
                change_type = "evidence_added"
            ledger.add_research_question_change(str(stored["rq_id"]), change_type, summary, review_id=review_id)
        saved.append(stored)
    return saved


def create_exploration_intent_for_rq(ledger: Ledger, rq: dict[str, Any], priority: str = "보통") -> tuple[str, str]:
    """Turn one backlog RQ into a concrete M1 search contract awaiting researcher approval."""
    intent_id = f"intent-{uuid.uuid4().hex[:12]}"
    question = str(rq.get("question", "")).strip()
    rationale = str(rq.get("rationale", "")).strip()
    gap = str(rq.get("gap_or_tension", "")).strip()
    context = str(rq.get("research_context", "")).strip()
    need = str(rq.get("exploration_need", "")).strip()
    intent = CurationIntent(
        intent_id=intent_id,
        title=f"RQ 탐색 · {question[:45]}",
        purpose=need or f"이 연구질문이 제기된 근거와 지식 공백을 선행연구에서 확인한다: {rationale}",
        question=question,
        research_context="\n".join(item for item in [
            f"Research question: {question}",
            f"Why this question emerged: {rationale}",
            f"Gap or tension: {gap}" if gap else "",
            f"Research context: {context}" if context else "",
        ] if item),
        labels=[], priority=priority if priority in {"높음", "보통", "낮음"} else "보통",
        expected_evidence=need or "질문의 전제, 반대 근거, 적용 조건, 관련 방법 및 사례를 확인할 수 있는 출처 기반 근거",
        completion_condition="질문의 핵심 공백에 대해 출처가 확인된 지식카드 또는 명시적인 미해결 지식 공백을 M2에 보고한다.",
        execution_mode="manual", created_by="m2",
    )
    case_id = ledger.create_case("research", question[:80])
    request_id = ledger.record(
        case_id, "decision_request", "m2", ["researcher"], "curation_intent",
        {"title": f"M1 탐색 Intent 승인: {intent.title}", "intent": intent.model_dump(mode="json"),
         "next_action": "승인 시 M1 실행함으로 전달", "research_question_id": rq.get("rq_id", "")},
        subject_id=intent_id,
    )
    ledger.link_research_question_intent(str(rq.get("rq_id", "")), intent_id, request_id)
    return intent_id, request_id



def auto_rq_priority_prompt(backlog: list[dict[str, Any]], max_select: int = 3) -> str:
    """Ask M2 to rank only currently actionable backlog questions for autonomous exploration."""
    from research_fellow.infrastructure.prompt_renderer import render_prompt

    return render_prompt("m2_auto_rq_priority.j2", backlog=backlog, max_select=max_select)


def parse_rq_priority_assessment(
    text: str, backlog: list[dict[str, Any]], *, limit: int = 3,
) -> list[dict[str, Any]]:
    """Parse RQ_ID/SCORE/REASON blocks and fall back deterministically when needed."""
    by_id = {str(item.get("rq_id", "")): item for item in backlog}
    ranked: list[dict[str, Any]] = []
    seen: set[str] = set()
    for block in re.split(r"(?im)^##\s*(?:Priority|RQ)\s*\d+\s*$", text or ""):
        fields: dict[str, str] = {}
        for line in block.splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip().lower().replace(" ", "_")
            if key in {"rq_id", "score", "reason"}:
                fields[key] = value.strip()
        rq_id = fields.get("rq_id", "")
        if rq_id not in by_id or rq_id in seen:
            continue
        try:
            score = max(1, min(5, int(float(fields.get("score", "0")))))
        except ValueError:
            score = 0
        if score <= 0:
            continue
        ranked.append({"rq": by_id[rq_id], "score": score, "reason": fields.get("reason", "")})
        seen.add(rq_id)

    if ranked:
        ranked.sort(key=lambda item: item["score"], reverse=True)
        return ranked[:limit]

    # Deterministic fallback: researcher interest first, then stronger provenance/context.
    def fallback_key(item: dict[str, Any]) -> tuple[int, int, int, str]:
        status_weight = 2 if item.get("status") == "interested" else 1
        evidence = len(item.get("source_card_ids", [])) + len(item.get("source_update_ids", []))
        context = sum(bool(str(item.get(key, "")).strip()) for key in ("rationale", "gap_or_tension", "research_context", "exploration_need"))
        return status_weight, evidence, context, str(item.get("updated_at", ""))

    result = []
    for rq in sorted(backlog, key=fallback_key, reverse=True)[:limit]:
        score = 5 if rq.get("status") == "interested" else 4
        result.append({
            "rq": rq, "score": score,
            "reason": "LLM 우선순위 평가를 읽지 못해 관심 상태, 근거 수, 연구 맥락 충실도를 기준으로 선택했습니다.",
        })
    return result


def auto_exploration_candidates(ledger: Ledger, limit: int = 100) -> list[dict[str, Any]]:
    """Questions eligible for auto mode: candidate/interested and not already handed to M1."""
    backlog = ledger.research_question_backlog(statuses=["candidate", "interested"], limit=limit)
    return [rq for rq in backlog if not ledger.research_question_intents(str(rq.get("rq_id", "")))]


def _intent_terms(value: str) -> set[str]:
    stop = {"the", "and", "for", "with", "from", "this", "that", "research", "study", "방법", "연구", "탐색", "관련"}
    return {token.lower() for token in re.findall(r"[A-Za-z0-9가-힣]{2,}", value) if token.lower() not in stop}


def find_duplicate_curation_intent(ledger: Ledger, question: str, purpose: str, threshold: float = 0.72) -> dict[str, Any] | None:
    """Find a materially duplicate M1 task already recorded in the ledger."""
    target = _intent_terms(f"{question} {purpose}")
    if not target:
        return None
    normalized_question = " ".join(re.findall(r"[A-Za-z0-9가-힣]+", question.lower()))
    for event in ledger.phenomena(type_="curation_intent"):
        payload = event.get("payload") or {}
        other_question = str(payload.get("question", ""))
        other_purpose = str(payload.get("purpose", ""))
        if not other_question:
            continue
        if normalized_question == " ".join(re.findall(r"[A-Za-z0-9가-힣]+", other_question.lower())):
            return event
        other = _intent_terms(f"{other_question} {other_purpose}")
        union = target | other
        similarity = len(target & other) / len(union) if union else 0.0
        if similarity >= threshold:
            return event
    return None


def create_auto_exploration_intent_for_rq(
    ledger: Ledger, rq: dict[str, Any], *, score: int, selection_reason: str,
) -> dict[str, Any]:
    """Create a ready M2→M1 Intent directly, bypassing researcher approval in explicit auto mode."""
    intent_id = f"intent-{uuid.uuid4().hex[:12]}"
    question = str(rq.get("question", "")).strip()
    rationale = str(rq.get("rationale", "")).strip()
    gap = str(rq.get("gap_or_tension", "")).strip()
    context = str(rq.get("research_context", "")).strip()
    need = str(rq.get("exploration_need", "")).strip()
    priority = "높음" if score >= 4 else "보통"
    duplicate = find_duplicate_curation_intent(ledger, question, need or rationale)
    if duplicate:
        existing_payload = duplicate.get("payload") or {}
        existing_intent_id = str(existing_payload.get("intent_id") or duplicate.get("subject_id") or "")
        if existing_intent_id:
            ledger.link_research_question_intent(str(rq.get("rq_id", "")), existing_intent_id, "")
        return {
            "rq_id": str(rq.get("rq_id", "")), "question": question, "score": score,
            "selection_reason": selection_reason, "intent": CurationIntent.model_validate(existing_payload),
            "phenomenon_id": duplicate["phenomenon_id"], "case_id": duplicate["case_id"], "reused": True,
        }

    intent = CurationIntent(
        intent_id=intent_id, title=f"자동 RQ 탐색 · {question[:45]}",
        purpose=need or f"이 연구질문의 핵심 지식 공백을 선행연구에서 확인한다: {rationale}",
        question=question,
        research_context="\n".join(item for item in [
            f"Research question: {question}",
            f"Why this question emerged: {rationale}",
            f"Gap or tension: {gap}" if gap else "",
            f"Research context: {context}" if context else "",
            f"Auto selection reason: {selection_reason}" if selection_reason else "",
        ] if item),
        labels=[], priority=priority,
        expected_evidence=need or "질문의 전제, 반대 근거, 적용 조건, 관련 방법 및 사례를 확인할 수 있는 출처 기반 근거",
        completion_condition="질문의 핵심 공백에 대해 출처가 확인된 지식카드 또는 명시적인 미해결 지식 공백을 M2와 연구자에게 보고한다.",
        execution_mode="auto", created_by="m2_auto",
    )
    case_id = ledger.create_case("research", f"AUTO · {question[:72]}")
    phenomenon_id = ledger.record(
        case_id, "curation_intent", "m2", ["m1"], "curation_intent",
        intent.model_dump(mode="json"), subject_id=intent_id, status="ready",
    )
    ledger.create_search_profile(intent.model_dump(mode="json"))
    ledger.link_research_question_intent(str(rq.get("rq_id", "")), intent_id, "")
    return {
        "rq_id": str(rq.get("rq_id", "")), "question": question, "score": score,
        "selection_reason": selection_reason, "intent": intent, "phenomenon_id": phenomenon_id, "case_id": case_id, "reused": False,
    }


def dispatch_top_research_questions(
    ledger: Ledger, ranked: list[dict[str, Any]], *, limit: int = 3, review_id: str | None = None,
) -> list[dict[str, Any]]:
    """Dispatch selected RQs to M1 and leave one researcher-visible audit report."""
    selected = ranked[:limit]
    if review_id:
        for item in ranked:
            rq_id = str(item["rq"].get("rq_id", ""))
            is_selected = item in selected
            ledger.update_review_question_selection(
                review_id, rq_id, score=int(item["score"]),
                reason=str(item.get("reason", "")), selected=is_selected,
            )
            if is_selected:
                reason = str(item.get("reason", "")).strip()
                suffix = f" 선정 이유: {reason}" if reason else ""
                ledger.add_research_question_change(
                    rq_id, "priority_changed", f"이번 연구상태 검토에서 중요도 {int(item['score'])}/5로 평가되어 자동 후속 탐색 대상으로 선정되었습니다.{suffix}", review_id=review_id,
                )
    dispatched = [
        create_auto_exploration_intent_for_rq(
            ledger, item["rq"], score=int(item["score"]), selection_reason=str(item.get("reason", "")),
        )
        for item in selected
    ]
    if dispatched:
        case_id = ledger.create_case("research", "M2 자동 연구질문 탐색")
        ledger.record(
            case_id, "advice_report", "m2", ["researcher"], "auto_exploration_dispatch",
            {
                "title": "M2 · 중요 연구질문 자동 탐색 보고",
                "report": "\n".join(
                    f"{idx}. [{item['score']}/5] {item['question']} → {item['intent'].title}"
                    for idx, item in enumerate(dispatched, 1)
                ),
                "selected": [
                    {
                        "rq_id": item["rq_id"], "question": item["question"], "score": item["score"],
                        "selection_reason": item["selection_reason"], "intent_id": item["intent"].intent_id,
                    }
                    for item in dispatched
                ],
            },
            status="completed",
        )
    return dispatched

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
    *, create_intent_requests: bool = True,
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
    for intent in (draft.intents[:3] if create_intent_requests else []):
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
