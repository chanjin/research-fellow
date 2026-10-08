"""Job-centred research workspace read model.

This module intentionally owns no research state. It projects existing durable
research questions, literature-discovery runs, paper-shelf state, and shared
phenomena into a single view of the Research Fellow's current job.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from research_fellow.storage import Ledger
from research_fellow.application.research_evidence import rq_evidence_bundle
from research_fellow.application.research_answer import evidence_fingerprint, latest_research_answer


@dataclass(frozen=True)
class ResearchQuestionView:
    rq_id: str
    question: str
    status: str
    rationale: str
    research_context: str
    exploration_need: str
    updated_at: str
    source_type: str = ""
    intent_count: int = 0
    evidence_source_count: int = 0
    current_stage: str = ""
    next_agent_action: str = ""
    human_attention_required: bool = False
    progress_detail: str = ""
    literature_round_completed: bool = False
    pending_knowledge_reviews: int = 0
    approved_knowledge_count: int = 0
    answer_confirmation_pending: bool = False
    answer_draft: str = ""
    answer_first_document: str = ""
    answer_version: int = 0
    answer_update_available: bool = False
    answer_work_in_progress: bool = False
    answer_work_stage: str = ""
    answer_work_status: str = ""
    direct_evidence_cards: int = 0
    direct_evidence_papers: int = 0
    supporting_knowledge_cards: int = 0
    supporting_papers: int = 0
    direct_evidence_titles: tuple[str, ...] = ()
    supporting_evidence_titles: tuple[str, ...] = ()
    attention_category: str = ""
    attention_phase: str = ""
    attention_round_label: str = ""
    researcher_comment: str = ""
    attention_item_count: int = 0
    attention_inputs: int = 0
    attention_reviews: int = 0
    attention_decisions: int = 0
    attention_exceptions: int = 0
    execution_status: str = ""
    execution_detail: str = ""
    paper_review_pending: int = 0
    paper_review_completed: int = 0
    lifecycle_phase: str = ""
    additional_literature_ready: bool = False

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass(frozen=True)
class EvidenceWorkView:
    item_id: str
    kind: str
    title: str
    status: str
    updated_at: str
    detail: str = ""
    count: int = 0

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass(frozen=True)
class KnowledgeProgressView:
    update_id: str
    title: str
    created_at: str
    status: str
    detail: str = ""

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass(frozen=True)
class AdvisoryView:
    rq_id: str
    question: str
    source_type: str
    status: str
    updated_at: str
    requester: str = ""

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()




@dataclass(frozen=True)
class PreviousResearchWorkView:
    work_id: str
    title: str
    status: str
    updated_at: str
    research_context: str = ""
    result_count: int = 0
    source_kind: str = "legacy_literature_discovery"

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass(frozen=True)
class NextActionView:
    action_id: str
    priority: str
    title: str
    reason: str
    target_type: str
    target_id: str

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()



def _rq_origin_match(payload: dict[str, Any], rq_id: str) -> bool:
    return any(str(link.get("origin_id") or "") == rq_id for link in (payload.get("origin_links") or []))


def _rq_research_round_state(ledger: Ledger, rq_id: str, intents: list[dict[str, Any]]) -> dict[str, Any]:
    intent_ids = {str(item.get("intent_id") or "") for item in intents if str(item.get("intent_id") or "")}
    literature_done = False
    for report in ledger.phenomena(type_="advice_report"):
        payload = dict(report.get("payload") or {})
        if report.get("subject_type") == "auto_literature_report" and report.get("status") == "completed" and str(payload.get("intent_id") or "") in intent_ids:
            literature_done = True
            break
    pending_knowledge = 0
    for req in ledger.phenomena(type_="decision_request", status="proposed"):
        if req.get("subject_type") not in {"knowledge_card", "knowledge_evidence"}:
            continue
        payload = dict(req.get("payload") or {})
        card = dict(payload.get("card") or {})
        if str(payload.get("intent_id") or "") in intent_ids or _rq_origin_match(card, rq_id):
            pending_knowledge += 1
    approved = 0
    for update in ledger.phenomena(type_="knowledge_update"):
        payload = dict(update.get("payload") or {})
        if _rq_origin_match(payload, rq_id):
            approved += 1
    answer_pending = any(
        req.get("subject_type") == "research_answer_draft" and str(req.get("subject_id") or "") == rq_id and req.get("status") == "proposed"
        for req in ledger.phenomena(type_="decision_request")
    )
    latest_answer = latest_research_answer(ledger, rq_id)
    answer_payload = dict((latest_answer or {}).get("payload") or {})
    answer_draft = str(answer_payload.get("report") or "")
    return {
        "literature_done": literature_done,
        "pending_knowledge": pending_knowledge,
        "approved": approved,
        "answer_pending": answer_pending,
        "answer_draft": answer_draft,
        "answer_first_document": str(answer_payload.get("first_document") or ""),
        "answer_version": int(answer_payload.get("answer_version") or (1 if latest_answer else 0)),
        "answer_evidence_fingerprint": str(answer_payload.get("evidence_fingerprint") or ""),
    }


def _active_questions(
    ledger: Ledger, memory_cards: list[dict[str, Any]], *, limit: int,
    attention_items: list[dict[str, Any]] | None = None,
) -> list[ResearchQuestionView]:
    rows = ledger.research_question_backlog(
        statuses=["exploring", "interested", "candidate", "hold", "resolved", "rejected"], limit=limit
    )
    result: list[ResearchQuestionView] = []
    for row in rows:
        rq_id = str(row.get("rq_id") or "")
        thread = ledger.research_question_thread(rq_id) or {}
        all_intents = ledger.research_question_intents(rq_id)
        # Historical local-PDF intake created synthetic intent-local-* links.
        # They are review context, not literature rounds, so exclude them from
        # the Research round projection and round counts.
        intents = [x for x in all_intents if not str(x.get("intent_id") or "").startswith("intent-local-")]
        intent_ids = {str(x.get("intent_id") or "") for x in intents if str(x.get("intent_id") or "")}
        rq_attention = []
        for attention in (attention_items or []):
            attention_row = dict(attention)
            attention_payload = dict(attention_row.get("payload") or {})
            attention_round_id = str(attention_row.get("round_id") or "")
            attention_subject_id = str(attention_row.get("subject_id") or "")
            attention_intent_id = str(attention_payload.get("intent_id") or "")
            if (
                attention_round_id in intent_ids
                or attention_round_id == rq_id
                or attention_subject_id == rq_id
                or attention_intent_id == rq_id
            ):
                rq_attention.append(attention_row)
        paper_tasks = [
            x for x in ledger.phenomena(type_="research_task")
            if str(x.get("subject_type") or "") == "paper_first_review"
            and str((x.get("payload") or {}).get("intent_id") or "") in intent_ids
        ]
        paper_review_pending = sum(1 for x in paper_tasks if x.get("status") == "ready")
        paper_review_completed = sum(1 for x in paper_tasks if x.get("status") == "completed")
        round_state = _rq_research_round_state(ledger, rq_id, intents)
        all_runs = ledger.auto_research_runs(statuses=("running", "needs_attention", "completed", "failed"), limit=200)
        runs = [x for x in all_runs if str(x.get("intent_id") or "") in intent_ids]
        answer_runs = [
            x for x in all_runs
            if str(x.get("intent_id") or "") == rq_id
            and str(x.get("current_stage") or "") in {"research_answer_draft", "research_answer_first_draft", "research_answer_enrichment"}
        ]
        latest_run = runs[0] if runs else None
        latest_answer_run = answer_runs[0] if answer_runs else None
        latest_intent_id = str((intents[0] if intents else {}).get("intent_id") or "")
        latest_intent_status = ""
        if latest_intent_id:
            for phenomenon in ledger.phenomena(type_="curation_intent"):
                payload = dict(phenomenon.get("payload") or {})
                intent_payload = dict(payload.get("intent") or {})
                candidate_id = str(intent_payload.get("intent_id") or phenomenon.get("subject_id") or "")
                if candidate_id == latest_intent_id:
                    latest_intent_status = str(phenomenon.get("status") or "")
                    break
        answer_run_status = str((latest_answer_run or {}).get("status") or "")
        answer_run_stage = str((latest_answer_run or {}).get("current_stage") or "")
        answer_work_in_progress = bool(
            not round_state.get("answer_draft")
            and (
                round_state.get("answer_pending")
                or answer_run_status in {"running", "needs_attention"}
            )
        )
        answer_work_stage = ""
        answer_work_status = ""
        if round_state.get("answer_draft"):
            answer_work_stage = "completed"
            answer_work_status = f"답변 v{int(round_state.get('answer_version') or 1)} 준비됨"
        elif answer_work_in_progress:
            if answer_run_stage == "research_answer_first_draft":
                answer_work_stage = "1/2"
                answer_work_status = "답변 작성 중 · 1/2 직접 근거 초안 입력 대기"
            elif answer_run_stage == "research_answer_enrichment":
                answer_work_stage = "2/2"
                answer_work_status = "답변 작성 중 · 1/2 완료 · 2/2 최종 답변 보강 입력 대기"
            elif round_state.get("answer_pending"):
                answer_work_stage = "queued"
                answer_work_status = "답변 작성 요청이 등록되었습니다 · Attention 작업 생성/확인 대기"
            else:
                answer_work_stage = "running"
                answer_work_status = "답변 작성 작업이 진행 중입니다"
        sources = ledger.research_question_sources(rq_id)
        status = str(row.get("status") or "candidate")
        evidence_bundle = rq_evidence_bundle(ledger, memory_cards, rq_id)
        evidence_counts = dict(evidence_bundle.get("counts") or {})
        current_evidence_fingerprint = evidence_fingerprint(evidence_bundle)
        answer_update_ready = bool(
            round_state["answer_draft"]
            and (not round_state.get("answer_evidence_fingerprint") or round_state.get("answer_evidence_fingerprint") != current_evidence_fingerprint)
        )
        direct_titles = tuple(
            [str(x.get("title") or "") for x in evidence_bundle.get("direct", {}).get("papers", [])[:4]]
            + [str(x.get("title") or "") for x in evidence_bundle.get("direct", {}).get("cards", [])[:4]]
        )
        supporting_titles = tuple(
            [str(x.get("title") or "") for x in evidence_bundle.get("supporting", {}).get("cards", [])[:4]]
            + [str(x.get("title") or "") for x in evidence_bundle.get("supporting", {}).get("papers", [])[:4]]
        )
        if round_state["answer_draft"]:
            current_stage = "answer_update_available" if answer_update_ready else "answer_ready"
            next_agent_action = (
                "새 근거가 반영되었으므로 필요하면 답변 업데이트를 실행합니다."
                if answer_update_ready else
                "현재 답변을 검토하고 필요하면 추가 문헌 조사를 요청합니다."
            )
            progress_detail = (
                f"답변 v{round_state['answer_version']} 이후 직접/보강 근거가 변경되어 업데이트할 수 있습니다."
                if answer_update_ready else
                f"직접 근거 기반 1차 문서와 기존 연구자산으로 보강한 답변 v{round_state['answer_version']}이 준비되었습니다."
            )
        elif round_state["answer_pending"]:
            current_stage = "answer_confirmation"
            next_agent_action = "연구자가 Attention에서 답변 초안 작성 여부를 확인합니다."
            progress_detail = f"승인 지식 {round_state['approved']}건이 연결되어 초안 작성 전 확인을 기다립니다."
        elif round_state["pending_knowledge"]:
            current_stage = "knowledge_review"
            next_agent_action = "연구자가 Attention에서 논문 기반 지식카드 후보를 검토합니다."
            progress_detail = f"1차 문헌조사 후 지식카드 후보 {round_state['pending_knowledge']}건이 승인 대기 중입니다."
        elif round_state["literature_done"] and round_state["approved"]:
            current_stage = "knowledge_updated"
            next_agent_action = "Research에서 누적 지식을 바탕으로 답변 초안을 작성하거나 추가 문헌 조사를 승인합니다."
            progress_detail = f"1차 문헌조사가 완료되고 승인 지식 {round_state['approved']}건이 M2 연구맥락에 축적되었습니다. 아직 새 Attention은 만들지 않습니다."
        elif round_state["literature_done"]:
            current_stage = "literature_review_complete"
            next_agent_action = "문헌 요약과 지식카드 후보를 검토해 지식자산 반영 여부를 결정합니다."
            progress_detail = "1차 문헌조사 보고가 완료되었습니다. 아직 승인 지식은 없습니다."
        elif status in {"interested", "candidate"} and not intents:
            current_stage = "research_question_review"
            next_agent_action = "M2가 기존 승인 지식과 현재 근거 공백을 확인합니다."
            progress_detail = "질문 접수 후 초기 연구 검토가 아직 연결되지 않았습니다."
        elif status == "exploring" and not sources:
            current_stage = "evidence_acquisition"
            next_agent_action = "M1이 연결된 탐색 Intent를 바탕으로 문헌 근거를 확보합니다."
            progress_detail = "후속 탐색 Intent가 연결되어 근거 확보 단계로 진행 중입니다."
        elif status == "exploring":
            current_stage = "evidence_review"
            next_agent_action = "M2가 확보된 근거를 연구질문과 비교해 다음 연구상태를 판단합니다."
            progress_detail = f"연결된 근거 {len(sources)}건을 검토하는 단계입니다."
        elif status == "hold":
            current_stage = "paused"
            next_agent_action = "보류 사유가 해소되거나 새 근거가 들어오면 다시 평가합니다."
            progress_detail = "현재 연구질문은 보류 상태입니다."
        else:
            current_stage = status
            next_agent_action = "현재 연구상태를 유지합니다."
            progress_detail = ""
        attention_category = ""
        attention_phase = ""
        attention_round_label = ""
        if rq_attention:
            current_attention = rq_attention[0]
            attention_category = str(current_attention.get("category") or "")
            attention_phase = str(current_attention.get("phase_label") or current_attention.get("title") or "").strip()
            attention_round_label = str(current_attention.get("round_label") or "").strip()
            category_label = {
                "inputs": "Inputs", "reviews": "Reviews", "decisions": "Decisions", "exceptions": "Exceptions",
            }.get(attention_category, "Attention")
            current_stage = "attention_required"
            next_agent_action = f"Attention > {category_label}에서 {attention_phase or '현재 사람 작업'}을 처리합니다."
            progress_detail = " · ".join(x for x in [attention_round_label, attention_phase] if x) or "현재 연구자 입력을 기다리고 있습니다."

        execution_status = ""
        execution_detail = ""
        if rq_attention:
            execution_status = "waiting_for_researcher"
            execution_detail = f"실제 Attention {len(rq_attention)}건이 생성되어 연구자 작업을 기다립니다."
        elif latest_answer_run is not None:
            run_status = str(latest_answer_run.get("status") or "")
            run_stage = str(latest_answer_run.get("current_stage") or "")
            if run_status == "needs_attention":
                execution_status = "attention_projection_missing"
                execution_detail = f"RQ 답변 run은 사람 입력 대기 상태(stage={run_stage or 'unknown'})이지만 연결된 Attention item이 없습니다."
            elif run_status == "running":
                execution_status = "agent_running"
                execution_detail = f"RQ 답변 작성이 진행 중입니다(stage={run_stage or 'unknown'})."
            elif run_status == "completed":
                execution_status = "answer_completed"
                execution_detail = "최근 RQ 답변 작성 run이 완료되었습니다."
            else:
                execution_status = "execution_error"
                execution_detail = f"RQ 답변 run 상태가 {run_status or 'unknown'}입니다(stage={run_stage or 'unknown'})."
        elif latest_run is not None:
            run_status = str(latest_run.get("status") or "")
            run_stage = str(latest_run.get("current_stage") or "")
            if run_status == "needs_attention":
                execution_status = "attention_projection_missing"
                execution_detail = f"M1 run은 사람 입력 대기 상태(stage={run_stage or 'unknown'})이지만 연결된 Attention item이 없습니다. Runtime/Attention projection을 확인해야 합니다."
            elif run_status == "running":
                execution_status = "agent_running"
                execution_detail = f"M1이 실행 중입니다(stage={run_stage or 'unknown'}). 아직 연구자 작업은 생성되지 않았습니다."
            elif run_status == "completed":
                execution_status = "round_completed"
                execution_detail = "최근 M1 문헌조사 run이 완료되었습니다."
            else:
                execution_status = "execution_error"
                execution_detail = f"M1 run 상태가 {run_status or 'unknown'}입니다(stage={run_stage or 'unknown'})."
        elif intents and status == "exploring":
            execution_status = "dispatch_pending"
            execution_detail = "탐색 Intent는 있지만 아직 M1 실행 run 또는 Attention item이 생성되지 않았습니다."

        # User-facing lifecycle phase is intentionally separate from the durable RQ status.
        # It answers "where is this research question in its job lifecycle?" while status
        # continues to preserve the canonical domain state.
        if status == "resolved":
            lifecycle_phase = "completed"
        elif status == "rejected":
            lifecycle_phase = "closed"
        elif status == "hold":
            lifecycle_phase = "hold"
        elif status in {"candidate", "interested"} and not intents:
            lifecycle_phase = "starting"
        elif status == "exploring" and len(intents) > 1:
            lifecycle_phase = "followup_literature"
        elif status == "exploring":
            lifecycle_phase = "initial_literature"
        else:
            lifecycle_phase = "starting"

        source_payload = dict(thread.get("source_payload") or {}) if isinstance(thread.get("source_payload"), dict) else {}
        researcher_comment = str(source_payload.get("researcher_comment") or source_payload.get("request_context") or row.get("research_context") or "").strip()
        attention_counts = {"inputs": 0, "reviews": 0, "decisions": 0, "exceptions": 0}
        for attention_row in rq_attention:
            category = str(attention_row.get("category") or "")
            if category in attention_counts:
                attention_counts[category] += 1

        result.append(ResearchQuestionView(
            rq_id=rq_id,
            question=str(row.get("question") or ""),
            status=status,
            rationale=str(row.get("rationale") or ""),
            research_context=str(row.get("research_context") or ""),
            exploration_need=str(row.get("exploration_need") or ""),
            updated_at=str(row.get("updated_at") or ""),
            source_type=str(thread.get("source_type") or ""),
            intent_count=len(intents),
            evidence_source_count=len(sources),
            current_stage=current_stage,
            next_agent_action=next_agent_action,
            human_attention_required=bool(rq_attention) or current_stage in {"knowledge_review", "answer_confirmation"},
            progress_detail=progress_detail,
            literature_round_completed=bool(round_state["literature_done"]),
            pending_knowledge_reviews=int(round_state["pending_knowledge"]),
            approved_knowledge_count=int(round_state["approved"]),
            answer_confirmation_pending=bool(round_state["answer_pending"]),
            answer_draft=str(round_state["answer_draft"]),
            answer_first_document=str(round_state.get("answer_first_document") or ""),
            answer_version=int(round_state.get("answer_version") or 0),
            answer_update_available=answer_update_ready,
            answer_work_in_progress=answer_work_in_progress,
            answer_work_stage=answer_work_stage,
            answer_work_status=answer_work_status,
            direct_evidence_cards=int(evidence_counts.get("direct_cards") or 0),
            direct_evidence_papers=int(evidence_counts.get("direct_papers") or 0),
            supporting_knowledge_cards=int(evidence_counts.get("supporting_cards") or 0),
            supporting_papers=int(evidence_counts.get("supporting_papers") or 0),
            direct_evidence_titles=direct_titles,
            supporting_evidence_titles=supporting_titles,
            attention_category=attention_category,
            attention_phase=attention_phase,
            attention_round_label=attention_round_label,
            researcher_comment=researcher_comment,
            attention_item_count=len(rq_attention),
            attention_inputs=attention_counts["inputs"],
            attention_reviews=attention_counts["reviews"],
            attention_decisions=attention_counts["decisions"],
            attention_exceptions=attention_counts["exceptions"],
            execution_status=execution_status,
            execution_detail=execution_detail,
            paper_review_pending=paper_review_pending,
            paper_review_completed=paper_review_completed,
            lifecycle_phase=lifecycle_phase,
            additional_literature_ready=bool(
                intents
                and round_state["literature_done"]
                and int(round_state["pending_knowledge"]) == 0
                and latest_intent_status == "completed"
            ),
        ))
    return result


def _evidence_work(ledger: Ledger, *, limit: int) -> list[EvidenceWorkView]:
    result: list[EvidenceWorkView] = []
    for run in ledger.unified_literature_discovery_runs(limit=limit):
        results = list(run.get("results") or [])
        result.append(EvidenceWorkView(
            item_id=str(run.get("run_id") or run.get("session_id") or ""),
            kind="literature_discovery",
            title=str(run.get("topic") or "Literature discovery"),
            status=str(run.get("status") or "completed"),
            updated_at=str(run.get("updated_at") or run.get("created_at") or ""),
            detail=str(run.get("search_summary") or run.get("query") or ""),
            count=len(results),
        ))
    for paper in ledger.shelf_papers(limit=limit):
        reading = str(paper.get("reading_status") or "unread")
        if reading not in {"unread", "reading"}:
            continue
        result.append(EvidenceWorkView(
            item_id=str(paper.get("paper_id") or ""),
            kind="paper_reading",
            title=str(paper.get("title") or "Untitled paper"),
            status=reading,
            updated_at=str(paper.get("updated_at") or ""),
            detail=str(paper.get("shelf_status") or "reference"),
            count=0,
        ))
    result.sort(key=lambda item: item.updated_at, reverse=True)
    return result[:limit]


def _knowledge_progress(ledger: Ledger, *, limit: int) -> list[KnowledgeProgressView]:
    reviewed = ledger.reviewed_knowledge_update_ids()
    result: list[KnowledgeProgressView] = []
    for row in ledger.phenomena(type_="knowledge_update")[:limit]:
        payload = row.get("payload") if isinstance(row.get("payload"), dict) else {}
        update_id = str(row.get("phenomenon_id") or "")
        title = str(
            payload.get("title")
            or payload.get("claim")
            or payload.get("finding")
            or "Knowledge updated"
        )
        detail = str(payload.get("summary") or payload.get("change_summary") or "")
        result.append(KnowledgeProgressView(
            update_id=update_id,
            title=title,
            created_at=str(row.get("created_at") or ""),
            status="reviewed" if update_id in reviewed else "new",
            detail=detail,
        ))
    return result


def _advisory_requests(ledger: Ledger, questions: list[ResearchQuestionView], *, limit: int) -> list[AdvisoryView]:
    result: list[AdvisoryView] = []
    for question in questions:
        source_type = question.source_type
        if source_type not in {"external_advisory", "researcher", "researcher_question"}:
            continue
        thread = ledger.research_question_thread(question.rq_id) or {}
        payload = thread.get("source_payload") if isinstance(thread.get("source_payload"), dict) else {}
        result.append(AdvisoryView(
            rq_id=question.rq_id,
            question=question.question,
            source_type=source_type,
            status=question.status,
            updated_at=question.updated_at,
            requester=str(payload.get("requester") or payload.get("recipient") or ""),
        ))
    return result[:limit]




def _previous_research_work(ledger: Ledger, *, limit: int) -> list[PreviousResearchWorkView]:
    """Compatibility projection for research work created before RQ/Intent became canonical."""
    workspaces = ledger.literature_discovery_workspaces()
    by_session: dict[str, dict[str, Any]] = {}
    for item in workspaces:
        session_id = str(item.get("m1-discovery-session-id") or "").strip()
        if session_id:
            by_session[session_id] = item
    result: list[PreviousResearchWorkView] = []
    for session in ledger.literature_discovery_sessions(limit=max(limit * 3, 30)):
        session_id = str(session.get("session_id") or "")
        saved = by_session.get(session_id, {})
        title = str(saved.get("work_title") or session.get("topic") or "Previous literature research")
        context = str(saved.get("m1-discovery-context") or session.get("research_context") or "")
        results = list(saved.get("m1-discovery-results") or session.get("results") or [])
        workspace_id = str(saved.get("workspace_id") or "")
        status = "saved" if workspace_id.startswith("draft:") else "completed"
        result.append(PreviousResearchWorkView(
            work_id=session_id, title=title, status=status,
            updated_at=str(saved.get("updated_at") or session.get("updated_at") or session.get("created_at") or ""),
            research_context=context, result_count=len(results),
        ))
    result.sort(key=lambda item: item.updated_at, reverse=True)
    return result[:limit]


def _next_actions(
    questions: list[ResearchQuestionView],
    evidence: list[EvidenceWorkView],
    knowledge: list[KnowledgeProgressView],
    *,
    limit: int,
) -> list[NextActionView]:
    actions: list[NextActionView] = []
    for rq in questions:
        if rq.status in {"interested", "candidate"} and rq.intent_count == 0:
            actions.append(NextActionView(
                action_id=f"explore:{rq.rq_id}", priority="high",
                title=f"Explore research question: {rq.question}",
                reason=rq.exploration_need or "No exploration intent is linked yet.",
                target_type="research_question", target_id=rq.rq_id,
            ))
        elif rq.status == "exploring" and rq.evidence_source_count == 0:
            actions.append(NextActionView(
                action_id=f"evidence:{rq.rq_id}", priority="medium",
                title=f"Acquire evidence for: {rq.question}",
                reason="The question is being explored but no evidence source is linked yet.",
                target_type="research_question", target_id=rq.rq_id,
            ))
    for item in evidence:
        if item.kind == "paper_reading" and item.status == "unread":
            actions.append(NextActionView(
                action_id=f"read:{item.item_id}", priority="medium",
                title=f"Review paper: {item.title}",
                reason="The paper is in the research shelf but has not been read yet.",
                target_type="paper", target_id=item.item_id,
            ))
    new_updates = [item for item in knowledge if item.status == "new"]
    if new_updates:
        actions.append(NextActionView(
            action_id="review-new-knowledge", priority="medium",
            title=f"Review {len(new_updates)} new knowledge update(s)",
            reason="New M1 knowledge has not yet been consumed by a completed M2 research-state review.",
            target_type="knowledge_update", target_id=new_updates[0].update_id,
        ))
    rank = {"high": 0, "medium": 1, "low": 2}
    actions.sort(key=lambda item: (rank.get(item.priority, 9), item.title.lower()))
    return actions[:limit]


def research_workspace_snapshot(
    ledger: Ledger, memory_cards: list[dict[str, Any]] | None = None, *, limit: int = 20,
    attention_items: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return a read-only job view of the current research lifecycle."""
    limit = max(1, min(int(limit), 100))
    questions = _active_questions(
        ledger, list(memory_cards or []), limit=limit, attention_items=list(attention_items or []),
    )
    evidence = _evidence_work(ledger, limit=limit)
    knowledge = _knowledge_progress(ledger, limit=limit)
    advisory = _advisory_requests(ledger, questions, limit=limit)
    previous_work = _previous_research_work(ledger, limit=limit)
    next_actions = _next_actions(questions, evidence, knowledge, limit=limit)

    question_counts: dict[str, int] = {}
    for item in questions:
        question_counts[item.status] = question_counts.get(item.status, 0) + 1
    evidence_counts = {
        "discovery": sum(1 for item in evidence if item.kind == "literature_discovery"),
        "papers_to_read": sum(1 for item in evidence if item.kind == "paper_reading" and item.status == "unread"),
        "papers_reading": sum(1 for item in evidence if item.kind == "paper_reading" and item.status == "reading"),
    }
    return {
        "counts": {
            "active_questions": sum(1 for item in questions if item.status not in {"resolved", "rejected"}),
            "total_questions": len(questions),
            "exploring_questions": question_counts.get("exploring", 0),
            "completed_questions": question_counts.get("resolved", 0),
            "evidence_items": len(evidence),
            "new_knowledge": sum(1 for item in knowledge if item.status == "new"),
            "advisory_requests": len(advisory),
            "previous_work": len(previous_work),
            "next_actions": len(next_actions),
        },
        "question_counts": question_counts,
        "evidence_counts": evidence_counts,
        "questions": [item.as_dict() for item in questions],
        "previous_work": [item.as_dict() for item in previous_work],
        "evidence": [item.as_dict() for item in evidence],
        "knowledge_progress": [item.as_dict() for item in knowledge],
        "advisory": [item.as_dict() for item in advisory],
        "next_actions": [item.as_dict() for item in next_actions],
    }
