from __future__ import annotations

import re
from pathlib import Path

import streamlit as st

from research_fellow.llm import OllamaDraftResult, ollama_draft, ollama_draft_result, ollama_status, set_llm_audit_log_path, set_llm_audit_logger
from research_fellow.application.claim_curation import (
    build_simple_claim_cards, discovery_prompt, parse_candidate_claims, submit_claim_cards,
)
from research_fellow.application.advising import (
    direction_prompt, draft_research_direction, latest_research_state, recent_knowledge_updates, recent_research_questions,
    parse_research_context_mapping, record_research_direction, record_update_report, research_context_mapping_prompt,
)
from research_fellow.application.prompt_tasks import knowledge_update_report_prompt
from research_fellow.application.meaning_summary import (
    attach_reports, build_fact_groups, delta_inputs, delta_meaning_summary_prompt, deterministic_delta_summary,
    latest_summary, meaning_summary_prompt, record_delta_summary,
)
from research_fellow.application.search_profiles import (
    abstract_relevance_prompt, attach_relevance, is_english_search_term, keyword_prompt, parse_keyword_plan, run_profile, shortlist_candidates,
)
from research_fellow.application.paper_batch import process_top_papers
from research_fellow.application.paper_shelf import document_from_shelf_path, paper_analysis_prompt, store_paper_upload, suggested_paper_labels
from research_fellow.application.duplicate_review import similar_approved_cards
from research_fellow.application.management import delete_knowledge_card, delete_knowledge_relation
from research_fellow.application.relations import lineage_dot, lineage_overview_prompt, propose_relation_candidates, relation_text_prompt
from research_fellow.infrastructure.document_reader import extract_document
from research_fellow.infrastructure.retrieval import KnowledgeRetriever, RetrievalResult
from research_fellow.infrastructure.knowledge_graph import build_graph, evidence_paths
from research_fellow.infrastructure.prompt_renderer import render_prompt
from research_fellow.memory import KnowledgeMemory, RelationMemory
from research_fellow.services import (
    complete_intent,
    create_external_case,
    decide_request,
    request_curation_intent,
)
from research_fellow.storage import Ledger
from research_fellow.ui.developer import render_developer_screen
from research_fellow.domain.research import ResearchState


ROOT = Path(__file__).parent
DATA = ROOT / "data"
EXTRACTION_CACHE = DATA / "extracted_documents"
ledger = Ledger(DATA / "research_fellow.db")
set_llm_audit_logger(ledger.record_llm_call)
set_llm_audit_log_path(DATA / "logs" / "llm_calls.jsonl")
memory = KnowledgeMemory(DATA / "knowledge_cards.jsonl")
relations = RelationMemory(DATA / "knowledge_relations.jsonl")
ledger.sync_knowledge_relations(relations.all())
retriever = KnowledgeRetriever(DATA / "retrieval_index.json")


def bullet_evidence(results: list[RetrievalResult]) -> str:
    if not results:
        return "No relevant approved knowledge is available."
    return "\n".join(
        f"- [{result.card['card_id']}] {result.card['title']} | Source: {result.card['provenance'].get('source_name', 'unknown')} "
        f"| Selection: {result.reason}\n  Claim: {result.card['claim'][:220]}"
        for result in results
    )


def report_for(question: str, results: list[RetrievalResult], model: str, use_ollama: bool) -> str:
    prompt = render_prompt("m2_research_review.j2", question=question, evidence=results)
    drafted = ollama_draft(prompt, model, use_ollama)
    if drafted:
        return drafted
    return f"""**Current interpretation**: {question}

**Available evidence**\n{bullet_evidence(results)}

**Recommendation**: Use this evidence as a starting point, while checking conditions, counterexamples, and application context separately.

**Next decision**: If more evidence is needed, create an M1 curation intent for researcher approval."""


def search_knowledge(query: str, semantic: bool, embedding_model: str, limit: int = 6) -> list[RetrievalResult]:
    return retriever.search(memory.all(), query, limit=limit, semantic=semantic, embedding_model=embedding_model)


def show_retrieval_results(results: list[RetrievalResult], detailed: bool = False) -> None:
    if not results:
        st.info("관련 승인 지식을 찾지 못했습니다. 키워드를 바꾸거나 M1 탐색 Intent를 제안하세요.")
        return
    visible = results if detailed else results[:3]
    st.caption(f"승인 지식에서 관련 카드 {len(results)}개를 찾았습니다. {'전체 후보와 점수 분석을 표시합니다.' if detailed else '상위 3개만 표시합니다.'}")
    for result in visible:
        card = result.card
        st.markdown(f"**{card['title']}**")
        st.caption(f"출처: {card['provenance'].get('source_name')} | 선정 이유: {result.reason}")
        st.write(f"주장: {card['claim']}")
        if card.get("explanation"):
            st.caption(f"보충 설명: {card['explanation']}")
        if card.get("labels"):
            st.caption(f"레이블: {', '.join(card['labels'])}")
        if card.get("evidence_excerpt"):
            st.caption(f"근거 발췌: {card['evidence_excerpt']}")
        if detailed:
            lexical = "없음" if result.lexical_score is None else f"{result.lexical_score:.2f}"
            semantic = "사용 안 함" if result.semantic_score is None else f"{result.semantic_score:.2f}"
            st.caption(f"검색 방식: {result.method} | 키워드 점수: {lexical} | 임베딩 유사도: {semantic} | 최종 정렬 점수: {result.score:.2f}")


def show_ollama_failure(result: OllamaDraftResult, model: str) -> None:
    """Show an actionable draft failure while preserving the non-LLM workflow."""
    st.error("Ollama 초안을 만들지 못했습니다.")
    st.caption(result.error or "원인을 확인하지 못했습니다.")
    if result.diagnostics:
        with st.expander("Ollama 응답 진단", expanded=True):
            st.json(result.diagnostics)
    with st.expander("Ollama 점검 명령", expanded=False):
        st.code(
            f"ollama list\n"
            f"ollama pull {model}\n"
            f"ollama run {model}",
            language="bash",
        )
        st.caption("`ollama list`에 모델이 있으면 Streamlit을 다시 실행한 뒤 재시도하세요.")


def show_relation_request(payload: dict[str, object]) -> None:
    """Readable approval projection; IDs remain available only as references."""
    source = payload.get("source_card_summary", {})
    target = payload.get("target_card_summary", {})
    relation = payload.get("relation", {})
    if not isinstance(source, dict) or not isinstance(target, dict) or not isinstance(relation, dict):
        st.json(payload)
        return
    st.markdown(f"**관계 제안 · {relation.get('relation_type', '')}**")
    st.write(payload.get("relation_summary", ""))
    left, right = st.columns(2)
    for column, heading, card in ((left, "출발 지식카드", source), (right, "도착 지식카드", target)):
        with column:
            st.markdown(f"**{heading}**")
            st.write(card.get("title") or card.get("card_id"))
            st.caption(f"주장: {card.get('claim', '')}")
            if card.get("explanation"):
                st.caption(f"보충 설명: {card['explanation']}")
            if card.get("labels"):
                st.caption(f"레이블: {', '.join(card['labels'])}")
    st.markdown("**관계 근거**")
    st.write(relation.get("evidence", ""))
    st.caption(f"적용 조건: {relation.get('conditions', '')} · 신뢰 수준: {relation.get('confidence', '')}")


def show_decision_request(item: dict[str, object]) -> None:
    """Human-readable approval projection; the ledger payload remains authoritative."""
    payload = item["payload"]
    subject_type = item["subject_type"]
    if subject_type == "knowledge_relation":
        show_relation_request(payload)
        return
    if subject_type == "knowledge_card":
        card = payload.get("card", {})
        st.markdown(f"**{card.get('title', '후보 지식카드')}**")
        st.write(f"주장: {card.get('claim', '')}")
        if card.get("explanation"):
            st.write(f"보충 설명: {card['explanation']}")
        if card.get("labels"):
            st.caption(f"레이블: {', '.join(card['labels'])}")
        provenance = card.get("provenance", {})
        if provenance:
            st.caption(f"출처: {provenance.get('source_name', '미상')}")
        if card.get("evidence_excerpt"):
            st.caption(f"근거 발췌: {card['evidence_excerpt']}")
        st.info("승인하면 이 카드만 승인 지식 JSONL에 저장되고 M2에 지식 업데이트가 통지됩니다.")
        return
    if subject_type == "curation_intent":
        intent = payload.get("intent", {})
        st.markdown(f"**{intent.get('title', 'M1 탐색 Intent')}**")
        st.write(f"목적: {intent.get('purpose', '후속 탐색의 목적을 연구자가 검토해야 합니다.')}")
        st.write(f"탐색 질문: {intent.get('question', '')}")
        st.write(f"연구 맥락: {intent.get('research_context', '이 Intent의 연구 맥락이 아직 명시되지 않았습니다.')}")
        if intent.get("labels"):
            st.caption(f"레이블: {', '.join(intent['labels'])}")
        st.caption(f"우선순위: {intent.get('priority', '보통')}")
        st.write(f"기대 근거: {intent.get('expected_evidence', '관련 근거 카드와 출처·조건·한계')}")
        st.write(f"완료 조건: {intent.get('completion_condition', '출처·조건·한계가 연결된 탐색 결과를 보고합니다.')}")
        st.info("승인하면 M1 실행함에 나타나며, 완료 뒤 M2와 연구자에게 탐색 결과가 통지됩니다.")
        return
    st.write(payload.get("next_action", "연구자 판단이 필요한 안건입니다."))


def show_m2_report_history(reports: list[dict[str, object]]) -> None:
    """Read-only, researcher-facing projection of persisted M2 advice reports."""
    if not reports:
        st.info("아직 저장된 M2 보고서가 없습니다. 연구 상태 검토 또는 M1 새 정보 요약을 실행하면 이곳에 남습니다.")
        return
    questions = []
    for item in reports:
        state = item["payload"].get("state") or {}
        question = state.get("question") if isinstance(state, dict) else None
        if question and question not in questions:
            questions.append(question)
    selected_question = st.selectbox("연구질문으로 필터", ["전체"] + questions, key="home-report-question")
    visible = [
        item for item in reports
        if selected_question == "전체" or (item["payload"].get("state") or {}).get("question") == selected_question
    ]
    st.caption(f"저장된 보고서 {len(visible)}건 · 최신순")
    for item in visible[:10]:
        payload = item["payload"]
        state = payload.get("state") or {}
        title = payload.get("title", "M2 보고서")
        question = state.get("question", "연구질문이 연결되지 않은 요약 보고서") if isinstance(state, dict) else "연구질문이 연결되지 않은 요약 보고서"
        with st.expander(f"{item['created_at'][:10]} · {title}"):
            st.caption(f"연구질문: {question}")
            st.markdown(payload.get("report", "보고서 본문이 없습니다."))
            evidence_count = len(payload.get("evidence_card_ids", []))
            update_count = len(payload.get("knowledge_update_ids", []))
            if evidence_count or update_count:
                st.caption(f"참조한 승인 지식 {evidence_count}건 · M1 업데이트 {update_count}건")


def show_intent_history() -> None:
    """Project M2 requests, M1 execution state, and resulting updates by case."""
    requests = [item for item in ledger.phenomena(type_="decision_request") if item["subject_type"] == "curation_intent"]
    executions = ledger.phenomena(type_="curation_intent")
    execution_by_case = {item["case_id"]: item for item in executions}
    updates_by_case: dict[str, list[dict[str, object]]] = {}
    for update in ledger.phenomena(type_="knowledge_update"):
        updates_by_case.setdefault(update["case_id"], []).append(update)
    choices = ["전체", "대기 승인", "승인됨", "실행 완료", "실행 실패"]
    selected_status = st.selectbox("Intent 상태", choices, key="home-intent-status")
    rows = []
    for request in requests:
        execution = execution_by_case.get(request["case_id"])
        visible_status = (
            "대기 승인" if request["status"] == "proposed" else
            "실행 완료" if execution and execution["status"] == "completed" else
            "실행 실패" if execution and execution["status"] == "failed" else
            "승인됨" if request["status"] == "approved" or execution else "보류/반려"
        )
        if selected_status == "전체" or selected_status == visible_status:
            rows.append((request, execution, updates_by_case.get(request["case_id"], []), visible_status))
    if not rows:
        st.info("선택한 상태의 M1 탐색 Intent가 없습니다.")
        return
    for request, execution, updates, visible_status in rows:
        intent = request["payload"].get("intent", {})
        with st.expander(f"{visible_status} · {request['created_at'][:10]} · {intent.get('title', 'M1 탐색 Intent')}"):
            st.write(f"연구질문: {intent.get('question', '')}")
            st.write(f"목적: {intent.get('purpose', '')}")
            st.write(f"연구 맥락: {intent.get('research_context', '')}")
            st.caption(f"생성일: {request['created_at']} · 우선순위: {intent.get('priority', '보통')}")
            if execution:
                st.caption(f"M1 실행 상태: {execution['status']} · {execution['created_at']}")
            if updates:
                st.markdown("**결과 지식 업데이트**")
                for update in updates:
                    st.write(f"- {update['payload'].get('title', 'M1 탐색 결과')}")
                    if update["payload"].get("finding"):
                        st.caption(update["payload"]["finding"])
            elif visible_status == "실행 완료":
                st.caption("완료 상태이지만 결과 업데이트가 아직 기록되지 않았습니다.")


def show_recent_activity() -> None:
    """Read-only activity projection grouped by day and research-question case."""
    phenomena = ledger.phenomena()
    if not phenomena:
        st.info("아직 표시할 연구 활동이 없습니다.")
        return
    st.caption("공유현상 원장과 승인 지식의 읽기 전용 투영입니다. 활동을 이 화면에서 새로 저장하지 않습니다.")
    st.markdown("**날짜별 타임라인**")
    by_day: dict[str, list[dict[str, object]]] = {}
    labels = {
        "research_update": "연구질문·진척 업데이트", "knowledge_update": "승인 지식 또는 M1 결과 업데이트",
        "advice_report": "M2 보고서", "decision_request": "연구자 판단 요청",
        "decision": "연구자 결정", "curation_intent": "M1 탐색 Intent",
    }
    for item in phenomena:
        by_day.setdefault(item["created_at"][:10], []).append(item)
    for day, items in sorted(by_day.items(), reverse=True)[:14]:
        with st.expander(f"{day} · 활동 {len(items)}건"):
            for item in reversed(items):
                payload = item["payload"]
                title = payload.get("title") or payload.get("finding") or ""
                st.write(f"{item['created_at'][11:19]} · {labels.get(item['phenomenon_type'], item['phenomenon_type'])} · {title}")

    st.markdown("**연구주제별 활동**")
    case_items: dict[str, list[dict[str, object]]] = {}
    for item in phenomena:
        case_items.setdefault(item["case_id"], []).append(item)
    groups: dict[str, list[dict[str, object]]] = {}
    for items in case_items.values():
        question = "기타 자료 지식화"
        for item in items:
            state = item["payload"].get("state")
            if isinstance(state, dict) and state.get("question"):
                question = state["question"]
                break
        groups.setdefault(question, []).extend(items)
    for question, items in sorted(groups.items(), key=lambda pair: max(item["created_at"] for item in pair[1]), reverse=True):
        with st.expander(f"{question} · 활동 {len(items)}건"):
            latest_state = next((item["payload"].get("state") for item in items if isinstance(item["payload"].get("state"), dict)), None)
            if latest_state and latest_state.get("researcher_note"):
                st.write(f"연구자 메모: {latest_state['researcher_note']}")
            for item in sorted(items, key=lambda row: row["created_at"], reverse=True):
                payload = item["payload"]
                if item["phenomenon_type"] == "advice_report":
                    st.write(f"M2 보고서 · {payload.get('title', '')}")
                elif item["phenomenon_type"] == "curation_intent":
                    st.write(f"M1 Intent · {payload.get('title', '')} · {item['status']}")
                elif item["phenomenon_type"] == "knowledge_update":
                    st.write(f"새 지식 · {payload.get('title', '')}")
                elif item["phenomenon_type"] == "research_update":
                    st.write("연구질문·진척 상태가 기록되었습니다.")


def meaning_summary_screen(model: str, use_ollama: bool) -> None:
    """Fact-first, persisted delta synthesis; source knowledge is never changed here."""
    st.header("연구 활동 의미 요약")
    st.caption("마지막 저장 요약 이후의 변화만 정리합니다. 새 승인 사실과 M2 해석을 분리하며, 원본 지식·관계·보고서는 변경하지 않습니다.")
    cards = memory.all()
    if not cards:
        st.info("승인된 지식카드가 있어야 의미 요약을 만들 수 있습니다.")
        return
    limit = st.select_slider("현재 전체 구조 보기 범위", options=[10, 15, 20, 30], value=20)
    visible_cards = cards[:limit]
    active_relations = relations.active_for_cards({card["card_id"] for card in visible_cards})
    reports = ledger.phenomena(recipient="researcher", type_="advice_report")
    previous = latest_summary(ledger)
    delta = delta_inputs(cards, relations.all(), reports, previous)
    groups, unmatched_reports = attach_reports(build_fact_groups(visible_cards, active_relations, limit), reports)
    fingerprint = (
        tuple(card["card_id"] for card in visible_cards),
        tuple(relation["relation_id"] for relation in active_relations),
        tuple(report["phenomenon_id"] for report in reports),
    )
    if st.session_state.get("meaning-summary-fingerprint") != fingerprint:
        st.session_state["meaning-summary-fingerprint"] = fingerprint
        st.session_state.pop("meaning-summary-result", None)
    linked = sum(len(group["reports"]) for group in groups)
    cols = st.columns(4)
    cols[0].metric("새 승인 지식", len(delta["delta"]["card_ids"]))
    cols[1].metric("새 승인 관계", len(delta["delta"]["relation_ids"]))
    cols[2].metric("새 M2 보고서", len(delta["delta"]["report_ids"]))
    cols[3].metric("현재 사실 묶음", len(groups))
    if previous:
        st.caption(f"기준선: {previous['created_at'][:16].replace('T', ' ')} 저장 요약 이후의 변화")
    else:
        st.info("저장된 요약이 없어 이번 실행은 현재 승인 지식을 초기 기준선으로 기록합니다.")
    if not any(delta["delta"].values()):
        st.info("마지막 저장 요약 이후 새로 승인·저장된 변화가 없습니다. 실행하면 ‘변화 없음’ 기록을 남길 수 있습니다.")
    st.caption("사실 묶음은 카드 레이블이나 LLM 추측이 아니라 승인된 관계의 연결 요소로만 구성됩니다. 관계 없는 카드는 독립 묶음으로 남습니다.")
    if st.button("이번 Delta 요약 만들고 저장", type="primary", key="meaning-summary-generate"):
        result = ollama_draft_result(delta_meaning_summary_prompt(delta), model, use_ollama)
        summary = result.text if result.ok else deterministic_delta_summary(delta)
        record_delta_summary(ledger, delta, summary, trigger="manual")
        st.session_state["meaning-summary-result"] = summary
        if not result.ok:
            st.warning(f"LLM 요약 대신 근거 목록형 기본 요약을 저장했습니다: {result.error}")
        else:
            st.success("이번 Delta 요약을 저장했습니다.")
    summary = st.session_state.get("meaning-summary-result")
    if summary:
        st.subheader("이번 Delta 요약")
        st.markdown(summary)
        st.caption("위 M2 문장은 저장된 읽기 전용 요약입니다. 승인 지식의 사실·관계 자체를 변경하지 않습니다.")
    history = ledger.phenomena(recipient="researcher", type_="activity_summary")
    st.subheader("저장된 Delta 요약 이력")
    if not history:
        st.caption("아직 저장된 Delta 요약이 없습니다.")
    for item in history[:20]:
        payload = item["payload"]
        change = payload.get("delta", {})
        with st.expander(f"{item['created_at'][:16].replace('T', ' ')} · 지식 {len(change.get('card_ids', []))} · 관계 {len(change.get('relation_ids', []))} · M2 보고서 {len(change.get('report_ids', []))}"):
            if payload.get("is_initial_baseline"):
                st.caption("초기 기준선")
            st.markdown(payload.get("summary", "저장된 요약 본문이 없습니다."))
    st.subheader("사실 근거 묶음")
    for index, group in enumerate(groups, start=1):
        relation_count, report_count = len(group["relations"]), len(group["reports"])
        title = group["cards"][0]["title"] if group["cards"] else "승인 지식"
        with st.expander(f"묶음 {index} · {title} · 카드 {len(group['cards'])} · 관계 {relation_count} · M2 보고서 {report_count}"):
            st.markdown("**승인된 사실**")
            for card in group["cards"]:
                st.write(f"- **{card['title']}** — {card['claim']}")
                st.caption(f"출처: {card.get('provenance', {}).get('source_name', '미상')} · 레이블: {', '.join(card.get('labels', []))}")
            st.markdown("**승인된 관계**")
            if group["relations"]:
                titles = {card["card_id"]: card["title"] for card in group["cards"]}
                for relation in group["relations"]:
                    st.write(f"- {titles.get(relation['source_card_id'], relation['source_card_id'])} → {titles.get(relation['target_card_id'], relation['target_card_id'])} · {relation['relation_type']} ({relation['confidence']})")
                    st.caption(f"근거: {relation['evidence']} · 조건: {relation['conditions']}")
            else:
                st.caption("이 카드 묶음에는 승인 관계가 없습니다.")
            if group["reports"]:
                st.markdown("**명시적으로 연결된 M2 보고서**")
                for report in group["reports"]:
                    state = report["payload"].get("state") or {}
                    st.write(f"- {report['payload'].get('title', 'M2 보고서')} · 연구질문: {state.get('question', '없음')}")
    if unmatched_reports:
        st.subheader("명시적으로 지식카드와 연결되지 않은 M2 보고서")
        st.caption("이 보고서들은 원장에 참조 카드 ID가 없어 특정 사실 묶음에 임의로 포함하지 않았습니다.")
        for report in unmatched_reports[:10]:
            state = report["payload"].get("state") or {}
            st.write(f"- {report['created_at'][:10]} · {report['payload'].get('title', 'M2 보고서')} · {state.get('question', '연결된 연구질문 없음')}")


def home(model: str, use_ollama: bool, semantic: bool, embedding_model: str) -> None:
    st.header("연구위원 홈")
    st.caption("연구자에게 필요한 판단과 M1·M2의 최근 공유현상을 한곳에서 봅니다.")
    pending = ledger.phenomena(recipient="researcher", type_="decision_request", status="proposed")
    updates = ledger.phenomena(recipient="researcher", type_="knowledge_update")
    active_relations = relations.active_for_cards({card["card_id"] for card in memory.all()})
    cols = st.columns(4)
    cols[0].metric("승인 대기", len(pending))
    cols[1].metric("승인 지식", len(memory.all()))
    cols[2].metric("최근 지식 업데이트", len(updates))
    cols[3].metric("승인 관계", len(active_relations))

    st.subheader("연구자 검토·승인함")
    if not pending:
        st.success("현재 연구자 판단이 필요한 안건이 없습니다.")
    else:
        selected = []
        select_all = st.checkbox("대기 안건 전체 선택", key="select-all-pending")
        for item in pending:
            label = item["payload"].get("title", item["subject_type"])
            checked = select_all or st.checkbox(label, key=f"pick-{item['phenomenon_id']}")
            if checked:
                selected.append(item)
            with st.expander(f"상세: {label}"):
                show_decision_request(item)
        note = st.text_input("공통 의견 (선택)", key="batch-note")
        left, middle, right = st.columns(3)
        if left.button("선택 안건 승인", disabled=not selected, type="primary"):
            for item in selected:
                decide_request(ledger, memory, item["phenomenon_id"], "approved", note, relations)
            st.rerun()
        if middle.button("선택 안건 보완 요청", disabled=not selected):
            for item in selected:
                decide_request(ledger, memory, item["phenomenon_id"], "deferred", note, relations)
            st.rerun()
        if right.button("선택 안건 반려", disabled=not selected):
            for item in selected:
                decide_request(ledger, memory, item["phenomenon_id"], "rejected", note, relations)
            st.rerun()

    st.subheader("M2 보고서 이력")
    reports = ledger.phenomena(recipient="researcher", type_="advice_report")
    show_m2_report_history(reports)

    st.subheader("승인 지식 검색")
    query = st.text_input("연구질문 또는 키워드", key="home-knowledge-query", placeholder="예: LLM generated design feasibility")
    if query.strip():
        results = search_knowledge(query, semantic, embedding_model, limit=10)
        show_retrieval_results(results)
        if results:
            options = {f"{item.card['title']} · {item.card['provenance'].get('source_name', '')}": item.card["card_id"] for item in results}
            chosen = st.multiselect("P2 계보 후보에 담을 카드 (최대 10개)", list(options), key="home-lineage-cards")
            if st.button("선택 카드를 P2 계보 목록으로 전달", disabled=not chosen, key="home-send-lineage"):
                current = st.session_state.setdefault("p2-lineage-selected-ids", [])
                for card_id in (options[label] for label in chosen):
                    if card_id not in current and len(current) < 10:
                        current.append(card_id)
                st.session_state["p2-lineage-selected-ids"] = current
                st.success(f"P2 계보 선택 목록에 카드 {len(current)}개를 저장했습니다. P2 화면에서 확인하세요.")

    st.subheader("M1 탐색 Intent 이력")
    show_intent_history()

    st.subheader("나의 최근 활동")
    show_recent_activity()

    st.subheader("최근 보고·업데이트")
    stream = ledger.phenomena(recipient="researcher")[:10]
    for item in stream:
        st.write(f"`{item['phenomenon_type']}` · **{item['producer']}** · {item['created_at']}")
        st.caption(item["payload"].get("title") or item["payload"].get("finding", ""))


def render_paper_shelf(model: str, use_ollama: bool) -> None:
    """M1's paper asset view; it is intentionally separate from approved cards."""
    st.subheader("중요 논문 서재")
    st.caption("중요 논문·보류 문헌·분석 메모를 보관합니다. 이 화면의 요약과 중요도는 승인 지식이나 논문 사실의 확정이 아닙니다.")
    with st.expander("논문 직접 등록", expanded=not ledger.shelf_papers()):
        with st.form("paper-shelf-register"):
            title = st.text_input("논문 제목")
            authors = st.text_input("저자 (쉼표로 구분)")
            year = st.text_input("발행 연도", max_chars=4)
            source_url = st.text_input("원문·DOI URL (선택)")
            uploaded = st.file_uploader("원문 파일 보관 (PDF/TXT/MD, 선택)", type=["pdf", "txt", "md"], key="paper-shelf-upload")
            submitted = st.form_submit_button("서재에 등록")
        if submitted:
            try:
                pdf_path = store_paper_upload(uploaded, DATA / "paper_shelf") if uploaded else ""
                paper = ledger.upsert_shelf_paper({
                    "title": title, "authors": [item.strip() for item in authors.split(",") if item.strip()],
                    "publication_year": year, "source_url": source_url, "pdf_path": pdf_path,
                    "shelf_status": "reference", "reading_status": "unread",
                })
                st.success(f"서재에 등록했습니다: {paper['title']}")
                st.rerun()
            except Exception as error:
                st.error(f"논문 등록에 실패했습니다: {error}")

    status_labels = {"core": "핵심 참고", "reference": "참고", "held": "보류", "excluded": "제외"}
    reading_labels = {"unread": "미읽음", "reading": "읽는 중", "read": "읽음"}
    all_papers = ledger.shelf_papers()
    filter_status = st.selectbox("상태 필터", ["all", *status_labels], format_func=lambda value: "전체" if value == "all" else status_labels[value], key="paper-shelf-filter")
    all_labels = sorted({label for paper in all_papers for label in paper.get("labels", [])}, key=str.casefold)
    selected_labels = st.multiselect("레이블 필터 (선택한 레이블을 모두 포함)", all_labels, key="paper-shelf-label-filter")
    papers = [
        paper for paper in all_papers
        if (filter_status == "all" or paper["shelf_status"] == filter_status)
        and set(selected_labels).issubset(set(paper.get("labels", [])))
    ]
    if not papers:
        st.info("등록된 논문이 없습니다. 탐색 로그의 후보를 추가하거나 원문을 직접 등록하세요.")
        return
    cards_by_id = {card["card_id"]: card for card in memory.all()}
    card_options = {f"{card['title']} [{card['card_id']}]": card["card_id"] for card in cards_by_id.values()}
    for paper in papers:
        with st.expander(f"{status_labels.get(paper['shelf_status'], paper['shelf_status'])} · {paper['title']}"):
            st.caption(f"{paper.get('publication_year') or '연도 미상'} · {', '.join(paper.get('authors', [])) or '저자 미상'} · {reading_labels.get(paper['reading_status'], paper['reading_status'])}")
            st.caption(f"레이블: {', '.join(paper.get('labels', [])) or '아직 없음'}")
            controls, actions = st.columns([3, 2])
            with controls:
                with st.form(f"paper-shelf-state-{paper['paper_id']}"):
                    shelf_status = st.selectbox("중요도", list(status_labels), index=list(status_labels).index(paper["shelf_status"]), format_func=lambda value: status_labels[value])
                    reading_status = st.selectbox("읽기 상태", list(reading_labels), index=list(reading_labels).index(paper["reading_status"]), format_func=lambda value: reading_labels[value])
                    state_saved = st.form_submit_button("상태 저장")
                if state_saved:
                    ledger.update_shelf_paper(paper["paper_id"], shelf_status=shelf_status, reading_status=reading_status)
                    st.success("서재 상태를 저장했습니다.")
                    st.rerun()
            with actions:
                if paper.get("source_url"):
                    st.link_button("원문 페이지 열기", paper["source_url"], key=f"shelf-url-{paper['paper_id']}")
                path = Path(paper["pdf_path"]) if paper.get("pdf_path") else None
                if path and path.exists():
                    st.download_button("보관 원문 내려받기", data=path.read_bytes(), file_name=path.name, key=f"shelf-download-{paper['paper_id']}")
                else:
                    st.caption("보관 원문 없음")
            analysis = ledger.paper_analysis(paper["paper_id"]) or {}
            question = st.text_area("이 논문을 읽는 연구 질문·활용 맥락", value=analysis.get("research_question", ""), key=f"shelf-question-{paper['paper_id']}")
            if st.button("본문 기반 서재 분석 요약 만들기", key=f"shelf-analyze-{paper['paper_id']}", disabled=not (path and path.exists())):
                try:
                    document = extract_document(document_from_shelf_path(str(path)), cache_dir=EXTRACTION_CACHE)
                    result = ollama_draft_result(paper_analysis_prompt(document, paper, question), model, use_ollama, profile="p1_card_draft")
                    if result.ok:
                        ledger.save_paper_analysis(paper["paper_id"], research_question=question, summary=result.text or "", researcher_note=analysis.get("researcher_note", ""), generated=True)
                        proposed_labels = suggested_paper_labels(result.text or "")
                        if proposed_labels:
                            ledger.update_shelf_paper(paper["paper_id"], shelf_status=paper["shelf_status"], reading_status=paper["reading_status"], labels=proposed_labels)
                        st.success("서재 분석과 레이블 제안을 저장했습니다. 이는 승인 지식이 아닌 M1 읽기용 해석입니다.")
                        st.rerun()
                    else:
                        show_ollama_failure(result, model)
                except Exception as error:
                    st.error(f"서재 분석에 실패했습니다: {error}")
            if analysis.get("summary"):
                st.markdown("**M1 분석 요약 (문헌 확인 / 해석 분리)**")
                st.markdown(analysis["summary"])
            with st.form(f"paper-shelf-note-{paper['paper_id']}"):
                labels = st.text_input("서재 레이블 (쉼표로 구분 · 최대 5개)", value=", ".join(paper.get("labels", [])), key=f"shelf-labels-{paper['paper_id']}")
                note = st.text_area("연구자 메모", value=analysis.get("researcher_note", ""), key=f"shelf-note-{paper['paper_id']}")
                selected = st.multiselect("연결된 승인 지식카드", list(card_options), default=[label for label, card_id in card_options.items() if card_id in ledger.paper_card_ids(paper["paper_id"])], key=f"shelf-links-{paper['paper_id']}")
                saved = st.form_submit_button("메모·카드 연결 저장")
            if saved:
                ledger.save_paper_analysis(paper["paper_id"], research_question=question, summary=analysis.get("summary", ""), researcher_note=note)
                ledger.update_shelf_paper(paper["paper_id"], shelf_status=paper["shelf_status"], reading_status=paper["reading_status"], labels=labels.split(","))
                ledger.set_paper_card_links(paper["paper_id"], [card_options[label] for label in selected])
                st.success("서재 레이블·메모·카드 연결을 저장했습니다.")
                st.rerun()


def m1_screen(model: str, use_ollama: bool, semantic: bool, embedding_model: str) -> None:
    st.header("M1 · 문헌조사·지식화 작업실")
    st.caption("M1은 문헌과 연구 노트를 탐색·구조화해 승인 후보 지식과 관계를 준비합니다. 연구자 질문이나 외부 자문에는 직접 답하지 않고, 검증 지식을 M2에 갱신합니다.")
    upload_tab, relation_tab, search_tab, queue_tab, shelf_tab, memory_tab = st.tabs(["자료 지식화", "관계·계보 정리", "승인 지식 조회", "문헌 탐색 작업", "중요 논문 서재", "승인 지식 목록"])
    with upload_tab:
        uploaded = st.file_uploader("논문 PDF·연구 노트", type=["pdf", "txt", "md"])
        source_kind = st.selectbox("자료 성격", ["외부 논문", "연구자의 확정 문서", "연구자의 아이디어 노트"])
        if uploaded and st.button("이 원문을 중요 논문 서재에 보관", key="claim-first-add-shelf"):
            try:
                document = extract_document(uploaded, cache_dir=EXTRACTION_CACHE)
                paper = ledger.upsert_shelf_paper({
                    "title": document.title, "authors": [document.author] if document.author else [],
                    "source_id": document.document_id, "pdf_path": store_paper_upload(uploaded, DATA / "paper_shelf"),
                    "shelf_status": "reference", "reading_status": "unread",
                })
                st.success(f"중요 논문 서재에 보관했습니다: {paper['title']}")
            except Exception as error:
                st.error(f"원문을 서재에 보관하지 못했습니다: {error}")
        st.caption("문서를 선택하면 Python이 본문을 추출한 뒤, LLM이 최대 10개의 주장·설명·레이블 카드 초안을 만듭니다.")
        if st.button("지식 카드 초안 만들기 (최대 10개)", disabled=uploaded is None, type="primary", key="claim-first-discover"):
            try:
                document = extract_document(uploaded, cache_dir=EXTRACTION_CACHE)
                result = ollama_draft_result(discovery_prompt(document), model, use_ollama)
                if result.ok:
                    candidates = parse_candidate_claims(result.text or "")
                    st.session_state["claim-first-document"] = document
                    st.session_state["claim-first-candidates"] = candidates
                    cards = build_simple_claim_cards(
                        document, source_kind, candidates, [],
                    )
                    st.session_state["claim-first-cards"] = cards
                    st.session_state.pop("claim-first-submitted", None)
                    st.success(f"후보 지식 카드 {len(cards)}개를 만들었습니다.")
                    if result.error:
                        st.warning(result.error)
                else:
                    show_ollama_failure(result, model)
            except Exception as error:
                st.error(f"Claim discovery failed: {error}")
        claim_document = st.session_state.get("claim-first-document")
        claim_cards = st.session_state.get("claim-first-cards", [])
        if claim_cards and claim_document and uploaded and claim_document.document_id == extract_document(uploaded, cache_dir=EXTRACTION_CACHE).document_id:
            st.subheader("후보 지식 카드")
            st.caption("기존 승인 지식과의 중복 가능성을 먼저 점검합니다. 유사 후보는 새 카드 기본 선택에서 제외하고, 기존 카드에 이 문헌의 근거·조건을 보강할지 검토하도록 안내합니다.")
            duplicate_matches = similar_approved_cards(claim_cards, memory.all())
            selected_cards = []
            for index, card in enumerate(claim_cards, start=1):
                st.markdown(f"**{card['title']}**")
                st.write(f"주장: {card['claim']}")
                if card.get("explanation"):
                    st.caption(f"보충 설명: {card['explanation']}")
                if any("가" <= character <= "힣" for field in ("title", "claim", "explanation", "conditions", "limits") for character in str(card.get(field, ""))):
                    st.warning("이 카드의 기계용 필드에 한글이 포함되어 있습니다. 영문으로 보정하지 않으면 승인함에 전송되지 않습니다. 원문 발췌는 원 언어를 유지해도 됩니다.")
                revised_labels = st.text_input(
                    f"Labels for claim {index}", value=", ".join(card.get("labels", [])), key=f"claim-first-labels-{index}",
                )
                matches = duplicate_matches.get(card["card_id"], [])
                if matches:
                    st.warning("기존 승인 지식과 유사합니다. 새 카드를 추가하기보다 기존 카드에 이 문헌의 근거·조건을 보강할지 우선 검토하세요.")
                    for match in matches:
                        st.caption(f"병합 검토 후보 · {match['title']} · 유사도 {match['score']:.2f} · {match['claim']}")
                include_label = "그래도 별도 카드로 승인함에 보내기" if matches else "Include this card"
                if st.checkbox(include_label, value=not bool(matches), key=f"claim-first-include-{index}"):
                    selected_cards.append({**card, "labels": [item.strip() for item in revised_labels.split(",") if item.strip()]})
                st.caption("Grounding: source document only. Exact excerpt, conditions, and limits have not been assessed.")
            for warning in st.session_state.get("claim-first-warnings", []):
                st.warning(warning)
            submitted = st.session_state.get("claim-first-submitted") == claim_document.document_id
            if st.button("선택한 카드를 연구자 승인함으로 보내기", disabled=submitted or not selected_cards, key="claim-first-submit"):
                request_ids = submit_claim_cards(ledger, claim_document, selected_cards, st.session_state.get("claim-first-warnings", []))
                st.session_state["claim-first-submitted"] = claim_document.document_id
                st.success(f"Sent {len(request_ids)} candidate cards. No card enters JSONL before approval.")

    if False:  # Legacy page-wise curation is intentionally hidden from the simplified M1 workflow.
        st.caption("작은 로컬 LLM 문맥을 위해 텍스트 구간별로 순차 초안을 만들고, 마지막에 별도 LLM 호출로 중복·포함 관계만 검토합니다.")
        progressive_upload = st.file_uploader("점진적으로 지식화할 PDF·노트", type=["pdf", "txt", "md"], key="progressive-upload")
        progressive_kind = st.selectbox("자료 성격", ["외부 논문", "연구자의 확정 문서", "연구자의 아이디어 노트"], key="progressive-kind")
        progressive_labels = st.text_input("기본 레이블 (선택)", key="progressive-labels", placeholder="예: agent engineering, evaluation")
        if progressive_upload:
            try:
                progressive_document = extract_document(progressive_upload, cache_dir=EXTRACTION_CACHE)
                st.caption(f"추출 텍스트 구간 {len(progressive_document.pages)}개 · {progressive_document.extraction_engine} · {'캐시 재사용' if progressive_document.cache_hit else '새 추출'}")
                page_numbers = [page.page_number for page in progressive_document.pages]
                if len(page_numbers) == 1:
                    start_page = end_page = page_numbers[0]
                    st.caption(f"이번 처리 범위: p.{start_page}")
                else:
                    start_page, end_page = st.select_slider(
                        "이번에 처리할 텍스트 구간 범위", options=page_numbers, value=(page_numbers[0], page_numbers[-1]), key="progressive-range",
                    )
                selected_pages = [page for page in progressive_document.pages if start_page <= page.page_number <= end_page]
                if st.button("페이지별 후보 카드 초안 만들기", type="primary", key="progressive-page-run"):
                    progress = st.progress(0, text="페이지별 초안을 준비합니다.")
                    def on_page(done: int, total: int, cache_hit: bool) -> None:
                        progress.progress(done / total, text=f"{done}/{total} 구간 {'캐시 재사용' if cache_hit else 'LLM 초안 생성'}")
                    def draft_for(prompt: str) -> str | None:
                        return ollama_draft_result(prompt, model, use_ollama).text
                    candidates, warnings = generate_page_candidates(
                        progressive_document, progressive_kind,
                        [item.strip() for item in progressive_labels.split(",") if item.strip()], model,
                        draft_for, CANDIDATE_DRAFT_CACHE, selected_pages, on_page,
                    )
                    progress.empty()
                    st.session_state["progressive-document"] = progressive_document
                    st.session_state["progressive-candidates"] = candidates
                    st.session_state["progressive-warnings"] = warnings
                    st.session_state.pop("progressive-kept", None)
                    st.session_state.pop("progressive-decisions", None)
                    st.session_state.pop("progressive-submitted", None)
                    st.success(f"구간별 후보 {len(candidates)}건을 만들었습니다.")
                candidates = st.session_state.get("progressive-candidates", [])
                stored_document = st.session_state.get("progressive-document")
                if candidates and stored_document and stored_document.document_id == progressive_document.document_id:
                    st.subheader("구간별 후보 (승인 전 임시 초안)")
                    for candidate in candidates:
                        card = candidate.card
                        st.markdown(f"**{card['title']}** — {card['claim']}")
                        st.caption(candidate.candidate_id)
                    for warning in st.session_state.get("progressive-warnings", []):
                        st.warning(warning)
                    pairs = candidate_pairs(candidates)
                    if pairs:
                        st.caption(f"Python이 의미 판단 없이 좁힌 후보 쌍 {len(pairs)}개를 M1 통합 프롬프트가 검토합니다.")
                        if st.button("LLM으로 중복·포함 관계 검토", key="progressive-consolidate"):
                            result = ollama_draft_result(consolidation_prompt(pairs), model, use_ollama)
                            kept, decisions, warnings = consolidate_candidates(candidates, result.text)
                            st.session_state["progressive-kept"] = kept
                            st.session_state["progressive-decisions"] = decisions
                            st.session_state["progressive-consolidation-warnings"] = warnings
                            if not result.ok:
                                st.warning(f"통합 LLM 초안을 만들지 못해 모든 후보를 유지했습니다: {result.error}")
                            st.success(f"통합 검토 후 연구자 검토 후보 {len(kept)}건입니다.")
                    else:
                        st.info("중복·포함 가능성이 높은 후보 쌍이 없어 모든 후보를 독립 카드로 유지합니다.")
                    kept = st.session_state.get("progressive-kept", candidates)
                    decisions = st.session_state.get("progressive-decisions", [])
                    if decisions:
                        st.subheader("LLM 통합 판단")
                        for decision in decisions:
                            st.caption(f"{decision.first_id} ↔ {decision.second_id} · {decision.relation} · {decision.action} — {decision.reason}")
                    for warning in st.session_state.get("progressive-consolidation-warnings", []):
                        st.warning(warning)
                    submitted = st.session_state.get("progressive-submitted") == progressive_document.document_id
                    if st.button("최종 후보를 연구자 승인함에 보내기", type="primary", key="progressive-submit", disabled=not kept or submitted):
                        request_ids = submit_progressive_candidates(ledger, progressive_document, kept, decisions, st.session_state.get("progressive-consolidation-warnings", []))
                        st.session_state["progressive-submitted"] = progressive_document.document_id
                        st.success(f"독립 후보 카드 {len(request_ids)}건을 연구자 승인함에 보냈습니다. 승인 전에는 JSONL에 저장되지 않습니다.")
                    elif submitted:
                        st.caption("이 문서 범위의 최종 후보는 이미 승인함에 보냈습니다.")
            except Exception as error:
                st.error(f"점진적 지식화에 실패했습니다: {error}")
    with relation_tab:
        cards = memory.all()
        selection_mode = st.radio(
            "계보에 표시할 카드 선정", ["최근 승인 카드", "검색 후 선택"], horizontal=True, key="p2-selection-mode",
        )
        selected_cards = cards[:10]
        if selection_mode == "검색 후 선택":
            selected_ids = st.session_state.setdefault("p2-lineage-selected-ids", [])
            by_id = {card["card_id"]: card for card in cards}
            lineage_query = st.text_input("계보 카드 검색", key="p2-lineage-query", placeholder="예: agent specification evaluation")
            search_hits = search_knowledge(lineage_query, semantic, embedding_model, limit=10) if lineage_query.strip() else []
            if search_hits:
                st.caption("검색 결과에서 카드를 추가하세요. 선택 목록은 다음 검색에도 유지됩니다.")
                for hit in search_hits:
                    card = hit.card
                    if card["card_id"] in selected_ids:
                        st.caption(f"선택됨 · {card['title']}")
                    elif st.button(f"계보에 추가 · {card['title']}", key=f"p2-add-{card['card_id']}", disabled=len(selected_ids) >= 10):
                        st.session_state["p2-lineage-selected-ids"] = [*selected_ids, card["card_id"]]
                        st.rerun()
            elif lineage_query.strip():
                st.info("검색된 승인 카드가 없습니다.")
            selected_cards = [by_id[card_id] for card_id in selected_ids if card_id in by_id][:10]
            st.subheader(f"계보 선택 카드 목록 ({len(selected_cards)}/10)")
            if selected_cards:
                for card in selected_cards:
                    left, right = st.columns([6, 1])
                    left.write(f"**{card['title']}** · {', '.join(card.get('labels', []))}")
                    if right.button("제거", key=f"p2-remove-{card['card_id']}"):
                        st.session_state["p2-lineage-selected-ids"] = [item for item in selected_ids if item != card["card_id"]]
                        st.rerun()
                if st.button("선택 카드 전체 비우기", key="p2-clear-selected"):
                    st.session_state["p2-lineage-selected-ids"] = []
                    st.rerun()
            else:
                st.caption("검색 결과에서 계보에 포함할 카드를 추가하세요.")
        selection_fingerprint = tuple(card["card_id"] for card in selected_cards)
        if st.session_state.get("p2-lineage-fingerprint") != selection_fingerprint:
            st.session_state["p2-lineage-fingerprint"] = selection_fingerprint
            st.session_state.pop("lineage-overview-result", None)
        st.caption(
            f"현재 계보 대상: 승인 카드 {len(selected_cards)}개 · "
            + ("최근 등록 순서" if selection_mode == "최근 승인 카드" else "연구자 검색·선택 결과")
        )
        if len(selected_cards) < 2:
            st.info("P2 관계 후보를 만들려면 승인된 지식 카드가 두 개 이상 필요합니다.")
        else:
            st.caption("M1이 현재 계보 대상 카드에서 관계 후보를 최대 3건 제안합니다. 연구자는 홈의 승인함에서 근거를 보고 승인·보완 요청·반려만 하면 됩니다.")
            if st.button("Gemma로 관계 후보 제안 만들기", type="primary"):
                def draft_for(source: dict[str, object], target: dict[str, object]) -> str | None:
                    return ollama_draft(relation_text_prompt(source, target), model, use_ollama)
                request_ids, warnings = propose_relation_candidates(
                    ledger, selected_cards, ledger.active_knowledge_relations({card["card_id"] for card in selected_cards}), draft_for,
                )
                st.success(f"관계 후보 {len(request_ids)}건을 연구자 승인함에 보냈습니다.")
                for warning in warnings:
                    st.warning(warning)
        st.subheader("승인된 개념·방법 계보 (최대 10개 카드)")
        if selected_cards:
            active_relations = ledger.active_knowledge_relations({card["card_id"] for card in selected_cards})
            st.graphviz_chart(lineage_dot(selected_cards, active_relations), use_container_width=True)
            if active_relations:
                for relation in active_relations:
                    st.caption(f"{relation['relation_type']} · {relation['confidence']} · {relation['evidence']}")
            else:
                st.caption("아직 승인된 관계가 없습니다. 그래프는 SQLite 승인 관계 테이블의 실행 시 투영입니다.")
            if len(selected_cards) >= 2 and active_relations:
                graph = build_graph(selected_cards, active_relations)
                by_title = {f"{card['title']} [{card['card_id']}]": card["card_id"] for card in selected_cards}
                source_label = st.selectbox("경로 출발 카드", list(by_title), key="p2-path-source")
                target_options = [label for label in by_title if by_title[label] != by_title[source_label]]
                target_label = st.selectbox("경로 도착 카드", target_options, key="p2-path-target")
                max_hops = st.select_slider("최대 관계 단계", options=[1, 2, 3], value=3, key="p2-path-hops")
                if st.button("승인 근거 경로 찾기", key="p2-evidence-path"):
                    st.session_state["p2-evidence-paths"] = evidence_paths(
                        graph, by_title[source_label], by_title[target_label], max_hops=max_hops,
                    )
                paths = st.session_state.get("p2-evidence-paths", [])
                if paths:
                    st.markdown("**승인 관계 기반 경로**")
                    titles = {card["card_id"]: card["title"] for card in selected_cards}
                    relation_index = {relation["relation_id"]: relation for relation in active_relations}
                    for path in paths:
                        steps = []
                        for index, relation_id in enumerate(path.relation_ids):
                            relation = relation_index[relation_id]
                            steps.append(f"{titles[path.card_ids[index]]} — **{relation['relation_type']}** → {titles[path.card_ids[index + 1]]}")
                            steps.append(f"근거 [{relation_id}]: {relation['evidence']}")
                        st.markdown("  \\n".join(steps))
                elif st.session_state.get("p2-evidence-paths") == []:
                    st.caption("선택한 방향과 단계 안에서 연결된 승인 관계가 없습니다.")
            if active_relations and st.button("계보 종합 의견 만들기", key="lineage-overview-generate"):
                overview = ollama_draft(lineage_overview_prompt(selected_cards, active_relations), model, use_ollama)
                if overview:
                    st.session_state["lineage-overview-result"] = overview
                else:
                    st.warning("계보 종합 의견 초안을 만들지 못했습니다.")
            if overview := st.session_state.get("lineage-overview-result"):
                st.subheader("계보 종합 의견")
                st.markdown(overview)
    with search_tab:
        st.caption("연구 질문과 관련된 승인 지식을 찾습니다. 기본 결과는 상위 3개 카드 전체를 보여 줍니다.")
        query = st.text_input("승인 지식 검색", key="p3-search-query", placeholder="예: agent specification evaluation")
        if st.button("P3 검색", disabled=not query.strip(), type="primary"):
            st.session_state["p3-results"] = search_knowledge(query, semantic, embedding_model, limit=10)
            st.session_state["p3-query"] = query
            st.session_state["p3-detailed"] = False
        search_results = st.session_state.get("p3-results", [])
        if search_results:
            show_retrieval_results(search_results, detailed=st.session_state.get("p3-detailed", False))
            if not st.session_state.get("p3-detailed", False) and len(search_results) > 3:
                if st.button("추가 분석 보기 (최대 10개 카드와 점수)", key="p3-detail"):
                    st.session_state["p3-detailed"] = True
                    st.rerun()
    with queue_tab:
        st.subheader("연구 Intent 탐색 작업 큐")
        st.caption("승인된 연구 Intent가 탐색 프로필로 대기합니다. 실행은 arXiv 초록 최대 100편 LLM 맥락 검토 → 상위 5편 본문 비교 → 유사도 상위 3편의 P1 카드 초안 순서입니다. 마지막 지식 편입은 연구자 승인함에서만 처리합니다.")
        ready_intents = ledger.phenomena(recipient="m1", type_="curation_intent", status="ready")
        profiles = ledger.search_profiles()
        all_profiles = ledger.search_profiles(include_deleted=True)
        queued_profiles = [profile for profile in profiles if profile["is_active"]]
        completed_profiles = [profile for profile in profiles if not profile["is_active"]]
        profile_intent_ids = {profile["intent_id"] for profile in all_profiles}
        missing = [intent for intent in ready_intents if str(intent["payload"].get("intent_id", "")) not in profile_intent_ids]
        registered_count = len(profiles)
        deleted_count = len(all_profiles) - registered_count
        st.caption(
            f"승인 Intent {len(ready_intents)}건 · 등록됨 {registered_count}건 · "
            f"새 등록 가능 {len(missing)}건 · 큐에서 삭제됨 {deleted_count}건"
        )
        migrate_label = f"승인 Intent {len(ready_intents)}건 중 새 {len(missing)}건을 M1 탐색 큐에 등록"
        if st.button(migrate_label, key="m1-migrate-search-profiles", disabled=not missing):
            for intent in missing:
                ledger.create_search_profile(intent["payload"])
            st.rerun()
        if not profiles:
            st.info("아직 승인된 탐색 프로필이 없습니다. M2가 제안한 Intent를 연구자 홈에서 승인하면 이곳에 나타납니다.")
        if queued_profiles:
            st.markdown(f"**대기 큐 · {len(queued_profiles)}건**")
        for profile_index, profile in enumerate(queued_profiles + completed_profiles):
            if profile_index == len(queued_profiles) and completed_profiles:
                st.divider()
                st.markdown(f"**완료된 탐색 이력 · {len(completed_profiles)}건**")
                st.caption("한 번 실행된 Intent는 큐와 주기 실행 대상에서 제거됩니다. 아래에는 로그·PDF·P1 후보 카드 검토만 남습니다.")
            state_label = "대기" if profile["is_active"] else "완료"
            with st.expander(f"{state_label} · {profile['title']} · {profile['cadence']}", expanded=profile["is_active"]):
                st.caption(f"원래 연구 질문: {profile['question']}")
                if profile["is_active"] and st.button("이 Intent를 탐색 큐에서 삭제", key=f"delete-profile-{profile['profile_id']}"):
                    ledger.delete_search_profile(profile["profile_id"])
                    st.success("탐색 큐에서 삭제했습니다. 기존 실행 로그는 감사 기록으로 보존됩니다.")
                    st.rerun()
                if not profile["is_active"]:
                    if st.button("이 Intent를 다시 탐색 큐에 넣기", key=f"requeue-profile-{profile['profile_id']}"):
                        ledger.update_search_profile(profile["profile_id"], context=profile["context"], keywords=profile["keywords"], core_terms=profile.get("core_terms", []), cadence=profile["cadence"], is_active=True)
                        st.rerun()
                with st.form(f"search-profile-form-{profile['profile_id']}"):
                    context = st.text_area("탐색 맥락", value=profile["context"], height=150)
                    keywords_text = st.text_area("선정 키워드 (한 줄에 하나 · 위가 더 중요)", value="\n".join(profile["keywords"]), height=120)
                    core_terms_text = st.text_input("핵심 5개어 AND (공백 또는 쉼표로 구분)", value=" ".join(profile.get("core_terms", [])), help="정확 구문 검색과 별도로 실행할, 유사어·불용어를 뺀 판별력 높은 5개어입니다.")
                    st.caption("검색은 각 정확 구문을 우선순위대로 따로 호출해 합치며, 전체 중복 제거어 AND와 핵심 5개어 AND를 별도 호출합니다. 여섯 구문을 한 식으로 AND 하지 않습니다.")
                    cadence = st.selectbox("주기", ["daily", "weekly", "manual"], index=["daily", "weekly", "manual"].index(profile["cadence"]))
                    is_active = st.checkbox("주기 탐색 활성화", value=profile["is_active"], disabled=cadence == "manual")
                    left, right = st.columns(2)
                    save = left.form_submit_button("맥락·키워드 저장")
                    regenerate = right.form_submit_button("저장한 맥락으로 영문 키워드 만들기")
                keywords = _lines(keywords_text)
                core_terms = [term for term in re.split(r"[\\s,]+", core_terms_text.strip()) if term]
                if save or regenerate:
                    try:
                        if regenerate:
                            refreshed = {**profile, "context": context, "keywords": keywords, "core_terms": core_terms}
                            generated = ollama_draft(keyword_prompt(refreshed), model, use_ollama)
                            generated_keywords, generated_core_terms = parse_keyword_plan(generated or "")
                            if generated_keywords:
                                ledger.update_search_profile(profile["profile_id"], context=context, keywords=generated_keywords, core_terms=generated_core_terms, cadence=cadence, is_active=is_active)
                            else:
                                st.warning("영문 키워드 초안을 만들지 못했습니다. Ollama 응답을 확인하세요.")
                        else:
                            ledger.update_search_profile(profile["profile_id"], context=context, keywords=keywords, core_terms=core_terms, cadence=cadence, is_active=is_active)
                        st.rerun()
                    except ValueError as error:
                        st.error(str(error))
                current = next(item for item in ledger.search_profiles() if item["profile_id"] == profile["profile_id"])
                invalid_keywords = [keyword for keyword in current["keywords"] if not is_english_search_term(keyword)]
                if not current["keywords"] or invalid_keywords:
                    st.warning("arXiv 탐색에는 영문 키워드가 필요합니다. 맥락을 저장한 뒤 ‘영문 키워드 만들기’를 실행하거나, 영문 키워드를 직접 입력하세요.")
                if st.button("지금 탐색·초록 100편 검토·상위 5편 본문 비교", key=f"run-profile-{profile['profile_id']}", disabled=not current["is_active"] or not current["keywords"] or bool(invalid_keywords)):
                    running = st.empty()
                    running.markdown('<div class="rf-running">🟠 수행 중 · arXiv 검색과 논문 본문 비교를 진행하고 있습니다</div>', unsafe_allow_html=True)
                    outcome = run_profile(ledger, current, "manual", reviewer=lambda prompt: ollama_draft(prompt, model, use_ollama, profile="abstract_triage"))
                    if outcome["status"] == "completed":
                        processed = process_top_papers(current, outcome["candidates"], DATA, lambda prompt: ollama_draft(prompt, model, use_ollama, profile="full_text_similarity"), make_cards=False)
                        top_three = sorted(processed, key=lambda item: item.get("full_text_similarity", 0), reverse=True)[:3]
                        drafted = process_top_papers(current, top_three, DATA, lambda prompt: ollama_draft(prompt, model, use_ollama, profile="p1_card_draft"), make_cards=True, review_full_text=False)
                        drafted = [{**item, "auto_selected_for_cards": True} for item in drafted]
                        for candidate in drafted:
                            if candidate.get("candidate_cards"):
                                doc = type("BatchDocument", (), {"title": candidate["title"]})()
                                submit_claim_cards(ledger, doc, candidate["candidate_cards"], ["자동 탐색·PDF 본문 비교를 거친 후보입니다. 원문 발췌·조건·한계를 승인 시 검토하세요."], memory.all())
                        merged = {item["source_id"]: item for item in outcome["candidates"]}
                        merged.update({item["source_id"]: item for item in processed})
                        merged.update({item["source_id"]: item for item in drafted})
                        ledger.update_search_run_candidates(outcome["run_id"], list(merged.values()))
                        st.success(f"초록 후보 {len(outcome['candidates'])}건을 기록하고, 본문 유사도 상위 3편의 P1 카드 초안까지 준비했습니다.")
                    elif outcome["status"] == "completed_no_candidates":
                        st.warning("arXiv 호출은 완료됐지만 이번 검색 전략으로는 후보가 없었습니다. 실행 이력에서 각 단계의 결과 수를 확인하세요. 이는 관련 논문이 없다는 뜻은 아닙니다.")
                    else:
                        st.error(outcome["error"])
                    running.empty()
                    st.rerun()
                runs = ledger.search_runs(profile["profile_id"], limit=5)
                if runs:
                    st.markdown("**탐색·논문 관련성 로그**")
                    for run in runs:
                        with st.expander(f"{run['created_at'][:16].replace('T', ' ')} · {run['trigger']} · {run['status']} · 후보 {len(run['candidates'])}"):
                            st.caption(f"검색 전략: {run['query']}")
                            if run["error"]:
                                st.error(run["error"])
                            if run["status"] == "completed_no_candidates":
                                st.warning("모든 검색 단계가 0건이었습니다. 이 기록은 현재 접근 가능한 arXiv 결과가 부족했음을 뜻할 뿐, 관련 문헌의 부재를 뜻하지 않습니다.")
                            shortlist = [candidate for candidate in run["candidates"] if candidate.get("abstract_shortlist")]
                            if shortlist:
                                st.markdown(f"**Intent 맥락 기반 본문 비교 대상 {len(shortlist)}편**")
                            else:
                                st.info("이전 실행입니다. 아래에서 초록 적합성 검토를 실행하면 상위 5편을 다시 선정합니다.")
                            for candidate in shortlist:
                                st.write(f"- **{candidate['title']}** ({candidate['published'][:10]})")
                                st.caption(f"arXiv · {', '.join(candidate['authors'][:4])} · {candidate['url']}")
                                st.link_button("arXiv 논문 페이지 열기", candidate["url"], key=f"open-abs-{run['run_id']}-{candidate['source_id']}")
                                if candidate.get("pdf_path"):
                                    pdf_path = Path(candidate["pdf_path"])
                                    if pdf_path.exists():
                                        st.download_button("저장 PDF 내려받기 (P1 업로드용)", data=pdf_path.read_bytes(), file_name=pdf_path.name, mime="application/pdf", key=f"download-pdf-{run['run_id']}-{candidate['source_id']}")
                                if st.button("중요 논문 서재에 추가", key=f"shelf-add-{run['run_id']}-{candidate['source_id']}"):
                                    saved = ledger.upsert_shelf_paper({
                                        "title": candidate["title"], "authors": candidate.get("authors", []),
                                        "publication_year": candidate.get("published", "")[:4], "source_url": candidate.get("url", ""),
                                        "source_id": candidate.get("source_id", ""), "pdf_path": candidate.get("pdf_path", ""),
                                        "shelf_status": "reference", "reading_status": "unread",
                                    })
                                    st.success(f"서재에 추가했습니다: {saved['title']}")
                                scopes = " · ".join(candidate.get("query_scopes", [candidate.get("query_scope", "기존 실행")]))
                                st.caption(f"발견 범위: {scopes} · 1차 맥락 점수: {candidate.get('context_match_score', '-')}")
                                st.caption(candidate["summary"][:360])
                                relevance = candidate.get("relevance", {})
                                st.caption(f"M2 초록-맥락 적합성: {relevance.get('level', 'unreviewed')} · {relevance.get('rationale', '')}")
                                if candidate.get("full_text_status") == "completed":
                                    st.markdown(f"**본문 비교 보고**  \n{candidate.get('full_text_review', '')}")
                                elif candidate.get("full_text_status") == "failed":
                                    st.warning(f"PDF 처리 실패: {candidate.get('full_text_error', '')}")
                                else:
                                    st.caption("상태: abstract_only_pending — 본문 확인·연구자 검토 전에는 승인 지식이 아닙니다.")
                            with st.expander(f"검색 로그 전체 초록 {len(run['candidates'])}편", expanded=False):
                                for candidate in run["candidates"]:
                                    st.write(f"- {candidate['title']} · 1차 맥락 점수 {candidate.get('context_match_score', '-')}")
                                    st.caption(candidate["summary"][:260])
                            if any(candidate.get("relevance", {}).get("level") == "unreviewed" for candidate in run["candidates"]):
                                if st.button("이 실행의 초록·맥락 적합성 검토", key=f"review-run-{run['run_id']}"):
                                    refreshed = shortlist_candidates(current, run["candidates"], lambda prompt: ollama_draft(prompt, model, use_ollama, profile="abstract_triage"))
                                    ledger.update_search_run_candidates(run["run_id"], refreshed)
                                    st.rerun()
                                    st.warning("초록 적합성 초안을 만들지 못했습니다.")
                            if st.button("상위 5편 PDF 수집·본문 비교 보고 만들기", key=f"p1-batch-{run['run_id']}", disabled=not run["candidates"]):
                                progress = st.progress(0, text="상위 5편을 순차 처리합니다.")
                                processed = process_top_papers(current, run["candidates"], DATA, lambda prompt: ollama_draft(prompt, model, use_ollama, profile="full_text_similarity"), make_cards=False)
                                top_three = sorted(processed, key=lambda item: item.get("full_text_similarity", 0), reverse=True)[:3]
                                drafted = process_top_papers(current, top_three, DATA, lambda prompt: ollama_draft(prompt, model, use_ollama, profile="p1_card_draft"), make_cards=True, review_full_text=False)
                                drafted = [{**item, "auto_selected_for_cards": True} for item in drafted]
                                for candidate in drafted:
                                    if candidate.get("candidate_cards"):
                                        doc = type("BatchDocument", (), {"title": candidate["title"]})()
                                        submit_claim_cards(ledger, doc, candidate["candidate_cards"], ["자동 탐색·PDF 본문 비교를 거친 후보입니다. 원문 발췌·조건·한계를 승인 시 검토하세요."], memory.all())
                                merged = {item["source_id"]: item for item in run["candidates"]}
                                merged.update({item["source_id"]: item for item in processed})
                                merged.update({item["source_id"]: item for item in drafted})
                                ledger.update_search_run_candidates(run["run_id"], list(merged.values()))
                                progress.progress(1.0, text="PDF 수집·본문 비교 보고 완료")
                                st.rerun()
                            completed = [candidate for candidate in shortlist if candidate.get("full_text_status") == "completed"]
                            if completed:
                                st.markdown("**본문 유사도 순위**")
                                for rank, candidate in enumerate(sorted(completed, key=lambda item: item.get("full_text_similarity", 0), reverse=True), start=1):
                                    marker = " · P1 카드 초안 생성 대상" if candidate.get("auto_selected_for_cards") else " · 이번 카드화에서는 제외 (로그 보존)"
                                    st.caption(f"{rank}위 · {candidate.get('full_text_similarity', 0)}/100 · {candidate['title']}{marker}")
                            paper_cards = [card for candidate in run["candidates"] if candidate.get("auto_selected_for_cards") for card in candidate.get("candidate_cards", [])]
                            if paper_cards:
                                st.markdown(f"**P1 본문 기반 후보 카드 {len(paper_cards)}건 · 업로드 P1과 동일한 승인 전 검토**")
                                for index, card in enumerate(paper_cards, start=1):
                                    st.markdown(f"**C{index}. {card['title']}**")
                                    st.write(f"주장: {card['claim']}")
                                    if card.get("explanation"):
                                        st.caption(f"보충 설명: {card['explanation']}")
                                    st.caption(f"레이블: {', '.join(card.get('labels', []))}")
                                st.caption("이 후보 카드는 자동으로 연구자 승인함에 등록되었습니다. 여기서는 P1과 같은 내용 검토용으로 다시 확인할 수 있습니다.")
    with shelf_tab:
        render_paper_shelf(model, use_ollama)
    with memory_tab:
        for card in memory.all():
            st.markdown(f"**{card['title']}** — {card['claim']}")
            st.caption(" · ".join(card.get("labels", [])))


def management_screen() -> None:
    st.header("지식 관리")
    st.caption("삭제는 원본 JSONL을 지우지 않고 삭제 표식을 남깁니다. 따라서 사례 타임라인과 감사 이력은 보존됩니다.")
    cards = memory.all()
    active_relations = relations.active_for_cards({card["card_id"] for card in cards})
    card_tab, relation_tab = st.tabs(["지식 카드", "계보 관계"])
    with card_tab:
        if not cards:
            st.info("삭제할 활성 지식 카드가 없습니다.")
        else:
            options = {f"{card['title']} [{card['card_id']}]": card["card_id"] for card in cards}
            selected = st.multiselect("삭제할 지식 카드", list(options))
            note = st.text_input("삭제 사유 (선택)", key="delete-card-note")
            confirmed = st.checkbox("선택한 카드와 연결된 계보 관계가 화면에서 제외됨을 확인했습니다.", key="confirm-card-delete")
            if st.button("선택 카드 삭제", disabled=not (selected and confirmed)):
                count = sum(delete_knowledge_card(ledger, memory, options[item], note) for item in selected)
                st.success(f"지식 카드 {count}건을 삭제 처리했습니다.")
                st.rerun()
    with relation_tab:
        if not active_relations:
            st.info("삭제할 활성 관계가 없습니다.")
        else:
            options = {f"{item['relation_type']} · {item['relation_id']}": item["relation_id"] for item in active_relations}
            selected = st.multiselect("삭제할 관계", list(options))
            note = st.text_input("삭제 사유 (선택)", key="delete-relation-note")
            confirmed = st.checkbox("선택한 관계를 계보에서 제외함을 확인했습니다.", key="confirm-relation-delete")
            if st.button("선택 관계 삭제", disabled=not (selected and confirmed)):
                count = sum(delete_knowledge_relation(ledger, relations, options[item], note) for item in selected)
                st.success(f"관계 {count}건을 삭제 처리했습니다.")
                st.rerun()


def _lines(value: str) -> list[str]:
    return [item.strip(" -•\t") for item in value.splitlines() if item.strip(" -•\t")]


def research_advisory_screen(model: str, use_ollama: bool, semantic: bool, embedding_model: str) -> None:
    """M2's concise, service-facing response to a researcher question."""
    st.subheader("연구자 질문 대응")
    st.caption("M2가 승인 지식·관계·검색 결과를 근거로 의견을 구성합니다. 답변과 내부 연구 판단은 분리하며, 근거 부족은 후속 탐색 필요로 표시합니다.")
    question = st.text_area("연구자 질문", placeholder="예: 다중 LLM 합의 평가는 설계 개념의 실현 가능성 판단에 어느 정도 신뢰할 수 있는가?", key="m2-service-question")
    context = st.text_area("현재 연구 맥락·제약 (선택)", placeholder="적용 대상, 비교하려는 대안, 현재 가설 또는 확인하려는 결정", key="m2-service-context")
    if st.button("근거 기반 연구 의견 만들기", type="primary", disabled=not question.strip(), key="m2-service-answer"):
        results = search_knowledge(f"{question}\n{context}", semantic, embedding_model)
        case_id = ledger.create_case("research", f"연구자 질문: {question[:80]}")
        state = ResearchState(question=question, researcher_note=context)
        ledger.record(
            case_id, "research_update", "researcher", ["m2"], "research_question",
            {"state": state.model_dump(mode="json")}, status="completed",
        )
        answer = report_for(question, results, model, use_ollama)
        ledger.record(
            case_id, "advice_report", "m2", ["researcher"], "research_question_response",
            {"title": "M2 · 연구자 질문 대응", "question": question, "context": context, "report": answer, "state": state.model_dump(mode="json"),
             "evidence_card_ids": [result.card["card_id"] for result in results]}, status="completed",
        )
        st.session_state["m2-service-results"] = results
        st.session_state["m2-service-answer-result"] = answer
    if "m2-service-answer-result" in st.session_state:
        show_retrieval_results(st.session_state.get("m2-service-results", []))
        st.markdown("### M2 의견")
        st.markdown(st.session_state["m2-service-answer-result"])
        if not st.session_state.get("m2-service-results", []):
            st.info("현재 승인 지식만으로는 충분한 근거를 찾지 못했습니다. 아래 ‘연구 상태·방향 검토’에서 M1 탐색 Intent를 제안할 수 있습니다.")


def m2_screen(model: str, use_ollama: bool, semantic: bool, embedding_model: str) -> None:
    st.header("M2 · 지식 기반 자문 작업실")
    st.caption("M2는 M1이 축적한 승인 지식을 근거로 연구자 질문과 외부 자문에 대응합니다. 지식 공백은 내부 보고와 M1 탐색 Intent 제안으로 전환합니다.")
    question_tab, state_tab, external_tab, updates_tab = st.tabs(["연구자 질문 대응", "연구 상태·방향 검토", "외부 자문 대응", "M1 새 정보 입력함"])
    with question_tab:
        research_advisory_screen(model, use_ollama, semantic, embedding_model)
    latest_state = latest_research_state(ledger)
    with state_tab:
        st.caption("연구질문을 직접 시작·기존 질문 심화·M1 새 정보 기반 신규 질문으로 관리합니다. 제안된 M1 Intent는 연구자 홈 승인함에서만 실행됩니다.")
        question_mode = st.radio("연구 질문 시작 방식", ["직접 새 질문", "기존 질문 심화", "M1 새 정보 기반 추천"], horizontal=True)
        question_seed = ""
        if question_mode == "기존 질문 심화":
            previous_questions = recent_research_questions(ledger)
            if previous_questions:
                question_seed = st.selectbox("심화할 기존 연구질문", previous_questions)
            else:
                st.info("아직 저장된 연구질문이 없습니다. 직접 새 질문을 입력하세요.")
        elif question_mode == "M1 새 정보 기반 추천":
            updates = recent_knowledge_updates(ledger)
            if st.button("M1 새 정보에서 연구질문 추천 받기", disabled=not updates, key="p4-question-suggest"):
                prompt = render_prompt("m2_research_question_suggestions.j2", updates=updates, existing_questions=recent_research_questions(ledger))
                suggested = ollama_draft(prompt, model, use_ollama)
                st.session_state["p4-question-suggestions"] = [line.strip() for line in (suggested or "").splitlines() if line.strip()][:3]
            suggestions = st.session_state.get("p4-question-suggestions", [])
            if suggestions:
                selected_suggestion = st.radio("추천 연구질문", suggestions, key="p4-question-suggestion-choice")
                question_seed = selected_suggestion.split(". ", 1)[-1]
            elif not updates:
                st.info("추천에 사용할 M1 새 정보가 없습니다.")
        question = st.text_area(
            "이번 연구 질문·진척", value=question_seed,
            placeholder="예: 에이전트 명세 우선 설계가 구현 품질과 검토 효율을 높이는가?",
            key=f"p4-question-{question_mode}-{question_seed}",
        )
        st.caption("선택한 연구질문에는 연구자 메모와 해석된 연구 맥락이 함께 기록됩니다. 이 맥락은 승인 지식카드가 아니라, 다음 검토와 M1 탐색의 기준이 되는 질문별 작업 상태입니다.")
        researcher_note = st.text_area(
            "연구자 메모 (선택)",
            value=latest_state.researcher_note if latest_state and latest_state.question == question else "",
            placeholder="예: 디자인 개념의 참신성만이 아니라 제조 가능성, 사용자 검증 가능성, 비용을 함께 평가해야 한다. 현재는 평가 기준의 비교가 부족하다.",
            key="p4-researcher-note",
        )
        if st.button("메모를 연구 맥락 초안으로 정리", disabled=not (question.strip() and researcher_note.strip()), key="p4-map-context"):
            mapped_result = ollama_draft_result(research_context_mapping_prompt(question, researcher_note), model, use_ollama)
            if mapped_result.ok:
                mapped = parse_research_context_mapping(mapped_result.text or "")
                st.session_state["p4-context-hypothesis"] = str(mapped["hypothesis"])
                st.session_state["p4-context-constraints"] = "\n".join(mapped["constraints"])
                st.session_state["p4-context-unresolved"] = "\n".join(mapped["unresolved"])
                st.session_state["p4-context-changes"] = "\n".join(mapped["changes"])
                st.success("연구자 확인용 맥락 초안을 만들었습니다. 아래에서 수정한 뒤 검토를 실행하세요.")
            else:
                st.warning(f"메모 정리 초안을 만들지 못했습니다: {mapped_result.error}")
        same_question_as_latest = bool(latest_state and latest_state.question == question)
        defaults = {
            "p4-context-hypothesis": latest_state.current_hypothesis if same_question_as_latest else "",
            "p4-context-constraints": "\n".join(latest_state.constraints) if same_question_as_latest else "",
            "p4-context-unresolved": "\n".join(latest_state.unresolved_issues) if same_question_as_latest else "",
            "p4-context-changes": "\n".join(latest_state.recent_evidence_changes) if same_question_as_latest else "",
            "p4-context-confidence": latest_state.confidence if same_question_as_latest else "medium",
        }
        for key, value in defaults.items():
            st.session_state.setdefault(key, value)
        with st.expander("연구 맥락 확인·보완", expanded=bool(researcher_note)):
            st.caption("현재 가설은 잠정적 답의 방향, 제약은 검토 범위, 미결 사항은 추가 확인할 질문, 최근 근거 변화는 이번 검토에 반영할 새 정보를 뜻합니다. 자유 메모를 정리한 값이므로 연구자가 수정·확정합니다.")
            hypothesis = st.text_area("현재 가설", key="p4-context-hypothesis")
            constraints = st.text_area("제약 (한 줄에 하나)", key="p4-context-constraints")
            unresolved = st.text_area("미결 사항 (한 줄에 하나)", key="p4-context-unresolved")
            changes = st.text_area("연구자가 인지한 최근 근거 변화 (한 줄에 하나)", key="p4-context-changes")
            confidence = st.select_slider("현재 확신 수준", options=["low", "medium", "high"], key="p4-context-confidence")
        if st.button("M2 연구 검토와 Intent 후보 만들기", disabled=not question.strip(), type="primary"):
            state = ResearchState(
                question=question, current_hypothesis=hypothesis or "아직 명시되지 않았습니다.",
                constraints=_lines(constraints), unresolved_issues=_lines(unresolved),
                recent_evidence_changes=_lines(changes), researcher_note=researcher_note, confidence=confidence,
            )
            results = search_knowledge(question, semantic, embedding_model)
            updates = recent_knowledge_updates(ledger)
            prompt = direction_prompt(state, results, updates)
            llm_result = ollama_draft_result(prompt, model, use_ollama)
            draft = draft_research_direction(state, results, updates, llm_result.text)
            _, request_ids = record_research_direction(ledger, state, results, updates, draft)
            st.session_state["p4-results"] = results
            st.session_state["p4-report"] = draft.report
            st.session_state["p4-intents"] = draft.intents
            if not llm_result.ok and use_ollama:
                st.warning(f"Ollama 초안 대신 근거 기반 기본 검토를 만들었습니다: {llm_result.error}")
            st.success(f"M2 보고와 M1 탐색 Intent 승인 안건 {len(request_ids)}건을 만들었습니다.")
        if "p4-report" in st.session_state:
            st.subheader("M2 연구 상태·방향 보고")
            show_retrieval_results(st.session_state.get("p4-results", []))
            st.markdown(st.session_state["p4-report"])
            intents = st.session_state.get("p4-intents", [])
            if intents:
                st.subheader("연구자 승인 대기 Intent")
                for intent in intents:
                    with st.expander(f"{intent.priority} · {intent.title}"):
                        st.write(f"**목적:** {intent.purpose}")
                        st.write(f"**질문:** {intent.question}")
                        st.write(f"**연구 맥락:** {intent.research_context}")
                        st.write(f"**기대 근거:** {intent.expected_evidence}")
                        st.write(f"**완료 조건:** {intent.completion_condition}")
                st.caption("승인·보완 요청·반려는 연구위원 홈의 승인함에서 다중 처리합니다.")
    with external_tab:
        external_advisory(model, use_ollama, semantic, embedding_model, embedded=True)
    with updates_tab:
        updates = recent_knowledge_updates(ledger)
        if not updates:
            st.info("M1에서 새로 통지한 지식 업데이트가 없습니다.")
        else:
            for update in updates:
                st.markdown(f"**{update['payload'].get('title', '지식 업데이트')}**")
                st.write(update["payload"].get("finding", "승인된 지식 카드 또는 관계가 의미 기억에 추가되었습니다."))
            if st.button("M1 새 정보의 M2 요약 보고 만들기"):
                state = latest_research_state(ledger)
                prompt = knowledge_update_report_prompt(memory.all()[-12:], state.question if state else "")
                result = ollama_draft_result(prompt, model, use_ollama)
                report = result.text or "## 핵심 요약\n최근 M1 새 정보를 현재 연구질문과 함께 검토해야 합니다.\n\n## 지식 공백\n- 추가 해석을 위해 연구자의 판단이 필요합니다."
                record_update_report(ledger, state, report, updates)
                st.session_state["p4-update-report"] = report
                if not result.ok and use_ollama:
                    st.warning(f"Ollama 초안 대신 기본 요약을 기록했습니다: {result.error}")
        if "p4-update-report" in st.session_state:
            st.subheader("M2 · M1 새 정보 요약")
            st.markdown(st.session_state["p4-update-report"])


def external_advisory(model: str, use_ollama: bool, semantic: bool, embedding_model: str, *, embedded: bool = False) -> None:
    if not embedded:
        st.header("외부 자문 · M2 전문성 기반 해석과 응답")
    st.caption("외부 요청을 즉시 답하지 않습니다. 먼저 M2 전문성에 비추어 문제와 답변 범위를 해석합니다.")
    requester = st.text_input("요청자")
    expertise = st.text_input("M2의 전문성", value="에이전트 공학과 도메인 전문 연구위원 설계")
    question = st.text_area("외부 자문 요청")
    context = st.text_area("요청 맥락·제약", placeholder="목적, 적용 환경, 원하는 답변 수준")
    if st.button("요청 해석안 만들기", disabled=not (requester and expertise and question), type="primary"):
        prompt = render_prompt(
            "m2_external_interpretation.j2", requester=requester, expertise=expertise,
            question=question, context=context,
        )
        interpretation = ollama_draft(prompt, model, use_ollama) or (
            f"## Actual problem\n{question}\n\n## Answerable scope\nReview of structure, evidence, and conditions from the perspective of {expertise}.\n\n"
            "## Out-of-scope items and assumptions\nOrganisation-specific financial, legal, and operational facts require separate confirmation.\n\n"
            "## Response strategy\nConfirm the scope, then provide conditional advice grounded in approved knowledge."
        )
        case_id = create_external_case(ledger, requester, question, context, interpretation)
        st.session_state["external_case"] = case_id
        st.session_state["external_interpretation"] = interpretation
    if "external_interpretation" in st.session_state:
        st.subheader("M2의 요청 해석·범위 확인")
        st.markdown(st.session_state["external_interpretation"])
        confirmed = st.checkbox("요청자가 이 해석과 답변 범위에 동의함")
        if st.button("근거 기반 자문 답변 만들기", disabled=not confirmed):
            results = search_knowledge(question, semantic, embedding_model)
            response_prompt = render_prompt(
                "m2_external_response.j2", expertise=expertise, question=question, evidence=results,
            )
            answer = ollama_draft(response_prompt, model, use_ollama) or report_for(question, results, model, False)
            ledger.record(
                st.session_state["external_case"], "advisory_exchange", "m2", ["external_requester", "researcher"],
                "advisory_response", {"title": "외부 자문 답변", "answer": answer, "evidence_count": len(results)}, status="completed",
            )
            show_retrieval_results(results)
            st.subheader("외부 자문 답변")
            st.markdown(answer)


def main() -> None:
    st.set_page_config(page_title="Research Fellow", layout="wide")
    st.markdown("""<style>
    [data-testid="stStatusWidget"], div[data-testid="stStatusWidget"], button[data-testid="stStatusWidget"], [data-testid="stToolbar"] [aria-label*="Running"] { background:#f79009 !important; color:#1f1300 !important; border-color:#f79009 !important; font-weight:700 !important; }
    [data-testid="stStatusWidget"] *, [data-testid="stToolbar"] [aria-label*="Running"] * { color:#1f1300 !important; }
    .rf-running { position: fixed; top: 0.55rem; right: 5.9rem; z-index: 999999; max-width: 30rem; padding: 0.42rem 0.75rem; border: 1px solid #b54708; border-radius: 0.45rem; background: #f79009; color: #1f1300; font-weight: 700; box-shadow: 0 2px 7px rgba(0,0,0,.22); }
    </style>""", unsafe_allow_html=True)
    st.sidebar.title("도메인 전문 연구위원")
    model = st.sidebar.text_input("Ollama 모델", value="gpt-oss:20b")
    use_ollama = st.sidebar.checkbox("Ollama 초안 사용", value=True)
    semantic = st.sidebar.checkbox("P3 Ollama 임베딩 검색", value=False)
    embedding_model = st.sidebar.text_input("임베딩 모델", value="nomic-embed-text", disabled=not semantic)
    connected, status = ollama_status(model)
    st.sidebar.caption(f"Ollama · {status}")
    if use_ollama and not connected:
        st.sidebar.warning("초안 없이도 P1·P2 흐름은 동작합니다.")
    screen = st.sidebar.radio("작업공간", ["연구위원 데스크", "M1 · 문헌조사·지식화", "M2 · 지식 기반 자문", "지식 베이스·운영", "개발·프롬프트"])
    if screen == "연구위원 데스크":
        home(model, use_ollama, semantic, embedding_model)
    elif screen == "M1 · 문헌조사·지식화":
        m1_screen(model, use_ollama, semantic, embedding_model)
    elif screen == "M2 · 지식 기반 자문":
        m2_screen(model, use_ollama, semantic, embedding_model)
    elif screen == "지식 베이스·운영":
        overview_tab, delta_tab, manage_tab = st.tabs(["승인 지식·관계", "연구 활동 Delta", "지식 관리"])
        with overview_tab:
            st.header("지식 베이스")
            st.caption("이 화면은 M1·M2가 함께 참조하는 승인 지식과 승인 관계의 읽기·관리 투영입니다.")
            cards = memory.all()
            active_relations = ledger.active_knowledge_relations({card["card_id"] for card in cards})
            st.metric("승인 지식카드", len(cards))
            st.metric("승인 관계", len(active_relations))
            query = st.text_input("승인 지식 검색", placeholder="예: multi LLM design feasibility", key="knowledge-base-query")
            if query.strip():
                show_retrieval_results(search_knowledge(query, semantic, embedding_model, limit=10), detailed=True)
        with delta_tab:
            meaning_summary_screen(model, use_ollama)
        with manage_tab:
            management_screen()
    elif screen == "개발·프롬프트":
        render_developer_screen(memory.all(), model, use_ollama, EXTRACTION_CACHE, ledger, DATA / "logs" / "llm_calls.jsonl")


if __name__ == "__main__":
    main()
