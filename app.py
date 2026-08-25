from __future__ import annotations

from pathlib import Path

import streamlit as st

from research_fellow.llm import OllamaDraftResult, ollama_draft, ollama_draft_result, ollama_status
from research_fellow.application.curation import curation_text_prompt
from research_fellow.application.advising import (
    direction_prompt, draft_research_direction, latest_research_state, recent_knowledge_updates,
    record_research_direction, record_update_report,
)
from research_fellow.application.prompt_tasks import knowledge_update_report_prompt
from research_fellow.application.management import delete_knowledge_card, delete_knowledge_relation
from research_fellow.application.relations import lineage_dot, propose_relation_candidates, relation_text_prompt
from research_fellow.infrastructure.document_reader import ExtractedDocument, extract_document, extracted_document_text
from research_fellow.infrastructure.retrieval import KnowledgeRetriever, RetrievalResult
from research_fellow.infrastructure.prompt_renderer import render_prompt
from research_fellow.memory import KnowledgeMemory, RelationMemory
from research_fellow.services import (
    complete_intent,
    create_document_candidates,
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
memory = KnowledgeMemory(DATA / "knowledge_cards.jsonl")
relations = RelationMemory(DATA / "knowledge_relations.jsonl")
retriever = KnowledgeRetriever(DATA / "retrieval_index.json")


def bullet_evidence(results: list[RetrievalResult]) -> str:
    if not results:
        return "승인된 관련 지식이 아직 없습니다."
    return "\n".join(
        f"- [{result.card['card_id']}] {result.card['title']} | 출처: {result.card['provenance'].get('source_name', '미상')} "
        f"({result.card['provenance'].get('page_or_section', '')}) | 선정: {result.reason}\n  주장: {result.card['claim'][:220]}"
        for result in results
    )


def report_for(question: str, results: list[RetrievalResult], model: str, use_ollama: bool) -> str:
    prompt = render_prompt("m2_research_review.j2", question=question, evidence=results)
    drafted = ollama_draft(prompt, model, use_ollama)
    if drafted:
        return drafted
    return f"""**현재 해석**: {question}

**확인된 근거**\n{bullet_evidence(results)}

**권고**: 위 근거를 출발점으로 하되, 조건·반례·적용 맥락을 별도로 확인해야 합니다.

**다음 판단**: 추가 탐색이 필요하면 아래에서 M1 탐색 Intent를 연구자 승인 안건으로 등록하세요."""


def search_knowledge(query: str, semantic: bool, embedding_model: str) -> list[RetrievalResult]:
    return retriever.search(memory.all(), query, semantic=semantic, embedding_model=embedding_model)


def show_retrieval_results(results: list[RetrievalResult]) -> None:
    if not results:
        st.info("관련 승인 지식을 찾지 못했습니다. 키워드를 바꾸거나 M1 탐색 Intent를 제안하세요.")
        return
    for result in results:
        card = result.card
        st.markdown(f"**[{card['card_id']}] {card['title']}** · `{result.method}` · 점수 {result.score:.2f}")
        st.caption(f"출처: {card['provenance'].get('source_name')} · {card['provenance'].get('page_or_section')} | 선정 이유: {result.reason}")
        st.write(card["claim"])


def show_ollama_failure(result: OllamaDraftResult, model: str) -> None:
    """Show an actionable draft failure while preserving the non-LLM workflow."""
    st.error("Ollama 초안을 만들지 못했습니다.")
    st.caption(result.error or "원인을 확인하지 못했습니다.")
    with st.expander("Ollama 점검 명령", expanded=False):
        st.code(
            f"ollama list\n"
            f"ollama pull {model}\n"
            f"ollama run {model}",
            language="bash",
        )
        st.caption("`ollama list`에 모델이 있으면 Streamlit을 다시 실행한 뒤 재시도하세요.")


def show_python_extraction(document: ExtractedDocument) -> None:
    """Expose the deterministic extraction result before any LLM drafting."""
    st.success(f"Python으로 {len(document.pages)}개 페이지/구간의 텍스트를 추출했습니다.")
    cache_label = "캐시 재사용" if document.cache_hit else "새 추출"
    st.caption(f"엔진: `{document.extraction_engine}` · {cache_label}")
    st.caption(document.extraction_note)
    st.download_button(
        "추출 텍스트(.txt) 다운로드",
        data=extracted_document_text(document).encode("utf-8"),
        file_name=f"{Path(document.file_name).stem}_extracted.txt",
        mime="text/plain",
        key=f"download-extracted-{document.document_id}",
    )
    for page in document.pages:
        heading = f"쪽/구간 {page.page_number}" + (f" · {page.section}" if page.section else "")
        with st.expander(heading):
            if page.blocks:
                for block in page.blocks:
                    location = f" · 좌표 {tuple(round(value, 1) for value in block.bbox)}" if block.bbox else ""
                    st.caption(f"{block.block_id}{location}")
                    st.text(block.text)
            else:
                st.text(page.text)
            if page.truncated:
                st.warning("이 페이지는 설정된 문자 수 제한으로 일부만 표시됩니다.")


def home(model: str, use_ollama: bool) -> None:
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
                st.json(item["payload"])
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

    st.subheader("최근 보고·업데이트")
    stream = ledger.phenomena(recipient="researcher")[:10]
    for item in stream:
        st.write(f"`{item['phenomenon_type']}` · **{item['producer']}** · {item['created_at']}")
        st.caption(item["payload"].get("title") or item["payload"].get("finding", ""))


def m1_screen(model: str, use_ollama: bool, semantic: bool, embedding_model: str) -> None:
    st.header("M1 · 자료 지식화와 승인 Intent 실행")
    upload_tab, relation_tab, search_tab, queue_tab, memory_tab = st.tabs(["P1 자료 지식화", "P2 지식 관계·계보", "P3 지식 검색", "승인 Intent", "승인 지식"])
    with upload_tab:
        uploaded = st.file_uploader("논문 PDF·연구 노트", type=["pdf", "txt", "md"])
        source_kind = st.selectbox("자료 성격", ["외부 논문", "연구자의 확정 문서", "연구자의 아이디어 노트"])
        labels_text = st.text_input("검색 레이블", placeholder="예: agent engineering, evaluation, governance")
        st.caption("1. Python(pypdf)으로 페이지별 텍스트를 추출·확인합니다. 이 단계는 LLM을 호출하지 않습니다.")
        if st.button("Python으로 텍스트 추출·확인", disabled=uploaded is None):
            try:
                st.session_state["m1-extracted-document"] = extract_document(uploaded, cache_dir=EXTRACTION_CACHE)
            except Exception as error:
                st.error(f"문서 추출에 실패했습니다: {error}")
        extracted = st.session_state.get("m1-extracted-document")
        if extracted and uploaded and extracted.file_name == uploaded.name:
            show_python_extraction(extracted)

        st.caption("2. 선택적으로 LLM이 추출된 텍스트를 바탕으로 후보 카드 초안을 작성합니다.")
        if st.button("Ollama 텍스트 초안 만들기", disabled=uploaded is None):
            try:
                document = extract_document(uploaded, cache_dir=EXTRACTION_CACHE)
                result = ollama_draft_result(curation_text_prompt(uploaded.name, source_kind, document.pages), model, use_ollama)
                if result.ok:
                    st.session_state["m1-text-draft"] = result.text
                else:
                    show_ollama_failure(result, model)
                    st.info("직접 초안을 입력하거나 원문 기반 후보 생성을 계속 사용할 수 있습니다.")
            except Exception as error:
                st.error(f"문서 추출에 실패했습니다: {error}")
        text_draft = st.text_area(
            "M1/LLM 후보 카드 초안 (선택)",
            placeholder="제목: ...\n주장: ...\n근거: 원문 1~3문장\n쪽수: 3\n인용: p.3, DOI ...\n레이블: 개념, 방법\n조건: ...\n한계: ...\n---\n제목: ...",
            key="m1-text-draft",
            help="구조화 JSON 대신 읽을 수 있는 텍스트 초안을 넣습니다. 누락 항목은 원문 쪽 근거로 보완하고 승인함에 경고를 표시합니다.",
        )
        if st.button("후보 지식 카드 만들기", disabled=uploaded is None, type="primary"):
            try:
                document = extract_document(uploaded, cache_dir=EXTRACTION_CACHE)
                request_ids, warnings = create_document_candidates(
                    ledger, uploaded.name, source_kind, document.pages,
                    [label.strip() for label in labels_text.split(",") if label.strip()], text_draft,
                )
                st.success(f"후보 지식 카드 {len(request_ids)}건을 연구자 승인함에 보냈습니다.")
                for warning in warnings:
                    st.warning(warning)
            except Exception as error:  # surfaced as a user-readable extraction error
                st.error(str(error))
    with relation_tab:
        cards = memory.all()
        if len(cards) < 2:
            st.info("P2 관계 후보를 만들려면 승인된 지식 카드가 두 개 이상 필요합니다.")
        else:
            st.caption("M1이 최근 승인 카드의 관계 후보를 최대 3건 제안합니다. 연구자는 홈의 승인함에서 근거를 보고 승인·보완 요청·반려만 하면 됩니다.")
            if st.button("Gemma로 관계 후보 제안 만들기", type="primary"):
                def draft_for(source: dict[str, object], target: dict[str, object]) -> str | None:
                    return ollama_draft(relation_text_prompt(source, target), model, use_ollama)
                request_ids, warnings = propose_relation_candidates(
                    ledger, cards, relations.active_for_cards({card["card_id"] for card in cards}), draft_for,
                )
                st.success(f"관계 후보 {len(request_ids)}건을 연구자 승인함에 보냈습니다.")
                for warning in warnings:
                    st.warning(warning)
        st.subheader("승인된 개념·방법 계보 (최대 10개 카드)")
        if cards:
            active_relations = relations.active_for_cards({card["card_id"] for card in cards})
            st.graphviz_chart(lineage_dot(cards, active_relations), use_container_width=True)
            if active_relations:
                for relation in active_relations:
                    st.caption(f"{relation['relation_type']} · {relation['confidence']} · {relation['evidence']}")
            else:
                st.caption("아직 승인된 관계가 없습니다. 그래프는 저장소가 아니라 승인 지식의 투영입니다.")
    with search_tab:
        st.caption("JSONL 승인 지식을 lexical로 항상 검색하고, 선택 시 로컬 Ollama 임베딩 결과를 함께 사용합니다.")
        query = st.text_input("승인 지식 검색", key="p3-search-query", placeholder="예: agent specification evaluation")
        if st.button("P3 검색", disabled=not query.strip(), type="primary"):
            show_retrieval_results(search_knowledge(query, semantic, embedding_model))
    with queue_tab:
        intents = ledger.phenomena(recipient="m1", type_="curation_intent", status="ready")
        if not intents:
            st.info("연구자가 승인한 M2 탐색 Intent가 없습니다.")
        for intent in intents:
            payload = intent["payload"]
            st.markdown(f"**{payload['title']}** · {payload['priority']}")
            st.caption(f"질문: {payload['question']} | 레이블: {', '.join(payload['labels'])}")
            if payload.get("purpose"):
                st.caption(f"목적: {payload['purpose']} | 기대 근거: {payload.get('expected_evidence', '')}")
            if payload.get("completion_condition"):
                st.caption(f"완료 조건: {payload['completion_condition']}")
            finding = st.text_area("탐색 결과 요약", key=f"finding-{intent['phenomenon_id']}")
            if st.button("M1 결과를 M2·연구자에게 보고", key=f"done-{intent['phenomenon_id']}", disabled=not finding.strip()):
                complete_intent(ledger, intent, finding)
                st.rerun()
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


def m2_screen(model: str, use_ollama: bool, semantic: bool, embedding_model: str) -> None:
    st.header("M2 · 연구 상태·방향")
    state_tab, updates_tab = st.tabs(["연구 상태 검토", "M1 새 정보 입력함"])
    latest_state = latest_research_state(ledger)
    with state_tab:
        st.caption("질문 하나로 검토를 시작합니다. 상세 입력은 선택 사항이며, 제안된 M1 Intent는 연구자 홈 승인함에서만 실행됩니다.")
        question = st.text_area(
            "이번 연구 질문·진척", value=latest_state.question if latest_state else "",
            placeholder="예: 에이전트 명세 우선 설계가 구현 품질과 검토 효율을 높이는가?",
        )
        with st.expander("연구 상태 보완 (선택)"):
            hypothesis = st.text_area("현재 가설", value=latest_state.current_hypothesis if latest_state else "")
            constraints = st.text_area("제약 (한 줄에 하나)", value="\n".join(latest_state.constraints) if latest_state else "")
            unresolved = st.text_area("미결 사항 (한 줄에 하나)", value="\n".join(latest_state.unresolved_issues) if latest_state else "")
            changes = st.text_area("연구자가 인지한 최근 근거 변화 (한 줄에 하나)", value="\n".join(latest_state.recent_evidence_changes) if latest_state else "")
            confidence = st.select_slider(
                "현재 확신 수준", options=["low", "medium", "high"],
                value=latest_state.confidence if latest_state else "medium",
            )
        if st.button("M2 연구 검토와 Intent 후보 만들기", disabled=not question.strip(), type="primary"):
            state = ResearchState(
                question=question, current_hypothesis=hypothesis or "아직 명시되지 않았습니다.",
                constraints=_lines(constraints), unresolved_issues=_lines(unresolved),
                recent_evidence_changes=_lines(changes), confidence=confidence,
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
                        st.write(f"**기대 근거:** {intent.expected_evidence}")
                        st.write(f"**완료 조건:** {intent.completion_condition}")
                st.caption("승인·보완 요청·반려는 연구위원 홈의 승인함에서 다중 처리합니다.")
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
                report = result.text or "## 핵심 요약\nM1의 최신 업데이트를 연구 질문과 연결해 검토해야 합니다.\n\n## 지식 공백\n- 추가 해석을 위한 연구자 판단이 필요합니다."
                record_update_report(ledger, state, report, updates)
                st.session_state["p4-update-report"] = report
                if not result.ok and use_ollama:
                    st.warning(f"Ollama 초안 대신 기본 요약을 기록했습니다: {result.error}")
        if "p4-update-report" in st.session_state:
            st.subheader("M2 · M1 새 정보 요약")
            st.markdown(st.session_state["p4-update-report"])


def external_advisory(model: str, use_ollama: bool, semantic: bool, embedding_model: str) -> None:
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
            f"실제 문제: {question}\n\n답할 수 있는 범위: {expertise} 관점의 구조·근거·조건 검토.\n\n"
            "범위 밖/전제: 개별 조직의 재무·법률·현장 사실은 별도 확인이 필요합니다.\n\n"
            "답변 전략: 범위를 확인한 뒤 승인 지식에 근거한 조건부 자문을 제공합니다."
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
    st.sidebar.title("도메인 전문 연구위원")
    model = st.sidebar.text_input("Ollama 모델", value="gemma4:e4b")
    use_ollama = st.sidebar.checkbox("Ollama 초안 사용", value=True)
    semantic = st.sidebar.checkbox("P3 Ollama 임베딩 검색", value=False)
    embedding_model = st.sidebar.text_input("임베딩 모델", value="nomic-embed-text", disabled=not semantic)
    connected, status = ollama_status(model)
    st.sidebar.caption(f"Ollama · {status}")
    if use_ollama and not connected:
        st.sidebar.warning("초안 없이도 P1·P2 흐름은 동작합니다.")
    screen = st.sidebar.radio("화면", ["연구위원 홈", "M1 자료·탐색", "M2 연구 방향", "외부 자문", "지식 관리", "개발·프롬프트"])
    if screen == "연구위원 홈":
        home(model, use_ollama)
    elif screen == "M1 자료·탐색":
        m1_screen(model, use_ollama, semantic, embedding_model)
    elif screen == "M2 연구 방향":
        m2_screen(model, use_ollama, semantic, embedding_model)
    elif screen == "지식 관리":
        management_screen()
    elif screen == "개발·프롬프트":
        render_developer_screen(memory.all(), model, use_ollama, EXTRACTION_CACHE)
    else:
        external_advisory(model, use_ollama, semantic, embedding_model)


if __name__ == "__main__":
    main()
