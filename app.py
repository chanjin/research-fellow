from __future__ import annotations

import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

import streamlit as st

from research_fellow.llm import OllamaDraftResult, gemini_api_available, gemini_draft_result, ollama_draft, ollama_draft_result, ollama_status, set_llm_audit_log_path, set_llm_audit_logger
from research_fellow.application.claim_curation import (
    build_simple_claim_cards, discovery_prompt, parse_candidate_claims, submit_claim_cards,
)
from research_fellow.application.advising import (
    auto_exploration_candidates, auto_rq_priority_prompt, create_exploration_intent_for_rq,
    direction_prompt, dispatch_top_research_questions, draft_research_direction, latest_research_state,
    parse_research_question_suggestions, parse_rq_priority_assessment, recent_knowledge_updates, recent_research_questions,
    parse_research_context_mapping, record_research_direction, record_update_report, research_context_mapping_prompt,
    store_research_question_candidates,
)
from research_fellow.application.advisory_workflow import (
    AdvisoryPlan, advisory_plan_prompt, advisory_synthesis_prompt, collect_evidence_clusters,
    deterministic_advisory, deterministic_subquestion_judgment, parse_advisory_plan,
    subquestion_judgment_prompt,
)
from research_fellow.application.episodic_memory import recall_act_spec, recall_context, store_advisory_episode, store_researcher_curation_episode
from research_fellow.application.prompt_tasks import knowledge_update_report_prompt, research_question_suggestions_prompt
from research_fellow.application.meaning_summary import (
    attach_reports, build_fact_groups, delta_inputs, delta_meaning_summary_prompt, deterministic_delta_summary,
    latest_summary, meaning_summary_prompt, record_delta_summary,
)
from research_fellow.application.search_profiles import (
    abstract_relevance_prompt, attach_relevance, is_english_search_term, keyword_prompt, parse_keyword_plan, run_profile, shortlist_candidates,
)
from research_fellow.application.paper_batch import process_top_papers
from research_fellow.application.auto_literature import execute_auto_literature_review
from research_fellow.application.research_cycle import execute_auto_research_cycle
from research_fellow.application.manual_recovery import external_recovery_prompt, validate_external_response
from research_fellow.application.m2_threads import build_m2_thread_review_prompt, extract_knowledge_gaps, extract_refined_question, validate_manual_m2_report
from research_fellow.application.paper_shelf import StoredPaperUpload, document_from_shelf_path, ensure_shelf_pdf, store_paper_upload, suggested_paper_labels
from research_fellow.application.paper_reading import parse_reading_questions, parse_reading_summary, reading_prompt, unconsumed_reading_sections
from research_fellow.application.ontology import ontology_context_dot, ontology_dot, search_cards_for_ontology
from research_fellow.application.duplicate_review import similar_approved_cards
from research_fellow.application.management import delete_knowledge_card, delete_knowledge_relation
from research_fellow.application.relations import (
    RELATION_TYPES, create_relation_candidate, lineage_dot, lineage_overview_prompt,
    parse_relation_batch_drafts, relation_batch_prompt,
)
from research_fellow.infrastructure.document_reader import extract_document, infer_bibliographic_metadata
from research_fellow.infrastructure.web_reader import WebPageExtractionError, fetch_web_page
from research_fellow.infrastructure.retrieval import KnowledgeRetriever, RetrievalResult
from research_fellow.infrastructure.episodic_retrieval import EpisodicRetriever
from research_fellow.infrastructure.prompt_renderer import render_prompt
from research_fellow.domain.episodes import EpisodicMemory
from research_fellow.memory import KnowledgeMemory, RelationMemory
from research_fellow.services import (
    complete_intent,
    create_external_case,
    decide_request,
    request_curation_intent,
)
from research_fellow.storage import Ledger
from research_fellow.workspace_sync import WorkspaceSync
from research_fellow.ui.developer import render_developer_screen
from research_fellow.domain.research import ResearchState


ROOT = Path(__file__).parent
DATA = Path(os.environ.get("RESEARCH_FELLOW_DATA_DIR", ROOT / "data")).expanduser()
CACHE = Path(os.environ.get("RESEARCH_FELLOW_CACHE_DIR", ROOT / ".cache" / "research-fellow")).expanduser()
DATA.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)
EXTRACTION_CACHE = CACHE / "extracted_documents"
LOCAL_DB = DATA / "research_fellow.db"
ledger = Ledger(LOCAL_DB)
set_llm_audit_logger(ledger.record_llm_call)
set_llm_audit_log_path(CACHE / "logs" / "llm_calls.jsonl")
# SQLite is now the canonical durable store. Existing JSONL files are imported
# once when the corresponding SQLite table is empty, then left untouched as legacy backup.
memory = KnowledgeMemory(LOCAL_DB, legacy_path=DATA / "knowledge_cards.jsonl")
relations = RelationMemory(LOCAL_DB, legacy_path=DATA / "knowledge_relations.jsonl")
retriever = KnowledgeRetriever(CACHE / "retrieval_index.json")
episodic_retriever = EpisodicRetriever(CACHE / "episodic_retrieval_index.json")


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
    drafted = llm_draft(prompt, model, use_ollama)
    if drafted:
        return drafted
    return f"""**Current interpretation**: {question}

**Available evidence**\n{bullet_evidence(results)}

**Recommendation**: Use this evidence as a starting point, while checking conditions, counterexamples, and application context separately.

**Next decision**: If more evidence is needed, create an M1 curation intent for researcher approval."""


def search_knowledge(query: str, semantic: bool, embedding_model: str, limit: int = 6) -> list[RetrievalResult]:
    return retriever.search(memory.all(), query, limit=limit, semantic=semantic, embedding_model=embedding_model)


def selected_llm_provider(workload: str = "internal") -> str:
    """Keep source-paper handling separate from internal knowledge work."""
    key = "llm-provider-paper" if workload == "paper" else "llm-provider-internal"
    return str(st.session_state.get(key, "ollama"))


def llm_draft_result(
    prompt: str, model: str, use_ollama: bool, profile: str | None = None,
    overrides: dict[str, object] | None = None, on_chunk: Callable[[str], None] | None = None,
) -> OllamaDraftResult:
    """Route paper source work and internal knowledge work independently."""
    workload = "paper" if profile in {"paper_reading", "full_text_similarity"} else "internal"
    if selected_llm_provider(workload) == "gemini":
        return gemini_draft_result(prompt, profile=profile)
    return ollama_draft_result(prompt, model, use_ollama, profile=profile, overrides=overrides, on_chunk=on_chunk)


def llm_draft(prompt: str, model: str, use_ollama: bool, profile: str | None = None) -> str | None:
    return llm_draft_result(prompt, model, use_ollama, profile=profile).text


def paper_draft_result(
    prompt: str, model: str, use_ollama: bool, profile: str, on_chunk: Callable[[str], None] | None = None,
) -> OllamaDraftResult:
    return llm_draft_result(prompt, model, use_ollama, profile=profile, on_chunk=on_chunk)


def persist_paper_reading_output(ledger: Ledger, paper: dict[str, Any], analysis: dict[str, Any], question: str, output: str, source_label: str) -> tuple[list[dict[str, Any]], list[str]]:
    """Persist API and researcher-pasted chat outputs through the same parser."""
    summary = parse_reading_summary(output)
    raw_output = output if source_label == "LLM API" else f"[생성 경로: {source_label}]\n\n{output}"
    ledger.save_paper_analysis(paper["paper_id"], research_question=question, summary=summary, reading_raw_output=raw_output, researcher_note=analysis.get("researcher_note", ""), generated=True)
    ledger.update_shelf_paper(paper["paper_id"], shelf_status=paper["shelf_status"], reading_status="read")
    return parse_reading_questions(output), unconsumed_reading_sections(output)


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


def show_evidence_clusters(clusters: list[tuple[object, object]]) -> None:
    """Show the algorithm-selected local graph neighbourhood, not an LLM guess."""
    for subquestion, cluster in clusters:
        with st.expander(f"근거 클러스터 · {subquestion.question}"):
            if not cluster.members:
                st.warning("직접 일치 카드나 연결된 승인 카드가 없습니다.")
                continue
            for member in cluster.members:
                card = member.result.card
                route = "시드 카드" if member.distance == 0 else " → ".join(member.relation_types)
                st.markdown(f"**[{card['card_id']}] {card['title']}**")
                st.caption(f"{member.distance}-hop · {route} · {member.result.reason}")
                st.write(card["claim"])


def show_recalled_episodes(act_spec: object) -> None:
    if not act_spec.recalled_episodes:
        st.caption("유사한 과거 질문·자문 사례가 리콜되지 않았습니다. 새 계획 경로로 진행합니다.")
        return
    st.markdown(f"**일화 리콜 · {act_spec.response_strategy}**")
    for recall in act_spec.recalled_episodes:
        episode = recall.episode
        with st.expander(f"[{episode.episode_id}] 유사도 {recall.score:.2f} · {episode.episode_type}"):
            st.write(episode.situation_summary)
            st.caption(f"과거 판단: {episode.decision_question}")
            st.caption(f"과거 답변 요약(현재 근거가 아닌 선례): {episode.answer_summary[:700]}")
            if episode.unresolved_items:
                st.caption("당시 미결 사항: " + "; ".join(episode.unresolved_items))


def execute_plan_first_advisory(
    plan: AdvisoryPlan, context: str, recipient: str, model: str, use_ollama: bool, semantic: bool, embedding_model: str,
) -> tuple[list[tuple[object, object]], list[str], str]:
    """LLMs judge and synthesize; card selection remains deterministic in retriever.cluster."""
    clusters = collect_evidence_clusters(
        plan, retriever, memory.all(), ledger.active_knowledge_relations({card["card_id"] for card in memory.all()}),
        context=context, semantic=semantic, embedding_model=embedding_model,
    )
    judgments: list[str] = []
    for subquestion, cluster in clusters:
        draft = llm_draft(subquestion_judgment_prompt(subquestion, cluster, context), model, use_ollama)
        judgments.append(draft or deterministic_subquestion_judgment(subquestion, cluster))
    answer = llm_draft(advisory_synthesis_prompt(plan, judgments, [cluster for _, cluster in clusters], recipient), model, use_ollama)
    return clusters, judgments, answer or deterministic_advisory(plan, judgments)


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
    if subject_type == "ontology_candidate":
        candidate = payload.get("ontology_candidate", {})
        st.markdown(f"**온톨로지 후보 · {candidate.get('statement', '')}**")
        st.caption(f"출처 논문: {candidate.get('paper_title', '미상')}")
        if candidate.get("evidence"):
            st.write("원문 근거: " + " · ".join(candidate["evidence"]))
        if candidate.get("researcher_comment"):
            st.caption(f"제안 메모: {candidate['researcher_comment']}")
        st.info("승인하면 이 일반화는 M1 관계·계보 정리의 온톨로지 정리 대상으로 남습니다. 아직 자동 추론 규칙으로 적용되지는 않습니다.")
        return
    if subject_type == "knowledge_card":
        card = payload.get("card", {})
        st.markdown(f"**주장(Claim) 후보 · {card.get('title', '후보 지식카드')}**")
        st.write(f"제안 주장: {card.get('claim', '')}")
        if card.get("explanation"):
            st.write(f"보충 설명: {card['explanation']}")
        if card.get("labels"):
            st.caption(f"레이블: {', '.join(card['labels'])}")
        provenance = card.get("provenance", {})
        if provenance:
            st.caption(f"출처: {provenance.get('source_name', '미상')}")
        if card.get("evidence_excerpt"):
            st.caption(f"근거 발췌: {card['evidence_excerpt']}")
        st.info("승인하면 이 주장이 지식카드로 저장되고 M2에 지식 업데이트가 통지됩니다.")
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



def show_auto_literature_reports() -> None:
    reports = [
        item for item in ledger.phenomena(recipient="researcher", type_="advice_report")
        if item.get("subject_type") == "auto_literature_report"
    ]
    if not reports:
        st.info("아직 자동 문헌탐색 보고가 없습니다.")
        return
    st.caption(f"자동 탐색 보고 {len(reports)}건 · 최신순. 초록/본문 비교 결과는 연구자 검토 전까지 승인 지식이 아닙니다.")
    for item in reports[:20]:
        payload = item.get("payload") or {}
        status = "완료" if item.get("status") == "completed" else "실패"
        with st.expander(f"{status} · {item['created_at'][:16].replace('T', ' ')} · {payload.get('title', 'M1 자동 문헌탐색 보고')}"):
            st.caption(
                f"초록 검토 {payload.get('abstract_review_count', 0)}편 · 본문 비교 {payload.get('fulltext_review_count', 0)}편 · Intent {payload.get('intent_id', '')}"
            )
            st.markdown(payload.get("report", "보고서 본문이 없습니다."))
            top_papers = payload.get("top_papers", [])
            if top_papers:
                st.markdown("**상위 논문 메타데이터**")
                for paper in top_papers:
                    citations = paper.get("citation_count")
                    citation_text = "확인 불가" if citations is None else f"{int(citations):,}회"
                    st.write(f"- **{paper.get('title', '')}** · {str(paper.get('published', ''))[:4]} · 인용 {citation_text} · 본문 적합성 {paper.get('full_text_similarity', 0)}/100")
                    if paper.get("url"):
                        st.caption(paper["url"])
                    if st.button("서재함에 추가", key=f"auto-report-shelf-{item['phenomenon_id']}-{paper.get('source_id', '')}"):
                        saved = ledger.upsert_shelf_paper({
                            "title": paper.get("title", ""), "authors": [],
                            "publication_year": str(paper.get("published", ""))[:4], "source_url": paper.get("url", ""),
                            "source_id": paper.get("source_id", ""), "pdf_path": paper.get("pdf_path", ""),
                            "shelf_status": "reference", "reading_status": "unread", "asset_type": "paper", "intake_source": "auto_search",
                        })
                        st.success(f"서재함에 추가했습니다: {saved['title']}")



def show_auto_retry_tasks(model: str, use_ollama: bool) -> None:
    failures = ledger.auto_research_failures(status="needs_attention", limit=50)
    if not failures:
        st.success("현재 Retry가 필요한 자동 연구 작업이 없습니다.")
        return
    st.caption(f"중간 실패로 연구자 확인이 필요한 작업 {len(failures)}건입니다. 기존 성공 결과와 로그는 보존하면서 해당 자동 작업을 다시 실행할 수 있습니다.")
    stage_labels = {
        "rq_generation": "새 지식 → 연구질문 생성",
        "rq_prioritization": "연구질문 중요도 평가",
        "search_strategy": "영문 검색전략 생성",
        "search_strategy_legacy": "영문 검색키워드 복구",
        "abstract_screening": "초록 스크리닝",
        "fulltext_review": "상위 논문 본문 검토",
        "synthesis": "문헌탐색 종합 보고",
    }

    def rerun_failure(failure: dict[str, object], *, abstract_batch_size: int = 20):
        context = dict(failure.get("context") or {})
        if failure.get("intent_id"):
            event = ledger.prepare_intent_for_retry(str(failure["intent_id"]))
            if not event:
                return None
            return execute_auto_literature_review(
                ledger, event, CACHE,
                keyword_drafter=lambda prompt: llm_draft(prompt, model, use_ollama),
                abstract_reviewer=lambda prompt: llm_draft(prompt, model, use_ollama, profile="abstract_triage"),
                fulltext_drafter=lambda prompt: paper_draft_result(prompt, model, use_ollama, "full_text_similarity").text,
                synthesis_drafter=lambda prompt: llm_draft(prompt, model, use_ollama),
                resume_run_id=str(failure.get("run_id") or context.get("resume_run_id") or ""),
                abstract_batch_size=abstract_batch_size,
            )
        return execute_auto_research_cycle(
            ledger, memory.all(), CACHE,
            rq_drafter=lambda prompt: llm_draft(prompt, model, use_ollama),
            priority_drafter=lambda prompt: llm_draft(prompt, model, use_ollama),
            keyword_drafter=lambda prompt: llm_draft(prompt, model, use_ollama),
            abstract_reviewer=lambda prompt: llm_draft(prompt, model, use_ollama, profile="abstract_triage"),
            fulltext_drafter=lambda prompt: paper_draft_result(prompt, model, use_ollama, "full_text_similarity").text,
            synthesis_drafter=lambda prompt: llm_draft(prompt, model, use_ollama),
            resume_run_id=str(failure.get("run_id") or ""),
        )
    for failure in failures:
        context = failure.get("context") or {}
        stage = failure.get("stage", "")
        title = stage_labels.get(stage, stage or "자동 연구 작업")
        with st.expander(f"주의 필요 · {title} · {failure.get('created_at', '')[:16].replace('T', ' ')}"):
            left, right = st.columns(2)
            left.write(f"**실패 원인:** `{failure.get('error_type', 'unknown')}`")
            right.write(f"**자동 Retry:** {failure.get('attempt_count', 0)} / 3")
            st.write(failure.get("error_message") or "오류 메시지가 없습니다.")
            st.info(f"권장 Retry 방법: {failure.get('recommended_action') or '실패 단계부터 다시 시도하세요.'}")
            if context.get("retry_hint"):
                st.caption(f"추가 제안: {context['retry_hint']}")
            if failure.get("intent_id"):
                st.caption(f"M1 Intent: {failure['intent_id']} · 앞 단계 결과는 유지하고 실패 단계부터 재개합니다.")
                linked_rqs = ledger.research_questions_for_intent(str(failure["intent_id"]))
                if linked_rqs:
                    st.caption("연결 RQ: " + " · ".join(str(item.get("question", ""))[:60] for item in linked_rqs))
                    st.info("M2 연구상태 검토와 RQ 생성은 이미 완료된 상태입니다. 따라서 새 지식카드가 입력함에서 사라진 것은 정상이며, 이 Retry는 M1 후속 문헌탐색만 재개합니다.")
            elif context.get("kind") == "auto_cycle":
                st.caption("M2 자동 연구 사이클의 RQ 생성/중요도 평가 단계에서 실패했습니다. 이 경우 새 지식카드는 미처리 상태로 남아 다시 시작할 수 있습니다.")

            if stage == "abstract_screening":
                st.caption(f"보존된 checkpoint: 초록 {context.get('batch_no', '?')}번째 batch 직전까지. 기존 검색전략·검색 후보·완료 batch는 다시 만들지 않습니다.")

            with st.expander("외부 LLM으로 수동 복구", expanded=False):
                st.caption("로컬/무료 LLM이 반복 실패한 경우 아래 프롬프트를 ChatGPT·Claude 등 외부 LLM에 붙여 넣고, 응답을 다시 가져와 이 단계만 복구할 수 있습니다.")
                recovery_text = external_recovery_prompt(failure)
                st.text_area(
                    "외부 LLM용 복구 프롬프트",
                    value=recovery_text,
                    height=360,
                    key=f"manual-recovery-prompt-{failure['failure_id']}",
                    help="영역 높이는 고정되어 있으며 긴 프롬프트는 내부 스크롤로 확인할 수 있습니다.",
                )
                external_response = st.text_area(
                    "외부 LLM 응답 붙여넣기",
                    value="",
                    height=280,
                    key=f"manual-recovery-response-{failure['failure_id']}",
                    placeholder="외부 LLM의 전체 응답을 여기에 붙여 넣으세요.",
                )
                st.caption("응답은 현재 stage의 기대 형식으로 검증한 뒤에만 checkpoint에 반영됩니다.")
                if st.button("외부 응답 검증 · 이 단계부터 재개", key=f"apply-manual-recovery-{failure['failure_id']}", type="primary"):
                    ok, validation_message = validate_external_response(stage, external_response, context)
                    ledger.record_manual_recovery_attempt(
                        failure["failure_id"],
                        run_id=str(failure.get("run_id") or ""),
                        stage=stage,
                        item_key=str(failure.get("item_key") or ""),
                        response_text=external_response,
                        validation_status="valid" if ok else "invalid",
                        validation_message=validation_message,
                        applied=ok,
                    )
                    if not ok:
                        st.error(f"외부 응답 검증 실패: {validation_message}")
                    else:
                        ledger.set_manual_recovery_override(
                            str(failure.get("run_id") or ""), stage=stage,
                            item_key=str(failure.get("item_key") or ""), response=external_response,
                        )
                        st.success("외부 응답이 검증되었습니다. 이 응답을 실패 stage의 LLM 결과로 사용해 재개합니다.")
                        # 기존 failure는 재개 전에 닫아 두어, 같은 run의 완료 판정에서
                        # 이미 수동 복구한 실패가 다시 unresolved로 집계되지 않게 합니다.
                        ledger.resolve_auto_research_failure(failure["failure_id"], status="manual_resolved")
                        with st.spinner("수동 복구 결과를 반영하고 다음 단계부터 계속 수행합니다."):
                            result = rerun_failure(failure)
                        if result and result.get("status") == "completed":
                            st.success("수동 복구 후 자동 연구 작업이 완료되었습니다.")
                        elif result and result.get("status") in {"needs_attention", "failed"}:
                            st.warning("해당 실패 stage는 수동으로 복구했지만 이후 단계에서 추가 확인이 필요합니다. 새 처리 필요 작업을 확인하세요.")
                        else:
                            st.info("수동 복구 결과를 반영했습니다. 현재 작업 상태를 다시 확인하세요.")
                        st.rerun()

            if st.button("실패 단계부터 재시도", key=f"retry-auto-{failure['failure_id']}", type="primary"):
                with st.spinner("보존된 checkpoint를 사용해 실패 단계부터 다시 수행하고 있습니다."):
                    result = None
                    if failure.get("intent_id"):
                        event = ledger.prepare_intent_for_retry(str(failure["intent_id"]))
                        if event:
                            result = execute_auto_literature_review(
                                ledger, event, CACHE,
                                keyword_drafter=lambda prompt: llm_draft(prompt, model, use_ollama),
                                abstract_reviewer=lambda prompt: llm_draft(prompt, model, use_ollama, profile="abstract_triage"),
                                fulltext_drafter=lambda prompt: paper_draft_result(prompt, model, use_ollama, "full_text_similarity").text,
                                synthesis_drafter=lambda prompt: llm_draft(prompt, model, use_ollama),
                                resume_run_id=str(failure.get("run_id") or context.get("resume_run_id") or ""),
                            )
                    else:
                        result = execute_auto_research_cycle(
                            ledger, memory.all(), CACHE,
                            rq_drafter=lambda prompt: llm_draft(prompt, model, use_ollama),
                            priority_drafter=lambda prompt: llm_draft(prompt, model, use_ollama),
                            keyword_drafter=lambda prompt: llm_draft(prompt, model, use_ollama),
                            abstract_reviewer=lambda prompt: llm_draft(prompt, model, use_ollama, profile="abstract_triage"),
                            fulltext_drafter=lambda prompt: paper_draft_result(prompt, model, use_ollama, "full_text_similarity").text,
                            synthesis_drafter=lambda prompt: llm_draft(prompt, model, use_ollama),
                            resume_run_id=str(failure.get("run_id") or ""),
                        )
                if result and result.get("status") in {"completed", "no_new_information"}:
                    ledger.resolve_auto_research_failure(failure["failure_id"])
                    st.success("Retry가 완료되었습니다. 기존 앞 단계 결과를 유지한 채 최신 결과를 반영합니다.")
                    st.rerun()
                else:
                    st.warning("Retry 후에도 처리가 완료되지 않았습니다. 새 실패 기록과 권장 조치를 확인하세요.")

            if stage == "abstract_screening" and st.button("추천 설정으로 Retry · batch 10편", key=f"retry-small-batch-{failure['failure_id']}"):
                event = ledger.prepare_intent_for_retry(str(failure.get("intent_id", ""))) if failure.get("intent_id") else None
                if event:
                    with st.spinner("완료된 batch를 유지하고 실패 지점부터 10편 단위로 재시도합니다."):
                        result = execute_auto_literature_review(
                            ledger, event, CACHE,
                            keyword_drafter=lambda prompt: llm_draft(prompt, model, use_ollama),
                            abstract_reviewer=lambda prompt: llm_draft(prompt, model, use_ollama, profile="abstract_triage"),
                            fulltext_drafter=lambda prompt: paper_draft_result(prompt, model, use_ollama, "full_text_similarity").text,
                            synthesis_drafter=lambda prompt: llm_draft(prompt, model, use_ollama),
                            resume_run_id=str(failure.get("run_id") or context.get("resume_run_id") or ""),
                            abstract_batch_size=10,
                        )
                    if result.get("status") == "completed":
                        ledger.resolve_auto_research_failure(failure["failure_id"])
                        st.success("추천 설정 Retry가 완료되었습니다.")
                        st.rerun()
                    else:
                        st.warning("10편 단위 Retry 후에도 완료되지 않았습니다. 새 실패 기록을 확인하세요.")

            if failure.get("intent_id"):
                linked_rqs = ledger.research_questions_for_intent(str(failure["intent_id"]))
                review_ids = []
                for rq in linked_rqs:
                    review_ids.extend(str(item.get("review_id", "")) for item in ledger.research_question_sources(str(rq.get("rq_id", ""))) if item.get("review_id"))
                review_ids = list(dict.fromkeys(review_ids))
                if review_ids:
                    rollback_review_id = review_ids[0]
                    with st.expander("전체 M2 검토를 되돌려 새 지식부터 다시 시작하려면"):
                        st.warning("이 기능은 기본 복구가 아닙니다. RQ와 Intent 기록은 감사 이력으로 남기고, 해당 검토 배치의 지식카드만 다시 '미처리 새 정보'로 돌립니다.")
                        if st.button("M2 검토 배치 되돌리기", key=f"rollback-review-{failure['failure_id']}-{rollback_review_id}"):
                            ledger.rollback_research_state_review(rollback_review_id)
                            st.success(f"{rollback_review_id}를 되돌렸습니다. 연결된 카드가 M1 새 정보 입력함에 다시 나타납니다.")
                            st.rerun()


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
        result = llm_draft_result(delta_meaning_summary_prompt(delta), model, use_ollama)
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
    # Paper reading already includes the researcher's evidence review and card
    # authoring. It therefore creates knowledge directly, not another approval
    # task. Legacy card requests remain in the ledger but are not work items.
    pending = [
        item for item in ledger.phenomena(recipient="researcher", type_="decision_request", status="proposed")
        if item["subject_type"] != "knowledge_card"
    ]
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

    st.subheader("처리 필요 작업 · Retry")
    show_auto_retry_tasks(model, use_ollama)

    st.subheader("M1 자동 문헌탐색 보고")
    show_auto_literature_reports()

    st.subheader("M2 보고서 이력")
    reports = [item for item in ledger.phenomena(recipient="researcher", type_="advice_report") if item.get("subject_type") != "auto_literature_report"]
    show_m2_report_history(reports)

    st.subheader("승인 지식 검색")
    query = st.text_input("연구질문 또는 키워드", key="home-knowledge-query", placeholder="예: LLM generated design feasibility")
    if query.strip():
        results = search_knowledge(query, semantic, embedding_model, limit=10)
        show_retrieval_results(results)
        if results:
            options = {f"{item.card['title']} · {item.card['provenance'].get('source_name', '')}": item.card["card_id"] for item in results}
            chosen = st.multiselect("P2 계보 후보에 담을 카드 (최대 20개)", list(options), key="home-lineage-cards")
            if st.button("선택 카드를 P2 계보 목록으로 전달", disabled=not chosen, key="home-send-lineage"):
                current = st.session_state.setdefault("p2-lineage-selected-ids", [])
                for card_id in (options[label] for label in chosen):
                    if card_id not in current and len(current) < 20:
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


def render_curation_precedents(situation: str, *, episode_types: set[str], semantic: bool, embedding_model: str) -> None:
    """Put prior researcher decisions before the next curation action."""
    episodes = [
        EpisodicMemory.model_validate(item)
        for item in ledger.episode_memories()
        if item.get("episode_type") in episode_types
    ]
    recalls = episodic_retriever.recall(episodes, situation, limit=3, semantic=semantic, embedding_model=embedding_model)
    if not recalls:
        return
    st.markdown("**유사한 과거 연구자 작업**")
    st.caption("과거의 등록·보류·무관 판단을 비교하되, 현재 논문의 근거와 조건은 다시 확인하세요.")
    cards = {card["card_id"]: card for card in memory.all()}
    for recall in recalls:
        episode = recall.episode
        decision_text = episode.decision_question
        decision_label = "등록" if "등록" in decision_text else "보류" if "보류" in decision_text else "무관" if "무관" in decision_text or "등록하지" in decision_text else "검토"
        age = _relative_time(episode.updated_at)
        episode_cards = [cards[card_id] for card_id in episode.evidence_card_ids if card_id in cards]
        claim_title = episode_cards[0]["title"] if episode_cards else _episode_claim_title(episode.answer_summary, episode.situation_summary)
        with st.expander(f"{age} · {decision_label} · {claim_title} · 유사도 {recall.score:.2f}"):
            st.caption(f"수행 시점: {episode.updated_at[:10]} · 당시 판단: {decision_text}")
            st.markdown("**작업 상황**")
            st.write(episode.situation_summary)
            st.markdown("**결정 결과**")
            st.write(episode.answer_summary)
            if episode.advisory_plan:
                st.caption("수행 단계: " + " → ".join(episode.advisory_plan))
            if episode_cards:
                st.markdown("**당시 등록·참조 지식카드**")
                for card in episode_cards:
                    render_knowledge_card(card, key_prefix=f"precedent-{episode.episode_id}")
            if episode.unresolved_items:
                st.caption("미결 사항: " + " · ".join(episode.unresolved_items))


def _relative_time(value: str) -> str:
    try:
        days = (datetime.now(UTC).date() - datetime.fromisoformat(value).date()).days
    except ValueError:
        return "시점 미상"
    return "오늘" if days <= 0 else "어제" if days == 1 else f"{days}일 전" if days < 30 else f"{days // 30}개월 전"


def _episode_claim_title(answer_summary: str, situation_summary: str) -> str:
    match = re.search(r"등록 카드:\s*([^\n]+)", answer_summary)
    if match:
        return match.group(1).strip()
    match = re.search(r"읽기 질문:\s*([^\n]+)", situation_summary)
    return match.group(1).strip()[:72] if match else "주장 미상"


def render_knowledge_card(card: dict[str, object], *, key_prefix: str = "card") -> None:
    """Compact, provenance-first presentation shared by cards and recalled precedents."""
    st.write(f"**{card.get('title', '제목 없음')}**")
    st.write(card.get("claim", "주장 없음"))
    source = (card.get("provenance") or {}).get("source_name", "출처 미상") if isinstance(card.get("provenance"), dict) else "출처 미상"
    st.caption(f"출처: {source} · 근거 수준: {card.get('evidence_level', '미상')}")
    if card.get("labels") or card.get("concepts"):
        st.caption(f"레이블: {', '.join(card.get('labels', [])) or '미입력'} · 개념: {', '.join(card.get('concepts', [])) or '미입력'}")
    st.caption(f"적용 대상: {', '.join(card.get('applies_to', [])) or '미입력'} · 조건: {card.get('conditions') or '미입력'}")
    with st.expander("근거·한계 보기", expanded=False):
        st.write(card.get("evidence_excerpt") or "근거 발췌 미입력")
        st.caption("한계·유보: " + str(card.get("limits") or "미입력"))
        supporting = card.get("supporting_evidence", [])
        if supporting:
            st.caption(f"추가 보강 근거 {len(supporting)}건")
            for item in supporting:
                st.write(f"- {item.get('source_name', '출처 미상')} · {item.get('evidence_excerpt', '')}")


def render_ontology_workspace(semantic: bool, embedding_model: str) -> None:
    """Facet-aware, multi-type ontology builder with contextual graph feedback."""
    st.subheader("온톨로지")
    st.caption("승인 지식카드에 여러 타입을 부여하고, Facet으로 타입을 묶습니다. 타입 간 관계는 Type↔Type 사이에 정의합니다.")
    cards = memory.all()
    cards_by_id = {card["card_id"]: card for card in cards}
    approved_relations = ledger.active_knowledge_relations()
    builder_tab, map_tab = st.tabs(["Ontology Builder", "Ontology Map"])

    def _sync_card_types(card_id: str, widget_key: str) -> None:
        ledger.set_card_ontology_types(card_id, list(st.session_state.get(widget_key, [])))
        st.session_state["ontology-focus-card-id"] = card_id
        st.session_state["ontology-flash"] = "타입 지정을 반영했습니다."

    with builder_tab:
        if flash := st.session_state.pop("ontology-flash", None):
            st.success(flash)
        search_col, graph_col = st.columns([1.45, 1], gap="large")
        with search_col:
            st.markdown("### 지식카드 탐색 · 타입 부여")
            query = st.text_input("카드 검색", placeholder="개념, 방법, 문제, 관계 맥락", key="ontology-card-query")
            f1, f2, f3 = st.columns(3)
            use_embedding = f1.checkbox("임베딩", value=semantic, key="ontology-use-embedding")
            use_keyword = f2.checkbox("키워드", value=True, key="ontology-use-keyword")
            use_relations = f3.checkbox("관계 확장", value=True, key="ontology-use-relations")
            sort_mode = st.radio("결과 정렬", ["관련도순", "미분류 카드 우선"], horizontal=True, key="ontology-sort-mode")
            if st.button("카드 찾기", key="ontology-search", disabled=not query.strip(), type="primary"):
                hits = search_cards_for_ontology(retriever, cards, approved_relations, query,
                    use_keyword=use_keyword, use_embedding=use_embedding, use_relations=use_relations,
                    embedding_model=embedding_model, limit=24)
                st.session_state["ontology-search-results"] = [{"card_id": h.result.card["card_id"], "score": round(h.result.score, 3), "reason": h.result.reason, "distance": h.relation_distance} for h in hits]

            facets = ledger.ontology_facets(); facet_by_id = {x["facet_id"]: x for x in facets}
            types = ledger.ontology_types(); type_by_id = {x["type_id"]: x for x in types}; type_ids = [x["type_id"] for x in types]
            current_types_by_card = {card["card_id"]: ledger.ontology_types_for_card(card["card_id"]) for card in cards}
            results = list(st.session_state.get("ontology-search-results", []))
            if sort_mode == "미분류 카드 우선":
                results.sort(key=lambda item: (bool(current_types_by_card.get(item["card_id"])), -float(item.get("score", 0))))
            if not results:
                st.info("검색어를 입력해 승인 지식카드를 찾으세요.")

            def _type_label(type_id: str) -> str:
                item = type_by_id.get(type_id, {})
                return f"{item.get('facet_name') or 'Facet 미지정'} · {item.get('name', type_id)}"

            for index, result in enumerate(results, 1):
                card = cards_by_id.get(result["card_id"])
                if not card: continue
                current_types = current_types_by_card.get(card["card_id"], [])
                current_ids = [x["type_id"] for x in current_types]
                with st.container(border=True):
                    st.markdown(f"**{index}. {card['title']}**"); st.write(card["claim"][:420])
                    relation_hint = f" · 관계 {result['distance']}-hop" if result["distance"] else ""
                    st.caption(f"검색 {result['score']:.3f} · {result['reason']}{relation_hint}")
                    if current_types:
                        st.markdown("현재 타입: " + " · ".join(f"`{x.get('facet_name') or '미분류'} / {x['name']}`" for x in current_types))
                    else: st.caption("현재 타입: 미지정")
                    key = f"ontology-inline-types-{card['card_id']}"
                    if key not in st.session_state: st.session_state[key] = current_ids
                    else: st.session_state[key] = [v for v in st.session_state[key] if v in set(type_ids)]
                    st.multiselect("타입", type_ids, key=key, format_func=_type_label,
                        on_change=_sync_card_types, args=(card["card_id"], key), placeholder="해당하는 타입을 모두 선택")
                    c1, c2 = st.columns([2.2, 1])
                    with c1:
                        with st.expander("+ 새 타입을 만들어 이 카드에 추가", expanded=False):
                            facet_options = [None, *[x["facet_id"] for x in facets]]
                            facet_id = st.selectbox("Facet", facet_options,
                                format_func=lambda v, names=facet_by_id: "Facet 미지정" if v is None else names[v]["name"],
                                key=f"ontology-inline-new-facet-{card['card_id']}")
                            new_name = st.text_input("새 타입 이름", placeholder="예: Non-functional Requirement", key=f"ontology-inline-new-name-{card['card_id']}")
                            new_desc = st.text_area("설명", height=90, key=f"ontology-inline-new-desc-{card['card_id']}")
                            if st.button("생성하고 추가", key=f"ontology-inline-create-{card['card_id']}", disabled=not new_name.strip()):
                                try:
                                    new_type = ledger.create_ontology_type(new_name, new_desc, facet_id)
                                    ledger.set_card_ontology_types(card["card_id"], [*current_ids, new_type["type_id"]])
                                    st.session_state["ontology-focus-card-id"] = card["card_id"]
                                    st.session_state["ontology-flash"] = f"{new_type['name']} 타입을 만들고 카드에 추가했습니다."
                                    st.rerun()
                                except ValueError as error: st.error(str(error))
                    with c2:
                        if st.button("그래프에서 보기", key=f"ontology-focus-{card['card_id']}"):
                            st.session_state["ontology-focus-card-id"] = card["card_id"]
                            st.rerun()

        with graph_col:
            st.markdown("### 현재 카드의 타입 구조")
            focus_card_id = st.session_state.get("ontology-focus-card-id")
            facets = ledger.ontology_facets(); types = ledger.ontology_types(); relations = ledger.ontology_type_relations()
            type_by_id = {x["type_id"]: x for x in types}
            focus_types = ledger.ontology_types_for_card(focus_card_id) if focus_card_id else []
            focus_ids = [x["type_id"] for x in focus_types]
            if focus_card_id and focus_card_id in cards_by_id: st.caption(f"포커스 카드 · {cards_by_id[focus_card_id]['title']}")
            if focus_types:
                for item in focus_types: st.markdown(f"- **{item.get('facet_name') or 'Facet 미지정'}** → {item['name']}")
                st.graphviz_chart(ontology_context_dot(types, relations, focus_ids, facets), use_container_width=True)
                local_relations = [r for r in relations if r["source_type_id"] in set(focus_ids) or r["target_type_id"] in set(focus_ids)]
                if local_relations:
                    st.markdown("**현재 카드 타입과 연결된 관계**")
                    for r in local_relations:
                        st.caption(f"{type_by_id.get(r['source_type_id'], {}).get('name', r['source_type_id'])} → {r['relation_name']} → {type_by_id.get(r['target_type_id'], {}).get('name', r['target_type_id'])}")
                else: st.info("이 카드에 부여된 타입들은 아직 다른 타입과 연결되지 않았습니다.")
                if len(types) >= 2:
                    with st.expander("+ 타입 관계 추가", expanded=not bool(local_relations)):
                        source_id = st.selectbox("출발 타입", [x["type_id"] for x in types], format_func=lambda v: f"{type_by_id[v].get('facet_name') or '미분류'} · {type_by_id[v]['name']}", key="ontology-context-rel-source")
                        targets = [x["type_id"] for x in types if x["type_id"] != source_id]
                        target_id = st.selectbox("도착 타입", targets, format_func=lambda v: f"{type_by_id[v].get('facet_name') or '미분류'} · {type_by_id[v]['name']}", key="ontology-context-rel-target") if targets else None
                        rel_name = st.text_input("관계 이름", placeholder="예: operates_on, constrains, requires", key="ontology-context-rel-name")
                        desc = st.text_area("관계 설명", height=90, key="ontology-context-rel-desc")
                        if st.button("관계 추가", type="primary", key="ontology-context-rel-save", disabled=not rel_name.strip() or not target_id):
                            try:
                                ledger.create_ontology_type_relation(source_id, target_id, rel_name, desc); st.session_state["ontology-flash"] = "타입 관계를 추가했습니다."; st.rerun()
                            except ValueError as error: st.error(str(error))
            elif types:
                st.graphviz_chart(ontology_dot(types, relations, facets), use_container_width=True)
                st.caption("카드에 하나 이상의 타입을 지정하면 해당 타입들과 1-hop 주변 구조가 표시됩니다.")
            else: st.info("아직 타입이 없습니다. 왼쪽 검색 결과 카드에서 새 타입을 만들어 시작하세요.")

    with map_tab:
        facets = ledger.ontology_facets(); facet_by_id = {x["facet_id"]: x for x in facets}
        types = ledger.ontology_types(); relations = ledger.ontology_type_relations(); type_by_id = {x["type_id"]: x for x in types}
        st.markdown("### 전체 Ontology Map")
        if types: st.graphviz_chart(ontology_dot(types, relations, facets), use_container_width=True)
        else: st.info("타입을 먼저 정의하면 전체 타입 그래프가 표시됩니다.")
        st.markdown("### Facet 관리")
        st.caption("Facet은 Type의 메타분류입니다. 그래프의 관계 노드가 아니라 타입을 묶는 의미 축입니다.")
        with st.expander("+ Facet 추가", expanded=not bool(facets)):
            fname = st.text_input("Facet 이름", placeholder="예: Requirement, Agent Case, Target Domain", key="ontology-facet-name")
            fdesc = st.text_area("Facet 설명", height=80, key="ontology-facet-description")
            if st.button("Facet 저장", type="primary", disabled=not fname.strip(), key="ontology-facet-save"):
                try: ledger.create_ontology_facet(fname, fdesc); st.rerun()
                except ValueError as error: st.error(str(error))
        for facet in facets: st.caption(f"**{facet['name']}** · Type {facet['type_count']}개 · {facet.get('description') or '설명 없음'}")
        if types:
            st.markdown("### 타입 관리")
            for item in types:
                with st.expander(f"[{item.get('facet_name') or 'Facet 미지정'}] {item['name']} · 카드 {item['card_count']}건"):
                    st.write(item.get("description") or "설명 없음")
                    facet_options = [None, *[x["facet_id"] for x in facets]]
                    selected = st.selectbox("Facet 변경", facet_options,
                        index=facet_options.index(item.get("facet_id")) if item.get("facet_id") in facet_options else 0,
                        format_func=lambda v, names=facet_by_id: "Facet 미지정" if v is None else names[v]["name"], key=f"ontology-type-facet-{item['type_id']}")
                    if st.button("Facet 반영", key=f"ontology-type-facet-save-{item['type_id']}"):
                        try: ledger.update_ontology_type(item["type_id"], name=item["name"], description=item.get("description") or "", facet_id=selected); st.rerun()
                        except ValueError as error: st.error(str(error))
                    for card_id in ledger.ontology_card_ids(item["type_id"]): st.write(f"- {cards_by_id.get(card_id, {}).get('title', card_id)}")
        if len(types) >= 2:
            st.markdown("### 타입 간 관계 정의")
            ids = [x["type_id"] for x in types]
            source_id = st.selectbox("출발 타입", ids, format_func=lambda v: f"{type_by_id[v].get('facet_name') or '미분류'} · {type_by_id[v]['name']}", key="ontology-map-rel-source")
            targets = [v for v in ids if v != source_id]
            target_id = st.selectbox("도착 타입", targets, format_func=lambda v: f"{type_by_id[v].get('facet_name') or '미분류'} · {type_by_id[v]['name']}", key="ontology-map-rel-target")
            rel_name = st.text_input("관계 이름", placeholder="예: operates_on, constrains, requires", key="ontology-map-rel-name")
            rel_desc = st.text_area("관계 설명", key="ontology-map-rel-description")
            if st.button("타입 관계 저장", type="primary", disabled=not rel_name.strip(), key="ontology-map-rel-save"):
                try: ledger.create_ontology_type_relation(source_id, target_id, rel_name, rel_desc); st.rerun()
                except ValueError as error: st.error(str(error))
        if relations:
            st.markdown("### 정의된 관계")
            for r in relations:
                cols = st.columns([5,1]); cols[0].write(f"**{type_by_id.get(r['source_type_id'],{}).get('name',r['source_type_id'])}** → `{r['relation_name']}` → **{type_by_id.get(r['target_type_id'],{}).get('name',r['target_type_id'])}**")
                if r.get("description"): cols[0].caption(r["description"])
                if cols[1].button("삭제", key=f"delete-ontology-rel-{r['relation_id']}"): ledger.delete_ontology_type_relation(r["relation_id"]); st.rerun()

def render_paper_shelf(model: str, use_ollama: bool, semantic: bool, embedding_model: str) -> None:
    """Paper assets are read and reviewed here, not converted to claims on intake."""
    st.subheader("서재함")
    st.caption("탐색 결과와 직접 업로드 자료가 한곳에 모입니다. 카드화는 논문 읽기·연구자 첨삭 이후에만 가능합니다.")
    status_labels = {"core": "핵심 참고", "reference": "참고", "held": "보류", "excluded": "제외"}
    reading_labels = {"unread": "미읽음", "reading": "읽는 중", "read": "읽음"}
    all_papers = ledger.shelf_papers()
    reviewed_ids = {paper["paper_id"] for paper in all_papers if ledger.paper_reading_questions(paper["paper_id"]) or ledger.paper_card_ids(paper["paper_id"])}
    knowledge_ids = {paper["paper_id"] for paper in all_papers if ledger.paper_card_ids(paper["paper_id"])}
    metric_a, metric_b, metric_c = st.columns(3)
    metric_a.metric("서재함", len(all_papers))
    metric_b.metric("리뷰 진행·완료", len(reviewed_ids))
    metric_c.metric("승인 지식 연결", len(knowledge_ids))
    query = st.text_input("논문 검색", placeholder="제목, 저자, 레이블", key="paper-shelf-query")
    filter_status = st.selectbox("중요도", ["all", *status_labels], format_func=lambda value: "전체" if value == "all" else status_labels[value], key="paper-shelf-filter")
    review_filter = st.selectbox("검토·지식화 현황", ["all", "reviewed", "knowledge_linked", "not_reviewed"], format_func={"all": "전체", "reviewed": "리뷰됨", "knowledge_linked": "승인 지식 연결됨", "not_reviewed": "아직 읽기 전"}.get)
    all_labels = sorted({label for paper in all_papers for label in paper.get("labels", [])}, key=str.casefold)
    selected_labels = st.multiselect("레이블 필터 (선택한 레이블을 모두 포함)", all_labels, key="paper-shelf-label-filter")
    papers = [
        paper for paper in all_papers
        if (filter_status == "all" or paper["shelf_status"] == filter_status)
        and set(selected_labels).issubset(set(paper.get("labels", [])))
        and (not query.strip() or query.casefold() in " ".join([paper["title"], *paper.get("authors", []), *paper.get("labels", [])]).casefold())
        and (review_filter == "all" or (review_filter == "reviewed" and paper["paper_id"] in reviewed_ids) or (review_filter == "knowledge_linked" and paper["paper_id"] in knowledge_ids) or (review_filter == "not_reviewed" and paper["paper_id"] not in reviewed_ids))
    ]
    if not papers:
        st.info("등록된 논문이 없습니다. 탐색 로그의 후보를 추가하거나 원문을 직접 등록하세요.")
        return
    cards_by_id = {card["card_id"]: card for card in memory.all()}
    card_options = {f"{card['title']} [{card['card_id']}]": card["card_id"] for card in cards_by_id.values()}
    for paper in papers:
        with st.expander(f"{status_labels.get(paper['shelf_status'], paper['shelf_status'])} · {paper['title']}"):
            st.caption(f"{paper.get('publication_year') or '연도 확인 필요'} · {', '.join(paper.get('authors', [])) or '저자 확인 필요'} · {reading_labels.get(paper['reading_status'], paper['reading_status'])}")
            st.caption(f"레이블: {', '.join(paper.get('labels', [])) or '아직 없음'}")
            analysis = ledger.paper_analysis(paper["paper_id"]) or {}
            suggested_labels = suggested_paper_labels(analysis.get("reading_raw_output", ""), max_labels=10)
            if suggested_labels:
                st.caption("M1 제안 레이블: " + ", ".join(suggested_labels))
            if analysis.get("summary"):
                st.markdown("### 논문 읽기 요약")
                st.markdown(analysis["summary"])
            if analysis.get("reading_raw_output"):
                with st.expander("모델이 생성한 논문 읽기 원문 보기"):
                    st.text(analysis["reading_raw_output"])
            controls, actions = st.columns([3, 2])
            with controls:
                with st.form(f"paper-shelf-state-{paper['paper_id']}"):
                    title = st.text_input("논문 제목", value=paper["title"])
                    authors_text = st.text_input("저자 (쉼표 구분)", value=", ".join(paper.get("authors", [])))
                    publication_year = st.text_input("발행 연도", value=paper.get("publication_year", ""), max_chars=4)
                    shelf_status = st.selectbox("중요도", list(status_labels), index=list(status_labels).index(paper["shelf_status"]), format_func=lambda value: status_labels[value])
                    reading_status = st.selectbox("읽기 상태", list(reading_labels), index=list(reading_labels).index(paper["reading_status"]), format_func=lambda value: reading_labels[value])
                    state_saved = st.form_submit_button("상태 저장")
                if state_saved:
                    ledger.update_shelf_paper(
                        paper["paper_id"], shelf_status=shelf_status, reading_status=reading_status,
                        title=title,
                        authors=[author.strip() for author in authors_text.split(",") if author.strip()],
                        publication_year=publication_year.strip(),
                    )
                    st.success("서재 상태를 저장했습니다.")
                    st.rerun()
            with actions:
                if paper.get("source_url"):
                    st.link_button("원문 페이지 열기", paper["source_url"], key=f"shelf-url-{paper['paper_id']}")
                path = Path(paper["pdf_path"]) if paper.get("pdf_path") and Path(paper["pdf_path"]).exists() else None
                if path:
                    st.download_button("보관 원문 내려받기", data=path.read_bytes(), file_name=path.name, key=f"shelf-download-{paper['paper_id']}")
                else:
                    st.caption("로컬 원문 없음 · arXiv 논문이면 읽기 실행 시 자동으로 다시 가져옵니다.")
                    if paper.get("source_url") and "arxiv.org" in paper.get("source_url", ""):
                        if st.button("원문 지금 가져오기", key=f"fetch-shelf-pdf-{paper['paper_id']}"):
                            try:
                                recovered = ensure_shelf_pdf(paper, DATA / "papers")
                                if not recovered:
                                    raise ValueError("이 논문의 원문 다운로드 주소를 만들 수 없습니다.")
                                ledger.update_shelf_pdf_path(paper["paper_id"], recovered)
                                st.success("원문을 로컬에 가져왔습니다.")
                                st.rerun()
                            except Exception as error:
                                st.error(f"원문 가져오기에 실패했습니다: {error}")
                with st.expander("서재함에서 삭제"):
                    st.caption("읽기 요약·질문·일반화 메모·이력·카드 연결이 함께 삭제됩니다. 이미 승인된 지식카드는 유지됩니다.")
                    with st.form(f"paper-shelf-delete-{paper['paper_id']}"):
                        confirmed = st.checkbox("위 내용을 확인했고 이 서재 항목을 삭제합니다.")
                        deleted = st.form_submit_button("이 논문을 서재함에서 삭제", type="secondary")
                    if deleted:
                        if not confirmed:
                            st.warning("삭제 내용을 확인한 뒤 체크해 주세요.")
                        else:
                            result = ledger.delete_shelf_paper(paper["paper_id"])
                            if result["paper"]:
                                st.success("서재 항목을 삭제했습니다. 승인 지식카드는 유지했습니다.")
                                st.rerun()
                            st.warning("이미 삭제되었거나 찾을 수 없는 서재 항목입니다.")
            if selected_llm_provider("paper") == "gemini":
                st.caption("본문 읽기 실행 환경: Gemini 외부 API · 이 논문의 원문과 프롬프트가 외부 API로 전송됩니다.")
            question = st.text_area("이 논문을 읽는 연구 질문·활용 맥락", value=analysis.get("research_question", ""), key=f"shelf-question-{paper['paper_id']}")
            st.markdown("**M1 논문 읽기 · 요약–해석–질문–첨삭**")
            st.caption("논문 요약, 현재 연구 맥락 해석, 추천 서재 레이블, 원문 근거 기반 읽기 질문을 한 번에 만듭니다.")
            has_reading_record = bool(analysis.get("reading_raw_output") or ledger.paper_reading_questions(paper["paper_id"]))
            reading_action_label = "M1 논문 다시 읽기" if has_reading_record else "M1 논문 읽기 시작"
            if st.button(reading_action_label, type="primary", key=f"paper-reading-{paper['paper_id']}"):
                try:
                    resolved_path = str(path) if path and path.exists() else ensure_shelf_pdf(paper, DATA / "papers")
                    if not resolved_path:
                        raise ValueError("로컬 원문이 없고 자동으로 원문을 가져올 수 없습니다. PDF를 직접 등록하거나 원문 URL을 확인하세요.")
                    if resolved_path != paper.get("pdf_path"):
                        ledger.update_shelf_pdf_path(paper["paper_id"], resolved_path)
                    document = extract_document(document_from_shelf_path(resolved_path), cache_dir=EXTRACTION_CACHE)
                    streamed_parts: list[str] = []
                    live_output = st.empty()

                    def show_stream(fragment: str) -> None:
                        streamed_parts.append(fragment)
                        live_output.text("".join(streamed_parts))

                    with st.spinner("논문을 읽고 있습니다. 생성되는 내용은 아래에 바로 표시됩니다."):
                        result = paper_draft_result(
                            reading_prompt(document, paper, question), model, use_ollama,
                            profile="paper_reading", on_chunk=None if selected_llm_provider("paper") == "gemini" else show_stream,
                        )
                    live_output.empty()
                    questions, unconsumed_sections = persist_paper_reading_output(ledger, paper, analysis, question, result.text or "", "LLM API") if result.ok else ([], [])
                    if unconsumed_sections:
                        with st.expander(f"파싱에 반영되지 않은 응답 섹션 {len(unconsumed_sections)}개", expanded=False):
                            st.caption("카드·요약에 쓰이지 않은 필드명 또는 섹션입니다. 이 내용을 복사해 파서 보완에 활용할 수 있습니다. 원본 응답은 별도로 보존됩니다.")
                            for index, section in enumerate(unconsumed_sections, start=1):
                                st.code(section, language="markdown")
                    if questions:
                        saved_question_ids = ledger.add_paper_reading_questions(paper["paper_id"], questions)
                        if saved_question_ids:
                            st.success(f"논문 요약과 원문 근거 읽기 질문 {len(saved_question_ids)}건을 추가했습니다.")
                        else:
                            st.info("논문 요약은 갱신했고, 같은 읽기 질문은 이미 저장되어 있어 중복 추가하지 않았습니다.")
                        st.rerun()
                    elif result.ok:
                        st.warning("원문 근거 형식의 읽기 질문을 해석하지 못했습니다.")
                    else:
                        show_ollama_failure(result, model)
                except Exception as error:
                    st.error(f"논문 읽기 질문 생성에 실패했습니다: {error}")
            with st.expander("외부 채팅으로 수동 처리", expanded=False):
                st.caption("API 한도·장애 시에 사용합니다. 생성한 프롬프트와 논문 발췌문을 Gemini 웹 또는 ChatGPT에 직접 전달한 뒤, 응답 전체를 붙여 넣으세요. 비공개 원문은 전송 전에 연구자가 판단해야 합니다.")
                prompt_key = f"manual-reading-prompt-{paper['paper_id']}"
                if st.button("외부 채팅용 M1 프롬프트 만들기", key=f"make-{prompt_key}"):
                    try:
                        resolved_path = str(path) if path and path.exists() else ensure_shelf_pdf(paper, DATA / "papers")
                        if not resolved_path:
                            raise ValueError("로컬 원문이 없고 자동으로 원문을 가져올 수 없습니다. PDF를 직접 등록하거나 원문 URL을 확인하세요.")
                        if resolved_path != paper.get("pdf_path"):
                            ledger.update_shelf_pdf_path(paper["paper_id"], resolved_path)
                        document = extract_document(document_from_shelf_path(resolved_path), cache_dir=EXTRACTION_CACHE)
                        st.session_state[prompt_key] = reading_prompt(document, paper, question)
                    except Exception as error:
                        st.error(f"M1 프롬프트 준비에 실패했습니다: {error}")
                manual_prompt = st.session_state.get(prompt_key, "")
                if manual_prompt:
                    st.code(manual_prompt, language="markdown")
                    manual_output = st.text_area("외부 채팅 응답 전체 붙여넣기", key=f"manual-reading-output-{paper['paper_id']}", height=260)
                    if st.button("붙여넣은 응답을 M1 결과로 적용", key=f"apply-manual-reading-{paper['paper_id']}", disabled=not manual_output.strip()):
                        questions, unconsumed_sections = persist_paper_reading_output(ledger, paper, analysis, question, manual_output.strip(), "외부 채팅 수동 입력")
                        if questions:
                            saved_question_ids = ledger.add_paper_reading_questions(paper["paper_id"], questions)
                            st.success(f"외부 채팅 응답에서 읽기 질문 {len(saved_question_ids)}건을 추가했습니다.")
                            st.rerun()
                        st.warning("요약은 저장했지만, 원문 근거 형식의 읽기 질문을 해석하지 못했습니다. 아래 원문과 미매칭 섹션을 확인하세요.")
                        if unconsumed_sections:
                            st.code("\n\n---\n\n".join(unconsumed_sections), language="markdown")
            reading_questions = ledger.paper_reading_questions(paper["paper_id"])
            reading_decision_labels = {
                "proposed": "결정 전", "registered": "지식카드 등록", "deferred": "보류", "irrelevant": "무관",
                # Historical states remain readable but are not used by the new flow.
                "promoted": "이전 승인 흐름", "approved": "이전 승인 흐름", "needs_revision": "이전 보완 흐름",
            }
            for item in reading_questions:
                with st.expander(f"{reading_decision_labels.get(item['status'], item['status'])} · {item['question']}", expanded=False):
                    st.write(item["tentative_answer"])
                    st.caption("근거: " + " · ".join(item["evidence"]))
                    st.caption("유보: " + item["uncertainty"])
                    if item.get("research_relevance"):
                        st.caption("연구 관련성: " + item["research_relevance"])
                    if item.get("suggested_ontology"):
                        st.caption("온톨로지 힌트: " + item["suggested_ontology"])
                    candidate_matches = similar_approved_cards(
                        [{"card_id": item["question_id"], "claim": item["tentative_answer"]}], memory.all(), threshold=0.62,
                    ).get(item["question_id"], [])
                    if candidate_matches:
                        st.warning("기존 승인 지식과 유사합니다. 같은 Claim이면 새 카드를 만들지 말고 아래에서 기존 카드의 근거를 보강하세요.")
                        for match in candidate_matches:
                            st.caption(f"유사 카드 · {match['title']} · 유사도 {match['score']:.2f} · {match['claim']}")
                    render_curation_precedents(
                        f"지식카드화 작업\n논문: {paper['title']}\n읽기 질문: {item['question']}\n잠정 해석: {item['tentative_answer']}",
                        episode_types={"paper_card_registration"}, semantic=semantic, embedding_model=embedding_model,
                    )
                    if item["status"] == "registered":
                        st.success("이 읽기 해석은 지식카드로 등록되었습니다.")
                        continue
                    with st.form(f"reading-review-{item['question_id']}"):
                        decision_options = ["register", "defer", "irrelevant"]
                        existing_decision = (
                            "register" if item["status"] in {"proposed", "promoted"} and len(item["evidence"]) >= 2
                            else "defer" if item["status"] == "deferred"
                            else "defer" if item["status"] in {"proposed", "promoted"}
                            else "irrelevant"
                        )
                        decision = st.radio("연구자 결정", decision_options, index=decision_options.index(existing_decision), horizontal=True, format_func={"register": "지식카드 등록", "defer": "보류", "irrelevant": "무관"}.get)
                        comment = st.text_area("근거 해석·첨삭", value=item["researcher_comment"])
                        st.caption("질문은 논문을 읽는 렌즈입니다. 카드 제목은 목록에서 구별하기 위한 짧은 요약이고, Claim은 근거와 조건을 포함한 완전한 주장입니다. 결정 전에도 제안값을 자유롭게 다듬을 수 있으며, 입력값은 ‘지식카드 등록’을 선택했을 때 카드에 반영됩니다.")
                        evidence_text = st.text_area(
                            "원문 근거 (한 줄에 하나 · 최소 2개)", value="\n".join(item["evidence"]),
                            help="각 근거는 p.N과 원문 단서가 있어야 합니다. 예: p.3 Method — evaluation setup",
                        )
                        card_title = st.text_input("카드 제목 (짧은 요약)", value=item.get("suggested_title", ""), help="Claim을 그대로 반복하지 말고, 목록·계보에서 구별할 수 있는 짧은 명사구로 작성합니다. 예: ‘명세 우선 설계의 품질 효과’")
                        card_claim = st.text_area("주장 (Claim)", value=item["tentative_answer"], help="근거와 적용 범위를 포함해 한 문장으로 독립적으로 이해되는 완전한 주장입니다.")
                        card_labels = st.text_input("레이블 (쉼표 구분, 선택)", value=item.get("suggested_labels", ""), help="M1 제안을 수정해 입력합니다. 관리·검색용 분류어이며 온톨로지 개념과 일치할 필요는 없습니다.")
                        card_concepts = st.text_input("핵심 개념 (쉼표 구분)", value=item.get("suggested_concepts", ""), help="M1 제안을 수정해 입력합니다. 나중에 카드 간 관계를 만들 도메인 개념입니다.")
                        card_applies_to = st.text_input("적용 대상 (쉼표 구분)", value=item.get("suggested_applies_to", ""), help="M1 제안을 수정해 입력합니다. 이 Claim이 다루는 객체·상황·과업의 유형입니다.")
                        card_conditions = st.text_area("적용 조건", value=item.get("suggested_conditions", ""), help="M1 제안을 수정해 입력합니다. 주장이 성립한 전제·관찰 범위·설계 제약입니다.")
                        card_limits = st.text_area("한계·유보", value=item.get("suggested_limits") or item["uncertainty"])
                        evidence_levels = {
                            "empirical": "실증 — 논문의 데이터·실험·사례가 직접 뒷받침",
                            "theoretical": "이론 — 개념적·논리적 논증이 중심",
                            "review": "문헌 종합 — 여러 선행 연구를 검토·종합",
                            "provisional": "잠정 — 연구자의 해석이거나 추가 검증 필요",
                        }
                        card_evidence_level = st.selectbox("근거 수준", list(evidence_levels), index=0, format_func=evidence_levels.get, help="논문이 Claim을 직접 얼마나 강하게 뒷받침하는지 고릅니다. 단일 실험 결과면 실증, 저자의 논증이면 이론, 리뷰 논문이면 문헌 종합, 본문을 넘어선 연구자 해석이면 잠정이 적합합니다.")
                        duplicate_mode = "separate"
                        duplicate_target_id = ""
                        if candidate_matches:
                            duplicate_mode = st.radio("유사 카드 처리", ["enrich", "separate", "defer"], horizontal=True, format_func={"enrich": "기존 카드 근거 보강", "separate": "별도 카드 등록", "defer": "보류"}.get)
                            target_options = {f"{match['title']} · {match['claim'][:60]}": match["card_id"] for match in candidate_matches}
                            duplicate_target_label = st.selectbox("근거를 보강할 기존 카드", list(target_options))
                            duplicate_target_id = target_options[duplicate_target_label]
                        saved = st.form_submit_button("판단 저장")
                    if saved:
                        final_evidence = [line.strip(" -•") for line in evidence_text.splitlines() if line.strip(" -•")]
                        if decision == "register" and (len(card_title.strip()) < 4 or len(card_claim.strip()) < 8):
                            st.error("지식카드 등록에는 4자 이상의 카드 제목과 8자 이상의 주장(Claim)이 필요합니다.")
                            continue
                        if decision == "register" and (len(final_evidence) < 2 or any(not re.search(r"\bp\.\s*\d+\b", evidence, flags=re.I) for evidence in final_evidence)):
                            st.error("지식카드 등록에는 p.N 형식의 독립 원문 근거를 최소 2개 입력해야 합니다. 근거가 하나뿐이면 보류로 남기세요.")
                            continue
                        if decision == "register" and duplicate_mode == "defer":
                            ledger.update_paper_reading_question(item["question_id"], researcher_comment=comment, status="deferred")
                            st.success("유사 카드 검토를 위해 보류로 저장했습니다.")
                            st.rerun()
                        if decision == "register" and duplicate_mode == "enrich":
                            enriched = memory.add_supporting_evidence(duplicate_target_id, {
                                "source_name": paper["title"], "paper_id": paper["paper_id"],
                                "reading_question": item["question"], "evidence_excerpt": "\n".join(final_evidence)[:1600],
                                "citation_markers": final_evidence, "conditions": card_conditions.strip(),
                                "limits": card_limits.strip(), "researcher_comment": comment.strip(),
                            })
                            existing_cards = ledger.paper_card_ids(paper["paper_id"])
                            ledger.set_paper_card_links(paper["paper_id"], [*existing_cards, enriched["card_id"]])
                            action_case_id = ledger.create_case("research", f"Knowledge evidence enrichment: {paper['title'][:72]}")
                            ledger.record(action_case_id, "knowledge_update", "m1", ["m2", "researcher"], "knowledge_card", {"title": f"기존 지식카드 근거 보강: {enriched['title']}", "card_id": enriched["card_id"], "paper_id": paper["paper_id"]}, enriched["card_id"], status="completed")
                            ledger.update_paper_reading_question(item["question_id"], researcher_comment=comment, status="registered")
                            st.success("새 카드를 만들지 않고 기존 지식카드에 이 논문의 근거를 보강했습니다.")
                            st.rerun()
                        action_case_id = ledger.create_case("research", f"Researcher paper curation: {paper['title'][:72]}")
                        if decision == "register":
                            card = memory.add({
                                "title": card_title.strip(), "source_kind": "external_paper" if paper.get("asset_type") in {"paper", "web_page"} else "researcher_idea_note",
                                "claim": card_claim.strip(), "explanation": "\n".join(part for part in [f"읽기 질문: {item['question']}", f"연구자 해석·첨삭: {comment.strip()}" if comment.strip() else ""] if part),
                                "labels": [value.strip() for value in card_labels.split(",") if value.strip()],
                                "concepts": [value.strip() for value in card_concepts.split(",") if value.strip()],
                                "applies_to": [value.strip() for value in card_applies_to.split(",") if value.strip()],
                                "evidence_level": card_evidence_level, "status": "verified",
                                "evidence_excerpt": "\n".join(final_evidence)[:1600], "evidence_pages": [],
                                "citation_markers": final_evidence, "conditions": card_conditions.strip(), "limits": card_limits.strip(),
                                "provenance": {"source_name": paper["title"], "paper_id": paper["paper_id"], "grounding": "paper_reading_researcher_registration", "reading_question": item["question"]},
                            })
                            existing_cards = ledger.paper_card_ids(paper["paper_id"])
                            ledger.set_paper_card_links(paper["paper_id"], [*existing_cards, card["card_id"]])
                            ledger.record(action_case_id, "knowledge_update", "m1", ["m2", "researcher"], "knowledge_card", {"title": f"논문 읽기에서 등록한 지식카드: {card['title']}", "card_id": card["card_id"], "paper_id": paper["paper_id"]}, card["card_id"], status="completed")
                            ledger.update_paper_reading_question(item["question_id"], researcher_comment=comment, status="registered")
                            store_researcher_curation_episode(
                                ledger, case_id=action_case_id, episode_type="paper_card_registration", paper_title=paper["title"],
                                situation=f"읽기 질문: {item['question']}\n원문 근거: {'; '.join(final_evidence)}",
                                decision="이 해석을 지식카드로 등록한다.", action_summary=f"등록 카드: {card['title']}\n주장: {card['claim']}",
                                action_steps=["원문 근거 확인", "잠정 해석 첨삭", "Claim·개념·조건·한계 입력", "지식카드 등록"],
                                evidence_card_ids=[card["card_id"]], unresolved_items=[card["limits"]] if card.get("limits") else [],
                            )
                        elif decision == "irrelevant":
                            ledger.update_paper_reading_question(item["question_id"], researcher_comment=comment, status="irrelevant")
                            store_researcher_curation_episode(
                                ledger, case_id=action_case_id, episode_type="paper_card_registration", paper_title=paper["title"],
                                situation=f"읽기 질문: {item['question']}\n원문 근거: {'; '.join(item['evidence'])}",
                                decision="이 해석은 현재 연구 주제의 지식카드로 등록하지 않는다.", action_summary=f"무관 판단 사유: {comment or '연구자 판단에 따라 현재 지식화 범위에서 제외'}",
                                action_steps=["원문 근거 확인", "현재 연구 주제와의 관련성 판단", "무관 처리"], evidence_card_ids=[], unresolved_items=[],
                            )
                        else:
                            ledger.update_paper_reading_question(item["question_id"], researcher_comment=comment, status="deferred")
                            store_researcher_curation_episode(
                                ledger, case_id=action_case_id, episode_type="paper_card_registration", paper_title=paper["title"],
                                situation=f"읽기 질문: {item['question']}\n원문 근거: {'; '.join(final_evidence)}",
                                decision="이 해석은 근거 또는 연구 맥락 확인이 더 필요해 보류한다.",
                                action_summary=f"보류 사유: {comment or '근거·조건을 추가 확인한 뒤 등록 또는 무관 판단'}",
                                action_steps=["원문 근거 확인", "Claim·조건 보완 필요성 판단", "보류 처리"],
                                evidence_card_ids=[], unresolved_items=[item["question"]],
                            )
                        st.success("연구자 판단을 저장했습니다.")
                        st.rerun()
            events = ledger.paper_asset_events(paper["paper_id"])
            if events:
                with st.expander(f"이 논문의 이력 · {len(events)}건"):
                    for event in events:
                        st.caption(f"{event['created_at'][:16].replace('T', ' ')} · {event['event_type']}")
            with st.form(f"paper-shelf-note-{paper['paper_id']}"):
                labels = st.text_input("연구자 서재 레이블 (쉼표로 구분 · 최대 5개)", value=", ".join(paper.get("labels", [])), key=f"shelf-labels-{paper['paper_id']}")
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
    upload_tab, ontology_tab, relation_tab, search_tab, queue_tab, memory_tab = st.tabs(["서재함", "온톨로지", "관계·계보 정리", "승인 지식 조회", "문헌 탐색 작업", "승인 지식 목록"])
    with upload_tab:
        st.caption("논문·연구 노트·웹페이지를 이곳에 넣고, 탐색에서 고른 논문도 같은 서재함에서 관리합니다. 등록 자체는 지식카드 생성이 아닙니다.")
        uploaded = st.file_uploader("논문 PDF·연구 노트", type=["pdf", "txt", "md"])
        source_kind = st.selectbox("자료 성격", ["외부 논문", "연구자의 확정 문서", "연구자의 아이디어 노트"])
        core_paper = st.checkbox("핵심 문헌으로 표시", key="asset-intake-core")
        if uploaded and st.button("서재함에 추가", key="claim-first-add-shelf"):
            try:
                document = extract_document(uploaded, cache_dir=EXTRACTION_CACHE)
                bibliography = infer_bibliographic_metadata(document)
                paper = ledger.upsert_shelf_paper({
                    "title": bibliography["title"], "authors": bibliography["authors"], "publication_year": bibliography["publication_year"],
                    "source_id": document.document_id, "pdf_path": store_paper_upload(uploaded, DATA / "paper_shelf"),
                    "shelf_status": "core" if core_paper else "reference", "reading_status": "unread",
                    "asset_type": "paper" if source_kind == "외부 논문" else "research_note", "intake_source": "upload",
                })
                st.success(f"서재함에 추가했습니다: {paper['title']}")
            except Exception as error:
                st.error(f"원문을 서재함에 보관하지 못했습니다: {error}")
        st.divider()
        st.markdown("**웹페이지 링크 등록**")
        st.caption("공개 HTML 페이지에서 본문형 텍스트를 추출해 Markdown으로 보관합니다. 메뉴·광고·스크립트·푸터 등 탐색 요소는 제거하며, 원본 URL은 함께 기록합니다.")
        with st.form("web-page-intake"):
            web_url = st.text_input("웹페이지 URL", placeholder="https://example.org/article")
            web_kind = st.selectbox("자료 성격", ["외부 논문·기술문서", "연구자의 확정 문서", "연구자의 아이디어 노트"], key="web-asset-kind")
            web_core = st.checkbox("핵심 문헌으로 표시", key="web-asset-core")
            add_web_page = st.form_submit_button("웹페이지를 서재함에 추가")
        if add_web_page:
            try:
                page = fetch_web_page(web_url)
                stored = StoredPaperUpload(name="web-page.md", content=page.markdown.encode("utf-8"))
                document = extract_document(stored, cache_dir=EXTRACTION_CACHE)
                paper = ledger.upsert_shelf_paper({
                    "title": page.title, "authors": [page.author] if page.author else [], "publication_year": page.publication_year,
                    "source_url": page.url, "source_id": document.document_id,
                    "pdf_path": store_paper_upload(stored, DATA / "paper_shelf"),
                    "shelf_status": "core" if web_core else "reference", "reading_status": "unread",
                    "asset_type": "web_page", "intake_source": "web",
                })
                st.success(f"정제한 웹페이지를 서재함에 추가했습니다: {paper['title']}")
                for warning in page.warnings:
                    st.warning(warning)
                st.rerun()
            except WebPageExtractionError as error:
                st.error(str(error))
            except Exception as error:
                st.error(f"웹페이지를 서재함에 보관하지 못했습니다: {error}")
        render_paper_shelf(model, use_ollama, semantic, embedding_model)
        st.info("직접 주장 추출은 이 흐름에서 사용하지 않습니다. 논문을 선택해 읽기 질문·근거·첨삭을 거친 항목만 그 자리에서 지식카드로 등록합니다.")
        if False and st.button("지식 카드 초안 만들기 (최대 10개)", disabled=uploaded is None, type="primary", key="claim-first-discover"):
            try:
                document = extract_document(uploaded, cache_dir=EXTRACTION_CACHE)
                result = llm_draft_result(discovery_prompt(document), model, use_ollama)
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
        if False and claim_cards and claim_document and uploaded and claim_document.document_id == extract_document(uploaded, cache_dir=EXTRACTION_CACHE).document_id:
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
                        return llm_draft_result(prompt, model, use_ollama).text
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
                            result = llm_draft_result(consolidation_prompt(pairs), model, use_ollama)
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
    with ontology_tab:
        render_ontology_workspace(semantic, embedding_model)
    with relation_tab:
        cards = memory.all()
        cards_by_id = {card["card_id"]: card for card in cards}
        all_approved_relations = ledger.active_knowledge_relations()
        selection_mode = st.radio(
            "계보에 표시할 카드 선정", ["최근 승인 카드", "검색 후 선택"], horizontal=True, key="p2-selection-mode",
        )
        selected_cards = cards[:20]
        if selection_mode == "검색 후 선택":
            selected_ids = st.session_state.setdefault("p2-lineage-selected-ids", [])
            by_id = cards_by_id
            lineage_query = st.text_input("소스 지식카드 키워드 검색", key="p2-lineage-query", placeholder="예: agent specification evaluation")
            if st.button("임베딩으로 지식카드 검색", key="p2-lineage-search", disabled=not lineage_query.strip()):
                hits = search_knowledge(lineage_query, semantic, embedding_model, limit=10)
                st.session_state["p2-lineage-search-results"] = [
                    {"card_id": hit.card["card_id"], "reason": hit.reason, "score": round(hit.score, 3)} for hit in hits
                ]
                st.session_state["p2-lineage-search-query-applied"] = lineage_query
            search_hits = [
                {**result, "card": by_id.get(result["card_id"])}
                for result in st.session_state.get("p2-lineage-search-results", [])
                if result["card_id"] in by_id
            ]
            if search_hits:
                st.markdown(f"#### 소스 지식카드 검색 결과 · {len(search_hits)}건")
                st.caption(f"검색어: {st.session_state.get('p2-lineage-search-query-applied', lineage_query)} · 결과에서 관계 탐색의 소스 카드를 선택하세요.")
                for hit in search_hits:
                    card = hit["card"]
                    info, action = st.columns([5, 2])
                    with info:
                        st.write(f"**{card['title']}**")
                        st.caption(f"유사도 {hit['score']:.3f} · 선정 이유: {hit['reason']}")
                        st.caption(card.get("claim", "")[:220])
                    with action:
                        if card["card_id"] in selected_ids:
                            st.caption("소스 선택됨")
                        elif st.button("관계 탐색 소스로 선택", key=f"p2-add-{card['card_id']}", disabled=len(selected_ids) >= 20):
                            st.session_state["p2-lineage-selected-ids"] = [*selected_ids, card["card_id"]]
                            st.rerun()
            elif st.session_state.get("p2-lineage-search-query-applied"):
                st.info("검색된 승인 카드가 없습니다.")
            selected_cards = [by_id[card_id] for card_id in selected_ids if card_id in by_id][:20]
            st.subheader(f"계보 선택 카드 목록 ({len(selected_cards)}/20)")
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
                st.caption("먼저 키워드로 소스 지식카드를 검색·선택하세요.")
        related_target_ids = st.session_state.get("p2-lineage-relation-target-ids", [])
        selected_card_ids = {card["card_id"] for card in selected_cards}
        for card_id in related_target_ids:
            card = by_id.get(card_id) if selection_mode == "검색 후 선택" else next((item for item in cards if item["card_id"] == card_id), None)
            if card and card["card_id"] not in selected_card_ids and len(selected_cards) < 20:
                selected_cards.append(card)
                selected_card_ids.add(card["card_id"])
        selection_fingerprint = tuple(card["card_id"] for card in selected_cards)
        if st.session_state.get("p2-lineage-fingerprint") != selection_fingerprint:
            st.session_state["p2-lineage-fingerprint"] = selection_fingerprint
            st.session_state.pop("lineage-overview-result", None)
        st.caption(
            f"현재 계보 대상: 승인 카드 {len(selected_cards)}개 · "
            + ("최근 등록 순서" if selection_mode == "최근 승인 카드" else "연구자 검색·선택 결과")
        )
        if not selected_cards:
            st.info("관계 탐색을 시작하려면 승인된 소스 지식카드 한 장을 선택하세요.")
        else:
            st.caption("2) 소스 카드 선택 → 3) 임베딩으로 타겟 카드 탐색 → 4) LLM 관계·근거 추천 → 5) 화면에서 직접 승인 순서로 진행합니다. 검색·추천은 버튼을 누를 때만 실행됩니다.")
            source_options = {f"{card['title']} [{card['card_id']}]": card for card in selected_cards}
            source_label = st.selectbox("관계를 탐색할 소스 카드", list(source_options), key="p2-relation-source")
            source_card = source_options[source_label]
            source_query = " ".join([
                str(source_card.get("title", "")), str(source_card.get("claim", "")),
                " ".join(source_card.get("concepts", [])), str(source_card.get("conditions", "")),
            ]).strip()
            all_active_relations = all_approved_relations
            existing_by_target: dict[str, list[dict[str, object]]] = {}
            for relation in all_active_relations:
                if str(relation["source_card_id"]) == str(source_card["card_id"]):
                    existing_by_target.setdefault(str(relation["target_card_id"]), []).append(relation)
            if st.button("임베딩으로 타겟 지식카드 탐색", key="p2-relation-target-search", disabled=not source_query):
                hits = search_knowledge(source_query, semantic, embedding_model, limit=15)
                st.session_state["p2-relation-target-results"] = [
                    {"card_id": hit.card["card_id"], "reason": hit.reason, "score": round(hit.score, 3)}
                    for hit in hits if hit.card["card_id"] != source_card["card_id"]
                ]
                st.session_state["p2-relation-target-source-id"] = source_card["card_id"]
            target_results = [
                item if isinstance(item, dict) else {"card_id": item, "reason": "이전 탐색 결과", "score": 0.0}
                for item in st.session_state.get("p2-relation-target-results", [])
            ]
            target_hits = [
                {**result, "card": next((card for card in cards if card["card_id"] == result["card_id"]), None)}
                for result in target_results
            ] if st.session_state.get("p2-relation-target-source-id") == source_card["card_id"] else []
            target_cards = [
                hit["card"] for hit in target_hits
                if hit["card"] is not None
            ] if st.session_state.get("p2-relation-target-source-id") == source_card["card_id"] else []
            target_cards = [card for card in target_cards if card is not None]
            if target_hits:
                st.markdown(f"#### 타겟 지식카드 탐색 결과 · {len(target_cards)}건")
                st.caption("소스 카드의 제목·주장·개념·조건을 기준으로 찾은 결과입니다. 이미 승인된 관계는 아래에서 따로 표시합니다.")
                with st.expander("타겟 검색 결과 자세히 보기", expanded=False):
                    for hit in target_hits:
                        card = hit["card"]
                        if card is None:
                            continue
                        st.write(f"**{card['title']}**")
                        st.caption(f"유사도 {hit['score']:.3f} · 선정 이유: {hit['reason']}")
                        st.caption(card.get("claim", "")[:260])
            existing_target_cards = [card for card in target_cards if card["card_id"] in existing_by_target]
            new_target_cards = [card for card in target_cards if card["card_id"] not in existing_by_target][:5]
            if existing_target_cards:
                st.markdown("#### 이미 승인된 관계")
                st.caption("이미 연결된 타겟은 LLM에 다시 보내지 않습니다. 필요하면 아래에서 기존 관계를 대체해 수정하세요.")
                for target_card in existing_target_cards:
                    for existing_relation in existing_by_target[target_card["card_id"]]:
                        with st.expander(f"{source_card['title']} → {target_card['title']} · 기존 {existing_relation['relation_type']}"):
                            with st.form(f"p2-existing-relation-edit-{existing_relation['relation_id']}"):
                                relation_type = st.selectbox("관계 유형", RELATION_TYPES, index=RELATION_TYPES.index(existing_relation["relation_type"]), key=f"p2-existing-type-{existing_relation['relation_id']}")
                                evidence = st.text_area("관계 근거", value=existing_relation["evidence"], key=f"p2-existing-evidence-{existing_relation['relation_id']}")
                                conditions = st.text_area("관계 적용 조건", value=existing_relation["conditions"], key=f"p2-existing-conditions-{existing_relation['relation_id']}")
                                confidence = st.selectbox("신뢰도", ["low", "medium", "high"], index=["low", "medium", "high"].index(existing_relation["confidence"]), key=f"p2-existing-confidence-{existing_relation['relation_id']}")
                                replace_relation = st.form_submit_button("기존 관계를 이 값으로 대체")
                            if replace_relation:
                                try:
                                    if not delete_knowledge_relation(ledger, relations, str(existing_relation["relation_id"]), "연구자가 관계·계보 화면에서 수정"):
                                        raise ValueError("기존 관계를 찾지 못했거나 이미 삭제되었습니다.")
                                    request_id, _ = create_relation_candidate(
                                        ledger, str(source_card["card_id"]), str(target_card["card_id"]), relation_type,
                                        "", evidence, conditions, confidence, source_card, target_card,
                                    )
                                    decide_request(ledger, memory, request_id, "approved", relation_memory=relations)
                                    replacement = ledger.phenomenon(request_id) or {}
                                    st.session_state["p2-lineage-focus-relation-id"] = replacement.get("payload", {}).get("relation", {}).get("relation_id", "")
                                    st.success("기존 관계를 수정한 관계로 대체했습니다.")
                                    st.rerun()
                                except ValueError as error:
                                    st.error(f"관계를 수정하지 못했습니다: {error}")
            if new_target_cards:
                st.caption("LLM 비교 타겟(최대 5개): " + " · ".join(card["title"] for card in new_target_cards))
                if st.button("LLM으로 관계 내용·근거 추천하기", type="primary", key="p2-relation-batch-draft"):
                    draft_text = llm_draft(relation_batch_prompt(source_card, new_target_cards), model, use_ollama)
                    drafts = parse_relation_batch_drafts(draft_text or "", str(source_card["card_id"]), new_target_cards)
                    st.session_state["p2-relation-drafts"] = drafts
                    st.session_state["p2-relation-draft-context"] = {
                        "source_card_id": source_card["card_id"],
                        "target_card_ids": [card["card_id"] for card in new_target_cards],
                    }
                    if drafts:
                        st.success(f"직접 검토할 관계 초안 {len(drafts)}건을 만들었습니다.")
                    else:
                        st.info("후보 타겟들 사이에서 방어 가능한 관계를 찾지 못했습니다. 이는 관계가 없다는 보수적 판단일 수 있습니다.")
            elif target_cards:
                st.info("탐색된 타겟은 모두 이미 승인 관계가 있습니다. 위의 기존 관계를 확인하거나 수정하세요.")
            elif st.session_state.get("p2-relation-target-source-id") == source_card["card_id"]:
                st.info("관련 타겟 카드를 찾지 못했습니다. 다른 소스 카드를 선택하거나 승인 지식을 더 추가하세요.")

            draft_context = st.session_state.get("p2-relation-draft-context", {})
            relation_drafts = st.session_state.get("p2-relation-drafts", [])
            if relation_drafts and draft_context.get("source_card_id") == source_card["card_id"]:
                cards_by_id = {card["card_id"]: card for card in cards}
                st.markdown("### 관계 초안 검토·승인")
                st.caption("승인은 즉시 관계·계보에 반영합니다. ‘승인함으로 보내기’는 판단을 나중으로 미룹니다.")
                for index, draft in enumerate(relation_drafts):
                    target_card = cards_by_id.get(draft["target_card_id"])
                    if not target_card:
                        continue
                    with st.expander(f"{source_card['title']} → {target_card['title']} · {draft['relation_type']}", expanded=True):
                        st.caption("소스 카드")
                        render_knowledge_card(source_card, key_prefix=f"relation-source-{index}")
                        st.caption("타겟 카드")
                        render_knowledge_card(target_card, key_prefix=f"relation-target-{index}")
                        with st.form(f"p2-relation-review-{source_card['card_id']}-{target_card['card_id']}"):
                            relation_type = st.selectbox("관계 유형", RELATION_TYPES, index=RELATION_TYPES.index(draft["relation_type"]), key=f"p2-draft-type-{target_card['card_id']}")
                            evidence = st.text_area("관계 근거", value=draft["evidence"], key=f"p2-draft-evidence-{target_card['card_id']}")
                            conditions = st.text_area("관계 적용 조건", value=draft["conditions"], key=f"p2-draft-conditions-{target_card['card_id']}")
                            confidence = st.selectbox("신뢰도", ["low", "medium", "high"], index=["low", "medium", "high"].index(draft["confidence"]), key=f"p2-draft-confidence-{target_card['card_id']}")
                            approve_now = st.form_submit_button("승인하고 계보에 반영", type="primary")
                            queue_for_later = st.form_submit_button("승인함으로 보내기")
                            discard = st.form_submit_button("이 후보 제외")
                        if approve_now or queue_for_later:
                            try:
                                request_id, warnings = create_relation_candidate(
                                    ledger, str(source_card["card_id"]), str(target_card["card_id"]), relation_type,
                                    "", evidence, conditions, confidence, source_card, target_card,
                                )
                                if approve_now:
                                    decide_request(ledger, memory, request_id, "approved", relation_memory=relations)
                                    approved_request = ledger.phenomenon(request_id) or {}
                                    st.session_state["p2-lineage-focus-relation-id"] = approved_request.get("payload", {}).get("relation", {}).get("relation_id", "")
                                    st.session_state["p2-lineage-relation-target-ids"] = list(dict.fromkeys([
                                        *st.session_state.get("p2-lineage-relation-target-ids", []), target_card["card_id"],
                                    ]))
                                    st.success("관계를 승인하고 계보에 반영했습니다.")
                                else:
                                    st.success("관계 후보를 연구자 승인함에 보냈습니다.")
                                for warning in warnings:
                                    st.warning(warning)
                                st.session_state["p2-relation-drafts"] = [
                                    item for item in relation_drafts
                                    if item["target_card_id"] != target_card["card_id"]
                                ]
                                st.rerun()
                            except ValueError as error:
                                st.error(f"관계 후보를 저장하지 못했습니다: {error}")
                        if discard:
                            st.session_state["p2-relation-drafts"] = [
                                item for item in relation_drafts
                                if item["target_card_id"] != target_card["card_id"]
                            ]
                            st.rerun()
        st.subheader(f"관계 승인 이력 · {len(all_approved_relations)}건")
        if all_approved_relations:
            with st.expander("승인된 관계 이력 보기", expanded=False):
                for relation in all_approved_relations[:30]:
                    source_title = cards_by_id.get(relation["source_card_id"], {}).get("title", relation["source_card_id"])
                    target_title = cards_by_id.get(relation["target_card_id"], {}).get("title", relation["target_card_id"])
                    st.write(f"**{source_title}** → **{target_title}** · `{relation['relation_type']}` · {relation['confidence']}")
                    st.caption(f"승인: {relation['approved_at'][:16].replace('T', ' ')} · 근거: {relation['evidence']}")
        else:
            st.caption("아직 승인된 관계가 없습니다.")

        focus_relation_id = str(st.session_state.get("p2-lineage-focus-relation-id", ""))
        focus_relation = next((item for item in all_approved_relations if item["relation_id"] == focus_relation_id), None)
        graph_cards, active_relations = selected_cards, ledger.active_knowledge_relations({card["card_id"] for card in selected_cards})
        graph_caption = "연구자가 선택한 카드 기준"
        if focus_relation:
            focus_nodes = {focus_relation["source_card_id"], focus_relation["target_card_id"]}
            surrounding_relations = [
                relation for relation in all_approved_relations
                if relation["source_card_id"] in focus_nodes or relation["target_card_id"] in focus_nodes
            ]
            surrounding_node_ids = list(dict.fromkeys([
                focus_relation["source_card_id"], focus_relation["target_card_id"],
                *[relation["source_card_id"] for relation in surrounding_relations],
                *[relation["target_card_id"] for relation in surrounding_relations],
            ]))[:20]
            graph_cards = [cards_by_id[card_id] for card_id in surrounding_node_ids if card_id in cards_by_id]
            graph_ids = {card["card_id"] for card in graph_cards}
            active_relations = [
                relation for relation in surrounding_relations
                if relation["source_card_id"] in graph_ids and relation["target_card_id"] in graph_ids
            ]
            graph_caption = "방금 승인한 관계의 양 끝 카드와 1-Hop 이웃 관계 기준"
            if st.button("선택 카드 기준 그래프로 돌아가기", key="p2-clear-lineage-focus"):
                st.session_state.pop("p2-lineage-focus-relation-id", None)
                st.rerun()
        graph_fingerprint = (tuple(card["card_id"] for card in graph_cards), tuple(relation["relation_id"] for relation in active_relations))
        if st.session_state.get("p2-lineage-graph-fingerprint") != graph_fingerprint:
            st.session_state["p2-lineage-graph-fingerprint"] = graph_fingerprint
            st.session_state.pop("lineage-overview-result", None)
        st.subheader(f"승인된 개념·방법 계보 (최대 20개 카드 · {graph_caption})")
        if graph_cards:
            st.graphviz_chart(lineage_dot(graph_cards, active_relations), use_container_width=True)
            if active_relations:
                for relation in active_relations:
                    st.caption(f"{relation['relation_type']} · {relation['confidence']} · {relation['evidence']}")
            else:
                st.caption("아직 승인된 관계가 없습니다. 그래프는 SQLite 승인 관계 테이블의 실행 시 투영입니다.")
            if active_relations:
                overview_prompt = lineage_overview_prompt(graph_cards, active_relations)
                if st.button("계보 종합 의견 만들기", key="lineage-overview-generate"):
                    overview = llm_draft(overview_prompt, model, use_ollama)
                    if overview:
                        st.session_state["lineage-overview-result"] = overview
                    else:
                        st.warning("계보 종합 의견 초안을 만들지 못했습니다.")
                with st.expander("외부 채팅으로 계보 종합 의견 만들기"):
                    st.caption("아래 프롬프트에는 현재 그래프의 승인 지식카드와 관계 근거가 포함됩니다. 공개해도 되는 정보인지 확인한 뒤 Gemini 또는 ChatGPT에 붙여넣으세요.")
                    st.code(overview_prompt, language="text")
                    manual_overview = st.text_area(
                        "외부 채팅의 계보 종합 의견 전체 붙여넣기",
                        key="lineage-overview-manual-output",
                        height=300,
                    )
                    if st.button(
                        "붙여넣은 의견을 계보 종합 의견으로 적용",
                        key="lineage-overview-apply-manual",
                        disabled=not manual_overview.strip(),
                    ):
                        st.session_state["lineage-overview-result"] = manual_overview.strip()
                        st.success("외부 채팅의 계보 종합 의견을 적용했습니다.")
                        st.rerun()
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
        st.caption("승인된 연구 Intent가 탐색 프로필로 대기합니다. 실행 결과는 논문 후보 서재함으로 보내며, 이 단계에서는 지식카드를 만들지 않습니다.")
        if selected_llm_provider("internal") == "gemini" or selected_llm_provider("paper") == "gemini":
            destinations = []
            if selected_llm_provider("internal") == "gemini":
                destinations.append("탐색 초안")
            if selected_llm_provider("paper") == "gemini":
                destinations.append("공개 arXiv 본문 비교")
            st.caption("Gemini 외부 API로 전송: " + " · ".join(destinations))
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
                            generated = llm_draft(keyword_prompt(refreshed), model, use_ollama)
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
                    outcome = run_profile(ledger, current, "manual", reviewer=lambda prompt: llm_draft(prompt, model, use_ollama, profile="abstract_triage"))
                    if outcome["status"] == "completed":
                        processed = process_top_papers(current, outcome["candidates"], DATA, lambda prompt: paper_draft_result(prompt, model, use_ollama, "full_text_similarity").text, make_cards=False)
                        merged = {item["source_id"]: item for item in outcome["candidates"]}
                        merged.update({item["source_id"]: item for item in processed})
                        ledger.update_search_run_candidates(outcome["run_id"], list(merged.values()))
                        st.success(f"초록 후보 {len(outcome['candidates'])}건을 기록하고, 상위 후보의 본문 비교 정보를 준비했습니다. 읽을 논문은 서재함에 추가하세요.")
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
                                citations = candidate.get("citation_count")
                                citation_text = "확인 불가" if citations is None else f"{int(citations):,}회"
                                st.caption(f"arXiv · {', '.join(candidate['authors'][:4])} · 인용 {citation_text} · {candidate['url']}")
                                st.link_button("arXiv 논문 페이지 열기", candidate["url"], key=f"open-abs-{run['run_id']}-{candidate['source_id']}")
                                if candidate.get("pdf_path"):
                                    pdf_path = Path(candidate["pdf_path"])
                                    if pdf_path.exists():
                                        st.download_button("저장 PDF 내려받기 (P1 업로드용)", data=pdf_path.read_bytes(), file_name=pdf_path.name, mime="application/pdf", key=f"download-pdf-{run['run_id']}-{candidate['source_id']}")
                                if st.button("서재함에 추가", key=f"shelf-add-{run['run_id']}-{candidate['source_id']}"):
                                    saved = ledger.upsert_shelf_paper({
                                        "title": candidate["title"], "authors": candidate.get("authors", []),
                                        "publication_year": candidate.get("published", "")[:4], "source_url": candidate.get("url", ""),
                                        "source_id": candidate.get("source_id", ""), "pdf_path": candidate.get("pdf_path", ""),
                                        "shelf_status": "reference", "reading_status": "unread", "asset_type": "paper", "intake_source": "search",
                                    })
                                    st.success(f"서재함에 추가했습니다: {saved['title']}")
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
                                    citations = candidate.get("citation_count")
                                    citation_text = "확인 불가" if citations is None else f"{int(citations):,}회"
                                    st.write(f"- {candidate['title']} · 인용 {citation_text} · 1차 맥락 점수 {candidate.get('context_match_score', '-')}")
                                    relevance = candidate.get("relevance", {})
                                    st.caption(f"M2 적합성 {relevance.get('level', 'unreviewed')} · {candidate['summary'][:260]}")
                            if any(candidate.get("relevance", {}).get("level") == "unreviewed" for candidate in run["candidates"]):
                                if st.button("이 실행의 초록·맥락 적합성 검토", key=f"review-run-{run['run_id']}"):
                                    refreshed = shortlist_candidates(current, run["candidates"], lambda prompt: llm_draft(prompt, model, use_ollama, profile="abstract_triage"))
                                    ledger.update_search_run_candidates(run["run_id"], refreshed)
                                    st.rerun()
                                    st.warning("초록 적합성 초안을 만들지 못했습니다.")
                            if st.button("상위 5편 PDF 수집·본문 비교 보고 만들기", key=f"p1-batch-{run['run_id']}", disabled=not run["candidates"]):
                                progress = st.progress(0, text="상위 5편을 순차 처리합니다.")
                                processed = process_top_papers(current, run["candidates"], DATA, lambda prompt: paper_draft_result(prompt, model, use_ollama, "full_text_similarity").text, make_cards=False)
                                merged = {item["source_id"]: item for item in run["candidates"]}
                                merged.update({item["source_id"]: item for item in processed})
                                ledger.update_search_run_candidates(run["run_id"], list(merged.values()))
                                progress.progress(1.0, text="PDF 수집·본문 비교 보고 완료")
                                st.rerun()
                            completed = [candidate for candidate in shortlist if candidate.get("full_text_status") == "completed"]
                            if completed:
                                st.markdown("**본문 유사도 순위**")
                                for rank, candidate in enumerate(sorted(completed, key=lambda item: item.get("full_text_similarity", 0), reverse=True), start=1):
                                    st.caption(f"{rank}위 · 본문 맥락 적합성 {candidate.get('full_text_similarity', 0)}/100 · {candidate['title']}")
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
    """M2's plan-first response to a researcher question."""
    st.subheader("연구자 질문 대응")
    st.caption("M2가 승인 지식·관계·검색 결과를 근거로 의견을 구성합니다. 답변과 내부 연구 판단은 분리하며, 근거 부족은 후속 탐색 필요로 표시합니다.")
    question = st.text_area("연구자 질문", placeholder="예: 다중 LLM 합의 평가는 설계 개념의 실현 가능성 판단에 어느 정도 신뢰할 수 있는가?", key="m2-service-question")
    context = st.text_area("현재 연구 맥락·제약 (선택)", placeholder="적용 대상, 비교하려는 대안, 현재 가설 또는 확인하려는 결정", key="m2-service-context")
    if st.button("답변 계획 만들기", type="primary", disabled=not question.strip(), key="m2-service-plan"):
        case_id = ledger.create_case("research", f"연구자 질문: {question[:80]}")
        state = ResearchState(question=question, researcher_note=context)
        ledger.record(
            case_id, "research_update", "researcher", ["m2"], "research_question",
            {"state": state.model_dump(mode="json")}, status="completed",
        )
        act_spec = recall_act_spec(
            ledger, episodic_retriever, situation=f"연구자 질문: {question}\n연구 맥락: {context}",
            active_card_ids={card["card_id"] for card in memory.all()}, semantic=semantic, embedding_model=embedding_model,
        )
        precedent = recall_context(act_spec)
        draft = llm_draft(advisory_plan_prompt("research_question", question, context, "researcher", precedent), model, use_ollama)
        plan = parse_advisory_plan(draft or "", "research_question", question)
        ledger.record(
            case_id, "advice_report", "m2", ["researcher"], "research_question_response",
            {"title": "M2 · 연구자 질문 답변 계획", "question": question, "context": context,
             "report": plan.decision_question, "state": state.model_dump(mode="json"),
             "advisory_plan": {"decision_question": plan.decision_question, "subquestions": [item.question for item in plan.subquestions]}}, status="completed",
        )
        st.session_state["m2-service-case"] = case_id
        st.session_state["m2-service-plan-result"] = plan
        st.session_state["m2-service-act-spec"] = act_spec
        st.session_state.pop("m2-service-answer-result", None)
        st.session_state.pop("m2-service-clusters", None)
    plan = st.session_state.get("m2-service-plan-result")
    if plan:
        act_spec = st.session_state.get("m2-service-act-spec")
        if act_spec:
            show_recalled_episodes(act_spec)
        st.subheader("답변 계획")
        st.write(f"**핵심 판단:** {plan.decision_question}")
        for index, subquestion in enumerate(plan.subquestions, start=1):
            st.write(f"{index}. {subquestion.question}")
        if st.button("계획에 따라 근거 수집·연구 의견 만들기", type="primary", key="m2-service-run-plan"):
            precedent = recall_context(act_spec) if act_spec else ""
            clusters, judgments, answer = execute_plan_first_advisory(plan, f"{context}\n\n{precedent}", "researcher", model, use_ollama, semantic, embedding_model)
            evidence_ids = [card_id for _, cluster in clusters for card_id in cluster.card_ids]
            relation_ids = [relation_id for _, cluster in clusters for relation_id in cluster.relation_ids]
            ledger.record(
                st.session_state["m2-service-case"], "advice_report", "m2", ["researcher"], "research_question_response",
                {"title": "M2 · 계획형 연구자 질문 대응", "question": question, "context": context, "report": answer,
                 "evidence_card_ids": list(dict.fromkeys(evidence_ids)), "evidence_relation_ids": list(dict.fromkeys(relation_ids)),
                 "subquestion_judgments": judgments}, status="completed",
            )
            episode = store_advisory_episode(
                ledger, case_id=st.session_state["m2-service-case"], episode_type="research_question",
                situation_summary=f"연구자 질문: {question}\n연구 맥락: {context}", decision_question=plan.decision_question,
                advisory_plan=[item.question for item in plan.subquestions], answer=answer,
                evidence_card_ids=list(dict.fromkeys(evidence_ids)), evidence_relation_ids=list(dict.fromkeys(relation_ids)),
                unresolved_items=[subquestion.question for subquestion, cluster in clusters if not cluster.members],
            )
            st.session_state["m2-service-episode-id"] = episode.episode_id
            st.session_state["m2-service-clusters"] = clusters
            st.session_state["m2-service-answer-result"] = answer
        if "m2-service-clusters" in st.session_state:
            show_evidence_clusters(st.session_state["m2-service-clusters"])
    if "m2-service-answer-result" in st.session_state:
        st.markdown("### M2 의견")
        st.markdown(st.session_state["m2-service-answer-result"])
        if not any(cluster.members for _, cluster in st.session_state.get("m2-service-clusters", [])):
            st.info("현재 승인 지식만으로는 충분한 근거를 찾지 못했습니다. 연구 상태·방향 검토에서 M1 탐색 Intent를 제안하세요.")
        episode_id = st.session_state.get("m2-service-episode-id")
        if episode_id and st.button("이 답변을 확인된 연구 선례로 표시", key="m2-confirm-episode"):
            ledger.update_episode_memory_outcome(episode_id, "confirmed", "연구자가 답변을 유사 사례의 재사용 가능한 선례로 확인함")
            st.success("다음 유사 질문에서 선례 기반 빠른 경로로 리콜할 수 있습니다.")



RQ_STATUS_LABELS = {
    "candidate": "후보",
    "interested": "관심",
    "exploring": "탐색중",
    "hold": "보류",
    "rejected": "제외",
}


def render_research_question_backlog() -> None:
    """Researcher-managed RQ lifecycle with evidence context and M1 handoff."""
    backlog = ledger.research_question_backlog(limit=100)
    st.subheader("Research Question Backlog")
    st.caption("질문을 한 번에 확정하지 않고 관심·탐색중·보류 상태로 관리합니다. 각 질문은 왜 도출되었는지와 M1 근거를 함께 보존합니다.")
    reviews = ledger.research_state_reviews(limit=1)
    if reviews:
        latest_review = reviews[0]
        review_changes = ledger.research_question_changes(review_id=str(latest_review.get("review_id", "")), limit=100)
        created = sum(1 for item in review_changes if item.get("change_type") == "created")
        strengthened = sum(1 for item in review_changes if item.get("change_type") == "evidence_added")
        failed = sum(1 for item in ledger.research_question_changes(limit=100) if item.get("change_type") == "exploration_failed" and item.get("created_at", "") >= str(latest_review.get("created_at", "")))
        mode_label = "자동" if latest_review.get("mode") == "auto" else "수동"
        summary_text = str(latest_review.get("summary") or "").strip()
        if not summary_text:
            summary_text = f"최근 {mode_label} 연구상태 검토에서 새 지식카드 {latest_review.get('source_card_count', 0)}건을 처리했습니다."
        extra = []
        if created or strengthened:
            extra.append(f"연구질문은 신규 {created}건, 기존 질문 보강 {strengthened}건의 변화가 있었습니다.")
        if failed:
            extra.append(f"후속 문헌탐색 {failed}건은 중간 실패로 재시도가 필요합니다.")
        st.info("**최근 변화**  \n" + summary_text + (" " + " ".join(extra) if extra else ""))
    if not backlog:
        st.info("아직 저장된 연구질문 후보가 없습니다. 위에서 M1 새 정보로 후보를 도출하거나 직접 질문을 시작하세요.")
        return

    counts = {status: sum(1 for item in backlog if item.get("status") == status) for status in RQ_STATUS_LABELS}
    cols = st.columns(5)
    for col, status in zip(cols, RQ_STATUS_LABELS):
        col.metric(RQ_STATUS_LABELS[status], counts[status])

    visible_labels = st.multiselect(
        "표시 상태",
        list(RQ_STATUS_LABELS),
        default=["candidate", "interested", "exploring", "hold"],
        format_func=lambda value: RQ_STATUS_LABELS[value],
        key="p4-rq-backlog-status-filter",
    )
    cards_by_id = {str(card.get("card_id", "")): card for card in memory.all()}
    visible = [item for item in backlog if item.get("status") in visible_labels]
    for rq in visible:
        rq_id = str(rq["rq_id"])
        label = RQ_STATUS_LABELS.get(str(rq.get("status", "candidate")), str(rq.get("status", "")))
        linked_intents = ledger.research_question_intents(rq_id)
        with st.expander(f"{label} · {rq['question']}", expanded=rq.get("status") in {"interested", "exploring"}):
            st.markdown(f"**왜 이 질문이 나왔나**  \n{rq.get('rationale', '')}")
            if rq.get("gap_or_tension"):
                st.markdown(f"**공백·긴장**  \n{rq['gap_or_tension']}")
            if rq.get("research_context"):
                st.markdown(f"**현재 연구와의 연결**  \n{rq['research_context']}")
            if rq.get("exploration_need"):
                st.markdown(f"**추가 탐색 필요**  \n{rq['exploration_need']}")
            source_ids = list(rq.get("source_card_ids", []))
            source_links = ledger.research_question_sources(rq_id)
            if source_ids:
                review_ids = list(dict.fromkeys(str(item.get("review_id", "")) for item in source_links if item.get("review_id")))
                st.markdown(f"**연결된 M1 근거 카드 {len(source_ids)}건 · 연구상태 검토 {len(review_ids)}회**")
                if source_links:
                    grouped: dict[str, list[dict]] = {}
                    for link in source_links:
                        grouped.setdefault(str(link.get("review_id", "")), []).append(link)
                    for review_id, links in grouped.items():
                        completed_at = str(links[0].get("review_completed_at", ""))[:19].replace("T", " ")
                        mode_label = "자동" if links[0].get("review_mode") == "auto" else "수동"
                        st.caption(f"{review_id} · {mode_label} 검토 · {completed_at or '진행 중'} · 근거 카드 {len(links)}건")
                        for link in links:
                            card_id = str(link.get("card_id", ""))
                            card = cards_by_id.get(card_id)
                            claim = card.get("claim", card.get("title", "")) if card else ""
                            st.write(f"- `{card_id}` · {claim}")
                else:
                    for card_id in source_ids:
                        card = cards_by_id.get(card_id)
                        st.write(f"- `{card_id}` · {card.get('claim', card.get('title', '')) if card else ''}")
            latest_change = ledger.latest_research_question_change(rq_id)
            if latest_change:
                st.caption(f"최근 변화 · {str(latest_change.get('created_at', ''))[:16].replace('T', ' ')} · {latest_change.get('summary', '')}")
            if linked_intents:
                st.caption(f"연결된 M1 탐색 Intent {len(linked_intents)}건")

            c1, c2, c3, c4, c5 = st.columns(5)
            if c1.button("관심", key=f"rq-interest-{rq_id}", disabled=rq.get("status") == "interested"):
                ledger.update_research_question_status(rq_id, "interested")
                st.rerun()
            if c2.button("보류", key=f"rq-hold-{rq_id}", disabled=rq.get("status") == "hold"):
                ledger.update_research_question_status(rq_id, "hold")
                st.rerun()
            if c3.button("제외", key=f"rq-reject-{rq_id}", disabled=rq.get("status") == "rejected"):
                ledger.update_research_question_status(rq_id, "rejected")
                st.rerun()
            if c4.button("이 질문 검토", key=f"rq-review-{rq_id}"):
                st.session_state["p4-selected-rq-id"] = rq_id
                st.session_state["p4-question-mode"] = "RQ Backlog에서 검토"
                st.rerun()
            if c5.button("M1 탐색 Intent", key=f"rq-intent-{rq_id}"):
                _, request_id = create_exploration_intent_for_rq(ledger, rq)
                st.session_state["p4-last-rq-intent-request"] = request_id
                st.rerun()

    if st.session_state.pop("p4-last-rq-intent-request", None):
        st.success("선택한 연구질문에서 M1 탐색 Intent 승인 안건을 만들었습니다. 연구자 홈 승인함에서 승인하면 M1 실행함으로 전달됩니다.")


M2_SOURCE_LABELS = {
    "m1_knowledge": "M1 새 지식",
    "researcher": "연구자 질문",
    "external_advisory": "외부 자문",
}


def _record_thread_report_to_desk(rq: dict[str, Any], report: dict[str, Any]) -> None:
    case_id = ledger.create_case("research", f"M2 Thread: {str(rq.get('question', ''))[:80]}")
    ledger.record(
        case_id, "advice_report", "m2", ["researcher"], "research_question_thread_report",
        {
            "title": f"M2 · {M2_SOURCE_LABELS.get(str(rq.get('source_type', 'm1_knowledge')), '연구질문')} 보고",
            "rq_id": rq.get("rq_id"), "question": rq.get("question", ""),
            "report_id": report.get("report_id"), "report": report.get("report_text", ""),
            "evidence_card_ids": report.get("evidence_card_ids", []),
            "knowledge_gaps": report.get("knowledge_gaps", ""),
            "generation_mode": report.get("generation_mode", ""),
        },
        status="completed",
    )


def _save_m2_thread_report(rq: dict[str, Any], context: str, evidence_ids: list[str], report_text: str, generation_mode: str) -> dict[str, Any]:
    gaps = extract_knowledge_gaps(report_text)
    report = ledger.save_m2_report(
        rq_id=str(rq["rq_id"]), report_text=report_text, context_text=context,
        evidence_card_ids=evidence_ids, generation_mode=generation_mode,
        report_type="external_advisory" if rq.get("source_type") == "external_advisory" else "research_review",
        knowledge_gaps=gaps,
    )
    _record_thread_report_to_desk(rq, report)
    return report


def _render_researcher_question_intake() -> None:
    st.subheader("새 연구자 질문")
    st.caption("연구자가 직접 제기한 질문은 후보 단계를 거치지 않고 바로 Research Question Thread로 시작합니다.")
    question = st.text_area(
        "연구자 질문",
        key="m2-thread-new-researcher-question",
        placeholder="예: Architect Agent의 working memory에 NFR dependency를 명시적으로 표현해야 하는가?",
        height=110,
    )
    comment = st.text_area(
        "연구자 코멘트 / 현재 맥락 (선택)",
        key="m2-thread-new-researcher-context",
        height=100,
    )
    if st.button(
        "Research Question Thread 시작",
        key="m2-thread-create-researcher",
        type="primary",
        disabled=not question.strip(),
    ):
        rq = ledger.create_research_question_thread(
            question=question,
            source_type="researcher",
            rationale="연구자가 직접 제기한 연구질문입니다.",
            research_context=comment,
            source_payload={"researcher_comment": comment},
            status="interested",
        )
        st.session_state["m2-selected-researcher-thread-id"] = str(rq["rq_id"])
        st.rerun()


def _render_external_question_intake(model: str, use_ollama: bool) -> None:
    st.subheader("새 외부 자문 요청")
    st.caption("외부 요청 원문을 보존하고, M2 전문성에 비추어 내부적으로 관리할 질문으로 해석한 뒤 Thread를 시작합니다.")
    requester = st.text_input("요청자", key="m2-thread-external-requester")
    request = st.text_area("외부 자문 요청 원문", key="m2-thread-external-request", height=140)
    context = st.text_area("요청 맥락·제약 (선택)", key="m2-thread-external-context", height=100)
    if st.button("M2 관점의 질문 해석안 만들기", key="m2-thread-interpret-external", disabled=not request.strip()):
        prompt = render_prompt(
            "m2_external_interpretation.j2", requester=requester or "external requester",
            expertise="에이전트 공학과 도메인 전문 연구위원 설계", question=request, context=context,
        )
        interpreted = llm_draft(prompt, model, use_ollama) or request
        st.session_state["m2-thread-external-interpretation"] = interpreted
    interpretation = st.session_state.get("m2-thread-external-interpretation", "")
    if interpretation:
        st.markdown("**M2 요청 해석**")
        st.markdown(interpretation)
    interpreted_question = st.text_area(
        "Thread에서 관리할 질문", value=st.session_state.get("m2-thread-external-question", request),
        key="m2-thread-external-question", height=100,
        placeholder="외부 요청을 M2 전문성에 비추어 내부적으로 관리할 질문으로 정리하세요.",
    )
    if st.button("외부 자문 Thread 시작", key="m2-thread-create-external", type="primary", disabled=not interpreted_question.strip()):
        rq = ledger.create_research_question_thread(
            question=interpreted_question, source_type="external_advisory",
            rationale="외부 자문 요청을 M2 전문성에 비추어 해석한 질문입니다.", research_context=context,
            source_payload={
                "requester": requester,
                "original_request": request,
                "request_context": context,
                "interpretation": interpretation,
            },
            status="interested",
        )
        st.session_state["m2-selected-external-thread-id"] = str(rq["rq_id"])
        st.rerun()


def _render_thread_detail(rq_id: str, model: str, use_ollama: bool, semantic: bool, embedding_model: str) -> None:
    rq = ledger.research_question_thread(rq_id)
    if not rq:
        st.warning("선택한 Research Question Thread를 찾을 수 없습니다.")
        return
    source_label = M2_SOURCE_LABELS.get(str(rq.get("source_type", "m1_knowledge")), str(rq.get("source_type", "")))
    st.subheader(str(rq.get("question", "")))
    st.caption(f"{source_label} · {RQ_STATUS_LABELS.get(str(rq.get('status')), str(rq.get('status', '')))} · {rq_id}")
    payload = rq.get("source_payload") or {}
    if rq.get("source_type") == "external_advisory":
        with st.expander("외부 요청 원문과 해석", expanded=False):
            st.write(f"**요청자:** {payload.get('requester', '')}")
            st.write(f"**원 요청:** {payload.get('original_request', '')}")
            if payload.get("interpretation"):
                st.markdown(str(payload["interpretation"]))
    if rq.get("rationale"):
        st.markdown(f"**도출 이유**  \n{rq['rationale']}")
    latest_change = ledger.latest_research_question_change(rq_id)
    if latest_change:
        st.info(f"최근 변화 · {str(latest_change.get('created_at', ''))[:16].replace('T', ' ')} · {latest_change.get('summary', '')}")

    status_cols = st.columns(4)
    for col, status, label in zip(status_cols, ["interested", "exploring", "hold", "rejected"], ["관심", "탐색중", "보류", "제외"]):
        if col.button(label, key=f"thread-status-{rq_id}-{status}", disabled=rq.get("status") == status):
            ledger.update_research_question_status(rq_id, status); st.rerun()

    source_ids = list(rq.get("source_card_ids", []))
    cards_by_id = {str(card.get("card_id", "")): card for card in memory.all()}
    if source_ids:
        with st.expander(f"질문의 출발 근거 카드 {len(source_ids)}건", expanded=False):
            for card_id in source_ids:
                card = cards_by_id.get(card_id)
                if card:
                    st.write(f"- `{card_id}` · {card.get('claim', card.get('title', ''))}")

    st.divider()
    st.markdown("### M2 Knowledge Review")
    st.caption("질문과 선택적 연구자 코멘트를 기준으로 관련 승인 지식카드를 고르고 M2 Report를 만듭니다. 제약·미결사항·지식공백은 보고서가 산출합니다.")
    default_context = str(rq.get("research_context", ""))
    context = st.text_area("이번 검토 코멘트 / 적용 맥락 (선택)", value=default_context, key=f"thread-context-{rq_id}", height=110)
    query = st.text_input("근거 지식카드 검색", value=str(rq.get("question", "")), key=f"thread-search-{rq_id}")
    if st.button("관련 지식카드 찾기", key=f"thread-search-run-{rq_id}", disabled=not query.strip()):
        hits = search_knowledge(query, semantic, embedding_model, limit=12)
        st.session_state[f"thread-hits-{rq_id}"] = [item.card["card_id"] for item in hits]
    hit_ids = st.session_state.get(f"thread-hits-{rq_id}", [])
    candidate_ids = list(dict.fromkeys([*source_ids, *hit_ids]))
    candidate_ids = [cid for cid in candidate_ids if cid in cards_by_id]
    options = {
        cid: f"{cards_by_id[cid].get('title', '')} · {cards_by_id[cid].get('claim', '')[:100]}"
        for cid in candidate_ids
    }
    selected_ids = st.multiselect(
        "이번 M2 Report에 사용할 지식카드", candidate_ids,
        default=source_ids[:8] if source_ids else candidate_ids[:6],
        format_func=lambda cid: options.get(cid, cid), key=f"thread-evidence-{rq_id}",
    )
    for cid in selected_ids[:8]:
        card = cards_by_id[cid]
        st.caption(f"`{cid}` · {card.get('claim', '')[:220]}")

    selected_cards = [cards_by_id[cid] for cid in selected_ids if cid in cards_by_id]
    prompt = build_m2_thread_review_prompt(
        question=str(rq.get("question", "")), context=context, cards=selected_cards,
        source_type=str(rq.get("source_type", "m1_knowledge")), source_payload=payload,
    )
    run_col, external_col = st.columns(2)
    with run_col:
        if st.button("현재 LLM으로 M2 Report 만들기", key=f"thread-report-internal-{rq_id}", type="primary"):
            result = llm_draft_result(prompt, model, use_ollama)
            if result.text:
                report = _save_m2_thread_report(rq, context, selected_ids, result.text, "internal_llm")
                st.session_state[f"thread-latest-report-{rq_id}"] = report["report_id"]
                st.success("M2 Report를 저장했습니다.")
                st.rerun()
            else:
                st.warning(f"M2 Report 생성에 실패했습니다: {result.error}")
    with external_col:
        st.caption("로컬/무료 LLM이 불안정하거나 여러 지식카드로 긴 프롬프트가 필요한 경우 외부 LLM을 사용할 수 있습니다.")

    with st.expander("외부 LLM으로 수동 M2 Review", expanded=False):
        prompt_key = f"thread-external-prompt-{rq_id}-{abs(hash(tuple(selected_ids))) % 100000}"
        st.text_area(
            "외부 LLM용 프롬프트", value=prompt, key=prompt_key, height=360,
            help="고정 높이 영역입니다. 긴 프롬프트는 내부 스크롤로 확인하고 전체 선택하여 복사할 수 있습니다.",
        )
        manual_response = st.text_area(
            "외부 LLM 응답 붙여넣기", key=f"thread-external-response-{rq_id}", height=280,
            placeholder="외부 LLM이 작성한 M2 Report 전체를 붙여넣으세요."
        )
        if st.button("외부 응답 검증 · M2 Report로 반영", key=f"thread-external-apply-{rq_id}", disabled=not manual_response.strip()):
            ok, message = validate_manual_m2_report(manual_response, valid_card_ids=set(selected_ids))
            if not ok:
                st.error(message)
            else:
                report = _save_m2_thread_report(rq, context, selected_ids, manual_response, "manual_external_llm")
                st.success(message + " M2 Report로 저장했습니다.")
                st.session_state[f"thread-latest-report-{rq_id}"] = report["report_id"]
                st.rerun()

    reports = ledger.m2_reports(rq_id, include_archived=True, limit=20)
    if reports:
        latest = reports[0]
        st.divider(); st.markdown("### 최신 M2 Report")
        st.caption(f"{latest['created_at'][:16].replace('T',' ')} · {latest['generation_mode']} · 근거 카드 {len(latest['evidence_card_ids'])}건")
        st.markdown(latest["report_text"])
        if latest.get("knowledge_gaps"):
            st.warning("**지식 보강 필요**  \n" + latest["knowledge_gaps"])
            if st.button("이 공백으로 M1 문헌 보강 Intent 만들기", key=f"thread-gap-intent-{rq_id}"):
                refreshed = ledger.research_question(rq_id) or rq
                _, request_id = create_exploration_intent_for_rq(ledger, refreshed)
                st.success(f"M1 탐색 Intent 승인 안건을 만들었습니다: {request_id}")
        refined = extract_refined_question(latest["report_text"])
        default_refine = refined or str(rq.get("question", ""))
        with st.expander("질문 보강·구체화", expanded=bool(refined)):
            new_question = st.text_area("구체화된 질문", value=default_refine, key=f"thread-refine-{rq_id}", height=90)
            reason = st.text_input("변경 이유", value="M2 Report와 보강된 지식을 반영한 질문 구체화", key=f"thread-refine-reason-{rq_id}")
            if st.button("질문 새 버전으로 반영", key=f"thread-refine-apply-{rq_id}", disabled=not new_question.strip() or new_question.strip() == str(rq.get("question", "")).strip()):
                ledger.refine_research_question(rq_id, new_question, reason)
                st.rerun()

    versions = rq.get("versions", [])
    linked_intents = ledger.research_question_intents(rq_id)
    with st.expander(f"Thread 이력 · 질문 버전 {len(versions)} · M2 Report {len(reports)} · M1 Intent {len(linked_intents)}", expanded=False):
        for version in versions:
            st.write(f"- v{version['version_no']} · {version['question']}")
            if version.get("change_reason"):
                st.caption(version["change_reason"])
        for report in reports:
            archive = "보관" if report.get("archived_at") else "활성"
            st.write(f"- Report `{report['report_id']}` · {archive} · {report['created_at'][:16].replace('T',' ')}")
        for intent in linked_intents:
            st.write(f"- M1 Intent `{intent['intent_id']}`")


_THREAD_STATUS_ORDER = {
    "exploring": 0,
    "interested": 1,
    "candidate": 2,
    "hold": 3,
    "rejected": 4,
}


def _sorted_threads(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # First keep the most recently changed question within each state, then force active work to the top.
    by_recent = sorted(items, key=lambda item: str(item.get("updated_at") or item.get("created_at") or ""), reverse=True)
    return sorted(by_recent, key=lambda item: _THREAD_STATUS_ORDER.get(str(item.get("status", "candidate")), 99))


def _thread_card_summary(rq: dict[str, Any]) -> str:
    rq_id = str(rq.get("rq_id", ""))
    reports = ledger.m2_reports(rq_id, include_archived=False, limit=1)
    intents = ledger.research_question_intents(rq_id)
    source_count = len(rq.get("source_card_ids", []))
    parts = []
    if source_count:
        parts.append(f"근거 {source_count}")
    if reports:
        parts.append("M2 Report 있음")
    if intents:
        parts.append(f"M1 Intent {len(intents)}")
    return " · ".join(parts) if parts else "아직 후속 작업 없음"


def _render_thread_list_for_source(
    source_type: str,
    model: str,
    use_ollama: bool,
    semantic: bool,
    embedding_model: str,
    *,
    session_key: str,
) -> None:
    backlog = ledger.research_question_backlog(limit=300)
    for item in backlog:
        if str(item.get("source_type") or "m1_knowledge") == "m1_knowledge":
            ledger.ensure_research_question_thread(str(item["rq_id"]), source_type="m1_knowledge")
    threads = []
    for item in backlog:
        thread = ledger.research_question_thread(str(item["rq_id"])) or item
        if str(thread.get("source_type", "m1_knowledge")) == source_type:
            threads.append(thread)
    threads = _sorted_threads(threads)

    st.markdown("### Research Question Threads")
    status_counts = {status: sum(1 for rq in threads if rq.get("status") == status) for status in RQ_STATUS_LABELS}
    metric_cols = st.columns(4)
    for col, status in zip(metric_cols, ["exploring", "interested", "candidate", "hold"]):
        col.metric(RQ_STATUS_LABELS[status], status_counts.get(status, 0))

    statuses = st.multiselect(
        "표시 상태",
        list(RQ_STATUS_LABELS),
        default=["exploring", "interested", "candidate", "hold"],
        format_func=lambda x: RQ_STATUS_LABELS[x],
        key=f"{session_key}-status-filter",
    )
    visible = [rq for rq in threads if rq.get("status") == "exploring" or rq.get("status") in statuses]
    visible = _sorted_threads(visible)
    if not visible:
        st.info("이 페이지에 표시할 Research Question Thread가 없습니다.")
        return

    active = [rq for rq in visible if rq.get("status") == "exploring"]
    if active:
        st.info(f"현재 작업중인 탐색중 Thread {len(active)}건을 목록 맨 위에 표시합니다.")

    selected_id = str(st.session_state.get(session_key) or "")
    if selected_id and not any(str(rq.get("rq_id")) == selected_id for rq in visible):
        selected_id = ""

    for rq in visible:
        rq_id = str(rq["rq_id"])
        status = str(rq.get("status", "candidate"))
        status_label = RQ_STATUS_LABELS.get(status, status)
        is_selected = rq_id == selected_id
        with st.container(border=True):
            top = st.columns([5, 1.4])
            with top[0]:
                st.markdown(f"#### {rq.get('question', '')}")
            with top[1]:
                if status == "exploring":
                    st.markdown("**🔵 탐색중**")
                else:
                    st.markdown(f"**{status_label}**")
            latest_change = ledger.latest_research_question_change(rq_id)
            summary = _thread_card_summary(rq)
            st.caption(summary)
            if latest_change and latest_change.get("summary"):
                st.write(f"최근 변화: {latest_change.get('summary')}")
            action_cols = st.columns([1.2, 4.8])
            if action_cols[0].button("닫기" if is_selected else "열기", key=f"{session_key}-open-{rq_id}", type="primary" if status == "exploring" and not is_selected else "secondary"):
                st.session_state[session_key] = "" if is_selected else rq_id
                st.rerun()
            action_cols[1].caption(f"{M2_SOURCE_LABELS.get(source_type, source_type)} · {rq_id}")
        if is_selected:
            with st.container(border=True):
                _render_thread_detail(rq_id, model, use_ollama, semantic, embedding_model)
            st.divider()


def _render_researcher_question_page(model: str, use_ollama: bool, semantic: bool, embedding_model: str) -> None:
    _render_researcher_question_intake()
    st.divider()
    _render_thread_list_for_source(
        "researcher", model, use_ollama, semantic, embedding_model,
        session_key="m2-selected-researcher-thread-id",
    )


def _render_external_advisory_page(model: str, use_ollama: bool, semantic: bool, embedding_model: str) -> None:
    _render_external_question_intake(model, use_ollama)
    st.divider()
    _render_thread_list_for_source(
        "external_advisory", model, use_ollama, semantic, embedding_model,
        session_key="m2-selected-external-thread-id",
    )

def _render_m1_new_information(model: str, use_ollama: bool, semantic: bool, embedding_model: str) -> None:
    updates = recent_knowledge_updates(ledger, limit=500)
    all_cards = memory.all()
    cards_by_id = {str(card.get("card_id", "")): card for card in all_cards}
    st.subheader("M1 새 정보 → 연구질문")
    st.caption("M2 연구상태 검토에서 아직 처리하지 않은 승인 지식카드입니다. 수동 모드는 RQ 후보를 Thread에 추가하고, 자동 모드는 Top 3 후속 문헌탐색까지 수행합니다.")
    st.metric("미처리 새 지식카드", len(updates))
    if updates:
        with st.expander(f"미처리 지식카드 {len(updates)}건 보기", expanded=False):
            for update in updates:
                cid = str((update.get("payload") or {}).get("card_id", ""))
                card = cards_by_id.get(cid, {})
                st.write(f"- `{cid}` · {card.get('claim', card.get('title', ''))}")
        valid_ids = {str((u.get("payload") or {}).get("card_id", "")) for u in updates if str((u.get("payload") or {}).get("card_id", ""))}
        prompt = research_question_suggestions_prompt(updates, recent_research_questions(ledger), all_cards, max_suggestions=8)
        manual_col, auto_col = st.columns(2)
        with manual_col:
            if st.button("수동 · 연구질문 후보 도출", key="m2-new-info-manual", type="primary"):
                parsed = parse_research_question_suggestions(llm_draft(prompt, model, use_ollama) or "", valid_card_ids=valid_ids, limit=8)
                if not parsed:
                    st.warning("RQ 후보를 읽지 못했습니다. 새 지식은 미처리 상태로 유지됩니다.")
                else:
                    review_id = ledger.create_research_state_review("manual", updates)
                    saved = store_research_question_candidates(ledger, parsed, updates, review_id=review_id)
                    for rq in saved:
                        ledger.ensure_research_question_thread(str(rq["rq_id"]), source_type="m1_knowledge", source_payload={"review_id": review_id})
                    ledger.complete_research_state_review(review_id, generated_rq_count=len(saved), selected_rq_count=0, summary=f"새 지식카드 {len(updates)}건에서 RQ {len(saved)}건을 도출했습니다.")
                    st.success(f"RQ Thread {len(saved)}건을 생성·보강했습니다."); st.rerun()
            with st.expander("외부 LLM으로 RQ 후보 도출", expanded=False):
                st.text_area("외부 LLM용 프롬프트", value=prompt, key="m2-rq-external-prompt", height=360)
                manual = st.text_area("외부 LLM 응답 붙여넣기", key="m2-rq-external-response", height=280)
                if st.button("외부 응답을 RQ Thread에 반영", key="m2-rq-external-apply", disabled=not manual.strip()):
                    parsed = parse_research_question_suggestions(manual, valid_card_ids=valid_ids, limit=8)
                    if not parsed:
                        st.error("RQ 블록을 읽지 못했습니다.")
                    else:
                        review_id = ledger.create_research_state_review("manual", updates)
                        saved = store_research_question_candidates(ledger, parsed, updates, review_id=review_id)
                        for rq in saved:
                            ledger.ensure_research_question_thread(str(rq["rq_id"]), source_type="m1_knowledge", source_payload={"review_id": review_id, "generation_mode": "manual_external_llm"})
                        ledger.complete_research_state_review(review_id, generated_rq_count=len(saved), selected_rq_count=0, summary=f"외부 LLM으로 새 지식카드 {len(updates)}건에서 RQ {len(saved)}건을 도출했습니다.")
                        st.success("RQ Thread에 반영했습니다."); st.rerun()
        with auto_col:
            if st.button("자동 · 새 지식 연구 사이클 실행", key="m2-new-info-auto", type="primary"):
                with st.spinner("RQ 생성·보강 → 중요도 평가 → M1 자동 문헌탐색을 수행합니다."):
                    result = execute_auto_research_cycle(
                        ledger, all_cards, CACHE,
                        rq_drafter=lambda p: llm_draft(p, model, use_ollama), priority_drafter=lambda p: llm_draft(p, model, use_ollama),
                        keyword_drafter=lambda p: llm_draft(p, model, use_ollama), abstract_reviewer=lambda p: llm_draft(p, model, use_ollama, profile="abstract_triage"),
                        fulltext_drafter=lambda p: paper_draft_result(p, model, use_ollama, "full_text_similarity").text,
                        synthesis_drafter=lambda p: llm_draft(p, model, use_ollama),
                    )
                st.session_state["m2-auto-result"] = result; st.rerun()
        auto_result = st.session_state.pop("m2-auto-result", None)
        if auto_result:
            if auto_result.get("status") in {"rq_generation_failed", "needs_attention"}:
                st.warning("자동 사이클 일부가 중단되었습니다. RQ까지 생성된 경우 Thread에는 보존되며, 연구위원 데스크의 Retry에서 후속 단계를 재개할 수 있습니다.")
            else:
                st.success(f"자동 연구 사이클 완료 · 새 지식 {auto_result.get('source_card_count',0)}건 · RQ 신규 {auto_result.get('new_rq_count',0)} / 보강 {auto_result.get('strengthened_rq_count',0)}")
    else:
        st.info("현재 M2가 처리해야 할 새 지식카드가 없습니다.")

    reviews = ledger.research_state_reviews(limit=6)
    if reviews:
        st.markdown("### 최근 연구상태 검토")
        for review in reviews:
            st.write(f"- {review['created_at'][:16].replace('T',' ')} · {'자동' if review.get('mode')=='auto' else '수동'} · 새 카드 {review.get('source_card_count',0)}건 → RQ {review.get('generated_rq_count',0)}건")
            if review.get("summary"):
                st.caption(review["summary"])

    st.divider()
    _render_thread_list_for_source(
        "m1_knowledge", model, use_ollama, semantic, embedding_model,
        session_key="m2-selected-m1-thread-id",
    )


def _render_m2_report_history() -> None:
    st.subheader("M2 Report History")
    st.caption("M2의 전체 보고서는 이 화면에서 관리합니다. 연구위원 데스크에는 최신 변화와 요약만 표시됩니다.")
    reports = ledger.m2_reports(include_archived=True, limit=300)
    if not reports:
        st.info("아직 Thread 기반 M2 Report가 없습니다.")
        return
    show_archived = st.checkbox("보관된 보고서 포함", value=True, key="m2-history-archived")
    cards_by_id = {str(card.get("card_id", "")): card for card in memory.all()}
    for report in reports:
        if not show_archived and report.get("archived_at"):
            continue
        rq = ledger.research_question_thread(str(report["rq_id"])) or {}
        source = M2_SOURCE_LABELS.get(str(rq.get("source_type", "m1_knowledge")), "M1 새 지식")
        state = "보관" if report.get("archived_at") else "활성"
        with st.expander(f"{report['created_at'][:16].replace('T',' ')} · {source} · {state} · {report['question_text']}"):
            st.caption(f"Report {report['report_id']} · {report['generation_mode']} · 근거 카드 {len(report['evidence_card_ids'])}건")
            if report.get("context_text"):
                st.markdown(f"**검토 맥락/코멘트**  \n{report['context_text']}")
            st.markdown(report["report_text"])
            if report.get("evidence_card_ids"):
                with st.expander("사용 지식카드", expanded=False):
                    for cid in report["evidence_card_ids"]:
                        card = cards_by_id.get(cid, {})
                        st.write(f"- `{cid}` · {card.get('claim', card.get('title', ''))}")
            c1, c2 = st.columns(2)
            if c1.button("보관 해제" if report.get("archived_at") else "보관", key=f"m2-report-archive-{report['report_id']}"):
                ledger.archive_m2_report(report["report_id"], archived=not bool(report.get("archived_at"))); st.rerun()
            if c2.button("보고서 제거", key=f"m2-report-delete-{report['report_id']}"):
                ledger.delete_m2_report(report["report_id"]); st.rerun()


def m2_screen(model: str, use_ollama: bool, semantic: bool, embedding_model: str) -> None:
    st.header("M2 · 지식 기반 자문 작업실")
    st.caption("질문 진입점은 별도 페이지로 관리하고, 질문이 정의된 이후의 검토·보고·M1 보강·질문 구체화 흐름은 동일한 Research Question Thread로 수행합니다.")
    pages = [
        "M1 새 지식 기반",
        "연구자 직접 질문",
        "외부 자문 요청",
        "M2 Report History",
    ]
    page = st.radio("M2 작업", pages, horizontal=True, key="m2-work-page")
    st.divider()
    if page == "M1 새 지식 기반":
        _render_m1_new_information(model, use_ollama, semantic, embedding_model)
    elif page == "연구자 직접 질문":
        _render_researcher_question_page(model, use_ollama, semantic, embedding_model)
    elif page == "외부 자문 요청":
        _render_external_advisory_page(model, use_ollama, semantic, embedding_model)
    else:
        _render_m2_report_history()

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
        interpretation = llm_draft(prompt, model, use_ollama) or (
            f"## Actual problem\n{question}\n\n## Answerable scope\nReview of structure, evidence, and conditions from the perspective of {expertise}.\n\n"
            "## Out-of-scope items and assumptions\nOrganisation-specific financial, legal, and operational facts require separate confirmation.\n\n"
            "## Response strategy\nConfirm the scope, then provide conditional advice grounded in approved knowledge."
        )
        case_id = create_external_case(ledger, requester, question, context, interpretation)
        st.session_state["external_case"] = case_id
        st.session_state["external_interpretation"] = interpretation
        st.session_state.pop("external-plan-result", None)
        st.session_state.pop("external-clusters", None)
        st.session_state.pop("external-answer-result", None)
    if "external_interpretation" in st.session_state:
        st.subheader("M2의 요청 해석·범위 확인")
        st.markdown(st.session_state["external_interpretation"])
        confirmed = st.checkbox("요청자가 이 해석과 답변 범위에 동의함")
        if st.button("자문 답변 계획 만들기", disabled=not confirmed, type="primary"):
            act_spec = recall_act_spec(
                ledger, episodic_retriever, situation=f"외부 자문 요청: {question}\n요청자: {requester}\n맥락: {context}",
                active_card_ids={card["card_id"] for card in memory.all()}, semantic=semantic, embedding_model=embedding_model,
            )
            draft = llm_draft(advisory_plan_prompt("external_advisory", question, context, requester, recall_context(act_spec)), model, use_ollama)
            plan = parse_advisory_plan(draft or "", "external_advisory", question)
            ledger.record(
                st.session_state["external_case"], "advisory_exchange", "m2", ["external_requester", "researcher"],
                "advisory_plan", {"title": "외부 자문 답변 계획", "interpretation": st.session_state["external_interpretation"],
                                  "decision_question": plan.decision_question,
                                  "subquestions": [item.question for item in plan.subquestions]}, status="completed",
            )
            st.session_state["external-plan-result"] = plan
            st.session_state["external-act-spec"] = act_spec
            st.session_state.pop("external-answer-result", None)
        plan = st.session_state.get("external-plan-result")
        if plan:
            act_spec = st.session_state.get("external-act-spec")
            if act_spec:
                show_recalled_episodes(act_spec)
            st.subheader("자문 답변 계획")
            st.write(f"**핵심 판단:** {plan.decision_question}")
            for index, subquestion in enumerate(plan.subquestions, start=1):
                st.write(f"{index}. {subquestion.question}")
            if st.button("계획에 따라 근거 수집·자문 답변 만들기", type="primary", key="external-run-plan"):
                precedent = recall_context(act_spec) if act_spec else ""
                clusters, judgments, answer = execute_plan_first_advisory(plan, f"{context}\n\n{precedent}", requester, model, use_ollama, semantic, embedding_model)
                evidence_ids = [card_id for _, cluster in clusters for card_id in cluster.card_ids]
                relation_ids = [relation_id for _, cluster in clusters for relation_id in cluster.relation_ids]
                ledger.record(
                    st.session_state["external_case"], "advisory_exchange", "m2", ["external_requester", "researcher"],
                    "advisory_response", {"title": "계획형 외부 자문 답변", "answer": answer,
                                          "evidence_card_ids": list(dict.fromkeys(evidence_ids)),
                                          "evidence_relation_ids": list(dict.fromkeys(relation_ids)),
                                          "subquestion_judgments": judgments}, status="completed",
                )
                episode = store_advisory_episode(
                    ledger, case_id=st.session_state["external_case"], episode_type="external_advisory",
                    situation_summary=f"외부 자문 요청: {question}\n요청자: {requester}\n맥락: {context}", decision_question=plan.decision_question,
                    advisory_plan=[item.question for item in plan.subquestions], answer=answer,
                    evidence_card_ids=list(dict.fromkeys(evidence_ids)), evidence_relation_ids=list(dict.fromkeys(relation_ids)),
                    unresolved_items=[subquestion.question for subquestion, cluster in clusters if not cluster.members],
                )
                st.session_state["external-episode-id"] = episode.episode_id
                st.session_state["external-clusters"] = clusters
                st.session_state["external-answer-result"] = answer
            if "external-clusters" in st.session_state:
                show_evidence_clusters(st.session_state["external-clusters"])
        if "external-answer-result" in st.session_state:
            st.subheader("외부 자문 답변")
            st.markdown(st.session_state["external-answer-result"])
            episode_id = st.session_state.get("external-episode-id")
            if episode_id and st.button("요청자 확인 후 재사용 가능한 자문 선례로 표시", key="external-confirm-episode"):
                ledger.update_episode_memory_outcome(episode_id, "confirmed", "외부 요청자 확인 후 재사용 가능한 자문 선례로 지정")
                st.success("다음 유사 자문에서 선례 기반 빠른 경로로 리콜할 수 있습니다.")



def render_workspace_sync() -> None:
    """Sidebar control for explicit local ↔ integrated-server database merge."""
    with st.sidebar.expander("작업공간 동기화", expanded=False):
        st.caption("집/학교에서는 로컬 DB로 작업하고, 필요할 때 서버 Workspace와 통합합니다.")
        st.caption(f"Local · {LOCAL_DB}")
        default_server = os.environ.get("RESEARCH_FELLOW_SERVER_DIR", "")
        if not default_server:
            legacy = os.environ.get("RESEARCH_FELLOW_SERVER_DB", "")
            if legacy:
                legacy_path = Path(legacy).expanduser()
                default_server = str(legacy_path.parent if legacy_path.suffix.lower() == ".db" else legacy_path)
        server_value = st.text_input(
            "서버 Workspace 디렉터리",
            value=default_server,
            placeholder="예: ~/Library/Mobile Documents/com~apple~CloudDocs/ResearchFellow",
            key="workspace-sync-server-dir",
            help="Google Drive/iCloud Drive/NAS의 폴더를 지정하세요. 앱이 이 안의 research_fellow.db를 자동으로 사용합니다.",
        ).strip()
        if not server_value:
            st.info("서버 Workspace 디렉터리를 지정하면 초기화 또는 통합 기능이 활성화됩니다.")
            return
        try:
            sync = WorkspaceSync(LOCAL_DB, server_value)
            st.caption(f"Server · {sync.server_db}")

            if not sync.server_exists:
                st.info("서버 DB가 아직 없습니다. 현재 Local DB를 서버의 초기 상태로 복사합니다.")
                counts = sync.initialization_counts()
                if counts:
                    labels = {
                        "paper_shelf": "논문",
                        "paper_analyses": "논문 분석",
                        "knowledge_cards": "지식카드",
                        "knowledge_relations": "카드 관계",
                        "ontology_facets": "Facet",
                        "ontology_types": "Type",
                        "ontology_type_relations": "Type 관계",
                        "ontology_card_assignments": "Card-Type",
                    }
                    summary_rows = [
                        {"데이터": labels.get(table, table), "건수": count}
                        for table, count in counts.items()
                        if count
                    ]
                    st.dataframe(summary_rows, use_container_width=True, hide_index=True, height=min(220, 38 + 35 * len(summary_rows)))
                if st.button("현재 Local DB로 서버 초기화", use_container_width=True, key="workspace-sync-init"):
                    sync.initialize_server_from_local()
                    st.success("서버 초기화 완료 · 현재 Local 상태를 동기화 기준으로 기록했습니다.")
                    st.rerun()
                return

            last = sync.last_run()
            if last:
                st.caption(f"최근 통합 · {last['completed_at']} · 적용 {last['applied_count']} · 충돌 {last['conflict_count']}")
            preview = sync.preview()
            counts = preview.counts()
            upload = sum(value for key, value in counts.items() if key.startswith("local→server:"))
            download = sum(value for key, value in counts.items() if key.startswith("server→local:"))
            conflicts = len(preview.conflicts)
            c1, c2, c3 = st.columns(3)
            c1.metric("서버로", upload)
            c2.metric("로컬로", download)
            c3.metric("충돌", conflicts)
            if preview.changes:
                rows = [
                    {
                        "방향": item.direction,
                        "동작": item.action,
                        "데이터": item.table,
                        "키": item.key,
                    }
                    for item in preview.changes
                ]
                st.dataframe(rows, use_container_width=True, hide_index=True, height=min(260, 38 + 35 * len(rows)))
            else:
                st.caption("현재 양쪽에서 병합할 변경이 없습니다.")
            if conflicts:
                st.warning("같은 레코드가 집/학교 양쪽에서 변경된 충돌은 자동 덮어쓰지 않습니다. 비충돌 변경만 통합됩니다.")
            if st.button("서버와 통합 실행", use_container_width=True, key="workspace-sync-apply"):
                result = sync.apply()
                st.success(f"통합 완료 · 비충돌 변경 {len(result.actionable)}건 적용 · 충돌 {len(result.conflicts)}건 보류")
                st.rerun()
        except Exception as exc:
            st.error(f"동기화 준비 실패: {exc}")

def main() -> None:
    st.set_page_config(page_title="Research Fellow", layout="wide")
    st.markdown("""<style>
    [data-testid="stStatusWidget"], div[data-testid="stStatusWidget"], button[data-testid="stStatusWidget"], [data-testid="stToolbar"] [aria-label*="Running"] { background:#f79009 !important; color:#1f1300 !important; border-color:#f79009 !important; font-weight:700 !important; }
    [data-testid="stStatusWidget"] *, [data-testid="stToolbar"] [aria-label*="Running"] * { color:#1f1300 !important; }
    .rf-running { position: fixed; top: 0.55rem; right: 5.9rem; z-index: 999999; max-width: 30rem; padding: 0.42rem 0.75rem; border: 1px solid #b54708; border-radius: 0.45rem; background: #f79009; color: #1f1300; font-weight: 700; box-shadow: 0 2px 7px rgba(0,0,0,.22); }
    </style>""", unsafe_allow_html=True)
    st.sidebar.title("도메인 전문 연구위원")
    render_workspace_sync()
    paper_provider = st.sidebar.radio(
        "본문 읽기·비교", ["gemini", "ollama"], horizontal=True,
        format_func={"ollama": "Ollama 로컬", "gemini": "Gemini 외부 API"}.get,
        key="llm-provider-paper-choice",
    )
    internal_provider = st.sidebar.radio(
        "내부 지식·M2 해석", ["ollama", "gemini"], horizontal=True,
        format_func={"ollama": "Ollama 로컬", "gemini": "Gemini 외부 API"}.get,
        key="llm-provider-internal-choice",
    )
    if "gemini" in {paper_provider, internal_provider} and not gemini_api_available():
        st.sidebar.warning("GEMINI_API_KEY가 없어 Gemini 외부 API를 선택할 수 없습니다.")
        paper_provider = "ollama" if paper_provider == "gemini" else paper_provider
        internal_provider = "ollama" if internal_provider == "gemini" else internal_provider
    st.session_state["llm-provider-paper"] = paper_provider
    st.session_state["llm-provider-internal"] = internal_provider
    model = st.sidebar.text_input("Ollama 모델", value="gpt-oss:20b", disabled="ollama" not in {paper_provider, internal_provider})
    use_ollama = "ollama" in {paper_provider, internal_provider}
    semantic = st.sidebar.checkbox("시드카드 임베딩 검색", value=True, help="질문과 표현이 다른 카드도 시드 후보로 찾습니다. 사용할 수 없으면 lexical 검색으로 자동 전환됩니다.")
    embedding_model = st.sidebar.text_input("임베딩 모델", value="nomic-embed-text", disabled=not semantic)
    if use_ollama:
        connected, status = ollama_status(model)
        st.sidebar.caption(f"Ollama · {status}")
        st.sidebar.caption("첫 모델 호출 시 Ollama 모델 기본 설정을 읽습니다. 논문 읽기는 Mac 메모리 사용을 위해 8K 컨텍스트를 명시합니다.")
        if not connected:
            st.sidebar.warning("초안 없이도 P1·P2 흐름은 동작합니다.")
    if paper_provider == "gemini":
        st.sidebar.caption("본문 원문은 Gemini 외부 API로 전송됩니다.")
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
        render_developer_screen(memory.all(), model, use_ollama, EXTRACTION_CACHE, ledger, CACHE / "logs" / "llm_calls.jsonl", provider=internal_provider)


if __name__ == "__main__":
    main()
