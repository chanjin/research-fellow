from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import uuid
from datetime import UTC, datetime
from dataclasses import dataclass
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import Any, Callable

# Always prefer this checkout's source tree over a previously installed package.
_LOCAL_SRC = Path(__file__).resolve().parent / "src"
if _LOCAL_SRC.is_dir() and str(_LOCAL_SRC) not in sys.path:
    sys.path.insert(0, str(_LOCAL_SRC))

import streamlit as st

import research_fellow.llm as llm_backend


@dataclass(frozen=True)
class _CompatDraftResult:
    """Compatibility result for older research_fellow.llm modules."""
    text: str | None
    error: str | None = None
    status_code: int | None = None
    diagnostics: dict[str, object] | None = None

    @property
    def ok(self) -> bool:
        return bool(self.text) and not self.error


# Do not couple the UI to one exact llm.py revision. Older project snapshots
# expose ollama_draft() but not OllamaDraftResult / ollama_draft_result().
OllamaDraftResult = getattr(llm_backend, "OllamaDraftResult", _CompatDraftResult)
ollama_draft = getattr(llm_backend, "ollama_draft", None)

def ollama_status(model: str) -> tuple[bool, str]:
    fn = getattr(llm_backend, "ollama_status", None)
    if callable(fn):
        return fn(model)
    return False, "현재 llm.py에는 ollama_status()가 없습니다."
gemini_api_available = getattr(llm_backend, "gemini_api_available", lambda: False)


def ui_text(ko: str, en: str) -> str:
    """Return a UI label in the currently selected presentation language.

    Internal IDs, DB values, and workflow keys stay unchanged; only visible UI copy
    should pass through this helper.
    """
    return en if st.session_state.get("response-language", "English") == "English" else ko


def render_origin_labels(value: dict[str, Any], *, prefix: str | None = None) -> None:
    """Render structured lineage as compact labels without losing stable IDs."""
    labels = origin_labels(value.get("origin_links", []))
    if labels:
        heading = prefix or ui_text("시작지점", "Origin")
        st.markdown(f"**{heading}** · " + " · ".join(f"`{label.replace('`', '')}`" for label in labels))

set_llm_audit_logger = getattr(llm_backend, "set_llm_audit_logger", lambda _logger: None)
set_llm_audit_log_path = getattr(llm_backend, "set_llm_audit_log_path", lambda _path: None)


def _coerce_draft_result(value: object, *, fallback_error: str | None = None) -> OllamaDraftResult:
    if hasattr(value, "text"):
        return value  # type: ignore[return-value]
    if isinstance(value, str):
        return OllamaDraftResult(value)
    if value is None:
        return OllamaDraftResult(None, fallback_error or "LLM returned no response.")
    return OllamaDraftResult(str(value))


def _backend_ollama_draft_result(
    prompt: str, model: str, enabled: bool, profile: str | None = None,
    overrides: dict[str, object] | None = None, on_chunk: Callable[[str], None] | None = None,
) -> OllamaDraftResult:
    fn = getattr(llm_backend, "ollama_draft_result", None)
    if callable(fn):
        try:
            return _coerce_draft_result(
                fn(prompt, model, enabled, profile=profile, overrides=overrides, on_chunk=on_chunk),
                fallback_error="Ollama returned no response.",
            )
        except TypeError:
            # Compatibility with intermediate revisions with fewer keyword arguments.
            try:
                return _coerce_draft_result(fn(prompt, model, enabled, profile=profile), fallback_error="Ollama returned no response.")
            except TypeError:
                return _coerce_draft_result(fn(prompt, model, enabled), fallback_error="Ollama returned no response.")
    try:
        draft_fn = getattr(llm_backend, "ollama_draft", None)
        if not callable(draft_fn):
            return OllamaDraftResult(None, "현재 llm.py에는 ollama_draft()가 없습니다.")
        text = draft_fn(prompt, model, enabled, profile=profile)
    except TypeError:
        draft_fn = getattr(llm_backend, "ollama_draft", None)
        if not callable(draft_fn):
            return OllamaDraftResult(None, "현재 llm.py에는 ollama_draft()가 없습니다.")
        text = draft_fn(prompt, model, enabled)
    if text and on_chunk:
        on_chunk(str(text))
    return _coerce_draft_result(text, fallback_error="Ollama returned no response.")


def _backend_gemini_draft_result(prompt: str, profile: str | None = None) -> OllamaDraftResult:
    fn = getattr(llm_backend, "gemini_draft_result", None)
    if not callable(fn):
        return OllamaDraftResult(None, "This llm.py revision does not provide Gemini API support.")
    try:
        return _coerce_draft_result(fn(prompt, profile=profile), fallback_error="Gemini returned no response.")
    except TypeError:
        return _coerce_draft_result(fn(prompt), fallback_error="Gemini returned no response.")
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
from research_fellow.application.prompt_tasks import (
    knowledge_grouping_prompt, knowledge_update_report_prompt, parse_knowledge_grouping,
    research_question_suggestions_prompt,
)
from research_fellow.application.meaning_summary import (
    attach_reports, build_fact_groups, delta_inputs, delta_meaning_summary_prompt, deterministic_delta_summary,
    latest_summary, meaning_summary_prompt, record_delta_summary,
)
from research_fellow.application.search_profiles import (
    abstract_relevance_prompt, attach_relevance, is_english_search_term, keyword_prompt, parse_keyword_plan, run_profile, shortlist_candidates,
    intent_discovery_task_prompt, record_external_intent_discovery, run_intent_discovery_plan, run_intent_discovery_task, scheduled_profiles,
)
from research_fellow.search_configuration import SEARCH_SOURCE_LABELS
from research_fellow.origin_lineage import cards_for_origin, merge_origin_links, normalize_origin_links, origin_labels, origin_research_context
from research_fellow.application.literature_discovery import (
    apply_discovery_triage, collect_arxiv_candidates, collect_multisource_candidates, discovery_search_plan_prompt,
    discovery_triage_prompt, external_literature_discovery_prompt, parse_discovery_search_plan,
    parse_external_literature_results, paper_access_links, download_discovery_pdf,
    google_scholar_url, build_paper_labels,
)
from research_fellow.application.paper_batch import process_top_papers
from research_fellow.application.auto_literature import execute_auto_literature_review
from research_fellow.application.research_cycle import execute_auto_research_cycle
from research_fellow.application.manual_recovery import external_recovery_prompt, validate_external_response
from research_fellow.application.m2_threads import build_m2_thread_review_prompt, extract_knowledge_gaps, extract_refined_question, validate_manual_m2_report

SEOUL_TZ = ZoneInfo("Asia/Seoul")

def _fmt_local_time(value: Any, *, with_seconds: bool = False) -> str:
    """Render stored UTC/ISO timestamps in the UI using Asia/Seoul time."""
    raw = str(value or "").strip()
    if not raw:
        return ""
    try:
        normalized = raw.replace("Z", "+00:00")
        dt = datetime.fromisoformat(normalized)
        if dt.tzinfo is None:
            # Existing Research Fellow timestamps were historically stored as UTC-like naive ISO strings.
            dt = dt.replace(tzinfo=UTC)
        local = dt.astimezone(SEOUL_TZ)
        return local.strftime("%Y-%m-%d %H:%M:%S" if with_seconds else "%Y-%m-%d %H:%M")
    except Exception:
        return raw[:19].replace("T", " ") if with_seconds else raw[:16].replace("T", " ")

from research_fellow.application.thread_documents import current_state_prompt, report_snapshot_prompt
from research_fellow.application.sensemaking import (
    sensemaking_answer_prompt, quick_search_plan_prompt, quick_literature_search,
    knowledge_card_candidate_prompt, parse_sensemaking_card_candidate,
)
from research_fellow.application.paper_shelf import StoredPaperUpload, document_from_shelf_path, document_from_source_url, ensure_shelf_pdf, pasted_paper_text_upload, store_paper_upload, suggested_paper_labels
from research_fellow.application.paper_reading import independent_card_context, parse_reading_questions, parse_reading_summary, reading_prompt, unconsumed_reading_sections
from research_fellow.application.ontology import ontology_context_dot, ontology_dot, ontology_plotly_figure, search_cards_for_ontology
from research_fellow.application.ontology_curation import (
    build_curation_context, parse_relation_suggestions, parse_type_suggestions,
    relation_suggestion_prompt, type_suggestion_prompt,
)
from research_fellow.application.ontology_evolution import ontology_delta_prompt, parse_ontology_delta
from research_fellow.application.paper_coauthor import (
    annotation_legend, apply_appendix_refresh, apply_review, apply_revisions,
    appendix_prompt as short_paper_appendix_prompt, draft_prompt as short_paper_draft_prompt,
    full_revision_prompt as short_paper_full_revision_prompt,
    manuscript_markdown, parse_manuscript, parse_review, review_prompt as short_paper_review_prompt,
    review_change_set, revision_prompt as short_paper_revision_prompt, revision_todo_timeline,
    revision_todos, sentences as manuscript_sentences,
    parse_todo_verification, todo_verification_prompt as short_paper_todo_verification_prompt,
    parse_resolution_proposal, resolution_proposal_prompt as short_paper_resolution_proposal_prompt,
    group_resolution_prompt as short_paper_group_resolution_prompt,
    parse_group_resolution, parse_todo_group_plan, search_revision_assets, selected_group_revisions,
    paper_proposal_prompt, parse_paper_proposal,
    writing_spec_guidance_prompt, parse_writing_spec_guidance,
    todo_reconciliation_prompt as short_paper_todo_reconciliation_prompt,
    parse_todo_reconciliation,
    revision_resolution_plan_prompt as short_paper_resolution_plan_prompt,
    parse_revision_resolution_plan,
    todo_grouping_prompt as short_paper_todo_grouping_prompt,
)
from research_fellow.application.paper_evidence import (
    assemble_paper_evidence_candidates, paper_evidence_query,
)
from research_fellow.application.short_paper_milestone import (
    WORKFLOW_STAGE_LABELS, frozen_milestone, latest_event_payload,
    manuscript_plain_text, normalize_writing_spec, project_workflow_stage,
    validate_short_paper,
)
from research_fellow.application.duplicate_review import similar_approved_cards
from research_fellow.application.management import delete_knowledge_card, delete_knowledge_relation
from research_fellow.application.relations import (
    RELATION_TYPES, create_relation_candidate, lineage_dot, lineage_overview_prompt,
    parse_relation_batch_drafts, relation_batch_prompt,
)
from research_fellow.infrastructure.document_reader import extract_document, extracted_document_text, infer_bibliographic_metadata
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
from research_fellow.prompt_profiles import apply_prompt_profile
from research_fellow.workspace_profiles import (
    BUILTIN_WORKSPACE_KEYS, delete_custom_workspace, get_workspace_profile,
    load_workspace_profiles, save_custom_workspace,
)
from research_fellow.workspace_archive import build_workspace_archive, restore_workspace_archive
from research_fellow.workspace_sync import AllWorkspacesSync
from research_fellow.ui.developer import render_developer_screen
from research_fellow.domain.research import CurationIntent, ResearchState


ROOT = Path(__file__).parent
DATA = Path(os.environ.get("RESEARCH_FELLOW_DATA_DIR", ROOT / "data")).expanduser()
DATA.mkdir(parents=True, exist_ok=True)
WORKSPACE_CONFIG = DATA / "workspace_profiles.json"
os.environ["RESEARCH_FELLOW_WORKSPACE_CONFIG"] = str(WORKSPACE_CONFIG)
WORKSPACE_PROFILES = load_workspace_profiles(WORKSPACE_CONFIG)


def _requested_workspace_key() -> str:
    env_key = os.environ.get("RESEARCH_FELLOW_WORKSPACE", "").strip()
    if env_key:
        return env_key
    try:
        value = st.query_params.get("workspace", "general")
        if isinstance(value, list):
            value = value[0] if value else "general"
        return str(value or "general")
    except Exception:
        return "general"


WORKSPACE_PROFILE = get_workspace_profile(_requested_workspace_key(), WORKSPACE_PROFILES)
WORKSPACE_KEY = WORKSPACE_PROFILE.key
_default_cache = ROOT / ".cache" / WORKSPACE_PROFILE.cache_name
_workspace_env_suffix = WORKSPACE_KEY.upper()
_cache_override = os.environ.get(f"RESEARCH_FELLOW_CACHE_DIR_{_workspace_env_suffix}", "").strip()
if not _cache_override and WORKSPACE_KEY == "general":
    _cache_override = os.environ.get("RESEARCH_FELLOW_CACHE_DIR", "").strip()
CACHE = Path(_cache_override or _default_cache).expanduser()
CACHE.mkdir(parents=True, exist_ok=True)
EXTRACTION_CACHE = CACHE / "extracted_documents"
_db_override = os.environ.get(f"RESEARCH_FELLOW_DB_FILENAME_{_workspace_env_suffix}", "").strip()
if not _db_override and WORKSPACE_KEY == "general":
    _db_override = os.environ.get("RESEARCH_FELLOW_DB_FILENAME", "").strip()
LOCAL_DB = DATA / (_db_override or WORKSPACE_PROFILE.db_filename)
ledger = Ledger(LOCAL_DB)
set_llm_audit_logger(ledger.record_llm_call)
set_llm_audit_log_path(CACHE / "logs" / "llm_calls.jsonl")
# SQLite is the canonical durable store. Only the general workspace imports the
# historical JSONL files; specialized workspaces start with an intentionally separate memory.
_general_legacy = WORKSPACE_KEY == "general"
memory = KnowledgeMemory(LOCAL_DB, legacy_path=(DATA / "knowledge_cards.jsonl") if _general_legacy else None)
relations = RelationMemory(LOCAL_DB, legacy_path=(DATA / "knowledge_relations.jsonl") if _general_legacy else None)
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


def lazy_page_limit(state_key: str, total: int, *, page_size: int = 40) -> int:
    """Render only an initial slice and let the researcher request more."""
    current = min(total, int(st.session_state.get(state_key, page_size)))
    if total > current:
        if st.button(
            ui_text(f"다음 {min(page_size, total-current)}건 불러오기", f"Load next {min(page_size, total-current)}"),
            key=f"{state_key}-load-more",
        ):
            st.session_state[state_key] = min(total, current + page_size)
            st.rerun()
        st.caption(ui_text(f"전체 {total}건 중 {current}건만 로딩했습니다.", f"Loaded {current} of {total}."))
    return current


def persistent_list_toggle(label: str, state_key: str, count: int) -> bool:
    """Keep long lists collapsed by default and remember the researcher's choice."""
    visible = bool(st.session_state.get(state_key, False))
    button_label = ui_text(f"{'접기' if visible else '펼치기'} · {label} {count}건", f"{'Collapse' if visible else 'Expand'} · {label} {count}")
    if st.button(button_label, key=f"{state_key}-toggle"):
        st.session_state[state_key] = not visible
        st.rerun()
    return visible


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
    prompt = apply_prompt_profile(
        prompt,
        st.session_state.get("response-language", "English"),
        requested_profile=profile,
        workspace_key=WORKSPACE_KEY,
    )
    if selected_llm_provider(workload) == "gemini":
        return _backend_gemini_draft_result(prompt, profile=profile)
    return _backend_ollama_draft_result(prompt, model, use_ollama, profile=profile, overrides=overrides, on_chunk=on_chunk)


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
        render_origin_labels(card)
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
        render_origin_labels(intent)
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
        with st.expander(f"{status} · {_fmt_local_time(item.get('created_at'))} · {payload.get('title', 'M1 자동 문헌탐색 보고')}"):
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
                    render_origin_labels(paper, prefix="탐색 시작지점")
                    if paper.get("url"):
                        st.caption(paper["url"])
                    if st.button("서재함에 추가", key=f"auto-report-shelf-{item['phenomenon_id']}-{paper.get('source_id', '')}"):
                        saved = ledger.upsert_shelf_paper({
                            "title": paper.get("title", ""), "authors": [],
                            "publication_year": str(paper.get("published", ""))[:4], "source_url": paper.get("url", ""),
                            "source_id": paper.get("source_id", ""), "abstract": paper.get("abstract", ""), "pdf_path": paper.get("pdf_path", ""),
                            "shelf_status": "reference", "reading_status": "unread", "asset_type": "paper", "intake_source": "auto_search",
                            "origin_links": paper.get("origin_links", []),
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
        with st.expander(f"주의 필요 · {title} · {_fmt_local_time(failure.get('created_at'))}"):
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
        st.caption(f"기준선: {_fmt_local_time(previous.get('created_at'))} 저장 요약 이후의 변화")
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
        with st.expander(f"{_fmt_local_time(item.get('created_at'))} · 지식 {len(change.get('card_ids', []))} · 관계 {len(change.get('relation_ids', []))} · M2 보고서 {len(change.get('report_ids', []))}"):
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
                render_origin_labels(card)
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
    st.header(WORKSPACE_PROFILE.label)
    st.markdown(f"**{ui_text(WORKSPACE_PROFILE.topic_ko, WORKSPACE_PROFILE.topic_en)}**")
    st.caption(ui_text(WORKSPACE_PROFILE.purpose, {
        "general": "A broad workspace for cross-domain interests and long-term research memory.",
        "agent_development": "A focused workspace for AI agent development, specification, workflows, memory, and evaluation.",
        "vision_ai": "A focused workspace for computer vision, multimodal AI, industrial visual inspection, and representation learning.",
    }.get(WORKSPACE_KEY, WORKSPACE_PROFILE.purpose)))
    st.divider()
    st.caption(ui_text("연구자에게 필요한 판단과 M1·M2의 최근 공유현상을 한곳에서 봅니다.", "Review researcher decisions and recent M1/M2 shared phenomena in one place."))
    # Paper reading already includes the researcher's evidence review and card
    # authoring. It therefore creates knowledge directly, not another approval
    # task. Legacy card requests remain in the ledger but are not work items.
    pending = [
        item for item in ledger.phenomena(recipient="researcher", type_="decision_request", status="proposed")
        if item["subject_type"] != "knowledge_card"
    ]
    updates = ledger.phenomena(recipient="researcher", type_="knowledge_update")
    approved_card_count = memory.count()
    active_relations = ledger.active_knowledge_relations()
    cols = st.columns(4)
    cols[0].metric(ui_text("승인 대기", "Pending approval"), len(pending))
    cols[1].metric(ui_text("승인 지식", "Approved knowledge"), approved_card_count)
    cols[2].metric(ui_text("최근 지식 업데이트", "Recent knowledge updates"), len(updates))
    cols[3].metric(ui_text("승인 관계", "Approved relations"), len(active_relations))

    st.subheader(ui_text("연구자 검토·승인함", "Researcher Review & Approval"))
    if not pending:
        st.success(ui_text("현재 연구자 판단이 필요한 안건이 없습니다.", "No items currently require researcher review."))
    else:
        selected = []
        select_all = st.checkbox(ui_text("대기 안건 전체 선택", "Select all pending items"), key="select-all-pending")
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


def _precedent_shared_terms(current: str, previous: str, limit: int = 6) -> list[str]:
    stop = {"논문", "연구", "질문", "지식카드", "작업", "현재", "대한", "관련", "판단", "등록", "the", "and", "for", "with", "from", "that", "this"}
    def toks(text: str) -> set[str]:
        return {t.lower() for t in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}|[가-힣]{2,}", text) if t.lower() not in stop}
    overlap = sorted(toks(current) & toks(previous), key=lambda x: (-len(x), x))
    return overlap[:limit]


def render_curation_precedents(situation: str, *, episode_types: set[str], semantic: bool, embedding_model: str) -> None:
    """Show prior researcher decisions as explainable precedents, not merely similar cards."""
    episodes = [
        EpisodicMemory.model_validate(item)
        for item in ledger.episode_memories()
        if item.get("episode_type") in episode_types
    ]
    recalls = episodic_retriever.recall(episodes, situation, limit=3, semantic=semantic, embedding_model=embedding_model)
    if not recalls:
        return
    st.markdown(ui_text("**참고할 만한 과거 연구자 판단**", "**Potentially relevant prior researcher decisions**"))
    st.caption(ui_text(
        "임베딩 유사도 자체가 근거는 아닙니다. 현재 작업과 겹치는 맥락, 당시 결정, 원천 논문을 보고 판단 기준만 참고하세요.",
        "Embedding similarity is not evidence by itself. Use the overlapping context, prior decision, and source papers only as decision precedents."
    ))
    cards = {card["card_id"]: card for card in memory.all()}
    for recall in recalls:
        episode = recall.episode
        decision_text = episode.decision_question
        decision_label = "등록" if "등록" in decision_text else "보류" if "보류" in decision_text else "무관" if "무관" in decision_text or "등록하지" in decision_text else "검토"
        age = _relative_time(episode.updated_at)
        episode_cards = [cards[card_id] for card_id in episode.evidence_card_ids if card_id in cards]
        shared_terms = _precedent_shared_terms(situation, episode.retrieval_text())
        sources = []
        for card in episode_cards:
            provenance = card.get("provenance") if isinstance(card.get("provenance"), dict) else {}
            source_name = str((provenance or {}).get("source_name", "")).strip()
            if source_name and source_name not in sources:
                sources.append(source_name)
        why = (
            ui_text("겹치는 맥락: ", "Shared context: ") + ", ".join(shared_terms)
            if shared_terms else
            ui_text("유사한 연구 판단 상황으로 검색됨", "Retrieved as a similar research-decision situation")
        )
        with st.expander(f"{age} · {decision_label} · {why}"):
            st.markdown(ui_text("**왜 참고하나**", "**Why this may help**"))
            st.write(why)
            st.caption(ui_text("당시 결정: ", "Prior decision: ") + decision_text)
            if sources:
                st.markdown(ui_text("**참고 논문**", "**Source papers**"))
                for source in sources[:4]:
                    st.write(f"- {source}")
            st.markdown(ui_text("**당시 작업 상황**", "**Prior work context**"))
            st.write(episode.situation_summary)
            st.markdown(ui_text("**결정 결과**", "**Decision outcome**"))
            st.write(episode.answer_summary)
            if episode.unresolved_items:
                st.caption(ui_text("미결 사항: ", "Open issues: ") + " · ".join(episode.unresolved_items))
            st.caption(ui_text(
                f"검색 보조 점수 {recall.score:.2f} · 결론을 복사하지 말고 판단 기준과 차이를 비교하세요.",
                f"Retrieval score {recall.score:.2f} · Compare decision criteria and differences rather than copying the conclusion."
            ))


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
    render_origin_labels(card)
    st.caption(f"적용 대상: {', '.join(card.get('applies_to', [])) or '미입력'} · 조건: {card.get('conditions') or '미입력'}")
    with st.expander("근거·한계 보기", expanded=False):
        st.write(card.get("evidence_excerpt") or "근거 발췌 미입력")
        st.caption("한계·유보: " + str(card.get("limits") or "미입력"))
        supporting = card.get("supporting_evidence", [])
        if supporting:
            st.caption(f"추가 보강 근거 {len(supporting)}건")
            for item in supporting:
                st.write(f"- {item.get('source_name', '출처 미상')} · {item.get('evidence_excerpt', '')}")


def render_ontology_change_review(model: str, use_ollama: bool) -> None:
    """Review new and existing card classifications, then publish a versioned snapshot."""
    raw_cards = memory.all()
    cards = [{**card, "current_ontology_types": ledger.ontology_types_for_card(str(card["card_id"]))} for card in raw_cards]
    cards_by_id = {str(card["card_id"]): card for card in cards}
    untyped = [card for card in cards if not card["current_ontology_types"]]
    typed = [card for card in cards if card["current_ontology_types"]]
    current = ledger.latest_ontology_version()
    st.markdown("### 온톨로지 변경 검토")
    st.caption("새 카드의 타입을 제안하고 기존 카드의 다중 타입 배정도 재검토합니다. 연구자의 승인 전에는 현재 온톨로지가 바뀌지 않습니다.")
    left, right, third = st.columns(3)
    left.metric("현재 버전", (current or {}).get("version_label", "초기 구조"))
    right.metric("검토 대기 카드", len(untyped))
    third.metric("승인 대기 변경안", len(ledger.ontology_change_reviews(status="proposed")))

    def _render_zoomable_type_graph(types: list[dict], relations: list[dict], facets: list[dict], highlights: dict[str, str], *, key: str) -> None:
        st.caption("점선 테두리는 Facet, 테두리 안의 상자는 Type입니다. Type 사이의 화살표는 관계를 나타냅니다.")
        st.graphviz_chart(ontology_dot(types, relations, facets, highlights), use_container_width=True)
        figure = ontology_plotly_figure(types, relations, facets, highlights)
        if figure is not None:
            with st.expander("확대·이동 가능한 그래프로 보기", expanded=False):
                st.caption("이 보기는 구조 탐색용입니다. Facet 구분은 노드 색상으로 표시됩니다. 마우스 휠로 확대·축소하고 드래그하여 이동할 수 있습니다.")
                st.plotly_chart(figure, use_container_width=True, config={"scrollZoom": True, "displaylogo": False, "responsive": True}, key=key)

    new_ids = st.multiselect(
        "이번 변경 검토에 포함할 새 지식카드", [str(card["card_id"]) for card in untyped],
        default=[str(card["card_id"]) for card in untyped],
        format_func=lambda card_id: str(cards_by_id.get(card_id, {}).get("title") or card_id),
        help="신규 카드는 개수 제한 없이 전체가 기본 선택됩니다. 필요한 경우 일부를 해제하여 나누어 검토할 수 있습니다.",
    )
    existing_ids = st.multiselect(
        "함께 재검토할 기존 지식카드 (선택)", [str(card["card_id"]) for card in typed],
        default=[], format_func=lambda card_id: (
            f"{cards_by_id.get(card_id, {}).get('title', card_id)} · "
            + ", ".join(str(item["name"]) for item in cards_by_id.get(card_id, {}).get("current_ontology_types", []))
        ),
        help="새 버전에서 기존 타입의 추가·제거가 필요한 카드를 선택합니다. 한 카드는 여러 타입을 유지할 수 있습니다.",
    )
    selected = list(dict.fromkeys([*new_ids, *existing_ids]))
    chosen = [cards_by_id[card_id] for card_id in selected if card_id in cards_by_id]
    prompt = ontology_delta_prompt(
        cards=chosen, facets=ledger.ontology_facets(), types=ledger.ontology_types(),
        relations=ledger.ontology_type_relations(),
    ) if chosen else ""
    if chosen:
        if st.button("내부 LLM으로 변경 Diff 만들기", type="primary", disabled=not selected, key="ontology-delta-generate"):
            result = llm_draft_result(prompt, model, use_ollama)
            if result.text:
                try:
                    proposal = parse_ontology_delta(result.text, cards=chosen, types=ledger.ontology_types())
                    review = ledger.create_ontology_change_review(source_card_ids=selected, proposal=proposal)
                    st.session_state["ontology-change-review-id"] = review["review_id"]
                    st.rerun()
                except ValueError as error:
                    st.error(str(error))
            else:
                st.error(f"변경 Diff 생성에 실패했습니다: {result.error}")
        with st.expander("외부 LLM으로 변경 Diff 만들기", expanded=False):
            st.caption("프롬프트를 복사하거나 필요한 경우 직접 수정한 뒤 외부 LLM에 전달하고, 받은 JSON 응답을 붙여넣으세요.")
            selection_key = abs(hash(tuple(selected))) % 1000000
            external_prompt = st.text_area(
                "외부 LLM용 프롬프트", value=prompt, height=420,
                key=f"ontology-delta-external-prompt-{selection_key}",
            )
            st.download_button("프롬프트 TXT 다운로드", data=external_prompt, file_name="ontology-delta-prompt.txt", mime="text/plain", key=f"ontology-delta-prompt-download-{selection_key}")
            manual_response = st.text_area(
                "외부 LLM JSON 응답", height=320,
                key=f"ontology-delta-external-response-{selection_key}",
                placeholder='{"summary":"...","reviewed_card_ids":["kc-..."],"assignments":[],"new_types":[],"relations":[],"warnings":[]}',
            )
            if st.button(
                "외부 응답 검증 · 변경안으로 저장", disabled=not chosen or not manual_response.strip(),
                key=f"ontology-delta-external-apply-{selection_key}",
            ):
                try:
                    proposal = parse_ontology_delta(manual_response, cards=chosen, types=ledger.ontology_types())
                    review = ledger.create_ontology_change_review(source_card_ids=selected, proposal=proposal)
                    st.session_state["ontology-change-review-id"] = review["review_id"]
                    st.success("외부 LLM 응답을 검증하고 온톨로지 변경안으로 저장했습니다.")
                    st.rerun()
                except (ValueError, TypeError) as error:
                    st.error(f"외부 응답을 반영할 수 없습니다: {error}")
    elif not untyped:
        st.info("새 미분류 카드는 없습니다. 기존 카드의 타입 배정을 재검토하려면 위에서 카드를 선택하세요.")

    versions = ledger.ontology_versions(limit=30)
    def _version_type_highlights(change: dict) -> dict[str, str]:
        changed_ids = list(change.get("changed_type_ids", []))
        if not changed_ids:
            changed_ids = [
                type_id for item in change.get("assignment_changes", [])
                for type_id in [*item.get("before_type_ids", []), *item.get("after_type_ids", [])]
            ]
        new_ids = {str(type_id) for type_id in change.get("new_type_ids", [])}
        highlights = {str(type_id): "changed" for type_id in changed_ids if str(type_id) not in new_ids}
        highlights.update({type_id: "new" for type_id in new_ids})
        return highlights

    def _render_change_legend() -> None:
        st.caption("그래프 변경 색상 · 🟩 신규 타입 · 🟨 이름·설명 또는 카드 배정이 변경된 기존 타입 · 그 외 색상은 Facet")

    last_version_id = st.session_state.get("ontology-last-published-version-id")
    highlighted = next((item for item in versions if item["version_id"] == last_version_id), None)
    if highlighted:
        st.divider(); st.markdown(f"### 방금 발행된 온톨로지 · {highlighted['version_label']}")
        st.success(highlighted.get("summary") or "온톨로지 새 버전을 발행했습니다.")
        snapshot = highlighted.get("snapshot") or {}; change = highlighted.get("change") or {}
        if snapshot.get("types"):
            _render_change_legend()
            _render_zoomable_type_graph(snapshot["types"], snapshot.get("relations", []), snapshot.get("facets", []), _version_type_highlights(change), key=f"published-ontology-{highlighted['version_id']}")
        for item in change.get("assignment_changes", []):
            type_names = {str(t["type_id"]): str(t["name"]) for t in snapshot.get("types", [])}
            before_names = [type_names.get(value, value) for value in item.get("before_type_ids", [])]
            after_names = [type_names.get(value, value) for value in item.get("after_type_ids", [])]
            st.write(f"- **{cards_by_id.get(item['card_id'], {}).get('title', item['card_id'])}**: {', '.join(before_names) or '타입 없음'} → {', '.join(after_names) or '타입 없음'}")

    reviews = ledger.ontology_change_reviews(status="proposed")
    if reviews:
        review_id = st.session_state.get("ontology-change-review-id") or reviews[0]["review_id"]
        review = next((item for item in reviews if item["review_id"] == review_id), reviews[0])
        st.divider(); st.markdown(f"### 제안 변경안 · {review['review_id']}")
        proposal = review["proposal"]
        st.write(proposal.get("summary") or "변경 요약이 없습니다.")
        for warning in proposal.get("warnings", []): st.warning(warning)
        preview_types = [dict(item) for item in ledger.ontology_types()]
        preview_relations = [dict(item) for item in ledger.ontology_type_relations()]
        preview_highlights: dict[str, str] = {}
        type_by_name = {str(item["name"]).casefold(): item for item in preview_types}
        preview_type_by_id = {str(item["type_id"]): item for item in preview_types}
        for item in proposal.get("type_updates", []):
            current_type = preview_type_by_id.get(str(item.get("type_id") or ""))
            if current_type:
                current_type["name"] = item.get("name") or current_type["name"]
                current_type["description"] = item.get("description") or current_type.get("description") or ""
                preview_highlights[str(current_type["type_id"])] = "changed"
        for item in proposal.get("assignments", []):
            existing = type_by_name.get(str(item.get("type") or "").casefold())
            if existing: preview_highlights[str(existing["type_id"])] = "changed"
        for index, item in enumerate(proposal.get("new_types", []), start=1):
            temporary_id = f"proposal-new-{index}"
            preview_types.append({"type_id":temporary_id,"name":item.get("name") or temporary_id,"description":item.get("description") or "","facet_id":None,"facet_name":item.get("facet") or "Facet 미지정","card_count":0})
            preview_highlights[temporary_id] = "new"
        preview_relations.extend(proposal.get("relations", []))
        for item in proposal.get("relation_changes", []):
            action = item.get("action")
            if action == "delete":
                preview_relations = [row for row in preview_relations if str(row.get("relation_id")) != str(item.get("relation_id"))]
            elif action == "update":
                for row in preview_relations:
                    if str(row.get("relation_id")) == str(item.get("relation_id")):
                        row.update({key: value for key, value in item.items() if value and key in {"source_type_id", "target_type_id", "relation_name", "description"}})
            elif action == "add":
                preview_relations.append(item)
        st.markdown("#### 제안 온톨로지 그래프")
        _render_change_legend()
        _render_zoomable_type_graph(preview_types, preview_relations, ledger.ontology_facets(), preview_highlights, key=f"proposal-ontology-{review['review_id']}")
        tabs = st.tabs(["카드 타입 배정", "신규 타입", "타입 관계", "연구자 검토"])
        with tabs[0]:
            grouped: dict[str, list[dict]] = {}
            for item in proposal.get("assignments", []): grouped.setdefault(str(item["card_id"]), []).append(item)
            for card_id in proposal.get("reviewed_card_ids", []):
                desired = grouped.get(str(card_id), [])
                current_names = [str(item["name"]) for item in cards_by_id.get(str(card_id), {}).get("current_ontology_types", [])]
                desired_names = [str(item["type"]) for item in desired]
                st.markdown(f"**{cards_by_id.get(str(card_id), {}).get('title', card_id)}**")
                st.write(f"현재: {', '.join(current_names) or '타입 없음'} → 제안: {', '.join(desired_names) or '타입 없음'}")
                for item in desired: st.caption(f"{item['type']} · {item.get('confidence', 'medium')} · {item.get('reason') or '근거 설명 없음'}")
        with tabs[1]:
            for item in proposal.get("new_types", []):
                with st.container(border=True):
                    st.success(f"신규 타입 · {item['facet'] or 'Facet 미지정'} / {item['name']}")
                    st.write(item.get("description") or "설명 없음"); st.caption(item.get("reason") or "생성 이유 없음")
            for item in proposal.get("type_updates", []):
                st.warning(f"타입 수정 · {item.get('type_id')} → {item.get('name')}")
                st.write(item.get("description") or "설명 없음"); st.caption(item.get("reason") or "수정 이유 없음")
        with tabs[2]:
            for item in proposal.get("relations", []):
                st.markdown(f"**{item['source_type']}** → `{item['relation_name']}` → **{item['target_type']}**")
                st.write(item.get("description") or "설명 없음"); st.caption(item.get("reason") or "제안 근거 없음")
            for item in proposal.get("relation_changes", []):
                st.markdown(f"**관계 {item.get('action')}** · {item.get('relation_id') or '신규'} · `{item.get('relation_name') or '이름 유지'}`")
                st.caption(item.get("reason") or "변경 이유 없음")
        with tabs[3]:
            st.caption("그래프를 확대·이동하며 아래 구조화 의견을 선택하거나 자유 의견을 입력하세요. 수정 Diff를 생성한 뒤 다시 검토·승인할 수 있습니다.")
            current_types = ledger.ontology_types(); current_relations = ledger.ontology_type_relations()
            type_options = [str(item["type_id"]) for item in current_types]
            type_lookup = {str(item["type_id"]): item for item in current_types}
            relation_options = [str(item["relation_id"]) for item in current_relations]
            relation_lookup = {str(item["relation_id"]): item for item in current_relations}
            opinion_kind = st.selectbox("구조화 의견", ["선택 안 함", "타입 이름·설명 수정", "관계 이름·설명 수정", "관계 추가", "관계 삭제"], key=f"ontology-opinion-kind-{review['review_id']}")
            structured_comment = ""
            if opinion_kind == "타입 이름·설명 수정" and type_options:
                type_id = st.selectbox("수정할 타입", type_options, format_func=lambda value: type_lookup[value]["name"], key=f"ontology-opinion-type-{review['review_id']}")
                revised_name = st.text_input("새 타입 이름", value=str(type_lookup[type_id]["name"]), key=f"ontology-opinion-type-name-{review['review_id']}")
                revised_description = st.text_area("새 타입 설명", value=str(type_lookup[type_id].get("description") or ""), key=f"ontology-opinion-type-desc-{review['review_id']}")
                structured_comment = f"타입 수정: type_id={type_id}, 이름='{revised_name}', 설명='{revised_description}'."
            elif opinion_kind in {"관계 이름·설명 수정", "관계 삭제"} and relation_options:
                relation_id = st.selectbox("대상 관계", relation_options, format_func=lambda value: f"{type_lookup.get(str(relation_lookup[value]['source_type_id']), {}).get('name', '?')} — {relation_lookup[value]['relation_name']} → {type_lookup.get(str(relation_lookup[value]['target_type_id']), {}).get('name', '?')}", key=f"ontology-opinion-relation-{review['review_id']}")
                if opinion_kind == "관계 삭제":
                    structured_comment = f"관계 삭제: relation_id={relation_id}."
                else:
                    relation_name = st.text_input("새 관계 이름", value=str(relation_lookup[relation_id]["relation_name"]), key=f"ontology-opinion-relation-name-{review['review_id']}")
                    relation_description = st.text_area("새 관계 설명", value=str(relation_lookup[relation_id].get("description") or ""), key=f"ontology-opinion-relation-desc-{review['review_id']}")
                    structured_comment = f"관계 수정: relation_id={relation_id}, 이름='{relation_name}', 설명='{relation_description}'."
            elif opinion_kind == "관계 추가" and len(type_options) >= 2:
                source_id = st.selectbox("출발 타입", type_options, format_func=lambda value: type_lookup[value]["name"], key=f"ontology-opinion-source-{review['review_id']}")
                target_id = st.selectbox("도착 타입", type_options, index=1, format_func=lambda value: type_lookup[value]["name"], key=f"ontology-opinion-target-{review['review_id']}")
                relation_name = st.text_input("관계 이름", key=f"ontology-opinion-new-relation-name-{review['review_id']}")
                relation_description = st.text_area("관계 설명", key=f"ontology-opinion-new-relation-desc-{review['review_id']}")
                structured_comment = f"관계 추가: source_type_id={source_id}, target_type_id={target_id}, 이름='{relation_name}', 설명='{relation_description}'."
            comment = st.text_area("자유 수정·보완 요청", value=review.get("researcher_comment", ""), placeholder="예: 이 카드의 기존 Method 타입은 제거하고 Claim과 Evaluation Context를 함께 부여해줘.", key=f"ontology-review-comment-{review['review_id']}")
            combined_comment = "\n".join(value for value in [structured_comment, comment.strip()] if value)
            review_cards = [cards_by_id[card_id] for card_id in review.get("source_card_ids", []) if card_id in cards_by_id]
            revision_prompt = ontology_delta_prompt(cards=review_cards, facets=ledger.ontology_facets(), types=current_types, relations=current_relations, comment=combined_comment)
            c1, c2, c3 = st.columns(3)
            if c1.button("코멘트 저장", key=f"ontology-review-comment-save-{review['review_id']}"):
                ledger.update_ontology_change_review_comment(review["review_id"], combined_comment); st.success("검토 의견을 저장했습니다.")
            if c2.button("의견 반영 Diff 재생성", disabled=not review_cards or not combined_comment, key=f"ontology-review-regenerate-{review['review_id']}"):
                result = llm_draft_result(revision_prompt, model, use_ollama)
                if result.text:
                    try:
                        revised = parse_ontology_delta(result.text, cards=review_cards, types=current_types)
                        new_review = ledger.create_ontology_change_review(source_card_ids=review["source_card_ids"], proposal=revised, base_version_id=review.get("base_version_id"))
                        ledger.update_ontology_change_review_comment(new_review["review_id"], combined_comment)
                        st.session_state["ontology-change-review-id"] = new_review["review_id"]; st.rerun()
                    except ValueError as error: st.error(str(error))
                else: st.error(f"수정 Diff 생성에 실패했습니다: {result.error}")
            if c3.button("검토 승인 · 새 버전 발행", type="primary", key=f"ontology-review-approve-{review['review_id']}"):
                try:
                    version = ledger.approve_ontology_change_review(review["review_id"], summary=proposal.get("summary", ""))
                    st.session_state["ontology-last-published-version-id"] = version["version_id"]
                    st.session_state.pop("ontology-change-review-id", None); st.rerun()
                except ValueError as error: st.error(str(error))
            with st.expander("외부 LLM으로 의견 반영 Diff 재생성", expanded=False):
                revised_external_prompt = st.text_area("수정 Diff 프롬프트", value=revision_prompt, height=420, key=f"ontology-revision-prompt-{review['review_id']}")
                revised_external_response = st.text_area("외부 LLM JSON 응답", height=300, key=f"ontology-revision-response-{review['review_id']}")
                if st.button("외부 수정 응답 검증 · 저장", disabled=not revised_external_response.strip(), key=f"ontology-revision-apply-{review['review_id']}"):
                    try:
                        revised = parse_ontology_delta(revised_external_response, cards=review_cards, types=current_types)
                        new_review = ledger.create_ontology_change_review(source_card_ids=review["source_card_ids"], proposal=revised, base_version_id=review.get("base_version_id"))
                        ledger.update_ontology_change_review_comment(new_review["review_id"], combined_comment)
                        st.session_state["ontology-change-review-id"] = new_review["review_id"]; st.rerun()
                    except (ValueError, TypeError) as error: st.error(f"외부 수정 응답을 반영할 수 없습니다: {error}")

    if versions:
        st.divider(); st.markdown("### 온톨로지 버전 이력")
        for version in versions:
            with st.expander(f"{version['version_label']} · {_fmt_local_time(version.get('approved_at'))} · {version.get('summary') or '변경 요약 없음'}", expanded=False):
                snapshot = version.get("snapshot") or {}; change = version.get("change") or {}
                st.caption(f"기반 버전: {version.get('base_version_id') or '없음'} · 반영 카드 {len(version.get('source_card_ids', []))}건")
                if snapshot.get("types"):
                    _render_change_legend()
                    _render_zoomable_type_graph(snapshot["types"], snapshot.get("relations", []), snapshot.get("facets", []), _version_type_highlights(change), key=f"history-ontology-{version['version_id']}")
                st.write(f"카드 타입 변경 {len(change.get('assignment_changes', []))}건 · 변경 타입 {len(change.get('changed_type_ids', []))}건 · 신규 타입 {len(change.get('new_type_ids', []))}건 · 신규 관계 {len(change.get('new_relation_ids', []))}건 · 수정 관계 {len(change.get('updated_relation_ids', []))}건 · 삭제 관계 {len(change.get('deleted_relation_ids', []))}건")


def render_ontology_workspace(model: str, use_ollama: bool, semantic: bool, embedding_model: str) -> None:
    """Facet-aware, multi-type ontology builder with contextual graph feedback."""
    st.subheader(ui_text("온톨로지", "Ontology"))
    st.caption(ui_text("승인 지식카드에 여러 타입을 부여하고, Facet으로 타입을 묶습니다. 타입 간 관계는 Type↔Type 사이에 정의합니다.", "Assign multiple Types to approved knowledge cards, group Types with Facets, and define relations between Types."))
    if flash := st.session_state.pop("ontology-flash", None):
        st.success(flash)
    cards = memory.all()
    cards_by_id = {card["card_id"]: card for card in cards}
    approved_relations = ledger.active_knowledge_relations()
    review_tab, map_tab = st.tabs([
        ui_text("변경 검토", "Change Review"),
        ui_text("온톨로지 맵", "Ontology Map"),
    ])

    def _card_display_name(card_id: str) -> str:
        card = cards_by_id.get(str(card_id), {})
        return str(card.get("title") or card.get("claim") or card_id).strip()[:90]

    def _type_display_name(type_row: dict) -> str:
        return f"{type_row.get('facet_name') or 'Facet 미지정'} = {type_row.get('name') or type_row.get('type_id')}"

    def _assignment_flash(card_id: str, type_row: dict) -> str:
        return (
            f"지식카드 ‘{_card_display_name(card_id)}’에 타입 "
            f"‘{_type_display_name(type_row)}’이 지정되었고 Ontology에 반영되었습니다."
        )

    def _render_manual_card_source_context(card: dict, key_suffix: str) -> None:
        """Show enough source context to assign additional Types confidently."""
        provenance = card.get("provenance") if isinstance(card.get("provenance"), dict) else {}
        source_paper = None
        paper_id = str((provenance or {}).get("paper_id") or "").strip()
        if paper_id:
            source_paper = ledger.shelf_paper(paper_id)
        if source_paper is None:
            source_name = str((provenance or {}).get("source_name") or "").strip().casefold()
            if source_name:
                source_paper = next(
                    (item for item in ledger.shelf_papers() if str(item.get("title", "")).strip().casefold() == source_name),
                    None,
                )
        source_analysis = ledger.paper_analysis(source_paper["paper_id"]) if source_paper else None
        with st.expander(ui_text("논문 · 원문 맥락 보기", "Paper · source context"), expanded=False):
            st.markdown(f"**{ui_text('논문 제목', 'Paper title')}**  \n{(source_paper or {}).get('title') or (provenance or {}).get('source_name') or ui_text('출처 제목 정보 없음', 'No source title')}" )
            abstract = str((source_paper or {}).get("abstract") or "").strip()
            if abstract:
                st.markdown(f"**{ui_text('초록', 'Abstract')}**")
                st.write(abstract)
            elif source_analysis and str(source_analysis.get("summary") or "").strip():
                st.markdown(f"**{ui_text('논문 분석 요약', 'Paper analysis summary')}**")
                st.write(source_analysis.get("summary"))
            if card.get("context"):
                st.markdown(f"**{ui_text('카드 맥락', 'Card context')}**")
                st.write(card.get("context"))
            if card.get("implication"):
                st.markdown(f"**{ui_text('설계·연구 함의', 'Research / design implication')}**")
                st.write(card.get("implication"))
            if card.get("source_excerpt"):
                st.markdown(f"**{ui_text('카드 주변 원문', 'Nearby source text')}**")
                st.text_area(
                    ui_text("주변 원문", "Nearby source text"), value=str(card.get("source_excerpt")),
                    height=180, key=f"ontology-manual-source-{key_suffix}", disabled=True, label_visibility="collapsed",
                )
            evidence_lines = [line.strip() for line in str(card.get("evidence_excerpt") or "").splitlines() if line.strip()]
            if evidence_lines:
                st.markdown(f"**{ui_text('직접 원문 근거', 'Direct source evidence')}**")
                for evidence in evidence_lines[:5]:
                    st.markdown(f"- {evidence}")

    def _sync_card_types(card_id: str, widget_key: str) -> None:
        selected_ids = list(st.session_state.get(widget_key, []))
        ledger.set_card_ontology_types(card_id, selected_ids)
        st.session_state["ontology-focus-card-id"] = card_id
        current_type_rows = {str(t["type_id"]): t for t in ledger.ontology_types()}
        selected_labels = [
            _type_display_name(current_type_rows[type_id])
            for type_id in selected_ids
            if type_id in current_type_rows
        ]
        if selected_labels:
            st.session_state["ontology-flash"] = (
                f"지식카드 ‘{_card_display_name(card_id)}’의 타입이 "
                f"‘{' / '.join(selected_labels)}’로 지정되었고 Ontology에 반영되었습니다."
            )
        else:
            st.session_state["ontology-flash"] = (
                f"지식카드 ‘{_card_display_name(card_id)}’의 타입 지정이 해제되었고 Ontology에 반영되었습니다."
            )

    with review_tab:
        render_ontology_change_review(model, use_ollama)

    # Legacy AI/manual tools share one deliberately secondary workspace at the
    # bottom of Change Review. Normal evolution should use the versioned flow.
    maintenance_tab = review_tab.expander(
        ui_text("고급 유지관리 · 복구 및 초기 구성", "Advanced Maintenance · recovery and initialization"),
        expanded=False,
    )
    ai_tab = maintenance_tab
    builder_tab = maintenance_tab

    with ai_tab:
        st.markdown("### 고급 유지관리")
        st.warning("일상적인 타입·관계 변경은 ‘변경 검토’를 사용하세요. 아래 기능은 초기 구조 구성, 긴급 복구, 개별 카드 진단을 위한 호환 도구입니다.")
        st.markdown("#### 레거시 AI 큐레이션 도구")
        st.caption("기존 단일 카드 중심 큐레이션 흐름입니다. 변경 이력과 승인 단위가 필요한 작업은 Change Review에서 수행하세요.")

        def _clear_ai_card_work_state(card_id: str) -> None:
            """Clear temporary curation state only when the researcher finishes this card."""
            st.session_state.pop("ontology-ai-target-card", None)
            for prefix in (
                "ontology-ai-type-result-",
                "ontology-ai-type-error-",
                "ontology-ai-external-type-response-",
                "ontology-ai-rel-result-",
                "ontology-ai-rel-error-",
                "ontology-ai-external-rel-response-",
            ):
                st.session_state.pop(f"{prefix}{card_id}", None)

        facets = ledger.ontology_facets()
        types = ledger.ontology_types()
        relations_now = ledger.ontology_type_relations()
        # Treat a card as classified only when it currently has at least one active Type.
        # Query card assignments directly so the metric/list reflects an approval immediately
        # after Streamlit reruns, including assignments created from AI Curation.
        current_types_by_card_ai = {
            str(card.get("card_id")): ledger.ontology_types_for_card(str(card.get("card_id")))
            for card in cards
        }
        untyped_cards = [
            card for card in cards
            if not current_types_by_card_ai.get(str(card.get("card_id")))
        ]
        a, b = st.columns(2)
        a.metric("미분류 지식카드", len(untyped_cards))
        b.metric("현재 Ontology Type", len(types))
        # A card should stay open after its first Type is assigned so the researcher can
        # add secondary Facet=Type assignments without searching for the card again.
        # "Untyped" is therefore a queue condition, not a curation-complete condition.
        untyped_ids = [str(card["card_id"]) for card in untyped_cards]
        current_ai_target = str(st.session_state.get("ontology-ai-target-card") or "")
        include_typed_cards = st.checkbox(
            ui_text("기존 타입이 있는 카드도 포함 · 추가 타입 지정/보정", "Include already typed cards · add/correct Types"),
            value=False, key="ontology-ai-include-typed",
            help=ui_text(
                "기존 카드에 두 번째·세 번째 Type을 추가하려면 켜세요. 같은 논문·원문 맥락과 AI 제안 기능을 그대로 사용할 수 있습니다.",
                "Enable this to reopen existing cards and add secondary Types with the same paper/source context and AI suggestions.",
            ),
        )
        target_ids = list(untyped_ids)
        if include_typed_cards:
            target_ids.extend(str(card["card_id"]) for card in cards if str(card["card_id"]) not in set(target_ids))
        if current_ai_target in cards_by_id and current_ai_target not in target_ids:
            target_ids.insert(0, current_ai_target)

        if not untyped_cards and not target_ids:
            st.success("현재 승인 지식카드는 모두 하나 이상의 타입이 지정되어 있습니다.")
            st.caption("위 옵션을 켜면 기존 타입 카드도 다시 열어 논문·원문 맥락을 보면서 추가 Type을 지정할 수 있습니다.")
        else:
            target_id = st.selectbox(
                "온톨로지 작업 카드",
                target_ids,
                format_func=lambda cid: (
                    ("[미분류] " if cid in untyped_ids else "[기존 타입 · 추가 지정] ")
                    + f"{cards_by_id[cid].get('title') or cards_by_id[cid].get('claim','')[:70]}"
                ),
                key="ontology-ai-target-card",
                help="타입을 하나 승인해도 현재 카드는 계속 열려 있습니다. 필요한 추가 타입을 모두 지정한 뒤 '이 카드 타입 지정 완료'를 누르세요.",
            )
            target = cards_by_id[target_id]
            assigned_now = ledger.ontology_types_for_card(target_id)
            if assigned_now:
                status_cols = st.columns([3, 1])
                status_cols[0].success(
                    f"미분류 해소 · 현재 {len(assigned_now)}개 Type 지정됨 · 추가 Type을 계속 지정할 수 있습니다."
                )
                if status_cols[1].button(
                    "이 카드 타입 지정 완료",
                    key=f"ontology-ai-complete-card-{target_id}",
                    help="현재 카드 작업을 닫고 다음 미분류 카드로 이동합니다. Type assignment는 그대로 유지됩니다.",
                ):
                    _clear_ai_card_work_state(target_id)
                    st.rerun()
            st.markdown(f"#### {target.get('title') or '제목 없음'}")
            st.write(target.get("claim") or "")
            if target.get("context"):
                st.caption("맥락 · " + str(target.get("context")))
            if target.get("implication"):
                st.info("연구/설계 함의 · " + str(target.get("implication")))

            provenance = target.get("provenance") or {}
            source_paper = None
            paper_id = str(provenance.get("paper_id") or "").strip()
            if paper_id:
                source_paper = ledger.shelf_paper(paper_id)
            if source_paper is None:
                source_name = str(provenance.get("source_name") or "").strip().casefold()
                if source_name:
                    source_paper = next((p for p in ledger.shelf_papers() if str(p.get("title", "")).strip().casefold() == source_name), None)
            source_analysis = ledger.paper_analysis(source_paper["paper_id"]) if source_paper else None
            paper_labels = list((source_paper or {}).get("labels") or [])
            with st.expander("논문 · 원문 맥락 보기", expanded=False):
                st.markdown(f"**논문 제목**  \n{(source_paper or {}).get('title') or provenance.get('source_name') or '출처 제목 정보 없음'}")
                abstract = str((source_paper or {}).get("abstract") or "").strip()
                if abstract:
                    st.markdown("**초록**")
                    st.write(abstract)
                elif source_analysis and str(source_analysis.get("summary") or "").strip():
                    st.markdown("**초록 미보존 · 논문 분석 요약**")
                    st.write(source_analysis.get("summary"))
                else:
                    st.caption("초록이 저장되어 있지 않습니다. 앞으로 arXiv/문헌탐색에서 서재함에 등록하는 논문은 초록도 함께 보존합니다.")
                if target.get("source_excerpt"):
                    st.markdown("**카드 주변 원문**")
                    st.text_area(
                        "주변 원문", value=str(target.get("source_excerpt")), height=180,
                        key=f"ontology-source-excerpt-{target_id}", disabled=True, label_visibility="collapsed",
                    )
                elif target.get("evidence_excerpt"):
                    st.markdown("**근거 원문**")
                    st.write(target.get("evidence_excerpt"))
                if target.get("evidence_excerpt") and target.get("source_excerpt"):
                    st.markdown("**직접 근거**")
                    st.write(target.get("evidence_excerpt"))

            # Human correction path: AI Curation must not become read-only after
            # suggestions are generated. Researchers can add/remove any existing
            # Types directly on the current card, before or after AI approval.
            with st.container(border=True):
                st.markdown("##### 현재 카드 타입 · 연구자 직접 보정")
                live_facets = ledger.ontology_facets()
                live_types = ledger.ontology_types()
                live_assigned = ledger.ontology_types_for_card(target_id)
                live_ids = [str(t["type_id"]) for t in live_assigned]
                type_options = [str(t["type_id"]) for t in live_types]
                type_label = {
                    str(t["type_id"]): f"{t.get('facet_name') or 'Facet 미지정'} = {t['name']}"
                    for t in live_types
                }
                selected_live_ids = st.multiselect(
                    "현재 카드에 지정할 Type",
                    options=type_options,
                    default=[type_id for type_id in live_ids if type_id in type_options],
                    format_func=lambda type_id: type_label.get(type_id, type_id),
                    key=f"ontology-ai-manual-types-{target_id}-{'-'.join(sorted(live_ids)) or 'none'}",
                    help="여러 Type을 동시에 지정할 수 있습니다. 기존 Type을 제거하면 카드에서 해당 Type assignment가 해제됩니다.",
                )
                mc1, mc2 = st.columns([1, 2])
                if mc1.button("타입 변경 반영", key=f"ontology-ai-manual-apply-{target_id}"):
                    before_ids = set(live_ids)
                    after_ids = set(selected_live_ids)
                    added_ids = after_ids - before_ids
                    removed_ids = before_ids - after_ids
                    ledger.set_card_ontology_types(target_id, selected_live_ids)
                    st.session_state["ontology-focus-card-id"] = target_id
                    type_by_id = {str(t["type_id"]): t for t in live_types}
                    changes = []
                    if added_ids:
                        changes.append("추가: " + ", ".join(_type_display_name(type_by_id[x]) for x in added_ids if x in type_by_id))
                    if removed_ids:
                        changes.append("제거: " + ", ".join(_type_display_name(type_by_id[x]) for x in removed_ids if x in type_by_id))
                    detail = " / ".join(changes) if changes else "변경 없음"
                    st.session_state["ontology-flash"] = (
                        f"지식카드 ‘{_card_display_name(target_id)}’의 타입 지정이 반영되었습니다. {detail}"
                    )
                    st.rerun()
                mc2.caption("AI 후보와 무관하게 기존 Type을 직접 추가·삭제할 수 있습니다.")

                with st.expander("+ 새 Type을 직접 만들어 현재 카드에 추가"):
                    if paper_labels:
                        st.caption("출처 논문 Labels · " + " · ".join(paper_labels))
                    else:
                        st.caption("출처 논문 Labels · 없음")
                    facet_options_manual = [None, *[str(f["facet_id"]) for f in live_facets], "__new__"]
                    manual_facet = st.selectbox(
                        "Facet (선택 사항)",
                        facet_options_manual,
                        format_func=lambda value: (
                            "Facet 미지정" if value is None else
                            "+ 새 Facet" if value == "__new__" else
                            next((f["name"] for f in live_facets if str(f["facet_id"]) == value), value)
                        ),
                        key=f"ontology-ai-manual-new-facet-{target_id}",
                    )
                    manual_new_facet_name = ""
                    if manual_facet == "__new__":
                        manual_new_facet_name = st.text_input(
                            "새 Facet 이름", key=f"ontology-ai-manual-new-facet-name-{target_id}"
                        )
                    manual_type_name = st.text_input(
                        "새 Type 이름", key=f"ontology-ai-manual-new-type-name-{target_id}"
                    )
                    manual_type_desc = st.text_area(
                        "Type 설명 (선택)", height=90, key=f"ontology-ai-manual-new-type-desc-{target_id}"
                    )
                    if st.button(
                        "새 Type 생성 · 현재 카드에 추가",
                        key=f"ontology-ai-manual-create-type-{target_id}",
                        disabled=not manual_type_name.strip(),
                    ):
                        try:
                            facet_id = manual_facet
                            if manual_facet == "__new__":
                                if not manual_new_facet_name.strip():
                                    raise ValueError("새 Facet 이름을 입력하거나 Facet 미지정을 선택하세요.")
                                existing_facet = next(
                                    (f for f in live_facets if str(f["name"]).casefold() == manual_new_facet_name.strip().casefold()),
                                    None,
                                )
                                facet_id = (
                                    existing_facet["facet_id"] if existing_facet
                                    else ledger.create_ontology_facet(manual_new_facet_name.strip())["facet_id"]
                                )
                            created = ledger.create_ontology_type(
                                manual_type_name.strip(), manual_type_desc.strip(), facet_id
                            )
                            current_ids = [str(t["type_id"]) for t in ledger.ontology_types_for_card(target_id)]
                            ledger.set_card_ontology_types(target_id, [*current_ids, str(created["type_id"])])
                            st.session_state["ontology-focus-card-id"] = target_id
                            created_row = {**created, "facet_name": next((f["name"] for f in ledger.ontology_facets() if str(f["facet_id"]) == str(created.get("facet_id"))), None)}
                            st.session_state["ontology-flash"] = _assignment_flash(target_id, created_row)
                            st.rerun()
                        except ValueError as error:
                            st.error(str(error))

            context = build_curation_context(
                retriever, target, cards, approved_relations, ledger,
                embedding_model=embedding_model, semantic=semantic, limit=10,
            )
            left, right = st.columns([1.05, 1.35], gap="large")
            with left:
                st.markdown("##### 유사 카드와 기존 타입")
                if not context.similar_cards:
                    st.info("유사 카드가 아직 충분히 검색되지 않았습니다. LLM은 기존 전체 타입도 함께 비교합니다.")
                for similar in context.similar_cards[:8]:
                    label = similar.get("title") or similar.get("claim", "")[:80]
                    with st.expander(f"{label} · {similar.get('similarity_score', 0):.2f}"):
                        st.write(similar.get("claim") or "")
                        similar_paper_labels = list(similar.get("source_paper_labels") or [])
                        if similar_paper_labels:
                            st.caption("출처 논문 레이블 · " + " · ".join(similar_paper_labels))
                        if similar.get("source_paper_title"):
                            st.caption("출처 논문 · " + str(similar.get("source_paper_title")))
                        assigned = similar.get("ontology_types") or []
                        if assigned:
                            st.caption("현재 타입 · 이 카드에 바로 적용 가능")
                            for type_idx, t in enumerate(assigned):
                                tc1, tc2 = st.columns([3, 1])
                                tc1.markdown(f"**{t.get('facet_name') or 'Facet 미지정'}** = {t['name']}")
                                if tc2.button(
                                    "이 타입 적용",
                                    key=f"ontology-ai-use-similar-type-{target_id}-{similar.get('card_id')}-{t['type_id']}-{type_idx}",
                                ):
                                    ledger.assign_cards_to_ontology_type(t["type_id"], [target_id])
                                    st.session_state["ontology-focus-card-id"] = target_id
                                    st.session_state["ontology-flash"] = _assignment_flash(target_id, t)
                                    st.rerun()
                        else:
                            st.caption("타입 미지정")

            with right:
                prompt = type_suggestion_prompt(context)
                st.session_state[f"ontology-ai-type-prompt-{target_id}"] = prompt
                c1, c2 = st.columns(2)
                if c1.button("AI 타입 후보 생성", type="primary", key=f"ontology-ai-generate-{target_id}"):
                    result = llm_draft_result(prompt, model, use_ollama, profile="ontology")
                    if result.text:
                        try:
                            st.session_state[f"ontology-ai-type-result-{target_id}"] = parse_type_suggestions(
                                result.text, existing_facets=facets, existing_types=types
                            )
                            st.session_state.pop(f"ontology-ai-type-error-{target_id}", None)
                        except ValueError as error:
                            st.session_state[f"ontology-ai-type-error-{target_id}"] = str(error)
                    else:
                        st.session_state[f"ontology-ai-type-error-{target_id}"] = result.error or "LLM 응답이 없습니다."
                c2.caption("긴 프롬프트이거나 로컬/무료 LLM이 불안정하면 아래 외부 LLM 경로를 사용하세요.")

                with st.expander("외부 LLM으로 타입 후보 만들기", expanded=bool(st.session_state.get(f"ontology-ai-type-error-{target_id}"))):
                    external_prompt = st.text_area(
                        "외부 LLM용 프롬프트",
                        value=prompt,
                        height=360,
                        key=f"ontology-ai-external-type-prompt-{target_id}",
                        help="고정 높이 영역입니다. 내용이 길면 내부 스크롤로 확인한 뒤 전체를 복사해 외부 LLM에 전달하세요.",
                    )
                    external_response = st.text_area(
                        "외부 LLM 응답 붙여넣기",
                        height=280,
                        key=f"ontology-ai-external-type-response-{target_id}",
                        placeholder="외부 LLM의 JSON 응답을 여기에 붙여넣으세요.",
                    )
                    if st.button("외부 응답 검증 · 후보로 반영", key=f"ontology-ai-apply-external-type-{target_id}", disabled=not external_response.strip()):
                        try:
                            st.session_state[f"ontology-ai-type-result-{target_id}"] = parse_type_suggestions(
                                external_response, existing_facets=facets, existing_types=types
                            )
                            st.session_state.pop(f"ontology-ai-type-error-{target_id}", None)
                            st.success("외부 응답을 검증해 타입 후보로 반영했습니다.")
                        except ValueError as error:
                            st.error(str(error))

                if error := st.session_state.get(f"ontology-ai-type-error-{target_id}"):
                    st.warning(f"타입 후보 생성 실패: {error}")

                suggestion = st.session_state.get(f"ontology-ai-type-result-{target_id}")
                if suggestion:
                    if suggestion.get("summary"):
                        st.info(suggestion["summary"])
                    for warning in suggestion.get("warnings", []):
                        st.warning(warning)
                    st.markdown("##### 타입 후보 · 연구자 승인")
                    facet_by_name = {str(f["name"]).casefold(): f for f in facets}
                    for idx, rec in enumerate(suggestion.get("recommendations", []), start=1):
                        action = rec.get("action")
                        candidate_name = rec.get("type") or "이름 미정"
                        facet_name = rec.get("facet") or "Facet 미지정"
                        with st.container(border=True):
                            st.markdown(f"**{idx}. {facet_name} = {candidate_name}**")
                            st.caption(f"제안: {action} · confidence {float(rec.get('confidence') or 0):.2f}")
                            if rec.get("reason"):
                                st.write(rec["reason"])
                            if action == "assign_existing" and rec.get("type_id"):
                                if st.button("기존 타입으로 승인", key=f"ontology-ai-approve-existing-{target_id}-{idx}"):
                                    ledger.assign_cards_to_ontology_type(rec["type_id"], [target_id])
                                    st.session_state["ontology-focus-card-id"] = target_id
                                    approved_type = next((t for t in ledger.ontology_types() if str(t["type_id"]) == str(rec["type_id"])), {"type_id": rec["type_id"], "name": candidate_name, "facet_name": facet_name})
                                    st.session_state["ontology-flash"] = _assignment_flash(target_id, approved_type)
                                    st.rerun()
                            else:
                                similar_types = rec.get("similar_types") or []
                                if similar_types:
                                    st.markdown("**유사 기존 타입 Top 5 · 차별점**")
                                    for comparison in similar_types[:5]:
                                        st.caption(
                                            f"{comparison.get('type','?')} · {comparison.get('similarity','')} — {comparison.get('difference','차이 설명 없음')}"
                                        )
                                facet_options = [None, *[f["facet_id"] for f in facets], "__new__"]
                                proposed_facet = facet_by_name.get(str(rec.get("facet") or "").casefold())
                                default_facet = proposed_facet["facet_id"] if proposed_facet else ("__new__" if rec.get("facet") else None)
                                selected_facet = st.selectbox(
                                    "Facet (선택 사항)", facet_options,
                                    index=facet_options.index(default_facet) if default_facet in facet_options else 0,
                                    format_func=lambda v: "Facet 미지정" if v is None else ("+ 새 Facet" if v == "__new__" else next((f["name"] for f in facets if f["facet_id"] == v), v)),
                                    key=f"ontology-ai-new-facet-select-{target_id}-{idx}",
                                )
                                new_facet_name = ""
                                if selected_facet == "__new__":
                                    new_facet_name = st.text_input(
                                        "새 Facet 이름", value=str(rec.get("facet") or ""),
                                        key=f"ontology-ai-new-facet-name-{target_id}-{idx}",
                                    )
                                edited_name = st.text_input(
                                    "Type 이름", value=str(candidate_name), key=f"ontology-ai-new-type-name-{target_id}-{idx}"
                                )
                                edited_desc = st.text_area(
                                    "Type 설명 (선택)", value=str(rec.get("description") or ""), height=90,
                                    key=f"ontology-ai-new-type-desc-{target_id}-{idx}",
                                )
                                if st.button("이 이름으로 신규 Type 승인", key=f"ontology-ai-create-type-{target_id}-{idx}", disabled=not edited_name.strip()):
                                    try:
                                        facet_id = selected_facet
                                        if selected_facet == "__new__":
                                            if not new_facet_name.strip():
                                                raise ValueError("새 Facet 이름을 입력하거나 Facet 미지정을 선택하세요.")
                                            existing_facet = facet_by_name.get(new_facet_name.strip().casefold())
                                            facet_id = existing_facet["facet_id"] if existing_facet else ledger.create_ontology_facet(new_facet_name.strip())["facet_id"]
                                        created = ledger.create_ontology_type(edited_name, edited_desc, facet_id)
                                        ledger.assign_cards_to_ontology_type(created["type_id"], [target_id])
                                        st.session_state["ontology-focus-card-id"] = target_id
                                        created_row = {**created, "facet_name": next((f["name"] for f in ledger.ontology_facets() if str(f["facet_id"]) == str(created.get("facet_id"))), None)}
                                        st.session_state["ontology-flash"] = _assignment_flash(target_id, created_row)
                                        st.rerun()
                                    except ValueError as error:
                                        st.error(str(error))

            # Relation proposal is a second LLM step and only becomes available
            # after the researcher has approved at least one Type for the card.
            approved_for_target = ledger.ontology_types_for_card(target_id)
            if approved_for_target:
                st.divider()
                st.markdown("### 승인된 Type을 기준으로 관계 후보 검토")
                st.caption("타입 승인과 관계 승인을 분리합니다. 여기서도 LLM은 후보만 제안하며 승인된 관계만 Ontology에 반영됩니다.")
                relation_prompt = relation_suggestion_prompt(
                    approved_for_target, ledger.ontology_types(), ledger.ontology_type_relations(), target
                )
                r1, r2 = st.columns(2)
                if r1.button("AI 관계 후보 생성", key=f"ontology-ai-rel-generate-{target_id}"):
                    result = llm_draft_result(relation_prompt, model, use_ollama)
                    if result.text:
                        try:
                            st.session_state[f"ontology-ai-rel-result-{target_id}"] = parse_relation_suggestions(
                                result.text, existing_types=ledger.ontology_types()
                            )
                            st.session_state.pop(f"ontology-ai-rel-error-{target_id}", None)
                        except ValueError as error:
                            st.session_state[f"ontology-ai-rel-error-{target_id}"] = str(error)
                    else:
                        st.session_state[f"ontology-ai-rel-error-{target_id}"] = result.error or "LLM 응답이 없습니다."
                r2.caption("관계 후보도 외부 LLM 수동 경로를 사용할 수 있습니다.")
                with st.expander("외부 LLM으로 관계 후보 만들기", expanded=bool(st.session_state.get(f"ontology-ai-rel-error-{target_id}"))):
                    st.text_area(
                        "외부 LLM용 관계 프롬프트", value=relation_prompt, height=360,
                        key=f"ontology-ai-external-rel-prompt-{target_id}",
                    )
                    relation_response = st.text_area(
                        "외부 LLM 관계 응답 붙여넣기", height=280,
                        key=f"ontology-ai-external-rel-response-{target_id}",
                    )
                    if st.button("외부 관계 응답 검증 · 후보로 반영", key=f"ontology-ai-external-rel-apply-{target_id}", disabled=not relation_response.strip()):
                        try:
                            st.session_state[f"ontology-ai-rel-result-{target_id}"] = parse_relation_suggestions(
                                relation_response, existing_types=ledger.ontology_types()
                            )
                            st.session_state.pop(f"ontology-ai-rel-error-{target_id}", None)
                            st.success("외부 응답을 관계 후보로 반영했습니다.")
                        except ValueError as error:
                            st.error(str(error))
                if error := st.session_state.get(f"ontology-ai-rel-error-{target_id}"):
                    st.warning(f"관계 후보 생성 실패: {error}")
                rel_result = st.session_state.get(f"ontology-ai-rel-result-{target_id}")
                if rel_result:
                    for warning in rel_result.get("warnings", []):
                        st.warning(warning)
                    current_relation_keys = {
                        (r["source_type_id"], r["target_type_id"], str(r["relation_name"]).casefold())
                        for r in ledger.ontology_type_relations()
                    }
                    for idx, rel in enumerate(rel_result.get("relations", []), start=1):
                        duplicate = (rel["source_type_id"], rel["target_type_id"], rel["relation_name"].casefold()) in current_relation_keys
                        with st.container(border=True):
                            st.markdown(f"**{rel['source_type']} → `{rel['relation_name']}` → {rel['target_type']}**")
                            if rel.get("reason"):
                                st.caption(rel["reason"])
                            if duplicate:
                                st.info("이미 동일한 Type 관계가 존재합니다.")
                            else:
                                edited_rel_name = st.text_input(
                                    "관계명", value=rel["relation_name"], key=f"ontology-ai-rel-name-{target_id}-{idx}"
                                )
                                edited_rel_desc = st.text_area(
                                    "관계 설명 (선택)", value=str(rel.get("description") or ""), height=80,
                                    key=f"ontology-ai-rel-desc-{target_id}-{idx}",
                                )
                                if st.button("관계 승인", key=f"ontology-ai-rel-approve-{target_id}-{idx}"):
                                    try:
                                        ledger.create_ontology_type_relation(
                                            rel["source_type_id"], rel["target_type_id"], edited_rel_name, edited_rel_desc
                                        )
                                        st.success("관계를 승인해 Ontology에 반영했습니다.")
                                        st.rerun()
                                    except ValueError as error:
                                        st.error(str(error))

    with builder_tab:
        st.divider()
        st.markdown("### 직접 편집·복구 도구")
        st.caption("LLM 없이 특정 카드 배정이나 관계를 즉시 복구해야 할 때만 사용합니다. 일반 변경은 Change Review를 권장합니다.")
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
                    _render_manual_card_source_context(card, f"builder-{card['card_id']}")
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
                                    created_row = {**new_type, "facet_name": next((f["name"] for f in ledger.ontology_facets() if str(f["facet_id"]) == str(new_type.get("facet_id"))), None)}
                                    st.session_state["ontology-flash"] = _assignment_flash(card["card_id"], created_row)
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
        st.markdown("### 현재 승인된 Ontology Map")
        st.caption("이 화면은 승인된 현재 구조를 탐색하는 읽기 전용 화면입니다. 변경은 ‘변경 검토’에서 제안·검토·승인하세요.")
        if types: st.graphviz_chart(ontology_dot(types, relations, facets), use_container_width=True)
        else: st.info("타입을 먼저 정의하면 전체 타입 그래프가 표시됩니다.")
        st.markdown("### Facet · Type 현황")
        st.caption("Facet은 Type의 메타분류이며 그래프에서는 점선 영역과 색상으로 표시됩니다.")
        for facet in facets:
            facet_types = [item for item in types if str(item.get("facet_id") or "") == str(facet["facet_id"])]
            with st.expander(f"{facet['name']} · Type {len(facet_types)}개"):
                st.write(facet.get("description") or "설명 없음")
                for item in facet_types:
                    st.markdown(f"- **{item['name']}** · 카드 {item['card_count']}건")
                    if item.get("description"): st.caption(item["description"])
        ungrouped_types = [item for item in types if not item.get("facet_id")]
        if ungrouped_types:
            with st.expander(f"Facet 미지정 · Type {len(ungrouped_types)}개"):
                for item in ungrouped_types: st.markdown(f"- **{item['name']}** · 카드 {item['card_count']}건")
        if types:
            st.markdown("### Type별 지식카드")
            for item in types:
                with st.expander(f"[{item.get('facet_name') or 'Facet 미지정'}] {item['name']} · 카드 {item['card_count']}건"):
                    st.write(item.get("description") or "설명 없음")
                    for card_id in ledger.ontology_card_ids(item["type_id"]): st.write(f"- {cards_by_id.get(card_id, {}).get('title', card_id)}")
        if relations:
            st.markdown("### 정의된 관계")
            for r in relations:
                st.write(f"**{type_by_id.get(r['source_type_id'],{}).get('name',r['source_type_id'])}** → `{r['relation_name']}` → **{type_by_id.get(r['target_type_id'],{}).get('name',r['target_type_id'])}**")
                if r.get("description"): st.caption(r["description"])

def render_paper_shelf(model: str, use_ollama: bool, semantic: bool, embedding_model: str) -> None:
    """Paper assets are read and reviewed here, not converted to claims on intake."""
    st.subheader("서재함")
    st.caption("탐색 결과와 직접 업로드 자료가 한곳에 모입니다. 카드화는 논문 읽기·연구자 첨삭 이후에만 가능합니다.")
    status_labels = {"core": "핵심 참고", "reference": "참고", "held": "보류", "excluded": "제외"}
    reading_labels = {"unread": "미읽음", "reading": "읽는 중", "read": "읽음"}
    total_papers = ledger.shelf_paper_count()
    st.metric("서재함 전체", total_papers)
    st.caption("논문 목록과 논문별 리뷰 상태는 목록을 펼칠 때만 불러옵니다. 펼친 상태는 현재 세션에서 유지됩니다.")
    if not persistent_list_toggle("서재함 논문", "paper-shelf-list-visible", total_papers):
        return
    visible_paper_count = lazy_page_limit("paper-shelf-visible-count", total_papers, page_size=30)
    all_papers = ledger.shelf_papers(limit=visible_paper_count)
    reviewed_ids = {paper["paper_id"] for paper in all_papers if ledger.paper_reading_questions(paper["paper_id"]) or ledger.paper_card_ids(paper["paper_id"])}
    knowledge_ids = {paper["paper_id"] for paper in all_papers if ledger.paper_card_ids(paper["paper_id"])}
    metric_a, metric_b, metric_c = st.columns(3)
    metric_a.metric("현재 로딩", len(all_papers))
    metric_b.metric("현재 로딩분 리뷰", len(reviewed_ids))
    metric_c.metric("현재 로딩분 지식 연결", len(knowledge_ids))
    query = st.text_input("논문 검색", placeholder="제목, 저자, 레이블", key="paper-shelf-query")
    filter_status = st.selectbox("중요도", ["all", *status_labels], format_func=lambda value: "전체" if value == "all" else status_labels[value], key="paper-shelf-filter")
    review_filter = st.selectbox("검토·지식화 현황", ["all", "reviewed", "knowledge_linked", "not_reviewed"], format_func={"all": "전체", "reviewed": "리뷰됨", "knowledge_linked": "승인 지식 연결됨", "not_reviewed": "아직 읽기 전"}.get)
    all_labels = sorted({label for paper in all_papers for label in paper.get("labels", [])}, key=str.casefold)
    selected_labels = st.multiselect("레이블 필터 (선택한 레이블을 모두 포함)", all_labels, key="paper-shelf-label-filter")
    if total_papers > len(all_papers):
        st.caption(f"검색·필터는 현재 불러온 최신 {len(all_papers)}편에 적용됩니다. 과거 자료가 필요하면 위 ‘다음 불러오기’를 누르세요.")
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
            render_origin_labels(paper, prefix=ui_text("발견 시작지점", "Discovery origin"))
            paper_lineage_context = origin_research_context(paper.get("origin_links", []))
            if paper_lineage_context:
                with st.expander(ui_text("이 문헌을 찾은 연구질문·의도 맥락", "Research question and intent behind this paper"), expanded=False):
                    st.text(paper_lineage_context)
                    st.caption(ui_text(
                        "이 맥락은 지식카드 본문에 복사하지 않고 발견·활용 계보(origin_links)로만 보존됩니다.",
                        "This context is kept only as discovery/use lineage (origin_links), not copied into the Knowledge Card body.",
                    ))
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
                with st.expander(ui_text("원문 URL 수정", "Edit source URL"), expanded=False):
                    with st.form(f"paper-source-url-{paper['paper_id']}"):
                        source_url_value = st.text_input(
                            ui_text("원문 URL", "Source URL"),
                            value=paper.get("source_url", ""),
                            placeholder="https://...",
                        )
                        source_url_saved = st.form_submit_button(ui_text("URL 저장", "Save URL"))
                    if source_url_saved:
                        ledger.update_shelf_paper(
                            paper["paper_id"],
                            shelf_status=paper["shelf_status"],
                            reading_status=paper["reading_status"],
                            source_url=source_url_value,
                        )
                        st.success(ui_text("원문 URL을 저장했습니다.", "Source URL saved."))
                        st.rerun()
                if paper.get("source_url"):
                    st.link_button(ui_text("원문 페이지 열기", "Open source page"), paper["source_url"], key=f"shelf-url-{paper['paper_id']}")
                path = Path(paper["pdf_path"]) if paper.get("pdf_path") and Path(paper["pdf_path"]).exists() else None
                if path:
                    st.download_button("보관 원문 내려받기", data=path.read_bytes(), file_name=path.name, key=f"shelf-download-{paper['paper_id']}")
                else:
                    st.caption(ui_text(
                        "보관 원문 파일 없음 · 원문 URL을 사용하거나 아래에서 다운로드한 파일·붙여넣은 텍스트를 등록할 수 있습니다.",
                        "No stored source file · Use the source URL or register a downloaded file / pasted text below.",
                    ))
                    with st.expander(ui_text("원문 파일·텍스트 보완", "Add source file or pasted text"),expanded=not bool(paper.get("source_url"))):
                        shelf_upload=st.file_uploader(
                            ui_text("다운로드한 PDF/TXT/MD", "Downloaded PDF/TXT/MD"),type=["pdf","txt","md"],
                            key=f"shelf-source-file-{paper['paper_id']}",
                        )
                        shelf_pasted_text=st.text_area(
                            ui_text("원문 텍스트 Copy/Paste", "Copy/paste paper text"),height=160,
                            key=f"shelf-source-text-{paper['paper_id']}",
                        )
                        shelf_file_col,shelf_text_col=st.columns(2)
                        if shelf_file_col.button(ui_text("원문 파일 연결", "Attach source file"),disabled=shelf_upload is None,key=f"shelf-source-file-save-{paper['paper_id']}"):
                            try:
                                saved=_store_manual_discovery_source(paper,shelf_upload,shelf_match=paper,intake_source="shelf_manual_file")
                                st.success(ui_text(f"원문 파일을 연결했습니다: {saved['title']}",f"Source file attached: {saved['title']}"));st.rerun()
                            except Exception as error:st.error(ui_text(f"원문 파일 등록 실패: {error}",f"Could not register source file: {error}"))
                        if shelf_text_col.button(ui_text("붙여넣은 원문 연결", "Attach pasted text"),disabled=not shelf_pasted_text.strip(),key=f"shelf-source-text-save-{paper['paper_id']}"):
                            try:
                                pasted_upload=pasted_paper_text_upload(str(paper.get("title") or "paper"),shelf_pasted_text)
                                saved=_store_manual_discovery_source(paper,pasted_upload,shelf_match=paper,intake_source="shelf_pasted_text")
                                st.success(ui_text(f"붙여넣은 원문을 연결했습니다: {saved['title']}",f"Pasted source attached: {saved['title']}"));st.rerun()
                            except Exception as error:st.error(ui_text(f"원문 텍스트 등록 실패: {error}",f"Could not register pasted text: {error}"))
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
            question = st.text_area(
                ui_text("이 논문을 읽는 연구 질문·활용 맥락", "Research question / intended use for reading this paper"),
                value=analysis.get("research_question", ""),
                key=f"shelf-question-{paper['paper_id']}",
            )
            qsave_col, _ = st.columns([1, 3])
            if qsave_col.button(ui_text("질문·활용 맥락 저장", "Save reading context"), key=f"save-shelf-question-{paper['paper_id']}"):
                ledger.save_paper_analysis(paper["paper_id"], research_question=question)
                st.success(ui_text("이 논문을 읽는 질문·활용 맥락을 저장했습니다.", "Reading question / context saved."))
                st.rerun()
            st.markdown("**M1 논문 읽기 · 요약–해석–질문–첨삭**")
            st.caption("논문 요약, 현재 연구 맥락 해석, 추천 서재 레이블, 원문 근거 기반 읽기 질문을 한 번에 만듭니다.")
            has_reading_record = bool(analysis.get("reading_raw_output") or ledger.paper_reading_questions(paper["paper_id"]))
            reading_action_label = "M1 논문 다시 읽기" if has_reading_record else "M1 논문 읽기 시작"
            if st.button(reading_action_label, type="primary", key=f"paper-reading-{paper['paper_id']}"):
                try:
                    if path and path.exists():
                        document = extract_document(document_from_shelf_path(str(path)), cache_dir=EXTRACTION_CACHE)
                    elif paper.get("source_url"):
                        document = extract_document(document_from_source_url(paper), cache_dir=EXTRACTION_CACHE)
                    else:
                        raise ValueError("읽을 원문이 없습니다. 로컬 PDF를 등록하거나 HTML 원문 URL을 입력해 주세요.")
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
                prompt_error_key=f"manual-reading-prompt-error-{paper['paper_id']}"
                if st.button("외부 채팅용 M1 프롬프트 만들기", key=f"make-{prompt_key}"):
                    st.session_state.pop(prompt_key,None);st.session_state.pop(prompt_error_key,None)
                    try:
                        if path and path.exists():
                            document = extract_document(document_from_shelf_path(str(path)), cache_dir=EXTRACTION_CACHE)
                        elif paper.get("source_url"):
                            document = extract_document(document_from_source_url(paper), cache_dir=EXTRACTION_CACHE)
                        else:
                            raise ValueError("읽을 원문이 없습니다. 로컬 PDF를 등록하거나 HTML 원문 URL을 입력해 주세요.")
                        st.session_state[prompt_key] = reading_prompt(document, paper, question)
                    except Exception as error:
                        st.session_state[prompt_error_key]=str(error)
                prompt_error=str(st.session_state.get(prompt_error_key) or "")
                if prompt_error:
                    st.warning(f"M1 프롬프트 준비 중 원문을 가져오지 못했습니다: {prompt_error}")
                    st.caption("사이트 접근을 반복하지 않고 아래 대체 원문으로 프롬프트를 생성할 수 있습니다.")
                fallback_file=st.file_uploader(
                    "대체 원문 파일 · PDF/TXT/MD",type=["pdf","txt","md"],
                    key=f"manual-reading-source-file-{paper['paper_id']}",
                )
                fallback_text=st.text_area(
                    "대체 원문 텍스트 Copy/Paste",height=160,key=f"manual-reading-source-text-{paper['paper_id']}",
                    placeholder="브라우저에서 확보한 논문 본문을 붙여 넣으세요.",
                )
                persist_fallback=st.checkbox(
                    "이 대체 원문을 서재함 논문에도 연결",value=True,key=f"manual-reading-source-persist-{paper['paper_id']}",
                    help="다음 M1 읽기와 M2 리비전에서도 같은 원문을 다시 사용할 수 있습니다.",
                )
                if st.button(
                    "대체 원문으로 M1 프롬프트 만들기",type="primary",
                    disabled=fallback_file is None and not fallback_text.strip(),key=f"manual-reading-source-build-{paper['paper_id']}",
                ):
                    try:
                        fallback_upload=fallback_file if fallback_file is not None else pasted_paper_text_upload(str(paper.get("title") or "paper"),fallback_text)
                        document=extract_document(fallback_upload,max_pages=60,chars_per_page=8000,cache_dir=EXTRACTION_CACHE)
                        st.session_state[prompt_key]=reading_prompt(document,paper,question)
                        st.session_state.pop(prompt_error_key,None)
                        if persist_fallback:
                            _store_manual_discovery_source(paper,fallback_upload,shelf_match=paper,intake_source="m1_manual_prompt_fallback")
                            st.success("대체 원문을 서재함에 연결하고 M1 프롬프트를 만들었습니다.")
                        else:st.success("대체 원문으로 M1 프롬프트를 만들었습니다. 원문 파일은 서재함에 저장하지 않았습니다.")
                    except Exception as error:st.error(f"대체 원문으로 M1 프롬프트를 만들지 못했습니다: {error}")
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
                    st.markdown("**원문 근거**")
                    for evidence in list(item["evidence"])[:5]:
                        st.markdown(f"- {evidence}")
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
                            "register" if item["status"] in {"proposed", "promoted"}
                            else "defer" if item["status"] == "deferred"
                            else "irrelevant"
                        )
                        decision = st.radio("연구자 결정", decision_options, index=decision_options.index(existing_decision), horizontal=True, format_func={"register": "지식카드 등록", "defer": "보류", "irrelevant": "무관"}.get)
                        comment = st.text_area("근거 해석·첨삭", value=item["researcher_comment"])
                        st.caption("질문은 논문을 읽는 렌즈입니다. 카드 제목은 목록에서 구별하기 위한 짧은 요약이고, Claim은 근거와 조건을 포함한 완전한 주장입니다. 결정 전에도 제안값을 자유롭게 다듬을 수 있으며, 입력값은 ‘지식카드 등록’을 선택했을 때 카드에 반영됩니다.")
                        evidence_text = st.text_area(
                            "원문 근거 (선택 · 한 줄에 하나 · 최대 5개)", value="\n".join(list(item["evidence"])[:5]),
                            help="확인 가능한 페이지와 원문 단서가 있으면 함께 남기세요. 근거 수나 p.N 형식은 카드 등록을 제한하지 않습니다.",
                        )
                        card_title = st.text_input("카드 제목 (짧은 요약)", value=item.get("suggested_title", ""), help="Claim을 그대로 반복하지 말고, 목록·계보에서 구별할 수 있는 짧은 명사구로 작성합니다. 예: ‘명세 우선 설계의 품질 효과’")
                        card_claim = st.text_area("주장 (Claim)", value=item["tentative_answer"], help="근거와 적용 범위를 포함해 한 문장으로 독립적으로 이해되는 완전한 주장입니다.")
                        card_context_default = independent_card_context(item, str(analysis.get("summary") or ""))
                        card_context = st.text_area(
                            "지식 맥락 · 논문 탐색·읽기 요약", value=card_context_default, height=150,
                            help="원 논문 자체의 과업·비교·문제 설정만 간결하게 남깁니다. 현재 작성 중인 논문, 연구질문, Revision To-do 맥락은 카드 본문에 넣지 않고 origin_links 계보로만 연결됩니다.",
                        )
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
                        if decision == "register" and len(final_evidence) > 5:
                            st.error("원문 근거는 핵심 위치 최대 5개까지만 유지해 주세요. 중복되거나 중요도가 낮은 근거를 줄인 뒤 다시 저장하세요.")
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
                                "research_context": card_context.strip(),
                                "origin_links": paper.get("origin_links", []),
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
                                "context": card_context.strip(),
                                "labels": [value.strip() for value in card_labels.split(",") if value.strip()],
                                "concepts": [value.strip() for value in card_concepts.split(",") if value.strip()],
                                "applies_to": [value.strip() for value in card_applies_to.split(",") if value.strip()],
                                "evidence_level": card_evidence_level, "status": "verified",
                                "evidence_excerpt": "\n".join(final_evidence)[:1600], "evidence_pages": [],
                                "citation_markers": final_evidence, "conditions": card_conditions.strip(), "limits": card_limits.strip(),
                                "provenance": {"source_name": paper["title"], "paper_id": paper["paper_id"], "grounding": "paper_reading_researcher_registration", "reading_question": item["question"]},
                                "origin_links": paper.get("origin_links", []),
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
                        st.caption(f"{_fmt_local_time(event.get('created_at'))} · {event['event_type']}")
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


def _discovery_paper_key(paper: dict[str, Any]) -> str:
    return (
        str(paper.get("source_id", "")).strip()
        or str(paper.get("source_url", "")).strip()
        or str(paper.get("html_url", "")).strip()
        or str(paper.get("pdf_url", "")).strip()
        or str(paper.get("title", "")).strip().lower()
    )


def _normalized_paper_title(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()


def _paper_identity_tokens(paper: dict[str, Any]) -> set[str]:
    tokens: set[str] = set()
    source_id = str(paper.get("source_id", "")).strip().lower()
    doi = str(paper.get("doi", "")).strip().lower()
    if source_id:
        tokens.add("id:" + source_id)
    if doi:
        doi = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", doi).strip()
        tokens.add("doi:" + doi)
    for field in ("source_url", "html_url", "pdf_url", "url"):
        url = str(paper.get(field, "")).strip().lower()
        if not url:
            continue
        doi_match = re.search(r"doi\.org/(10\.[^?#\s]+)", url)
        if doi_match:
            tokens.add("doi:" + doi_match.group(1).rstrip("/"))
        arxiv_match = re.search(r"arxiv\.org/(?:abs|html|pdf)/(\d{4}\.\d{4,5})(?:v\d+)?", url)
        if arxiv_match:
            tokens.add("arxiv:" + arxiv_match.group(1))
    if source_id:
        arxiv_match = re.fullmatch(r"(?:arxiv:)?(\d{4}\.\d{4,5})(?:v\d+)?", source_id)
        if arxiv_match:
            tokens.add("arxiv:" + arxiv_match.group(1))
    title = _normalized_paper_title(str(paper.get("title", "")))
    if title:
        tokens.add("title:" + title)
    return tokens


def _discovery_shelf_match(paper: dict[str, Any], shelf_papers: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidate_tokens = _paper_identity_tokens(paper)
    if not candidate_tokens:
        return None
    strong = {token for token in candidate_tokens if not token.startswith("title:")}
    candidate_title = next((token for token in candidate_tokens if token.startswith("title:")), "")
    for shelf_paper in shelf_papers:
        shelf_tokens = _paper_identity_tokens(shelf_paper)
        shelf_strong = {token for token in shelf_tokens if not token.startswith("title:")}
        if strong and shelf_strong and strong.intersection(shelf_strong):
            return shelf_paper
        shelf_title = next((token for token in shelf_tokens if token.startswith("title:")), "")
        if candidate_title and candidate_title == shelf_title:
            return shelf_paper
    return None


def _store_manual_discovery_source(
    paper: dict[str,Any], upload: Any, *, shelf_match: dict[str,Any] | None=None,
    intake_source: str="llm_discovery_manual_source",
) -> dict[str,Any]:
    """Attach a researcher-supplied PDF/text source to a discovery result or shelf row."""
    document=extract_document(upload,max_pages=60,chars_per_page=8000,cache_dir=EXTRACTION_CACHE)
    links=paper_access_links(paper);existing=shelf_match or {}
    published=str(paper.get("publication_year") or paper.get("published") or existing.get("publication_year") or "")[:4]
    source_id=str(paper.get("source_id") or existing.get("source_id") or document.document_id).strip()
    return ledger.upsert_shelf_paper({
        "paper_id":str(existing.get("paper_id") or paper.get("paper_id") or ""),
        "title":str(paper.get("title") or existing.get("title") or document.title or "Untitled paper").strip(),
        "authors":list(paper.get("authors") or existing.get("authors") or []),
        "publication_year":published if len(published)==4 else "",
        "source_url":str(links.get("html_url") or links.get("source_url") or existing.get("source_url") or ""),
        "source_id":source_id,"pdf_path":store_paper_upload(upload,DATA / "paper_shelf"),
        "abstract":str(paper.get("summary") or paper.get("abstract") or existing.get("abstract") or "").strip(),
        "labels":list(paper.get("labels") or existing.get("labels") or build_paper_labels(paper)),
        "shelf_status":str(existing.get("shelf_status") or "reference"),
        "reading_status":str(existing.get("reading_status") or "unread"),
        "asset_type":"paper","intake_source":intake_source,
        "origin_links":paper.get("origin_links") or existing.get("origin_links") or [],
    })


def _discovery_reference_match(paper: dict[str, Any], references: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidate_tokens = _paper_identity_tokens(paper)
    if not candidate_tokens:
        return None
    strong = {token for token in candidate_tokens if not token.startswith("title:")}
    candidate_title = next((token for token in candidate_tokens if token.startswith("title:")), "")
    for reference in references:
        ref_paper = dict(reference.get("paper", {}))
        ref_tokens = _paper_identity_tokens(ref_paper)
        ref_strong = {token for token in ref_tokens if not token.startswith("title:")}
        if strong and ref_strong and strong.intersection(ref_strong):
            return reference
        ref_title = next((token for token in ref_tokens if token.startswith("title:")), "")
        if candidate_title and candidate_title == ref_title:
            return reference
    return None


def _minimal_discovery_history_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    minimal: list[dict[str, Any]] = []
    for paper in results:
        links = paper_access_links(paper)
        minimal.append({
            "title": str(paper.get("title", "")).strip(),
            "source_id": str(paper.get("source_id", "")).strip(),
            "source_url": links.get("source_url", ""),
            "html_url": links.get("html_url", ""),
            "pdf_url": links.get("pdf_url", ""),
            "origin_links": normalize_origin_links(paper.get("origin_links", [])),
            "history_detail": False,
        })
    return minimal


_DISCOVERY_WORKSPACE_KEYS = (
    "m1-discovery-topic", "m1-discovery-context", "m1-discovery-count",
    "m1-discovery-context-card-ids", "m1-discovery-results", "m1-discovery-plan",
    "m1-discovery-summary", "m1-discovery-source", "m1-discovery-session-id",
    "m1-discovery-run-id", "m1-discovery-history-selected",
)


def _discovery_workspace_id(profile_id: str = "") -> str:
    return f"intent:{profile_id}" if profile_id else "ad-hoc"


def _current_discovery_workspace_id() -> str:
    profile_id = str(st.session_state.get("m1-discovery-active-profile-id", "")).strip()
    draft_id = str(st.session_state.get("m1-discovery-active-draft-id", "")).strip()
    return _discovery_workspace_id(profile_id) if profile_id else draft_id or "ad-hoc"


def _save_discovery_workspace() -> None:
    payload = {
        key: st.session_state.get(key)
        for key in _DISCOVERY_WORKSPACE_KEYS
        if key in st.session_state
    }
    profile_id = str(st.session_state.get("m1-discovery-active-profile-id", "")).strip()
    if payload:
        payload["profile_id"] = profile_id
        payload["work_title"] = str(payload.get("work_title") or st.session_state.get("m1-discovery-work-title") or payload.get("m1-discovery-topic") or "탐색 작업").strip()
        ledger.save_literature_discovery_workspace(payload, workspace_id=_current_discovery_workspace_id())


def _restore_discovery_workspace() -> None:
    if st.session_state.get("m1-discovery-workspace-restored"):
        return
    saved = ledger.literature_discovery_workspace(workspace_id=_current_discovery_workspace_id()) or {}
    for key in _DISCOVERY_WORKSPACE_KEYS:
        if key not in st.session_state and key in saved:
            st.session_state[key] = saved[key]
    st.session_state["m1-discovery-workspace-restored"] = True


def _open_discovery_workspace(profile_id: str, profile: dict[str, Any]) -> None:
    """Switch the workbench without discarding another Intent's in-progress state."""
    current_profile_id = str(st.session_state.get("m1-discovery-active-profile-id", "")).strip()
    if current_profile_id != profile_id:
        _save_discovery_workspace()
    for key in _DISCOVERY_WORKSPACE_KEYS:
        st.session_state.pop(key, None)
    saved = ledger.literature_discovery_workspace(workspace_id=_discovery_workspace_id(profile_id)) or {}
    for key in _DISCOVERY_WORKSPACE_KEYS:
        if key in saved:
            st.session_state[key] = saved[key]
    st.session_state.setdefault("m1-discovery-topic", str(profile.get("question") or profile.get("title") or ""))
    st.session_state.setdefault("m1-discovery-context", str(profile.get("context") or ""))
    st.session_state.setdefault("m1-discovery-count", 12)
    st.session_state.setdefault("m1-discovery-context-card-ids", [])
    st.session_state["m1-discovery-active-profile-id"] = profile_id
    st.session_state["m1-discovery-active-draft-id"] = ""
    st.session_state["m1-discovery-loaded-profile-id"] = profile_id
    st.session_state["m1-discovery-workspace-restored"] = True


def _open_discovery_draft(workspace_id: str) -> None:
    _save_discovery_workspace()
    for key in _DISCOVERY_WORKSPACE_KEYS:
        st.session_state.pop(key, None)
    saved = ledger.literature_discovery_workspace(workspace_id=workspace_id) or {}
    for key in _DISCOVERY_WORKSPACE_KEYS:
        if key in saved:
            st.session_state[key] = saved[key]
    st.session_state["m1-discovery-work-title"] = str(saved.get("work_title") or saved.get("m1-discovery-topic") or "탐색 작업")
    st.session_state["m1-discovery-active-profile-id"] = ""
    st.session_state["m1-discovery-active-draft-id"] = workspace_id
    st.session_state["m1-discovery-loaded-profile-id"] = ""
    st.session_state["m1-discovery-workspace-restored"] = True


def _selected_discovery_history_results(
    results: list[dict[str, Any]], selected_keys: set[str]
) -> list[dict[str, Any]]:
    stored: list[dict[str, Any]] = []
    for paper in results:
        key = _discovery_paper_key(paper)
        links = paper_access_links(paper)
        if key in selected_keys:
            detail = dict(paper)
            detail["source_url"] = links.get("source_url", "")
            detail["html_url"] = links.get("html_url", "")
            detail["pdf_url"] = links.get("pdf_url", "")
            detail["history_detail"] = True
            stored.append(detail)
        else:
            stored.append({
                "title": str(paper.get("title", "")).strip(),
                "source_id": str(paper.get("source_id", "")).strip(),
                "source_url": links.get("source_url", ""),
                "html_url": links.get("html_url", ""),
                "pdf_url": links.get("pdf_url", ""),
                "origin_links": normalize_origin_links(paper.get("origin_links", [])),
                "history_detail": False,
            })
    return stored


def m1_screen(model: str, use_ollama: bool, semantic: bool, embedding_model: str) -> None:
    st.header(ui_text("M1 · 문헌조사·지식화 작업실", "M1 · Literature & Knowledge Workspace"))
    st.caption(ui_text("M1은 문헌과 연구 노트를 탐색·구조화해 승인 후보 지식과 관계를 준비합니다. 연구자 질문이나 외부 자문에는 직접 답하지 않고, 검증 지식을 M2에 갱신합니다.", "M1 explores and structures literature and research notes to prepare candidate knowledge and relations. It does not answer researcher questions directly; validated knowledge is passed to M2."))
    discovery_tab, upload_tab, knowledge_tab, ontology_tab = st.tabs([
        ui_text("문헌 탐색", "Literature Discovery"),
        ui_text("서재함", "Paper Shelf"),
        ui_text("승인 지식", "Approved Knowledge"),
        ui_text("온톨로지", "Ontology"),
    ])
    # Search tasks are an execution state of Literature Discovery. Search, list,
    # relations, and lineage are projections of the same approved knowledge base.
    queue_tab = discovery_tab
    knowledge_browse_tab, relation_tab = knowledge_tab.tabs([
        ui_text("검색·목록", "Search & List"),
        ui_text("관계·계보 보완", "Relations & Lineage Maintenance"),
    ])
    search_tab = knowledge_browse_tab
    memory_tab = knowledge_browse_tab
    with discovery_tab:
        _restore_discovery_workspace()
        st.caption(ui_text(
            "궁금한 연구주제를 빠르게 탐색해 관련 논문의 윤곽을 파악합니다. 이 기능은 정식 문헌리뷰가 아니라 탐색용이며, 마음에 드는 논문을 원문 링크로 확인한 뒤 서재함에 넣어 정식 읽기로 이어갑니다.",
            "Quickly explore a research topic to understand the literature landscape. This is exploratory discovery, not a systematic review. Inspect source pages, then move promising papers to the shelf for formal reading."
        ))

        # Show the whole backlog before any individual search form. Selecting an
        # Intent below loads its question/context into the shared discovery form.
        queue_overview_profiles = ledger.search_profiles()
        active_queue_profiles = [item for item in queue_overview_profiles if item.get("is_active")]
        completed_queue_profiles = [item for item in queue_overview_profiles if not item.get("is_active")]
        due_profile_ids = {str(item["profile_id"]) for item in scheduled_profiles(ledger)}
        st.subheader(ui_text("연구 Intent 탐색 작업 큐", "Research Intent discovery queue"))
        st.caption(ui_text(
            "이 표는 실행 화면이 아니라 전체 Intent의 대기·진행·완료 상태를 확인하는 목록입니다. 실제 탐색은 바로 아래 ‘논문 탐색 작업대’에서 수행합니다.",
            "This table is a queue overview, not an execution screen. Run and review searches in the Literature Discovery Workbench directly below.",
        ))
        queue_metric_1, queue_metric_2, queue_metric_3 = st.columns(3)
        queue_metric_1.metric(ui_text("대기 Intent", "Queued Intents"), len(active_queue_profiles))
        queue_metric_2.metric(ui_text("주기 도래", "Due now"), len(due_profile_ids))
        queue_metric_3.metric(ui_text("완료", "Completed"), len(completed_queue_profiles))
        if active_queue_profiles:
            queue_rows = []
            for item in active_queue_profiles:
                profile_id = str(item["profile_id"])
                saved_workspace = ledger.literature_discovery_workspace(workspace_id=_discovery_workspace_id(profile_id)) or {}
                saved_results = list(saved_workspace.get("m1-discovery-results") or [])
                if item.get("cadence") == "manual":
                    execution_state = ui_text("연구자 수동 실행", "Researcher-run")
                elif profile_id in due_profile_ids:
                    execution_state = ui_text("실행 주기 도래", "Due now")
                else:
                    execution_state = ui_text("다음 주기 대기", "Waiting for cadence")
                queue_rows.append({
                    ui_text("상태", "Status"): execution_state,
                    ui_text("작업 진행", "Work progress"): ui_text(f"결과 {len(saved_results)}편 보존", f"{len(saved_results)} results saved") if saved_results else ui_text("탐색 전", "Not started"),
                    ui_text("시작지점", "Origin"): " · ".join(origin_labels(item.get("origin_links", []))) or "-",
                    ui_text("제목", "Title"): item.get("title", ""),
                    ui_text("주기", "Cadence"): item.get("cadence", ""),
                    ui_text("최근 실행", "Last run"): _fmt_local_time(item.get("last_run_at")) or "-",
                })
            if persistent_list_toggle(ui_text("대기 Intent 목록", "Queued Intents"), "m1-intent-queue-list-visible", len(queue_rows)):
                st.dataframe(queue_rows, hide_index=True, use_container_width=True)
        else:
            st.info(ui_text(
                "현재 실행 가능한 연구 Intent가 없습니다. 승인된 Intent가 등록되면 이곳에 표시됩니다.",
                "There are no runnable Research Intents. Approved Intents will appear here.",
            ))

        st.divider()
        st.subheader(ui_text("논문 탐색 작업대", "Literature Discovery Workbench"))
        st.caption(ui_text(
            "① 시작점 선택 → ② LLM 문헌 탐색 → ③ 결과 검토·원문 확보 → ④ Reference/서재함 등록 → ⑤ 탐색 완료의 순서로 작업합니다. Intent별 진행 결과는 서로 분리되어 보존됩니다.",
            "Work through: ① choose an origin → ② LLM discovery → ③ review and obtain sources → ④ add to References/Shelf → ⑤ complete. Each Intent keeps an independent workspace.",
        ))
        saved_discovery_drafts = ledger.literature_discovery_workspaces(prefix="draft:")
        if saved_discovery_drafts and persistent_list_toggle(
            ui_text("보관된 탐색 작업", "Saved discovery work"), "m1-saved-discovery-list-visible", len(saved_discovery_drafts)
        ):
            draft_by_id = {str(item["workspace_id"]): item for item in saved_discovery_drafts}
            selected_draft_id = st.selectbox(
                ui_text("다시 열 탐색 작업", "Saved discovery work to reopen"), list(draft_by_id),
                format_func=lambda value: f"{draft_by_id[value].get('work_title') or draft_by_id[value].get('m1-discovery-topic') or value} · {len(draft_by_id[value].get('m1-discovery-results') or [])}{ui_text('편', ' papers')}",
                key="m1-saved-discovery-selection",
            )
            if st.button(ui_text("선택 작업을 작업대에서 열기", "Open selected work in the workbench"), key="m1-open-saved-discovery", use_container_width=True):
                st.session_state["m1-discovery-queue-selection"] = ""
                _open_discovery_draft(selected_draft_id)
                st.rerun()
        queue_profile_by_id = {str(item["profile_id"]): item for item in active_queue_profiles}
        pending_queue_profile_id = str(st.session_state.pop("m1-discovery-pending-profile-id", ""))
        if pending_queue_profile_id in queue_profile_by_id:
            # The detailed queue is rendered later on the same page. Stage its
            # selection before the top selectbox is instantiated on this rerun.
            st.session_state["m1-discovery-queue-selection"] = pending_queue_profile_id
            st.session_state["m1-discovery-loaded-profile-id"] = ""
        if st.session_state.pop("m1-discovery-reset-queue-selection", False):
            st.session_state["m1-discovery-queue-selection"] = ""
        queue_select_col, queue_clear_col = st.columns([5, 1])
        with queue_select_col:
            selected_queue_profile_id = st.selectbox(
                ui_text("작업대에서 열 연구 Intent", "Research Intent to open in the workbench"),
                options=[""] + list(queue_profile_by_id),
                format_func=lambda value: ui_text("Intent를 선택하세요", "Select an Intent") if not value else str(queue_profile_by_id[value].get("title") or value),
                key="m1-discovery-queue-selection",
            )
        with queue_clear_col:
            st.write("")
            clear_discovery = st.button(ui_text("내용 Clear", "Clear"), key="m1-discovery-clear", use_container_width=True)
        if clear_discovery:
            for key in (
                "m1-discovery-results", "m1-discovery-plan", "m1-discovery-summary",
                "m1-discovery-source", "m1-discovery-session-id", "m1-discovery-run-id",
                "m1-discovery-history-selected", "m1-discovery-external-response",
                "m1-discovery-external-prompt", "m1-discovery-external-prompt-signature",
            ):
                st.session_state.pop(key, None)
            st.session_state["m1-discovery-topic"] = ""
            st.session_state["m1-discovery-context"] = ""
            st.session_state["m1-discovery-context-card-ids"] = []
            ledger.clear_literature_discovery_workspace(workspace_id=_discovery_workspace_id(selected_queue_profile_id))
            # Preserve the selection marker so Clear does not immediately refill it.
            st.session_state["m1-discovery-loaded-profile-id"] = selected_queue_profile_id
            st.rerun()
        if selected_queue_profile_id and selected_queue_profile_id != st.session_state.get("m1-discovery-loaded-profile-id"):
            selected_profile = queue_profile_by_id[selected_queue_profile_id]
            _open_discovery_workspace(selected_queue_profile_id, selected_profile)
            st.rerun()
        if not selected_queue_profile_id and st.session_state.get("m1-discovery-loaded-profile-id"):
            _open_discovery_workspace("", {})
            st.session_state["m1-discovery-loaded-profile-id"] = ""
            st.rerun()
        active_workspace_results = list(st.session_state.get("m1-discovery-results") or [])
        active_workspace_selection = list(st.session_state.get("m1-discovery-history-selected") or [])
        active_draft_id = str(st.session_state.get("m1-discovery-active-draft-id", "")).strip()
        workbench_state_cols = st.columns(4)
        workbench_state_cols[0].metric(ui_text("시작점", "Origin"), ui_text("연구 Intent", "Research Intent") if selected_queue_profile_id else ui_text("지식카드/자유 질문", "Cards / Free question"))
        workbench_state_cols[1].metric(
            ui_text("현재 작업", "Current workspace"),
            str(queue_profile_by_id.get(selected_queue_profile_id, {}).get("title") or st.session_state.get("m1-discovery-work-title") or ui_text("새 탐색", "New discovery"))[:36],
        )
        workbench_state_cols[2].metric(ui_text("탐색 결과", "Results"), f"{len(active_workspace_results)}{ui_text('편', ' papers')}")
        workbench_state_cols[3].metric(ui_text("상세 보존 선택", "Selected for detail"), f"{len(active_workspace_selection)}{ui_text('편', ' papers')}")
        if not selected_queue_profile_id and active_workspace_results:
            default_work_title = str(st.session_state.get("m1-discovery-work-title") or st.session_state.get("m1-discovery-topic") or "문헌 탐색 작업").strip()
            archive_col, new_col = st.columns([2, 1])
            with archive_col:
                work_title = st.text_input(ui_text("보관할 탐색 작업 이름", "Saved discovery work title"), value=default_work_title, key="m1-discovery-save-work-title")
            with new_col:
                st.write("")
                st.write("")
                if st.button(ui_text("현재 결과 보관 · 새 탐색", "Save current results · New discovery"), type="primary", key="m1-save-current-start-new", use_container_width=True):
                    st.session_state["m1-discovery-work-title"] = work_title.strip() or default_work_title
                    _save_discovery_workspace()
                    if not active_draft_id:
                        current_payload = ledger.literature_discovery_workspace(workspace_id="ad-hoc") or {}
                        draft_id = f"draft:{uuid.uuid4().hex[:12]}"
                        current_payload["work_title"] = work_title.strip() or default_work_title
                        current_payload["profile_id"] = ""
                        ledger.save_literature_discovery_workspace(current_payload, workspace_id=draft_id)
                        ledger.clear_literature_discovery_workspace(workspace_id="ad-hoc")
                    for key in _DISCOVERY_WORKSPACE_KEYS:
                        st.session_state.pop(key, None)
                    for key in ("m1-discovery-external-response", "m1-discovery-external-prompt", "m1-discovery-external-prompt-signature"):
                        st.session_state.pop(key, None)
                    st.session_state["m1-discovery-topic"] = ""
                    st.session_state["m1-discovery-context"] = ""
                    st.session_state["m1-discovery-context-card-ids"] = []
                    st.session_state["m1-discovery-count"] = 12
                    st.session_state["m1-discovery-work-title"] = ""
                    st.session_state["m1-discovery-active-draft-id"] = ""
                    st.session_state["m1-discovery-active-profile-id"] = ""
                    st.session_state["m1-discovery-direct-run-message"] = ui_text("현재 탐색 결과를 보관하고 빈 작업대를 열었습니다.", "Saved the current discovery results and opened a blank workbench.")
                    st.rerun()
        if selected_queue_profile_id:
            st.success(ui_text(
                "선택한 Intent를 이 작업공간에서 바로 수정·실행할 수 있습니다. 실행 결과는 같은 Intent 이력으로 기록됩니다.",
                "You can edit and run the selected Intent here. The result remains linked to the same Intent history.",
            ))
            selected_direct_profile = queue_profile_by_id[selected_queue_profile_id]
            st.markdown(ui_text("#### 1–2단계 · Intent 확인 및 LLM 탐색", "#### Steps 1–2 · Confirm Intent and run LLM discovery"))
            render_origin_labels(selected_direct_profile)
            has_direct_plan=bool(st.session_state.get("m1-discovery-plan"))
            has_direct_results=st.session_state.get("m1-discovery-results") is not None
            has_review_selection=bool(st.session_state.get("m1-discovery-history-selected"))
            workflow_cols=st.columns(4)
            workflow_cols[0].metric(ui_text("1. Intent", "1. Intent"),ui_text("선택됨", "Selected"))
            workflow_cols[1].metric(ui_text("2. LLM 탐색", "2. LLM discovery"),ui_text("응답 반영", "Applied") if has_direct_results else ui_text("응답 대기", "Awaiting response"))
            workflow_cols[2].metric(ui_text("3. 논문 검색", "3. Paper search"),ui_text("완료", "Done") if has_direct_results else ui_text("실행 전", "Pending"))
            workflow_cols[3].metric(ui_text("4. 결과 검토", "4. Review"),ui_text("선택됨", "Selected") if has_review_selection else ui_text("검토 전", "Pending"))
            direct_topic=st.text_area(
                ui_text("탐색 질문", "Search question"),key="m1-discovery-topic",height=90,
                help=ui_text("M2에서 등록한 질문입니다. 이번 실행에 한해 구체화할 수 있습니다.","The question registered by M2. You may refine it for this run."),
            )
            direct_context=st.text_area(
                ui_text("탐색 맥락·필요 근거·완료 조건", "Context, evidence need, and completion condition"),
                key="m1-discovery-context",height=150,
                help=ui_text("논문 작성 To-do, 필요한 근거와 탐색 완료 조건을 함께 확인합니다.","Review the paper To-do, required evidence, and search completion condition together."),
            )
            direct_count=st.slider(ui_text("이번 실행에서 검토할 논문 수", "Papers to inspect in this run"),5,20,12,1,key="m1-discovery-count")
            direct_execution_profile = {
                **selected_direct_profile,
                "question": str(direct_topic or selected_direct_profile.get("question") or "").strip(),
                "context": str(direct_context or selected_direct_profile.get("context") or "").strip(),
            }
            direct_external_prompt = external_literature_discovery_prompt(
                direct_execution_profile["question"], direct_execution_profile["context"], int(direct_count),
                ["arXiv", "Semantic Scholar", "Crossref", "Google Scholar", "publisher and conference sites"],
            )
            direct_prompt_digest = hashlib.sha256(
                f"{direct_execution_profile['question']}\n{direct_execution_profile['context']}\n{direct_count}".encode("utf-8")
            ).hexdigest()[:12]
            st.markdown(ui_text("##### 기본 · 외부 LLM 문헌 탐색", "##### Default · External-LLM literature discovery"))
            st.caption(ui_text(
                "웹 검색이 가능한 외부 LLM이 논문 탐색·검증·Intent 맥락 정리를 한 번에 수행합니다. 프롬프트를 전달하고 JSON 응답을 붙여 넣으세요.",
                "A web-enabled external LLM discovers, verifies, and organises papers for the Intent in one task. Send the prompt and paste its JSON response.",
            ))
            direct_prompt_text = st.text_area(
                ui_text("외부 LLM용 편집 가능한 탐색 프롬프트", "Editable discovery prompt for an external LLM"),
                value=direct_external_prompt, height=360,
                key=f"m1-direct-intent-external-prompt-{selected_queue_profile_id}-{direct_prompt_digest}",
            )
            direct_external_response = st.text_area(
                ui_text("외부 LLM의 JSON 응답", "External LLM JSON response"), height=260,
                key=f"m1-direct-intent-external-response-{selected_queue_profile_id}-{direct_prompt_digest}",
                placeholder=ui_text("웹 검색 가능한 LLM의 전체 JSON 응답을 붙여 넣으세요.", "Paste the complete JSON response from a web-enabled LLM."),
            )
            if st.button(
                ui_text("외부 LLM 탐색 결과 검증 · 반영", "Validate and apply external-LLM results"),
                type="primary", key="m1-direct-intent-external-apply", use_container_width=True,
                disabled=not direct_external_response.strip(),
            ):
                direct_execution_profile = {
                    **selected_direct_profile,
                    "question": str(direct_topic or selected_direct_profile.get("question") or "").strip(),
                    "context": str(direct_context or selected_direct_profile.get("context") or "").strip(),
                }
                try:
                    direct_outcome = record_external_intent_discovery(
                        ledger, direct_execution_profile, direct_external_response,
                        trigger="manual_external_llm", max_results=int(direct_count),
                    )
                    st.session_state["m1-discovery-results"] = list(direct_outcome.get("candidates", []))
                    st.session_state["m1-discovery-plan"] = {}
                    st.session_state["m1-discovery-summary"] = direct_outcome.get("search_summary", "")
                    st.session_state["m1-discovery-source"] = "external"
                    st.session_state["m1-discovery-run-id"] = direct_outcome.get("run_id", "")
                    st.session_state.pop("m1-discovery-session-id", None)
                    st.session_state["m1-discovery-history-selected"] = []
                    _save_discovery_workspace()
                    st.session_state["m1-discovery-direct-run-message"] = ui_text(
                        f"‘{selected_direct_profile.get('title', '')}’ Intent에 외부 LLM 탐색 결과 {len(direct_outcome.get('candidates', []))}편을 반영했습니다.",
                        f"Applied {len(direct_outcome.get('candidates', []))} external-LLM results to ‘{selected_direct_profile.get('title', '')}’.",
                    )
                    st.rerun()
                except ValueError as error:
                    st.error(str(error))

            with st.expander(ui_text("보조 · 내부 LLM + arXiv 키워드 검색", "Fallback · Internal LLM + arXiv keyword search"), expanded=False):
                st.caption(ui_text(
                    "외부 LLM을 사용할 수 없을 때만 사용합니다. 내부 LLM이 검색식을 만들고 arXiv API 후보를 선별합니다.",
                    "Use only when an external LLM is unavailable. The internal LLM creates queries and screens candidates returned by the arXiv API.",
                ))
                direct_internal_prompt = intent_discovery_task_prompt(direct_execution_profile, ledger.search_runs(selected_queue_profile_id, limit=5), int(direct_count))
                st.text_area(ui_text("내부 검색식 생성 프롬프트", "Internal query-generation prompt"), value=direct_internal_prompt, height=260, key=f"m1-direct-intent-internal-prompt-{selected_queue_profile_id}")
                run_selected_intent = st.button(ui_text("내부 LLM + arXiv 실행", "Run internal LLM + arXiv"), key="m1-direct-intent-run", use_container_width=True)
                if run_selected_intent:
                    try:
                        with st.spinner(ui_text("검색식 생성 → arXiv 검색 → 초록 평가 중", "Generating queries → searching arXiv → triaging abstracts")):
                            direct_outcome = run_intent_discovery_task(
                                ledger, direct_execution_profile, trigger="manual_internal_arxiv",
                                planner=lambda _prompt: llm_draft(direct_internal_prompt, model, use_ollama, profile="search_strategy"),
                                triage_drafter=lambda prompt: llm_draft(prompt, model, use_ollama, profile="abstract_triage"),
                                sources=["arxiv"], max_results=int(direct_count),
                            )
                        if direct_outcome["status"] == "failed":
                            st.error(direct_outcome["error"])
                        else:
                            direct_plan = dict(direct_outcome.get("plan", {})); direct_plan["sources"] = ["arXiv"]
                            st.session_state["m1-discovery-results"] = list(direct_outcome.get("candidates", []))
                            st.session_state["m1-discovery-plan"] = direct_plan
                            st.session_state["m1-discovery-summary"] = direct_plan.get("scope_summary", "")
                            st.session_state["m1-discovery-source"] = "internal"
                            st.session_state["m1-discovery-run-id"] = direct_outcome.get("run_id", "")
                            st.session_state["m1-discovery-history-selected"] = []
                            _save_discovery_workspace(); st.rerun()
                    except Exception as error:
                        st.error(ui_text(f"내부 arXiv 탐색에 실패했습니다: {error}", f"Internal arXiv search failed: {error}"))
        if direct_run_message := st.session_state.pop("m1-discovery-direct-run-message", ""):
            st.success(str(direct_run_message))
        if selected_queue_profile_id and st.session_state.get("m1-discovery-results") is not None:
            preview_results=list(st.session_state.get("m1-discovery-results") or [])
            preview_plan=st.session_state.get("m1-discovery-plan") or {}
            if preview_results:
                st.success(ui_text(f"이번 탐색 결과 반영됨 · 후보 {len(preview_results)}편",f"Current discovery results applied · {len(preview_results)} candidates"))
                if preview_plan.get("scope_summary"):st.info(str(preview_plan.get("scope_summary")))
                with st.expander(ui_text("후보 논문 빠른 확인", "Quick candidate review"),expanded=True):
                    for paper in preview_results:
                        links=paper_access_links(paper);title=str(paper.get("title") or ui_text("제목 없음","Untitled"))
                        url=links.get("html_url") or links.get("source_url") or links.get("pdf_url")
                        st.markdown(f"- [{title}]({url})" if url else f"- **{title}**")
                    st.caption(ui_text("Reference List·서재함 추가와 상세 선택은 아래 ‘현재 탐색 결과 관리’에서 수행합니다.","Use Current result management below to add papers to References or the shelf."))
            else:
                st.warning(ui_text("이번 탐색에서는 후보 논문을 찾지 못했습니다. 질문·맥락을 수정해 다시 실행할 수 있습니다.","No candidates were found. Refine the question or context and run again."))
            complete_col, keep_col = st.columns([1, 2])
            with complete_col:
                if st.button(ui_text("이 Intent 탐색 완료", "Complete this Intent discovery"), type="primary", key=f"m1-complete-intent-{selected_queue_profile_id}"):
                    _save_discovery_workspace()
                    ledger.complete_search_profile(selected_queue_profile_id)
                    _open_discovery_workspace("", {})
                    st.session_state["m1-discovery-queue-selection"] = ""
                    st.session_state["m1-discovery-loaded-profile-id"] = ""
                    st.session_state["m1-discovery-direct-run-message"] = ui_text(
                        f"‘{selected_direct_profile.get('title', '')}’ 탐색을 완료 처리했습니다. 결과와 이력은 보존됩니다.",
                        f"Completed discovery for ‘{selected_direct_profile.get('title', '')}’. Results and history were preserved.",
                    )
                    st.rerun()
            with keep_col:
                st.caption(ui_text("완료 전까지 다른 Intent로 전환해도 현재 결과가 이 Intent 작업공간에 보존됩니다.", "Until completion, you can switch Intents and return to these saved results."))

        discovery_history_all = ledger.unified_literature_discovery_runs(limit=100)
        references = ledger.literature_references(limit=500)
        history_shelf_papers = ledger.shelf_papers()
        history_origin_filter = st.selectbox(
            ui_text("탐색 이력 출발점", "Discovery history origin"),
            ["all", "knowledge_cards", "research_intent"],
            format_func=lambda value: {
                "all": ui_text("전체", "All"),
                "knowledge_cards": ui_text("지식카드 기반", "Knowledge Card"),
                "research_intent": ui_text("M2 연구 Intent 기반", "M2 Research Intent"),
            }[value], key="m1-unified-discovery-origin-filter",
        )
        discovery_history = [
            item for item in discovery_history_all
            if history_origin_filter == "all" or item.get("origin_type") == history_origin_filter
        ]
        st.markdown(ui_text("#### 보조 패널 · 과거 탐색과 Reference", "#### Supporting panels · History and References"))
        st.caption(ui_text("현재 작업대의 실행·검토와 별개로 필요할 때만 펼쳐보는 기록 영역입니다.", "These records are separate from the active workbench; expand them only when needed."))
        with st.expander(
            ui_text(f"통합 문헌 탐색 이력 · {len(discovery_history)}건", f"Unified literature discovery history · {len(discovery_history)}"),
            expanded=False,
        ):
            st.caption(ui_text(
                "지식카드 기반 일회성 탐색과 M2 연구 Intent 기반 주기 실행을 같은 형식으로 보여줍니다. 개별 실행은 완료되지만 활성 Intent 프로필은 다음 주기에 다시 실행됩니다.",
                "Ad-hoc Knowledge Card searches and periodic M2 Research Intent runs share one history. Each run completes, while an active Intent profile can run again on its next cadence.",
            ))
            if not discovery_history:
                st.caption(ui_text(
                    "아직 저장된 탐색 이력이 없습니다. 탐색을 실행하면 검색 맥락과 결과 논문 제목·링크가 자동 저장됩니다.",
                    "No saved discovery sessions yet. Running a discovery automatically stores the search context plus result titles and links.",
                ))
            for history_index, history in enumerate(discovery_history):
                history_topic = str(history.get("topic", "")).strip() or ui_text("제목 없는 탐색", "Untitled discovery")
                origin_label = ui_text("지식카드", "Knowledge Card") if history.get("origin_type") == "knowledge_cards" else ui_text("M2 연구 Intent", "M2 Research Intent")
                mode_label = {
                    "ad_hoc": ui_text("일회성", "Ad-hoc"),
                    "manual": ui_text("연구자 실행", "Researcher-run"),
                    "periodic": ui_text("주기 실행", "Periodic"),
                }.get(str(history.get("execution_mode")), ui_text("실행", "Run"))
                history_source = ui_text("내부 LLM", "Internal LLM") if history.get("trigger") == "internal" else str(history.get("trigger") or ui_text("외부/자동", "External/automatic"))
                history_results = list(history.get("results", []))
                reference_count = sum(1 for paper in history_results if _discovery_reference_match(paper, references))
                shelf_count = sum(1 for paper in history_results if _discovery_shelf_match(paper, history_shelf_papers))
                result_count = len(history_results)
                label = f"[{origin_label} · {mode_label}] {history_topic} · {result_count}{ui_text('편', ' papers')}"
                with st.expander(label, expanded=False):
                    st.caption(
                        f"{_fmt_local_time(history.get('created_at'))} · {history_source} · {history.get('status', 'completed')} · "
                        + ui_text(
                            f"Reference List {reference_count}편 · 서재함 {shelf_count}편",
                            f"Reference List {reference_count} · Shelf {shelf_count}",
                        )
                    )
                    context_text = str(history.get("research_context", "")).strip()
                    render_origin_labels(history)
                    if context_text:
                        st.markdown(ui_text("**검색 맥락**", "**Research context**"))
                        st.write(context_text)
                    summary_text = str(history.get("search_summary", "")).strip()
                    if summary_text:
                        st.markdown(ui_text("**탐색 결과 요약**", "**Discovery summary**"))
                        st.write(summary_text)
                    if history.get("query"):
                        st.caption(ui_text("검색 전략: ", "Search strategy: ") + str(history["query"]))
                    if history.get("error"):
                        st.error(str(history["error"]))
                    st.markdown(ui_text(f"**결과 논문 · {result_count}편**", f"**Papers found · {result_count}**"))
                    if not history_results:
                        st.caption(ui_text("저장된 결과 논문이 없습니다.", "No result papers were stored."))
                    for paper_index, paper in enumerate(history_results, start=1):
                        title = str(paper.get("title", "")).strip() or ui_text("제목 없음", "Untitled")
                        ref_match = _discovery_reference_match(paper, references)
                        shelf_match = _discovery_shelf_match(paper, history_shelf_papers)
                        links = paper_access_links(paper)
                        status_parts = []
                        if ref_match:
                            status_parts.append(ui_text("Reference List", "Reference List"))
                        if shelf_match:
                            status_parts.append(ui_text("서재함", "Shelf"))
                        status_text = " · ".join(status_parts) if status_parts else ui_text("탐색 결과", "Discovery only")
                        paper_col, status_col, link_col = st.columns([6, 2, 1.4])
                        with paper_col:
                            st.markdown(f"{paper_index}. {title}")
                        with status_col:
                            st.caption(status_text)
                        with link_col:
                            open_url = links.get("html_url") or links.get("source_url") or links.get("pdf_url")
                            if open_url:
                                st.link_button(ui_text("원문", "Source"), open_url, key=f"m1-history-source-{history.get('run_id','')}-{paper_index}")
                    if history_index < len(discovery_history) - 1:
                        st.caption(ui_text(
                            "이 기록은 당시 탐색 결과를 요약해서 보여줍니다. 관심 논문은 Reference List, 본격적으로 읽을 논문은 서재함에서 계속 관리합니다.",
                            "This log summarizes the discovery at that time. Keep promising papers in the Reference List and papers selected for full reading in the shelf.",
                        ))

        with st.expander(
            ui_text(f"참고문헌 리스트 · {len(references)}편", f"Reference list · {len(references)} papers"),
            expanded=False,
        ):
            if not references:
                st.caption(ui_text(
                    "아직 선택한 참고문헌이 없습니다. 문헌 탐색 결과에서 관심 논문을 선택해 Reference List에 추가하세요.",
                    "No selected references yet. Choose papers from discovery results and add them to the Reference List.",
                ))
            else:
                reference_topics = sorted({str(item.get("topic", "")).strip() for item in references if str(item.get("topic", "")).strip()})
                topic_filter = st.selectbox(
                    ui_text("주제별 보기", "Filter by topic"),
                    options=[""] + reference_topics,
                    format_func=lambda value: ui_text("전체", "All") if not value else value,
                    key="m1-reference-topic-filter",
                )
                visible_refs = [item for item in references if not topic_filter or item.get("topic") == topic_filter]
                for ref_index, ref in enumerate(visible_refs):
                    paper = dict(ref.get("paper", {}))
                    title = str(paper.get("title", "")).strip() or ui_text("제목 없음", "Untitled")
                    st.markdown(f"**{title}**")
                    st.caption(f"{ref.get('topic','')} · {ui_text('상태', 'status')} {ref.get('status','selected')}")
                    render_origin_labels(paper, prefix=ui_text("탐색 시작지점", "Discovery origin"))
                    labels = list(ref.get("labels", []))
                    if labels:
                        st.caption(ui_text("Labels · ", "Labels · ") + " · ".join(labels))
                    links = paper_access_links(paper)
                    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
                    if links.get("html_url"):
                        c1.link_button(ui_text("HTML 원문", "HTML full text"), links["html_url"], key=f"m1-ref-html-{ref_index}-{ref.get('reference_id','')}")
                    elif links.get("source_url"):
                        c1.link_button(ui_text("원문", "Source"), links["source_url"], key=f"m1-ref-source-{ref_index}-{ref.get('reference_id','')}")
                    c2.link_button("Google Scholar", google_scholar_url(paper), key=f"m1-ref-scholar-{ref_index}-{ref.get('reference_id','')}")
                    if links.get("pdf_url"):
                        c3.link_button("PDF", links["pdf_url"], key=f"m1-ref-pdf-{ref_index}-{ref.get('reference_id','')}")
                    with c4:
                        if st.button(ui_text("서재함에 추가", "Add to shelf"), key=f"m1-ref-shelf-{ref_index}-{ref.get('reference_id','')}"):
                            saved = ledger.upsert_shelf_paper({
                                "title": title,
                                "authors": list(paper.get("authors", [])),
                                "publication_year": str(paper.get("published", ""))[:4],
                                "source_url": links.get("html_url") or links.get("source_url", ""),
                                "source_id": str(paper.get("source_id", "")).strip(),
                                "pdf_path": "",
                                "abstract": str(paper.get("summary", "")).strip(),
                                "labels": labels,
                                "shelf_status": "reference",
                                "reading_status": "unread",
                                "asset_type": "paper",
                                "intake_source": "literature_reference",
                                "origin_links": paper.get("origin_links", []),
                            })
                            ledger.update_literature_reference_status(ref.get("reference_id", ""), "shelved")
                            st.success(ui_text(f"서재함에 추가했습니다: {saved['title']}", f"Added to shelf: {saved['title']}"))
                    if ref_index < len(visible_refs) - 1:
                        st.divider()

        st.markdown(ui_text(
            "### 작업대 · 3–4단계 결과 검토·등록" if selected_queue_profile_id else "### 작업대 · 지식카드/자유 질문으로 새 탐색",
            "### Workbench · Steps 3–4 Review and register results" if selected_queue_profile_id else "### Workbench · New discovery from cards or a free question",
        ))
        st.caption(ui_text(
            "선택한 Intent의 결과를 검토하고 Reference·서재함으로 넘깁니다." if selected_queue_profile_id else "승인 지식카드를 출발점으로 선택하거나 질문과 맥락을 직접 작성해 탐색을 시작합니다.",
            "Review the selected Intent's results and move papers to References or the Shelf." if selected_queue_profile_id else "Select approved Knowledge Cards as an origin, or enter a question and context directly.",
        ))
        approved_cards_for_discovery = memory.all()
        discovery_card_by_id = {str(card["card_id"]): card for card in approved_cards_for_discovery}
        discovery_card_ids = st.multiselect(
            ui_text("탐색의 출발점이 될 지식카드 (선택)", "Knowledge Cards to use as discovery context (optional)"),
            options=list(discovery_card_by_id),
            format_func=lambda card_id: str(discovery_card_by_id[card_id].get("title") or card_id),
            key="m1-discovery-context-card-ids",
            disabled=bool(selected_queue_profile_id),
            help=ui_text(
                "선택한 카드의 주장·조건·한계·개념으로 탐색 대상과 연구 맥락 초안을 구성합니다.",
                "Build an editable topic and context from the selected cards' claims, conditions, limits, and concepts.",
            ),
        )
        if st.button(
            ui_text("선택 카드로 탐색 문맥 구성", "Build discovery context from selected cards"),
            key="m1-discovery-build-from-cards", disabled=not discovery_card_ids or bool(selected_queue_profile_id),
        ):
            selected_context_cards = [discovery_card_by_id[card_id] for card_id in discovery_card_ids]
            concepts = list(dict.fromkeys(
                str(concept).strip() for card in selected_context_cards
                for concept in card.get("concepts", []) if str(concept).strip()
            ))
            titles = [str(card.get("title") or "").strip() for card in selected_context_cards if str(card.get("title") or "").strip()]
            st.session_state["m1-discovery-topic"] = ui_text(
                f"{', '.join(titles[:3])}와 관련된 최신 근거, 상충 결과 및 미해결 연구 문제",
                f"Recent evidence, conflicting findings, and open questions related to {', '.join(titles[:3])}",
            )
            context_lines = [ui_text("선택 지식카드에서 출발한 탐색입니다.", "This discovery starts from selected approved Knowledge Cards.")]
            for card in selected_context_cards:
                context_lines.extend([
                    f"- {card.get('title') or card['card_id']}: {card.get('claim') or ''}",
                    f"  {ui_text('조건', 'Conditions')}: {card.get('conditions') or ui_text('미기재', 'not stated')}",
                    f"  {ui_text('한계', 'Limits')}: {card.get('limits') or ui_text('미기재', 'not stated')}",
                ])
            if concepts:
                context_lines.append(f"{ui_text('핵심 개념', 'Core concepts')}: {', '.join(concepts[:20])}")
            context_lines.append(ui_text(
                "기존 주장을 반복하기보다 이를 보완·반박하거나 적용 범위를 구체화하는 문헌을 우선 탐색합니다.",
                "Prioritize literature that extends, challenges, or clarifies the scope of the existing claims.",
            ))
            st.session_state["m1-discovery-context"] = "\n".join(context_lines)
            st.rerun()
        if selected_queue_profile_id:
            topic=str(st.session_state.get("m1-discovery-topic") or "")
            context=str(st.session_state.get("m1-discovery-context") or "")
            target_count=int(st.session_state.get("m1-discovery-count") or 12)
            st.caption(ui_text("질문·맥락·논문 수는 위 ‘선택 Intent 실행 작업공간’에서 수정합니다. 아래에서는 결과와 외부 LLM 대안을 관리합니다.","Edit the question, context, and count in the Selected Intent workspace above. Manage results and the external-LLM alternative below."))
        else:
            topic = st.text_area(
                ui_text("무엇을 찾아보고 싶은가?", "What do you want to explore?"),
                key="m1-discovery-topic", height=100,
                placeholder=ui_text("예: Agentic workflow와 role-driven autonomous agent의 차이를 다룬 연구", "e.g. Research comparing agentic workflows with role-driven autonomous agents"),
            )
            context = st.text_area(
                ui_text("연구 맥락 · 관심 관점 (선택)", "Research context / angle (optional)"),
                key="m1-discovery-context", height=90,
                placeholder=ui_text("왜 이 주제가 궁금한지, 특히 보고 싶은 관점이나 제외할 범위를 적습니다.", "Add why this matters, the angle you care about, or what should be excluded."),
            )
            target_count = st.slider(ui_text("확인할 논문 수", "Number of papers to inspect"), min_value=5, max_value=20, value=12, step=1, key="m1-discovery-count")
        discovery_origin_values = [{
            "origin_type": "m2_knowledge", "origin_id": card_id,
            "label": str(discovery_card_by_id[card_id].get("title") or card_id),
            "source_card_ids": [card_id],
            "research_title": topic.strip(),
            "research_question": topic.strip(),
            "research_context": context.strip(),
        } for card_id in discovery_card_ids]
        if not discovery_origin_values and topic.strip():
            origin_digest = hashlib.sha256(f"{topic.strip()}\n{context.strip()}".encode("utf-8")).hexdigest()[:12]
            discovery_origin_values.append({
                "origin_type": "researcher_question", "origin_id": f"rq-search-{origin_digest}",
                "label": topic.strip()[:120], "source_card_ids": [],
                "research_title": topic.strip(), "research_question": topic.strip(),
                "research_context": context.strip(),
            })
        discovery_origin_links = normalize_origin_links(discovery_origin_values)
        source_options = SEARCH_SOURCE_LABELS
        # The default route is a web-enabled external LLM.  The internal
        # fallback is intentionally narrow and uses only arXiv keyword search.
        selected_sources = ["arxiv"]

        internal_col, external_col = st.columns(2)
        with internal_col:
            if st.button(ui_text("보조 · 내부 LLM + arXiv", "Fallback · Internal LLM + arXiv"), disabled=not topic.strip(), key="m1-discovery-internal"):
                try:
                    with st.spinner(ui_text("검색전략 생성 → 다중 학술소스 후보 수집 → 초록 빠른 비교 중", "Generating search plan → retrieving candidates from multiple scholarly sources → triaging abstracts")):
                        selected_profile = queue_profile_by_id.get(selected_queue_profile_id)
                        if selected_profile:
                            execution_profile = {**selected_profile, "question": topic.strip(), "context": context.strip()}
                            outcome = run_intent_discovery_task(
                                ledger, execution_profile, trigger="manual_researcher",
                                planner=lambda prompt: llm_draft(prompt, model, use_ollama, profile="search_strategy"),
                                triage_drafter=lambda prompt: llm_draft(prompt, model, use_ollama, profile="abstract_triage"),
                                sources=selected_sources, max_results=target_count,
                            )
                            if outcome["status"] == "failed":
                                st.error(outcome["error"])
                            else:
                                results = list(outcome.get("candidates", []))
                                plan = dict(outcome.get("plan", {}))
                                plan["sources"] = [source_options.get(key, key) for key in selected_sources]
                                st.session_state["m1-discovery-results"] = results
                                st.session_state["m1-discovery-plan"] = plan
                                st.session_state["m1-discovery-summary"] = plan.get("scope_summary", "")
                                st.session_state["m1-discovery-source"] = "internal"
                                st.session_state["m1-discovery-run-id"] = outcome.get("run_id", "")
                                st.session_state.pop("m1-discovery-session-id", None)
                                st.session_state["m1-discovery-history-selected"] = []
                                _save_discovery_workspace()
                                st.rerun()
                        else:
                            plan_raw = llm_draft(discovery_search_plan_prompt(topic, context, target_count), model, use_ollama, profile="search_strategy") or ""
                            plan = parse_discovery_search_plan(plan_raw)
                            candidates = collect_multisource_candidates(topic, context, plan["queries"], selected_sources, max_results=target_count)
                            plan["sources"] = [source_options.get(key, key) for key in selected_sources]
                            if not candidates:
                                st.session_state["m1-discovery-results"] = []
                                st.session_state["m1-discovery-plan"] = plan
                                st.session_state["m1-discovery-summary"] = plan.get("scope_summary", "")
                                st.session_state["m1-discovery-source"] = "internal"
                                _save_discovery_workspace()
                                st.warning(ui_text("현재 조건으로 후보 논문을 찾지 못했습니다. 연구 맥락을 조정하거나 외부 LLM 탐색을 사용해 보세요.", "No candidate papers were found. Adjust the research context or try external-LLM discovery."))
                            else:
                                triage_raw = llm_draft(discovery_triage_prompt(topic, context, candidates, target_count), model, use_ollama, profile="abstract_triage") or ""
                                results = apply_discovery_triage(candidates, triage_raw, target_count)
                                results = [{**paper, "origin_links": discovery_origin_links} for paper in results]
                                st.session_state["m1-discovery-results"] = results
                                st.session_state["m1-discovery-plan"] = plan
                                st.session_state["m1-discovery-summary"] = plan.get("scope_summary", "")
                                st.session_state["m1-discovery-source"] = "internal"
                                st.session_state.pop("m1-discovery-run-id", None)
                                saved_session = ledger.save_literature_discovery_session(
                                    topic=topic, research_context=context, target_count=target_count,
                                    discovery_source="internal", search_plan=plan,
                                    search_summary=plan.get("scope_summary", ""), results=_minimal_discovery_history_results(results),
                                    origin_card_ids=discovery_card_ids,
                                )
                                st.session_state["m1-discovery-session-id"] = saved_session.get("session_id", "")
                                st.session_state["m1-discovery-history-selected"] = []
                                _save_discovery_workspace()
                                st.rerun()
                except Exception as error:
                    st.error(ui_text(f"빠른 문헌 탐색에 실패했습니다: {error}", f"Quick literature discovery failed: {error}"))
        with external_col:
            st.caption(ui_text("웹 검색이 가능한 외부 LLM을 사용하면 arXiv 밖의 논문도 함께 탐색할 수 있습니다.", "A web-enabled external LLM can also discover papers beyond arXiv."))

        with st.expander(ui_text("기본 · 외부 LLM으로 문헌 탐색", "Default · Discover with an external LLM"), expanded=True):
            external_source_names = ["arXiv", "Semantic Scholar", "Crossref", "Google Scholar", "publisher and conference sites"]
            prompt_signature = f"{topic.strip()}\n---CONTEXT---\n{context.strip()}\n---COUNT---\n{target_count}\n---SOURCES---\n{'|'.join(external_source_names)}"
            previous_signature = st.session_state.get("m1-discovery-external-prompt-signature")
            if topic.strip() and prompt_signature != previous_signature:
                st.session_state["m1-discovery-external-prompt"] = external_literature_discovery_prompt(topic, context, target_count, external_source_names)
                st.session_state["m1-discovery-external-prompt-signature"] = prompt_signature
            elif not topic.strip() and prompt_signature != previous_signature:
                st.session_state["m1-discovery-external-prompt"] = ui_text(
                    "먼저 위의 ‘무엇을 찾아보고 싶은가?’를 입력하세요.",
                    "Enter ‘What do you want to explore?’ above first.",
                )
                st.session_state["m1-discovery-external-prompt-signature"] = prompt_signature

            st.caption(ui_text(
                "위의 연구주제와 연구 맥락을 바꾸면 외부 LLM용 검색 프롬프트도 자동으로 갱신됩니다. 아래 프롬프트 전체를 웹 검색이 가능한 외부 LLM에 전달하세요.",
                "The external-LLM search prompt is regenerated automatically from the topic and research context above. Copy the full prompt below into a web-enabled external LLM.",
            ))
            if st.button(
                ui_text("현재 입력으로 검색 프롬프트 다시 만들기", "Regenerate prompt from current inputs"),
                disabled=not topic.strip(),
                key="m1-discovery-external-regenerate",
            ):
                st.session_state["m1-discovery-external-prompt"] = external_literature_discovery_prompt(topic, context, target_count, external_source_names)
                st.session_state["m1-discovery-external-prompt-signature"] = prompt_signature
                st.rerun()

            st.text_area(
                ui_text("외부 LLM용 검색 프롬프트", "Prompt for external LLM"),
                height=360,
                key="m1-discovery-external-prompt",
                help=ui_text(
                    "연구주제·연구 맥락·확인할 논문 수가 반영된 프롬프트입니다. 필요하면 복사 전에 직접 수정할 수 있습니다.",
                    "This prompt reflects the research topic, research context, and target paper count. You may edit it before copying if needed.",
                ),
            )
            pasted = st.text_area(ui_text("외부 LLM의 JSON 응답 붙여넣기", "Paste the external LLM JSON response"), height=280, key="m1-discovery-external-response")
            if st.button(ui_text("외부 LLM 결과 반영", "Apply external LLM results"), disabled=not pasted.strip(), key="m1-discovery-external-apply"):
                try:
                    parsed = parse_external_literature_results(pasted, target_count)
                    parsed["papers"] = [{**paper, "origin_links": discovery_origin_links} for paper in parsed["papers"]]
                    st.session_state["m1-discovery-results"] = parsed["papers"]
                    st.session_state["m1-discovery-summary"] = parsed.get("search_summary", "")
                    st.session_state["m1-discovery-plan"] = {}
                    st.session_state["m1-discovery-source"] = "external"
                    st.session_state.pop("m1-discovery-run-id", None)
                    saved_session = ledger.save_literature_discovery_session(
                        topic=topic, research_context=context, target_count=target_count,
                        discovery_source="external", search_plan={},
                        search_summary=parsed.get("search_summary", ""), results=_minimal_discovery_history_results(parsed["papers"]),
                        origin_card_ids=discovery_card_ids,
                    )
                    st.session_state["m1-discovery-session-id"] = saved_session.get("session_id", "")
                    st.session_state["m1-discovery-history-selected"] = []
                    _save_discovery_workspace()
                    st.rerun()
                except ValueError as error:
                    st.error(str(error))

        discovery_results = st.session_state.get("m1-discovery-results", [])
        if discovery_results:
            if st.session_state.get("m1-discovery-source") == "internal":
                plan_sources = (st.session_state.get("m1-discovery-plan", {}) or {}).get("sources", [])
                source_label = ui_text("내부 LLM + ", "Internal LLM + ") + (", ".join(plan_sources) if plan_sources else "arXiv")
            else:
                source_label = ui_text("외부 LLM", "External LLM")
            st.markdown(ui_text(f"**빠른 탐색 결과 · {len(discovery_results)}편 · {source_label}**", f"**Quick discovery · {len(discovery_results)} papers · {source_label}**"))
            st.caption(ui_text(
                "현재 결과는 임시 작업본으로 보존됩니다. 다른 페이지를 다녀와도 유지되며, ‘내용 Clear’ 또는 다음 탐색 실행 시 교체됩니다.",
                "These results are retained as the current working set across page changes. Clear them with ‘Clear’ or replace them by running another discovery.",
            ))
            summary = str(st.session_state.get("m1-discovery-summary", "")).strip()
            if summary:
                st.info(summary)
            plan = st.session_state.get("m1-discovery-plan", {})
            if isinstance(plan, dict) and plan.get("queries"):
                with st.expander(ui_text("사용한 검색식 보기", "View search queries"), expanded=False):
                    for index, query in enumerate(plan.get("queries", []), start=1):
                        st.code(f"Q{index}: {query}")
                    for note in plan.get("search_notes", []):
                        st.caption(f"- {note}")

            st.info(ui_text(
                "탐색 히스토리에는 검색 주제·맥락과 모든 결과의 제목·링크가 기본 저장됩니다. 아래에서 참고할 논문을 선택하면 해당 논문만 빠른 해석·관련성·주의점·초록/요약까지 상세 보존합니다.",
                "Discovery history keeps the search topic/context plus titles and links for all results by default. Select papers below to preserve their quick take, relevance rationale, caution, and abstract/summary in detail.",
            ))
            selected_default = set(st.session_state.get("m1-discovery-history-selected", []))
            result_options = {_discovery_paper_key(p): p for p in discovery_results}
            session_id = str(st.session_state.get("m1-discovery-session-id", "")).strip()
            intent_run_id = str(st.session_state.get("m1-discovery-run-id", "")).strip()
            discovery_history_id = session_id or intent_run_id
            selected_keys: list[str] = []
            discovery_shelf_papers = ledger.shelf_papers()

            st.caption(ui_text(
                "각 논문 카드의 ‘상세 보존’ 체크박스로 관심 논문을 바로 선택하세요. 체크하지 않은 논문도 제목과 원문 링크는 탐색 히스토리에 남습니다.",
                "Select papers directly with the ‘Preserve details’ checkbox on each paper card. Unchecked papers still retain their title and source link in discovery history.",
            ))

            for index, paper in enumerate(discovery_results, start=1):
                title = str(paper.get("title", "")).strip() or ui_text("제목 없음", "Untitled")
                shelf_match = _discovery_shelf_match(paper, discovery_shelf_papers)
                year = str(paper.get("published", ""))[:4]
                score = paper.get("relevance_score")
                score_text = f" · {ui_text('관련도', 'relevance')} {score}/100" if score is not None else ""
                paper_key = _discovery_paper_key(paper)
                st.markdown(f"**{index}. {title}**")
                render_origin_labels(paper, prefix=ui_text("탐색 시작지점", "Discovery origin"))
                if shelf_match:
                    reading_status = str(shelf_match.get("reading_status", "unread"))
                    shelf_status = str(shelf_match.get("shelf_status", "reference"))
                    st.success(ui_text(
                        f"서재함 등록됨 · {shelf_status} · {reading_status}",
                        f"Already in shelf · {shelf_status} · {reading_status}",
                    ))
                checkbox_key = f"m1-discovery-detail-{session_id or 'unsaved'}-{index}"
                preserve_detail = st.checkbox(
                    ui_text("상세 보존", "Preserve details"),
                    value=paper_key in selected_default,
                    key=checkbox_key,
                    help=ui_text(
                        "체크하면 빠른 해석·관련성·주의점·초록/요약까지 탐색 히스토리에 저장되고, Reference List 추가 대상에도 포함됩니다.",
                        "Checked papers keep quick take, relevance rationale, caution, and abstract/summary in discovery history and are also eligible for Reference List actions.",
                    ),
                )
                if preserve_detail:
                    selected_keys.append(paper_key)
                authors = ", ".join(paper.get("authors", [])[:6])
                meta = " · ".join(part for part in [year, authors] if part)
                if meta or score_text:
                    st.caption((meta or "") + score_text)
                discovery_sources = list(paper.get("discovery_sources", []))
                labels = list(paper.get("labels", [])) or build_paper_labels(paper)
                if discovery_sources:
                    st.caption(ui_text("발견 소스 · ", "Found via · ") + " · ".join(discovery_sources))
                if labels:
                    st.caption("Labels · " + " · ".join(labels))
                if paper.get("quick_take"):
                    st.write(paper["quick_take"])
                if paper.get("why_relevant"):
                    st.caption(ui_text("왜 참고할 만한가 · ", "Why it may matter · ") + str(paper["why_relevant"]))
                if paper.get("caution"):
                    st.caption(ui_text("주의 · ", "Caution · ") + str(paper["caution"]))
                if paper.get("summary"):
                    with st.expander(ui_text("초록/요약 보기", "View abstract / summary"), expanded=False):
                        st.write(paper["summary"])
                links = paper_access_links(paper)
                source_url = links.get("source_url", "")
                html_url = links.get("html_url", "")
                pdf_url = links.get("pdf_url", "")
                action_html, action_source, action_pdf, action_scholar, action_shelf = st.columns([1, 1, 1, 1, 1])
                if html_url:
                    action_html.link_button(ui_text("HTML 원문", "HTML full text"), html_url, key=f"m1-discovery-html-{index}-{paper.get('source_id', '')}")
                elif source_url:
                    action_html.link_button(ui_text("원문 열기", "Open source"), source_url, key=f"m1-discovery-open-{index}-{paper.get('source_id', '')}")
                if source_url and source_url != html_url:
                    action_source.link_button(ui_text("arXiv/출처", "arXiv / source"), source_url, key=f"m1-discovery-source-{index}-{paper.get('source_id', '')}")
                if pdf_url:
                    action_pdf.link_button(ui_text("PDF 열기", "Open PDF"), pdf_url, key=f"m1-discovery-pdf-{index}-{paper.get('source_id', '')}")
                action_scholar.link_button("Google Scholar", google_scholar_url(paper), key=f"m1-discovery-scholar-{index}-{paper.get('source_id', '')}")
                with action_shelf:
                    if shelf_match:
                        st.caption(ui_text("서재함 등록됨", "In shelf"))
                    elif st.button(ui_text("서재함에 추가", "Add to shelf"), key=f"m1-discovery-add-{index}-{paper.get('source_id', '')}"):
                        try:
                            saved = ledger.upsert_shelf_paper({
                                "title": title,
                                "authors": list(paper.get("authors", [])),
                                "publication_year": year if len(year) == 4 else "",
                                "source_url": html_url or source_url,
                                "source_id": str(paper.get("source_id", "")).strip(),
                                "pdf_path": "",
                                "abstract": str(paper.get("summary", "")).strip(),
                                "labels": labels,
                                "shelf_status": "reference",
                                "reading_status": "unread",
                                "asset_type": "paper",
                                "intake_source": "llm_discovery",
                                "origin_links": paper.get("origin_links", []),
                            })
                            st.success(ui_text(f"서재함에 추가했습니다: {saved['title']}", f"Added to shelf: {saved['title']}"))
                            st.rerun()
                        except Exception as error:
                            st.error(str(error))
                if pdf_url and not shelf_match:
                    if st.button(ui_text("PDF 내려받아 서재함에 보관", "Download PDF into shelf"), key=f"m1-discovery-save-pdf-{index}-{paper.get('source_id', '')}"):
                        try:
                            local_pdf = download_discovery_pdf(paper, DATA / "paper_shelf")
                            saved = ledger.upsert_shelf_paper({
                                "title": title,
                                "authors": list(paper.get("authors", [])),
                                "publication_year": year if len(year) == 4 else "",
                                "source_url": html_url or source_url,
                                "source_id": str(paper.get("source_id", "")).strip(),
                                "pdf_path": local_pdf,
                                "abstract": str(paper.get("summary", "")).strip(),
                                "labels": labels,
                                "shelf_status": "reference",
                                "reading_status": "unread",
                                "asset_type": "paper",
                                "intake_source": "llm_discovery",
                                "origin_links": paper.get("origin_links", []),
                            })
                            st.success(ui_text(f"PDF를 내려받아 서재함에 보관했습니다: {saved['title']}", f"Downloaded the PDF into the shelf: {saved['title']}"))
                            st.rerun()
                        except Exception as error:
                            st.error(ui_text(f"PDF 자동 보관 실패: {error}", f"Could not download the PDF: {error}"))
                elif not html_url and source_url:
                    st.caption(ui_text("HTML 원문이 확인되지 않아 출처 페이지를 엽니다. PDF URL이 확인되면 자동 보관할 수 있습니다.", "No HTML full-text link was identified. Open the source page; if a direct PDF URL is available, it can be stored automatically."))
                stored_path_value=str((shelf_match or {}).get("pdf_path") or "").strip()
                has_stored_source=bool(stored_path_value) and Path(stored_path_value).exists()
                if not html_url and not pdf_url and not has_stored_source:
                    manual_source_key=f"{index}-{hashlib.sha256(paper_key.encode('utf-8')).hexdigest()[:12]}"
                    with st.expander(ui_text("원문 직접 등록 · 다운로드 파일 또는 텍스트", "Register source manually · downloaded file or text"),expanded=False):
                        st.caption(ui_text(
                            "Google Scholar·출판사에서 원문을 내려받은 뒤 PDF/TXT/MD를 등록하거나, 확보한 본문 전체를 붙여 넣으세요. 등록한 원문은 서재함의 동일 논문과 M1 읽기·M2 리비전에 사용됩니다.",
                            "Download the paper from Google Scholar or the publisher and register a PDF/TXT/MD file, or paste the full text. The source is attached to the same shelf paper for M1 reading and M2 revision.",
                        ))
                        manual_file=st.file_uploader(
                            ui_text("다운로드한 원문 파일", "Downloaded source file"),type=["pdf","txt","md"],
                            key=f"m1-discovery-manual-file-{manual_source_key}",
                        )
                        pasted_text=st.text_area(
                            ui_text("원문 텍스트 Copy/Paste", "Copy/paste paper text"),height=180,
                            key=f"m1-discovery-pasted-text-{manual_source_key}",
                            placeholder=ui_text("초록이 아니라 분석에 사용할 논문 본문을 붙여 넣으세요.", "Paste the paper body to be analyzed, not only the abstract."),
                        )
                        file_col,text_col=st.columns(2)
                        if file_col.button(ui_text("다운로드 파일 등록", "Register downloaded file"),disabled=manual_file is None,key=f"m1-discovery-register-file-{manual_source_key}"):
                            try:
                                saved=_store_manual_discovery_source({**paper,"labels":labels},manual_file,shelf_match=shelf_match,intake_source="llm_discovery_manual_file")
                                st.success(ui_text(f"원문 파일을 서재함 논문에 연결했습니다: {saved['title']}",f"Source file attached to shelf paper: {saved['title']}"));st.rerun()
                            except Exception as error:st.error(ui_text(f"원문 파일 등록 실패: {error}",f"Could not register source file: {error}"))
                        if text_col.button(ui_text("붙여넣은 원문 등록", "Register pasted text"),disabled=not pasted_text.strip(),key=f"m1-discovery-register-text-{manual_source_key}"):
                            try:
                                pasted_upload=pasted_paper_text_upload(title,pasted_text)
                                saved=_store_manual_discovery_source({**paper,"labels":labels},pasted_upload,shelf_match=shelf_match,intake_source="llm_discovery_pasted_text")
                                st.success(ui_text(f"붙여넣은 원문을 서재함 논문에 연결했습니다: {saved['title']}",f"Pasted source attached to shelf paper: {saved['title']}"));st.rerun()
                            except Exception as error:st.error(ui_text(f"원문 텍스트 등록 실패: {error}",f"Could not register pasted text: {error}"))
                st.divider()

            st.session_state["m1-discovery-history-selected"] = list(selected_keys)
            action_save_history, action_add_reference = st.columns(2)
            with action_save_history:
                if st.button(
                    ui_text("체크 논문 상세정보 저장", "Save checked paper details"),
                    disabled=not discovery_history_id,
                    key="m1-discovery-history-save-selection",
                    use_container_width=True,
                ):
                    stored_results = _selected_discovery_history_results(discovery_results, set(selected_keys))
                    if session_id:
                        updated = ledger.update_literature_discovery_session_results(
                            session_id, stored_results, search_summary=str(st.session_state.get("m1-discovery-summary", ""))
                        )
                    else:
                        ledger.update_search_run_candidates(intent_run_id, stored_results)
                        updated = {"run_id": intent_run_id} if intent_run_id else None
                    if updated:
                        st.success(ui_text(
                            f"탐색 기록을 갱신했습니다. 상세 보존 {len(selected_keys)}편 · 나머지는 제목/링크만 저장됩니다.",
                            f"Discovery history updated. {len(selected_keys)} papers are preserved in detail; the rest keep title/link only.",
                        ))
                    else:
                        st.error(ui_text("탐색 이력을 찾지 못했습니다.", "The discovery session could not be found."))
            with action_add_reference:
                if st.button(
                    ui_text("체크 논문을 참고문헌 리스트에 추가", "Add checked papers to Reference List"),
                    disabled=not selected_keys,
                    key="m1-discovery-add-references",
                    use_container_width=True,
                ):
                    added = 0
                    for key in selected_keys:
                        paper = result_options.get(key)
                        if not paper:
                            continue
                        labels = list(paper.get("labels", [])) or build_paper_labels(paper)
                        ledger.upsert_literature_reference(
                            topic=topic, research_context=context, session_id=discovery_history_id,
                            paper=paper, labels=labels, status="selected",
                        )
                        added += 1
                    st.success(ui_text(f"참고문헌 리스트에 {added}편을 추가했습니다.", f"Added {added} papers to the Reference List."))
                    st.rerun()

    with upload_tab:
        st.caption(ui_text("논문·연구 노트·웹페이지를 이곳에 넣고, 탐색에서 고른 논문도 같은 서재함에서 관리합니다. 등록 자체는 지식카드 생성이 아닙니다.", "Add papers, research notes, and web pages here. Papers selected from search are managed in the same shelf. Adding an item does not create a knowledge card by itself."))
        uploaded = st.file_uploader(ui_text("논문 PDF·연구 노트", "Paper PDF or research note"), type=["pdf", "txt", "md"])
        source_kind = st.selectbox(ui_text("자료 성격", "Source type"), ["외부 논문", "연구자의 확정 문서", "연구자의 아이디어 노트"], format_func=lambda v: {"외부 논문": ui_text("외부 논문", "External paper"), "연구자의 확정 문서": ui_text("연구자의 확정 문서", "Confirmed researcher document"), "연구자의 아이디어 노트": ui_text("연구자의 아이디어 노트", "Researcher idea note")}.get(v, v))
        core_paper = st.checkbox(ui_text("핵심 문헌으로 표시", "Mark as core paper"), key="asset-intake-core")
        if uploaded and st.button(ui_text("서재함에 추가", "Add to Paper Shelf"), key="claim-first-add-shelf"):
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
        render_ontology_workspace(model, use_ollama, semantic, embedding_model)
    with relation_tab:
        st.markdown("### 관계·계보 검토 및 보완")
        st.caption("온톨로지가 Type 수준의 구조를 제공하므로, 여기서는 승인 지식카드 사이의 근거·반박·보완 관계와 출처 계보를 확인하고 필요한 부분만 수정합니다.")
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
        st.markdown("### 승인 지식 검색")
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
        st.divider()
        st.subheader("고급 관리 · Intent 정책 및 실행 이력")
        st.caption("일반적인 M2 논문 Intent 탐색은 상단 작업공간에서 완료합니다. 아래 영역은 주기 변경, 재등록, 삭제, 과거 Run 로그 등 관리가 필요할 때만 사용합니다.")
        if selected_llm_provider("internal") == "gemini" or selected_llm_provider("paper") == "gemini":
            destinations = []
            if selected_llm_provider("internal") == "gemini":
                destinations.append("탐색 초안")
            if selected_llm_provider("paper") == "gemini":
                destinations.append("공개 arXiv 본문 비교")
            st.caption("Gemini 외부 API로 전송: " + " · ".join(destinations))
        ready_intents = ledger.phenomena(recipient="m1", type_="curation_intent", status="ready")
        pending_intent_requests = [
            item for item in ledger.phenomena(recipient="researcher", type_="decision_request", status="proposed")
            if item.get("subject_type")=="curation_intent"
        ]
        profiles = ledger.search_profiles()
        all_profiles = ledger.search_profiles(include_deleted=True)
        queued_profiles = [profile for profile in profiles if profile["is_active"]]
        completed_profiles = [profile for profile in profiles if not profile["is_active"]]
        profile_intent_ids = {profile["intent_id"] for profile in all_profiles}
        missing = [intent for intent in ready_intents if str(intent["payload"].get("intent_id", "")) not in profile_intent_ids]
        registered_count = len(profiles)
        deleted_count = len(all_profiles) - registered_count
        st.caption(
            f"승인 대기 {len(pending_intent_requests)}건 · 승인 Intent {len(ready_intents)}건 · 등록됨 {registered_count}건 · "
            f"새 등록 가능 {len(missing)}건 · 큐에서 삭제됨 {deleted_count}건"
        )
        migrate_label = f"승인 Intent {len(ready_intents)}건 중 새 {len(missing)}건을 M1 탐색 큐에 등록"
        if st.button(migrate_label, key="m1-migrate-search-profiles", disabled=not missing):
            for intent in missing:
                ledger.create_search_profile(intent["payload"])
            st.rerun()
        if pending_intent_requests:
            st.markdown(f"**승인 대기 Intent · {len(pending_intent_requests)}건**")
            st.caption("아직 실행 큐는 아닙니다. 연구자 홈의 검토·승인함에서 승인하면 같은 제목으로 실행 프로필이 생성됩니다.")
            for request in pending_intent_requests:
                intent=(request.get("payload") or {}).get("intent") or {}
                title=str(intent.get("title") or "M1 탐색 Intent")
                with st.expander(f"승인 대기 · {title}"):
                    st.markdown(f"**등록 예정 제목**  \n{title}")
                    st.write(f"탐색 질문: {intent.get('question','')}")
                    st.caption(f"요청 ID: {request.get('phenomenon_id','')} · 승인 위치: 연구자 홈 > 연구자 검토·승인함")
            st.divider()
        if not profiles:
            st.info("아직 승인되어 실행 가능한 탐색 프로필이 없습니다. 위 승인 대기 Intent를 연구자 홈에서 승인하면 동일 제목으로 이곳에 등록됩니다.")
        if queued_profiles:
            st.markdown(f"**대기 큐 · {len(queued_profiles)}건**")
        for profile_index, profile in enumerate(queued_profiles + completed_profiles):
            if profile_index == len(queued_profiles) and completed_profiles:
                st.divider()
                st.markdown(f"**완료된 탐색 이력 · {len(completed_profiles)}건**")
                st.caption("한 번 실행된 Intent는 큐와 주기 실행 대상에서 제거됩니다. 아래에는 로그·PDF·P1 후보 카드 검토만 남습니다.")
            state_label = "대기" if profile["is_active"] else "완료"
            with st.expander(f"{state_label} · {profile['title']} · {profile['cadence']}", expanded=False):
                st.caption(f"원래 연구 질문: {profile['question']}")
                render_origin_labels(profile)
                if profile["is_active"] and st.button(
                    "위 논문 탐색 작업대에서 열기",
                    key=f"load-profile-into-direct-discovery-{profile['profile_id']}",
                    help="이 Intent 전용 작업공간을 상단에서 엽니다. 다른 Intent의 진행 결과는 그대로 보존됩니다.",
                ):
                    st.session_state["m1-discovery-pending-profile-id"] = str(profile["profile_id"])
                    st.rerun()
                if profile["is_active"] and st.button("이 Intent를 탐색 큐에서 삭제", key=f"delete-profile-{profile['profile_id']}"):
                    ledger.delete_search_profile(profile["profile_id"])
                    st.success("탐색 큐에서 삭제했습니다. 기존 실행 로그는 감사 기록으로 보존됩니다.")
                    st.rerun()
                if not profile["is_active"]:
                    if st.button("이 Intent를 다시 탐색 큐에 넣기", key=f"requeue-profile-{profile['profile_id']}"):
                        ledger.update_search_profile_policy(profile["profile_id"], context=profile["context"], cadence=profile["cadence"], is_active=True)
                        st.rerun()
                with st.form(f"search-profile-form-{profile['profile_id']}"):
                    context = st.text_area("탐색 맥락", value=profile["context"], height=150)
                    cadence = st.selectbox("주기", ["daily", "weekly", "manual"], index=["daily", "weekly", "manual"].index(profile["cadence"]))
                    is_active = st.checkbox("탐색 프로필 활성화", value=profile["is_active"])
                    save = st.form_submit_button("LLM Task 정책 저장")
                if save:
                    ledger.update_search_profile_policy(profile["profile_id"], context=context, cadence=cadence, is_active=is_active)
                    st.rerun()
                current = next(item for item in ledger.search_profiles() if item["profile_id"] == profile["profile_id"])
                st.caption("탐색 실행은 상단 작업대에서만 수행합니다. 아래에는 정책과 실행 이력만 표시합니다.")
                runs = ledger.search_runs(profile["profile_id"], limit=5)
                if runs:
                    st.markdown("**최근 실행 상세**")
                    st.caption("아래 실행은 상단 통합 탐색 이력에도 같은 Discovery Run으로 표시됩니다. 이 영역은 후보 재검토와 PDF 처리용 상세 도구입니다.")
                    for run in runs:
                        with st.expander(f"{_fmt_local_time(run.get('created_at'))} · {run['trigger']} · {run['status']} · 후보 {len(run['candidates'])}"):
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
                                render_origin_labels(candidate, prefix="탐색 시작지점")
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
                                        "source_id": candidate.get("source_id", ""), "abstract": candidate.get("summary", ""), "pdf_path": candidate.get("pdf_path", ""),
                                        "shelf_status": "reference", "reading_status": "unread", "asset_type": "paper", "intake_source": "search",
                                        "origin_links": candidate.get("origin_links", profile.get("origin_links", [])),
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
        st.divider()
        st.markdown("### 전체 승인 지식카드")
        total_cards = memory.count()
        st.caption("목록은 기본적으로 접혀 있으며, 펼친 상태는 현재 세션에서 유지됩니다.")
        if persistent_list_toggle("승인 지식카드", "m1-approved-card-list-visible", total_cards):
            visible_card_count = lazy_page_limit("m1-approved-card-visible-count", total_cards)
            for card in memory.all(limit=visible_card_count):
                st.markdown(f"**{card['title']}** — {card['claim']}")
                st.caption(" · ".join(card.get("labels", [])))


def management_screen() -> None:
    st.header("지식 관리")
    st.caption("삭제는 원본 JSONL을 지우지 않고 삭제 표식을 남깁니다. 따라서 사례 타임라인과 감사 이력은 보존됩니다.")
    total_cards = memory.count()
    visible_card_count = lazy_page_limit("knowledge-management-visible-count", total_cards)
    cards = memory.all(limit=visible_card_count)
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
    "completed": "완료",
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
    cols = st.columns(len(RQ_STATUS_LABELS))
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
                        completed_at = _fmt_local_time(links[0].get("review_completed_at"), with_seconds=True)
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
                st.caption(f"최근 변화 · {_fmt_local_time(latest_change.get('created_at'))} · {latest_change.get('summary', '')}")
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


def _update_thread_current_state(
    *, thread_kind: str, thread_id: str, title: str, current_question: str, model: str, use_ollama: bool,
    conversation: list[dict[str, Any]] | None = None, reports: list[dict[str, Any]] | None = None, generation_mode: str = "internal_llm",
) -> dict[str, Any] | None:
    prior = ledger.thread_current_state(thread_kind, thread_id) or {}
    prompt = current_state_prompt(
        thread_kind=thread_kind, title=title, current_question=current_question,
        prior_state=str(prior.get("body_text", "")), conversation=conversation or [], reports=reports or [],
    )
    result = llm_draft_result(prompt, model, use_ollama, profile="thread_state")
    if not result.text:
        return None
    return ledger.save_thread_current_state(
        thread_kind=thread_kind, thread_id=thread_id, current_question=current_question,
        body_text=result.text, generation_mode=generation_mode,
    )


def _update_rq_current_state(rq_id: str, model: str, use_ollama: bool) -> dict[str, Any] | None:
    rq = ledger.research_question_thread(rq_id) or {}
    if not rq:
        return None
    versions = list(rq.get("versions") or [])
    version_turns = [
        {"role": "question_version", "content": f"v{v.get('version_no')}: {v.get('question','')} / {v.get('change_reason','')}"}
        for v in versions[-6:]
    ]
    return _update_thread_current_state(
        thread_kind="research_question", thread_id=rq_id,
        title=str(rq.get("question", "Research Question")), current_question=str(rq.get("question", "")),
        model=model, use_ollama=use_ollama, conversation=version_turns,
        reports=ledger.m2_reports(rq_id, include_archived=False, limit=3),
    )


def _render_current_state_and_reports(
    *, thread_kind: str, thread_id: str, title: str, current_question: str, model: str, use_ollama: bool,
    refresh_callback: Callable[[], dict[str, Any] | None],
) -> None:
    st.markdown("### Current State")
    state = ledger.thread_current_state(thread_kind, thread_id)
    if state:
        st.caption(f"마지막 업데이트 · {_fmt_local_time(state.get('updated_at'))} · {state.get('generation_mode','')}")
        with st.container(border=True):
            st.markdown(str(state.get("body_text", "")))
    else:
        st.info("아직 Current State 문서가 없습니다. 첫 답변/검토 이후 자동 생성되며 수동으로도 만들 수 있습니다.")

    c1, c2 = st.columns(2)
    if c1.button("현재 상태 다시 정리", key=f"state-refresh-{thread_kind}-{thread_id}"):
        updated = refresh_callback()
        if updated:
            st.success("Current State를 최신 상태로 갱신했습니다.")
            st.rerun()
        else:
            st.warning("Current State 갱신용 LLM 응답을 얻지 못했습니다.")
    if c2.button("현재 결론 보고서 만들기", key=f"state-report-{thread_kind}-{thread_id}", disabled=not bool(state)):
        current = ledger.thread_current_state(thread_kind, thread_id) or {}
        prompt = report_snapshot_prompt(
            thread_kind=thread_kind, title=title, current_state=str(current.get("body_text", "")),
            current_question=current_question,
        )
        result = llm_draft_result(prompt, model, use_ollama, profile="report_snapshot")
        if result.text:
            ledger.create_thread_report_snapshot(
                thread_kind=thread_kind, thread_id=thread_id, title=title,
                body_text=result.text, generation_mode="internal_llm",
            )
            st.success("현재 시점의 결론 보고서를 Snapshot으로 저장했습니다.")
            st.rerun()
        else:
            st.warning(f"보고서 생성에 실패했습니다: {result.error}")

    snapshots = ledger.thread_report_snapshots(thread_kind, thread_id, include_archived=True, limit=30)
    if snapshots:
        with st.expander(f"결론 보고서 이력 {len(snapshots)}건", expanded=False):
            for item in snapshots:
                state_label = "보관" if item.get("archived_at") else "활성"
                with st.container(border=True):
                    st.markdown(f"**{item.get('title','')}**")
                    st.caption(f"{_fmt_local_time(item.get('created_at'))} · {state_label} · {item.get('generation_mode','')}")
                    st.markdown(str(item.get("body_text", "")))
                    a, d = st.columns(2)
                    if a.button("보관 해제" if item.get("archived_at") else "보관", key=f"snapshot-archive-{item['report_id']}"):
                        ledger.archive_thread_report_snapshot(str(item["report_id"]), archived=not bool(item.get("archived_at")))
                        st.rerun()
                    if d.button("보고서 제거", key=f"snapshot-delete-{item['report_id']}"):
                        ledger.delete_thread_report_snapshot(str(item["report_id"]))
                        st.rerun()


M2_SOURCE_LABELS = {
    "m1_knowledge": "M1 새 지식",
    "researcher": "연구자 질문",
    "external_advisory": "외부 자문",
    "sensemaking": "Research Sensemaking",
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
    # Keep the question visually prominent without dominating the page.
    st.markdown(f"**{str(rq.get('question', ''))}**")
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
        st.info(f"최근 변화 · {_fmt_local_time(latest_change.get('created_at'))} · {latest_change.get('summary', '')}")

    _render_current_state_and_reports(
        thread_kind="research_question", thread_id=rq_id, title=str(rq.get("question", "")),
        current_question=str(rq.get("question", "")), model=model, use_ollama=use_ollama,
        refresh_callback=lambda: _update_rq_current_state(rq_id, model, use_ollama),
    )

    status_cols = st.columns(5)
    status_actions = [
        ("interested", ui_text("관심", "Interested")),
        ("exploring", ui_text("탐색중", "Exploring")),
        ("hold", ui_text("보류", "On hold")),
        ("completed", ui_text("완료", "Completed")),
        ("rejected", ui_text("제외", "Rejected")),
    ]
    for col, (status, label) in zip(status_cols, status_actions):
        if col.button(label, key=f"thread-status-{rq_id}-{status}", disabled=rq.get("status") == status):
            ledger.update_research_question_status(rq_id, status)
            st.rerun()

    source_ids = list(rq.get("source_card_ids", []))
    cards_by_id = {str(card.get("card_id", "")): card for card in memory.all()}
    if source_ids:
        with st.expander(f"질문의 출발 근거 카드 {len(source_ids)}건", expanded=False):
            for card_id in source_ids:
                card = cards_by_id.get(card_id)
                if card:
                    st.markdown(f"**`{card_id}` · {card.get('claim', card.get('title', ''))}**")
                    paper_title = _card_source_paper_title(card)
                    if paper_title:
                        st.caption(f"참고논문 · {paper_title}")

    st.divider()
    st.markdown("### 연구 판단안 작성")
    st.caption("연구 논점과 적용 맥락을 기준으로 승인 지식을 연결하여, 권고·선택지·반론·다음 행동을 갖춘 연구 판단안을 만듭니다. 지식공백은 M1 근거보강 요청으로 이어집니다.")
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
        "이번 판단안에 사용할 승인 지식카드", candidate_ids,
        default=source_ids[:8] if source_ids else candidate_ids[:6],
        format_func=lambda cid: options.get(cid, cid), key=f"thread-evidence-{rq_id}",
    )
    for cid in selected_ids[:8]:
        card = cards_by_id[cid]
        st.caption(f"`{cid}` · {card.get('claim', '')[:220]}")
        paper_title = _card_source_paper_title(card)
        if paper_title:
            st.caption(f"참고논문 · {paper_title}")

    selected_cards = [cards_by_id[cid] for cid in selected_ids if cid in cards_by_id]
    prompt = build_m2_thread_review_prompt(
        question=str(rq.get("question", "")), context=context, cards=selected_cards,
        source_type=str(rq.get("source_type", "m1_knowledge")), source_payload=payload,
    )
    run_col, external_col = st.columns(2)
    with run_col:
        if st.button("현재 LLM으로 연구 판단안 만들기", key=f"thread-report-internal-{rq_id}", type="primary"):
            result = llm_draft_result(prompt, model, use_ollama)
            if result.text:
                report = _save_m2_thread_report(rq, context, selected_ids, result.text, "internal_llm")
                _update_rq_current_state(rq_id, model, use_ollama)
                st.session_state[f"thread-latest-report-{rq_id}"] = report["report_id"]
                st.success("연구 판단안을 저장하고 Current State를 갱신했습니다.")
                st.rerun()
            else:
                st.warning(f"M2 Report 생성에 실패했습니다: {result.error}")
    with external_col:
        st.caption("로컬/무료 LLM이 불안정하거나 여러 지식카드로 긴 프롬프트가 필요한 경우 외부 LLM을 사용할 수 있습니다.")

    with st.expander("외부 LLM으로 수동 연구 판단안 작성", expanded=False):
        prompt_key = f"thread-external-prompt-{rq_id}-{abs(hash(tuple(selected_ids))) % 100000}"
        st.text_area(
            "외부 LLM용 프롬프트", value=prompt, key=prompt_key, height=360,
            help="고정 높이 영역입니다. 긴 프롬프트는 내부 스크롤로 확인하고 전체 선택하여 복사할 수 있습니다.",
        )
        manual_response = st.text_area(
            "외부 LLM 응답 붙여넣기", key=f"thread-external-response-{rq_id}", height=280,
            placeholder="외부 LLM이 작성한 연구 판단안 전체를 붙여넣으세요."
        )
        if st.button("외부 응답 검증 · 연구 판단안으로 반영", key=f"thread-external-apply-{rq_id}", disabled=not manual_response.strip()):
            ok, message = validate_manual_m2_report(manual_response, valid_card_ids=set(selected_ids))
            if not ok:
                st.error(message)
            else:
                report = _save_m2_thread_report(rq, context, selected_ids, manual_response, "manual_external_llm")
                _update_rq_current_state(rq_id, model, use_ollama)
                st.success(message + " 연구 판단안으로 저장하고 Current State를 갱신했습니다.")
                st.session_state[f"thread-latest-report-{rq_id}"] = report["report_id"]
                st.rerun()

    reports = ledger.m2_reports(rq_id, include_archived=True, limit=20)
    if reports:
        latest = reports[0]
        st.divider(); st.markdown("### 최신 연구 판단안")
        st.caption(f"{_fmt_local_time(latest.get('created_at'))} · {latest['generation_mode']} · 근거 카드 {len(latest['evidence_card_ids'])}건")
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
            reason = st.text_input("변경 이유", value="연구 판단안과 보강된 지식을 반영한 질문 구체화", key=f"thread-refine-reason-{rq_id}")
            if st.button("질문 새 버전으로 반영", key=f"thread-refine-apply-{rq_id}", disabled=not new_question.strip() or new_question.strip() == str(rq.get("question", "")).strip()):
                ledger.refine_research_question(rq_id, new_question, reason)
                _update_rq_current_state(rq_id, model, use_ollama)
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
            st.write(f"- Report `{report['report_id']}` · {archive} · {_fmt_local_time(report.get('created_at'))}")
        for intent in linked_intents:
            st.write(f"- M1 Intent `{intent['intent_id']}`")


_THREAD_STATUS_ORDER = {
    "exploring": 0,
    "interested": 1,
    "candidate": 2,
    "hold": 3,
    "completed": 4,
    "rejected": 5,
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
                st.markdown(f"**{rq.get('question', '')}**")
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


def _render_all_question_threads(model: str, use_ollama: bool, semantic: bool, embedding_model: str) -> None:
    """Unified issue board for question threads originating from M1, researcher, or external advisory."""
    backlog = ledger.research_question_backlog(limit=500)
    for item in backlog:
        if str(item.get("source_type") or "m1_knowledge") == "m1_knowledge":
            ledger.ensure_research_question_thread(str(item["rq_id"]), source_type="m1_knowledge")

    allowed_sources = ["m1_knowledge", "researcher", "external_advisory"]
    threads: list[dict[str, Any]] = []
    for item in backlog:
        thread = ledger.research_question_thread(str(item["rq_id"])) or item
        if str(thread.get("source_type", "m1_knowledge")) in allowed_sources:
            threads.append(thread)

    pending_statuses = {"candidate", "interested", "exploring", "hold"}
    completed_statuses = {"completed"}
    pending_count = sum(1 for rq in threads if str(rq.get("status", "candidate")) in pending_statuses)
    completed_count = sum(1 for rq in threads if str(rq.get("status", "candidate")) in completed_statuses)
    rejected_count = sum(1 for rq in threads if str(rq.get("status", "candidate")) == "rejected")

    st.subheader(ui_text("전체 Research Question Threads", "All Research Question Threads"))
    st.caption(ui_text(
        "M1 새 지식, 연구자 직접 질문, 외부 자문에서 시작된 질문을 하나의 이슈 보드에서 관리합니다. 기본 화면은 아직 결론나지 않은 Pending 질문입니다.",
        "Manage questions originating from M1 new knowledge, researcher questions, and external advisory requests in one issue board. Pending questions are shown by default.",
    ))
    c1, c2, c3 = st.columns(3)
    c1.metric(ui_text("Pending", "Pending"), pending_count)
    c2.metric(ui_text("Completed", "Completed"), completed_count)
    c3.metric(ui_text("Excluded", "Excluded"), rejected_count)

    view = st.radio(
        ui_text("질문 보기", "Question view"),
        ["pending", "completed", "all"],
        horizontal=True,
        key="m2-all-thread-view",
        format_func=lambda value: {
            "pending": ui_text("Pending Issues", "Pending Issues"),
            "completed": ui_text("완료 질문", "Completed"),
            "all": ui_text("전체", "All"),
        }[value],
    )
    selected_sources = st.multiselect(
        ui_text("질문 출처", "Question sources"),
        allowed_sources,
        default=allowed_sources,
        key="m2-all-thread-source-filter",
        format_func=lambda value: M2_SOURCE_LABELS.get(value, value),
    )

    if view == "pending":
        visible = [rq for rq in threads if str(rq.get("status", "candidate")) in pending_statuses]
    elif view == "completed":
        visible = [rq for rq in threads if str(rq.get("status", "candidate")) in completed_statuses]
    else:
        visible = list(threads)
    visible = [rq for rq in visible if str(rq.get("source_type", "m1_knowledge")) in selected_sources]
    visible = _sorted_threads(visible)

    if not visible:
        st.info(ui_text("조건에 맞는 질문이 없습니다.", "No questions match the current filters."))
        return

    selected_id = str(st.session_state.get("m2-selected-all-thread-id") or "")
    if selected_id and not any(str(rq.get("rq_id")) == selected_id for rq in visible):
        selected_id = ""

    for rq in visible:
        rq_id = str(rq.get("rq_id", ""))
        source_type = str(rq.get("source_type", "m1_knowledge"))
        status = str(rq.get("status", "candidate"))
        is_selected = rq_id == selected_id
        with st.container(border=True):
            top = st.columns([5.2, 1.6])
            with top[0]:
                # Compact question typography for issue-board scanning.
                st.markdown(f"**{rq.get('question', '')}**")
            with top[1]:
                st.caption(RQ_STATUS_LABELS.get(status, status))
            st.caption(f"{M2_SOURCE_LABELS.get(source_type, source_type)} · {_thread_card_summary(rq)}")
            latest_change = ledger.latest_research_question_change(rq_id)
            if latest_change and latest_change.get("summary"):
                st.caption(f"{ui_text('최근 변화', 'Latest change')} · {latest_change.get('summary')}")
            cols = st.columns([1.2, 1.4, 4.4])
            if cols[0].button(ui_text("닫기", "Close") if is_selected else ui_text("열기", "Open"), key=f"m2-all-open-{rq_id}"):
                st.session_state["m2-selected-all-thread-id"] = "" if is_selected else rq_id
                st.rerun()
            if status not in {"completed", "rejected"}:
                if cols[1].button(ui_text("완료", "Complete"), key=f"m2-all-complete-{rq_id}"):
                    ledger.update_research_question_status(rq_id, "completed")
                    st.session_state["m2-selected-all-thread-id"] = ""
                    st.rerun()
            elif status == "completed":
                if cols[1].button(ui_text("다시 열기", "Reopen"), key=f"m2-all-reopen-{rq_id}"):
                    ledger.update_research_question_status(rq_id, "interested")
                    st.rerun()
            cols[2].caption(rq_id)
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
    if theme_flash := st.session_state.pop("m2-theme-start-flash", None):
        st.success(theme_flash)
    if updates:
        update_by_id = {
            str((update.get("payload") or {}).get("card_id", "")): update
            for update in updates
            if str((update.get("payload") or {}).get("card_id", ""))
        }
        display_by_id: dict[str, str] = {}
        for cid, update in update_by_id.items():
            card = cards_by_id.get(cid, {})
            provenance = card.get("provenance") if isinstance(card.get("provenance"), dict) else {}
            source_name = str((provenance or {}).get("source_name", "")).strip()
            claim = str(card.get("claim") or card.get("title") or cid).strip().replace("\n", " ")
            short_claim = claim[:95] + ("…" if len(claim) > 95 else "")
            display_by_id[cid] = f"{short_claim} · {source_name}" if source_name else short_claim

        valid_group_ids = set(update_by_id)
        group_prompt = knowledge_grouping_prompt(updates, all_cards, max_groups=6)
        group_cols = st.columns([1.5, 4.5])
        if group_cols[0].button(
            ui_text("AI 연구주제 그룹 제안", "Suggest research-theme groups"),
            key="m2-new-info-group-suggest", type="secondary",
        ):
            with st.spinner(ui_text("새 지식을 연구주제 관점에서 묶고 있습니다.", "Grouping new knowledge by research theme.")):
                raw_grouping = llm_draft(group_prompt, model, use_ollama) or ""
                st.session_state["m2-new-info-groups"] = parse_knowledge_grouping(
                    raw_grouping, valid_card_ids=valid_group_ids, limit=6
                )
                st.session_state["m2-new-info-group-source"] = "internal_llm"
            st.rerun()
        group_cols[1].caption(ui_text(
            "LLM이 주장·맥락·함의·원천 논문을 함께 보고 연구주제 그룹과 서로 다른 Research Question Alternatives를 제안합니다. 연구자는 하나를 선택·수정한 뒤 바로 Thread를 시작할 수 있습니다.",
            "The LLM proposes research-theme groups and distinct Research Question Alternatives using claims, context, implications, and source papers. Choose and edit one alternative, then start the Thread directly.",
        ))

        with st.expander(
            ui_text("외부 LLM으로 연구주제 그룹 제안", "Suggest research-theme groups with an external LLM"),
            expanded=bool(st.session_state.get("m2-new-info-group-external-error")),
        ):
            st.caption(ui_text(
                "로컬/무료 LLM 대신 ChatGPT·Claude 등 외부 LLM을 사용할 수 있습니다. 아래 프롬프트 전체를 복사해 외부 LLM에 전달하고, JSON 응답을 다시 붙여 넣으세요.",
                "Use ChatGPT, Claude, or another external LLM instead of the local/free model. Copy the full prompt below, then paste the returned JSON response here.",
            ))
            st.text_area(
                ui_text("외부 LLM용 그룹핑 프롬프트", "Grouping prompt for external LLM"),
                value=group_prompt,
                key="m2-new-info-group-external-prompt",
                height=360,
                help=ui_text(
                    "고정 높이 영역입니다. 전체 프롬프트를 복사해서 외부 LLM에 전달하세요.",
                    "This is a fixed-height area. Copy the complete prompt to the external LLM.",
                ),
            )
            external_grouping = st.text_area(
                ui_text("외부 LLM 응답 붙여넣기", "Paste external LLM response"),
                key="m2-new-info-group-external-response",
                height=280,
                placeholder=ui_text(
                    "외부 LLM이 반환한 연구주제 그룹 JSON 전체를 붙여 넣으세요.",
                    "Paste the complete research-theme grouping JSON returned by the external LLM.",
                ),
            )
            if st.button(
                ui_text("외부 LLM 그룹 제안 반영", "Apply external LLM group suggestions"),
                key="m2-new-info-group-external-apply",
                type="primary",
                disabled=not external_grouping.strip(),
            ):
                parsed_groups = parse_knowledge_grouping(
                    external_grouping, valid_card_ids=valid_group_ids, limit=6
                )
                if not parsed_groups:
                    st.session_state["m2-new-info-group-external-error"] = True
                    st.error(ui_text(
                        "그룹 제안을 해석하지 못했습니다. 외부 LLM이 프롬프트에서 요구한 JSON 형식으로 응답했는지 확인해 주세요.",
                        "Could not parse the group suggestions. Check that the external LLM returned the JSON format requested in the prompt.",
                    ))
                else:
                    st.session_state["m2-new-info-groups"] = parsed_groups
                    st.session_state["m2-new-info-group-external-error"] = False
                    st.session_state["m2-new-info-group-source"] = "external_llm"
                    st.success(ui_text(
                        f"외부 LLM의 연구주제 그룹 {len(parsed_groups)}건을 반영했습니다.",
                        f"Applied {len(parsed_groups)} research-theme groups from the external LLM.",
                    ))
                    st.rerun()

        proposed_groups = list(st.session_state.get("m2-new-info-groups", []))
        if proposed_groups:
            st.markdown(ui_text("#### 제안된 연구주제 그룹", "#### Suggested research-theme groups"))
            group_source = st.session_state.get("m2-new-info-group-source", "")
            if group_source == "external_llm":
                st.caption(ui_text("제안 생성 · 외부 LLM", "Generated by · External LLM"))
            elif group_source == "internal_llm":
                st.caption(ui_text("제안 생성 · 내부 LLM", "Generated by · Internal LLM"))
            for index, group in enumerate(proposed_groups, start=1):
                group_ids = [cid for cid in group.get("card_ids", []) if cid in valid_group_ids]
                if not group_ids:
                    continue
                with st.container(border=True):
                    st.markdown(f"**{index}. {group.get('name', '')}** · {len(group_ids)} cards")
                    if group.get("research_focus"):
                        st.write(group["research_focus"])
                    if group.get("why_together"):
                        st.caption(ui_text("함께 볼 이유 · ", "Why together · ") + group["why_together"])
                    alternatives = list(group.get("research_question_alternatives", []))
                    if not alternatives and group.get("candidate_question"):
                        alternatives = [{
                            "title": "",
                            "question": str(group.get("candidate_question", "")),
                            "rationale": str(group.get("why_together", "")),
                            "research_context": str(group.get("research_focus", "")),
                            "gap_or_tension": "",
                            "exploration_need": "",
                        }]
                    if alternatives:
                        st.markdown(ui_text("**Research Theme Alternatives**", "**Research Theme Alternatives**"))
                        alt_index = st.radio(
                            ui_text("연구 방향 선택", "Choose a research angle"),
                            options=list(range(len(alternatives))),
                            format_func=lambda alt_i, alts=alternatives: (
                                f"{alts[alt_i].get('title') + ' · ' if alts[alt_i].get('title') else ''}{alts[alt_i].get('question', '')}"
                            ),
                            key=f"m2-theme-alt-choice-{index}",
                            label_visibility="collapsed",
                        )
                        selected_alt = alternatives[int(alt_index)]
                        st.caption(ui_text(
                            "선택한 제안은 출발점입니다. 아래에서 연구자가 질문과 맥락을 직접 수정한 뒤 바로 Thread를 시작할 수 있습니다.",
                            "The selected alternative is only a starting point. Edit the question and context below, then start the Thread directly.",
                        ))
                        edit_key = f"m2-theme-edit-{index}-{alt_index}"
                        edited_question = st.text_area(
                            ui_text("Research Question", "Research Question"),
                            value=str(selected_alt.get("question", "")),
                            key=f"{edit_key}-question", height=100,
                        )
                        edited_rationale = st.text_area(
                            ui_text("도출 이유 / Why now", "Rationale / Why now"),
                            value=str(selected_alt.get("rationale") or group.get("why_together", "")),
                            key=f"{edit_key}-rationale", height=80,
                        )
                        edited_context = st.text_area(
                            ui_text("연구 맥락", "Research context"),
                            value=str(selected_alt.get("research_context") or group.get("research_focus", "")),
                            key=f"{edit_key}-context", height=90,
                        )
                        with st.expander(ui_text("공백·긴장 / 후속 탐색 필요 수정", "Edit gap/tension and exploration need"), expanded=False):
                            edited_gap = st.text_area(
                                ui_text("공백 / 긴장", "Gap / tension"),
                                value=str(selected_alt.get("gap_or_tension", "")),
                                key=f"{edit_key}-gap", height=70,
                            )
                            edited_need = st.text_area(
                                ui_text("추가 탐색 필요", "Exploration need"),
                                value=str(selected_alt.get("exploration_need", "")),
                                key=f"{edit_key}-need", height=70,
                            )
                    else:
                        edited_question = ""
                        edited_rationale = str(group.get("why_together", ""))
                        edited_context = str(group.get("research_focus", ""))
                        edited_gap = ""
                        edited_need = ""
                        st.warning(ui_text(
                            "이 그룹에는 Research Question Alternative가 없습니다. 그룹 제안을 다시 생성해 주세요.",
                            "This group has no research-question alternatives. Regenerate the theme suggestions.",
                        ))
                    with st.expander(ui_text("그룹의 지식카드 상세 보기", "View knowledge-card details"), expanded=False):
                        for cid in group_ids:
                            card = cards_by_id.get(cid, {})
                            provenance = card.get("provenance") if isinstance(card.get("provenance"), dict) else {}
                            st.markdown(f"**{card.get('title') or cid}**")
                            if (provenance or {}).get("source_name"):
                                st.caption(ui_text("참고논문 · ", "Source paper · ") + str(provenance.get("source_name")))
                            st.write(card.get("claim") or "")
                            if card.get("context"):
                                st.caption(ui_text("맥락 · ", "Context · ") + str(card.get("context")))
                            if card.get("implication"):
                                st.caption(ui_text("함의 · ", "Implication · ") + str(card.get("implication")))
                            if card.get("limits"):
                                st.caption(ui_text("한계 · ", "Limits · ") + str(card.get("limits")))
                    if st.button(
                        ui_text("이 연구주제로 시작", "Start with this research theme"),
                        key=f"m2-start-proposed-theme-{index}", type="primary",
                        disabled=not edited_question.strip(),
                    ):
                        group_updates = [update_by_id[cid] for cid in group_ids if cid in update_by_id]
                        rq_block = f"""## RQ 1
Question: {edited_question.strip()}
Why now: {edited_rationale.strip()}
Gap/Tension: {edited_gap.strip()}
Research Context: {edited_context.strip()}
Source Card IDs: {', '.join(group_ids)}
Exploration Need: {edited_need.strip()}
"""
                        parsed = parse_research_question_suggestions(
                            rq_block, valid_card_ids=set(group_ids), limit=1
                        )
                        if not parsed:
                            st.error(ui_text(
                                "수정한 연구질문을 저장 가능한 형식으로 만들지 못했습니다.",
                                "Could not convert the edited research theme into a storable research question.",
                            ))
                        else:
                            review_id = ledger.create_research_state_review("manual", group_updates)
                            saved = store_research_question_candidates(
                                ledger, parsed, group_updates, review_id=review_id
                            )
                            for rq in saved:
                                ledger.ensure_research_question_thread(
                                    str(rq["rq_id"]),
                                    source_type="m1_knowledge",
                                    source_payload={
                                        "review_id": review_id,
                                        "generation_mode": "theme_alternative",
                                        "theme_name": str(group.get("name", "")),
                                        "theme_focus": str(group.get("research_focus", "")),
                                        "why_together": str(group.get("why_together", "")),
                                        "alternative_index": int(alt_index) if alternatives else 0,
                                        "source_card_ids": group_ids,
                                    },
                                )
                            ledger.complete_research_state_review(
                                review_id, generated_rq_count=len(saved), selected_rq_count=len(saved),
                                summary=ui_text(
                                    f"연구주제 그룹 ‘{group.get('name','')}’의 Alternative를 연구자가 수정·확인하여 RQ Thread {len(saved)}건을 시작했습니다.",
                                    f"Started {len(saved)} RQ Thread(s) after the researcher edited and confirmed an alternative from theme group '{group.get('name','')}'.",
                                ),
                            )
                            if saved:
                                st.session_state["m2-selected-m1-thread-id"] = str(saved[0]["rq_id"])
                                st.session_state["m2-theme-start-flash"] = ui_text(
                                    "선택한 연구주제로 Research Question Thread를 시작했습니다. 아래 M1 새 지식 기반 Thread에서 바로 이어갈 수 있습니다.",
                                    "Started a Research Question Thread from the selected theme. Continue in the M1-knowledge Thread below.",
                                )
                            st.rerun()

        existing_selection = st.session_state.get("m2-new-info-selected-cards", [])
        if existing_selection and any(value not in valid_group_ids for value in existing_selection):
            st.session_state["m2-new-info-selected-cards"] = []
        selected_ids = st.multiselect(
            ui_text("연구질문 생성에 사용할 새 지식 선택", "Select new knowledge for research-question generation"),
            options=list(update_by_id),
            format_func=lambda cid: display_by_id.get(cid, cid),
            key="m2-new-info-selected-cards",
            help=ui_text(
                "AI 제안 그룹을 그대로 선택하거나, 연구자가 관련 카드 몇 건만 다시 조정해 하나의 연구질문 생성 batch로 처리합니다. 선택하지 않은 카드는 미처리 상태로 남습니다.",
                "Use an AI-proposed group or adjust the cards manually. Unselected cards remain unreviewed for later batches.",
            ),
        )
        selected_updates = [update_by_id[cid] for cid in selected_ids if cid in update_by_id]
        st.caption(ui_text(
            f"선택 {len(selected_updates)} / 미처리 {len(updates)}건 · 선택하지 않은 지식은 다음 batch에 남습니다.",
            f"Selected {len(selected_updates)} of {len(updates)} unreviewed items · unselected knowledge remains for a later batch."
        ))
        with st.expander(ui_text(f"미처리 지식카드 {len(updates)}건 보기", f"View {len(updates)} unreviewed knowledge cards"), expanded=False):
            for update in updates:
                cid = str((update.get("payload") or {}).get("card_id", ""))
                card = cards_by_id.get(cid, {})
                provenance = card.get("provenance") if isinstance(card.get("provenance"), dict) else {}
                source_name = str((provenance or {}).get("source_name", "")).strip()
                st.write(f"- {'✓' if cid in selected_ids else '○'} `{cid}` · {card.get('claim', card.get('title', ''))}")
                if source_name:
                    st.caption(ui_text("참고논문 · ", "Source paper · ") + source_name)
        valid_ids = set(selected_ids)
        prompt = research_question_suggestions_prompt(selected_updates, recent_research_questions(ledger), all_cards, max_suggestions=8) if selected_updates else ""
        manual_col, auto_col = st.columns(2)
        with manual_col:
            if st.button(ui_text("수동 · 연구질문 후보 도출", "Manual · Generate RQ candidates"), key="m2-new-info-manual", type="primary", disabled=not selected_updates):
                parsed = parse_research_question_suggestions(llm_draft(prompt, model, use_ollama) or "", valid_card_ids=valid_ids, limit=8)
                if not parsed:
                    st.warning("RQ 후보를 읽지 못했습니다. 새 지식은 미처리 상태로 유지됩니다.")
                else:
                    review_id = ledger.create_research_state_review("manual", selected_updates)
                    saved = store_research_question_candidates(ledger, parsed, selected_updates, review_id=review_id)
                    for rq in saved:
                        ledger.ensure_research_question_thread(str(rq["rq_id"]), source_type="m1_knowledge", source_payload={"review_id": review_id})
                    ledger.complete_research_state_review(review_id, generated_rq_count=len(saved), selected_rq_count=0, summary=f"선택한 새 지식카드 {len(selected_updates)}건에서 RQ {len(saved)}건을 도출했습니다.")
                    st.success(f"RQ Thread {len(saved)}건을 생성·보강했습니다."); st.rerun()
            with st.expander("외부 LLM으로 RQ 후보 도출", expanded=False):
                st.text_area(ui_text("외부 LLM용 프롬프트", "Prompt for external LLM"), value=prompt, key="m2-rq-external-prompt", height=360)
                manual = st.text_area("외부 LLM 응답 붙여넣기", key="m2-rq-external-response", height=280)
                if st.button(ui_text("외부 응답을 RQ Thread에 반영", "Apply external response to RQ Threads"), key="m2-rq-external-apply", disabled=(not manual.strip() or not selected_updates)):
                    parsed = parse_research_question_suggestions(manual, valid_card_ids=valid_ids, limit=8)
                    if not parsed:
                        st.error("RQ 블록을 읽지 못했습니다.")
                    else:
                        review_id = ledger.create_research_state_review("manual", selected_updates)
                        saved = store_research_question_candidates(ledger, parsed, selected_updates, review_id=review_id)
                        for rq in saved:
                            ledger.ensure_research_question_thread(str(rq["rq_id"]), source_type="m1_knowledge", source_payload={"review_id": review_id, "generation_mode": "manual_external_llm"})
                        ledger.complete_research_state_review(review_id, generated_rq_count=len(saved), selected_rq_count=0, summary=f"외부 LLM으로 선택한 새 지식카드 {len(selected_updates)}건에서 RQ {len(saved)}건을 도출했습니다.")
                        st.success("RQ Thread에 반영했습니다."); st.rerun()
        with auto_col:
            if st.button(ui_text("자동 · 선택 지식 연구 사이클 실행", "Auto · Run cycle for selected knowledge"), key="m2-new-info-auto", type="primary", disabled=not selected_updates):
                with st.spinner(ui_text("선택한 지식으로 RQ 생성·보강 → 중요도 평가 → M1 자동 문헌탐색을 수행합니다.", "Generating/refining RQs from selected knowledge → prioritizing → running M1 literature follow-up.")):
                    result = execute_auto_research_cycle(
                        ledger, all_cards, CACHE,
                        rq_drafter=lambda p: llm_draft(p, model, use_ollama), priority_drafter=lambda p: llm_draft(p, model, use_ollama),
                        keyword_drafter=lambda p: llm_draft(p, model, use_ollama), abstract_reviewer=lambda p: llm_draft(p, model, use_ollama, profile="abstract_triage"),
                        fulltext_drafter=lambda p: paper_draft_result(p, model, use_ollama, "full_text_similarity").text,
                        synthesis_drafter=lambda p: llm_draft(p, model, use_ollama),
                        source_updates=selected_updates,
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
            st.write(f"- {_fmt_local_time(review.get('created_at'))} · {'자동' if review.get('mode')=='auto' else '수동'} · 새 카드 {review.get('source_card_count',0)}건 → RQ {review.get('generated_rq_count',0)}건")
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
        with st.expander(f"{_fmt_local_time(report.get('created_at'))} · {source} · {state} · {report['question_text']}"):
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



def _card_source_paper_title(card: dict[str, Any]) -> str:
    """Best-effort source-paper label for a knowledge card."""
    provenance = card.get("provenance") or {}
    paper_id = str(provenance.get("paper_id") or "").strip()
    if paper_id:
        paper = ledger.shelf_paper(paper_id)
        if paper and str(paper.get("title") or "").strip():
            return str(paper["title"]).strip()
    source_name = str(provenance.get("source_name") or "").strip()
    if source_name and not source_name.lower().startswith("sensemaking:"):
        return source_name
    return ""


def _render_knowledge_card_candidate_preview(card: dict[str, Any]) -> None:
    st.markdown("#### 생성된 지식카드 후보")
    with st.container(border=True):
        st.markdown(f"**{card.get('title', '제목 없음')}**")
        if card.get("claim"):
            st.markdown(f"**Claim**  \n{card.get('claim', '')}")
        if card.get("context"):
            st.markdown(f"**Context**  \n{card.get('context', '')}")
        if card.get("implication"):
            st.markdown(f"**Implication**  \n{card.get('implication', '')}")
        if card.get("evidence_excerpt"):
            st.markdown(f"**Evidence**  \n{card.get('evidence_excerpt', '')}")
        if card.get("conditions"):
            st.markdown(f"**Conditions**  \n{card.get('conditions', '')}")
        if card.get("limits"):
            st.markdown(f"**Limits**  \n{card.get('limits', '')}")
        labels = list(card.get("labels") or [])
        if labels:
            st.caption("Labels · " + " · ".join(str(x) for x in labels))



def _sensemaking_quick_preview(text: str, limit: int = 360) -> str:
    """Compact default preview for a long Research Fellow answer."""
    value = " ".join(str(text or "").strip().split())
    if not value:
        return ""
    # Prefer the first 2-3 sentences, then hard-cap for predictable UI density.
    import re
    parts = re.split(r"(?<=[.!?。！？])\s+", value)
    preview = " ".join(parts[:3]).strip()
    if not preview or len(preview) > limit:
        preview = value[:limit].rstrip()
    if len(value) > len(preview):
        preview = preview.rstrip(" .") + "…"
    return preview


def _render_sensemaking_evidence(turn: dict[str, Any], cards_by_id: dict[str, dict[str, Any]], *, key_suffix: str) -> None:
    evidence_ids = list(turn.get("evidence_card_ids") or [])
    papers = list(turn.get("quick_papers") or [])
    if evidence_ids:
        with st.expander(f"기존 지식 연결 {len(evidence_ids)}건", expanded=False):
            for cid in evidence_ids:
                card = cards_by_id.get(str(cid), {})
                st.markdown(f"**`{cid}` · {card.get('claim', card.get('title', ''))}**")
                paper_title = _card_source_paper_title(card)
                if paper_title:
                    st.caption(f"참고논문 · {paper_title}")
    if papers:
        with st.expander(f"외부 문헌 맥락 {len(papers)}편", expanded=False):
            for paper in papers[:20]:
                cites = paper.get("citation_count")
                cite_text = "확인 불가" if cites is None else f"{int(cites):,}회"
                st.markdown(f"**{paper.get('title','')}** · {str(paper.get('published',''))[:4]} · 인용 {cite_text}")
                st.caption(str(paper.get("summary", ""))[:600])


def _render_sensemaking_thread(thread_id: str, model: str, use_ollama: bool, semantic: bool, embedding_model: str) -> None:
    thread = ledger.sensemaking_thread(thread_id)
    if not thread:
        st.warning("Sensemaking Thread를 찾을 수 없습니다.")
        return
    turns = ledger.sensemaking_turns(thread_id)
    latest_user = next((t for t in reversed(turns) if t.get("role") == "user"), None)
    latest_assistant = next((t for t in reversed(turns) if t.get("role") == "assistant"), None)
    # A newly created thread already contains its first user turn.  Treat a trailing
    # user turn with no following assistant turn as the question waiting for
    # interpretation instead of asking the researcher to type it again.
    pending_user = turns[-1] if turns and turns[-1].get("role") == "user" else None
    current_question = str(latest_user.get("content", "")) if latest_user else str(thread.get("title", ""))
    st.markdown(f"## {thread['title']}")
    if thread.get("linked_rq_id"):
        rq = ledger.research_question(str(thread["linked_rq_id"]))
        st.success(f"Research Question Thread로 연결됨 · {rq.get('question','') if rq else thread['linked_rq_id']}")

    _render_current_state_and_reports(
        thread_kind="sensemaking", thread_id=thread_id, title=str(thread.get("title", "")),
        current_question=current_question, model=model, use_ollama=use_ollama,
        refresh_callback=lambda: _update_thread_current_state(
            thread_kind="sensemaking", thread_id=thread_id, title=str(thread.get("title", "")),
            current_question=current_question, model=model, use_ollama=use_ollama,
            conversation=ledger.sensemaking_turns(thread_id), reports=[],
        ),
    )

    cards_by_id = {str(card.get("card_id", "")): card for card in memory.all()}

    # Latest exchange first: this is the working surface immediately below Current State.
    if latest_user or latest_assistant:
        st.markdown("### 최신 대화")
        with st.container(border=True):
            if latest_user:
                st.markdown("**연구자**")
                st.markdown(str(latest_user.get("content", "")))
            if latest_assistant:
                # Keep the working surface dense: show only the quick interpretation content.
                st.caption(_sensemaking_quick_preview(str(latest_assistant.get("content", ""))))
                with st.expander("연구위원 전체 답변 보기", expanded=False):
                    st.markdown(str(latest_assistant.get("content", "")))
                _render_sensemaking_evidence(latest_assistant, cards_by_id, key_suffix="latest")

    # Previous exchanges are compact and reverse-chronological.
    previous_turns = list(turns)
    if latest_assistant in previous_turns:
        previous_turns.remove(latest_assistant)
    if latest_user in previous_turns:
        previous_turns.remove(latest_user)
    if previous_turns:
        st.markdown("### 이전 질의응답")
        # Pair adjacent user/assistant turns, then show newest pairs first.
        pairs: list[tuple[dict[str, Any] | None, dict[str, Any] | None]] = []
        pending_user: dict[str, Any] | None = None
        for turn in previous_turns:
            if turn.get("role") == "user":
                if pending_user is not None:
                    pairs.append((pending_user, None))
                pending_user = turn
            else:
                pairs.append((pending_user, turn))
                pending_user = None
        if pending_user is not None:
            pairs.append((pending_user, None))
        for idx, (user_turn, assistant_turn) in enumerate(reversed(pairs), start=1):
            q = str((user_turn or {}).get("content", ""))
            label = _sensemaking_quick_preview(q, 110) or "이전 대화"
            with st.expander(f"{idx}. {label}", expanded=False):
                if user_turn:
                    st.markdown("**연구자**")
                    st.markdown(q)
                if assistant_turn:
                    st.caption(_sensemaking_quick_preview(str(assistant_turn.get("content", ""))))
                    with st.expander("전체 답변", expanded=False):
                        st.markdown(str(assistant_turn.get("content", "")))
                    _render_sensemaking_evidence(assistant_turn, cards_by_id, key_suffix=f"prev-{idx}")

    st.divider()
    if pending_user:
        pending_question = str(pending_user.get("content", "")).strip()
        st.markdown("### 처음 질문")
        with st.container(border=True):
            st.markdown(pending_question)
            st.caption("이 질문은 아직 연구위원이 해석하지 않았습니다.")

        quick_lit = st.checkbox("Quick Literature 10~20편 포함", value=False, key=f"sm-pending-quick-{thread_id}")
        if st.button("빠르게 해석하기", key=f"sm-pending-answer-{thread_id}", type="primary", disabled=not pending_question):
            hits = search_knowledge(pending_question, semantic, embedding_model, limit=8)
            cards = [item.card for item in hits]
            papers: list[dict[str, Any]] = []
            # Exclude the unanswered user turn from conversation history because
            # it is supplied separately as the current question.
            prior_turns = turns[:-1]
            if quick_lit:
                plan_text = llm_draft(quick_search_plan_prompt(pending_question, prior_turns), model, use_ollama, profile="search_strategy") or ""
                result = quick_literature_search(plan_text, max_papers=20)
                papers = result["papers"]
            prompt = sensemaking_answer_prompt(
                thread_title=str(thread["title"]), conversation=prior_turns,
                question=pending_question, cards=cards, papers=papers,
            )
            answer = llm_draft(prompt, model, use_ollama, profile="sensemaking") or "현재 LLM 응답을 얻지 못했습니다. 외부 LLM 수동 응답 경로를 사용해 주세요."
            ledger.add_sensemaking_turn(
                thread_id, "assistant", answer,
                evidence_card_ids=[str(c.get("card_id", "")) for c in cards],
                quick_papers=papers,
            )
            _update_thread_current_state(
                thread_kind="sensemaking", thread_id=thread_id, title=str(thread.get("title", "")),
                current_question=pending_question, model=model, use_ollama=use_ollama,
                conversation=ledger.sensemaking_turns(thread_id), reports=[],
            )
            st.rerun()

        with st.expander("외부 LLM으로 첫 질문 빠르게 해석", expanded=False):
            hits = search_knowledge(pending_question, semantic, embedding_model, limit=8)
            cards = [item.card for item in hits]
            prior_turns = turns[:-1]
            prompt = sensemaking_answer_prompt(
                thread_title=str(thread["title"]), conversation=prior_turns,
                question=pending_question, cards=cards, papers=[],
            )
            st.text_area("외부 LLM용 프롬프트", value=prompt, key=f"sm-pending-ext-prompt-{thread_id}", height=360)
            manual = st.text_area("외부 LLM 응답 붙여넣기", key=f"sm-pending-ext-response-{thread_id}", height=280)
            if st.button("수동 응답을 Thread에 반영", key=f"sm-pending-ext-apply-{thread_id}", disabled=not manual.strip()):
                ledger.add_sensemaking_turn(
                    thread_id, "assistant", manual.strip(),
                    evidence_card_ids=[str(c.get("card_id", "")) for c in cards],
                    generation_mode="manual_external_llm",
                )
                _update_thread_current_state(
                    thread_kind="sensemaking", thread_id=thread_id, title=str(thread.get("title", "")),
                    current_question=pending_question, model=model, use_ollama=use_ollama,
                    conversation=ledger.sensemaking_turns(thread_id), reports=[],
                    generation_mode="manual_external_llm",
                )
                st.rerun()
    else:
        st.markdown("### 이어서 물어보기")
        question = st.text_area(
            "추가 질문·주장·사례", key=f"sm-follow-{thread_id}", height=110,
            placeholder="짧게 물어보세요. 기존 지식으로 먼저 답하고, 필요하면 10~20편만 빠르게 확인합니다.",
        )
        quick_lit = st.checkbox("Quick Literature 10~20편 포함", value=False, key=f"sm-quick-{thread_id}")
        if st.button("빠르게 해석하기", key=f"sm-answer-{thread_id}", type="primary", disabled=not question.strip()):
            ledger.add_sensemaking_turn(thread_id, "user", question.strip())
            hits = search_knowledge(question, semantic, embedding_model, limit=8)
            cards = [item.card for item in hits]
            papers: list[dict[str, Any]] = []
            if quick_lit:
                plan_text = llm_draft(quick_search_plan_prompt(question, turns), model, use_ollama, profile="search_strategy") or ""
                result = quick_literature_search(plan_text, max_papers=20)
                papers = result["papers"]
            prompt = sensemaking_answer_prompt(
                thread_title=str(thread["title"]), conversation=turns,
                question=question, cards=cards, papers=papers,
            )
            answer = llm_draft(prompt, model, use_ollama, profile="sensemaking") or "현재 LLM 응답을 얻지 못했습니다. 외부 LLM 수동 응답 경로를 사용해 주세요."
            ledger.add_sensemaking_turn(
                thread_id, "assistant", answer,
                evidence_card_ids=[str(c.get("card_id", "")) for c in cards],
                quick_papers=papers,
            )
            _update_thread_current_state(
                thread_kind="sensemaking", thread_id=thread_id, title=str(thread.get("title", "")),
                current_question=question.strip(), model=model, use_ollama=use_ollama,
                conversation=ledger.sensemaking_turns(thread_id), reports=[],
            )
            st.rerun()

        with st.expander("외부 LLM으로 이번 질문 빠르게 해석", expanded=False):
            if question.strip():
                hits = search_knowledge(question, semantic, embedding_model, limit=8)
                cards = [item.card for item in hits]
                prompt = sensemaking_answer_prompt(thread_title=str(thread["title"]), conversation=turns, question=question, cards=cards, papers=[])
                st.text_area("외부 LLM용 프롬프트", value=prompt, key=f"sm-ext-prompt-{thread_id}", height=360)
                manual = st.text_area("외부 LLM 응답 붙여넣기", key=f"sm-ext-response-{thread_id}", height=280)
                if st.button("수동 응답을 Thread에 반영", key=f"sm-ext-apply-{thread_id}", disabled=not manual.strip()):
                    ledger.add_sensemaking_turn(thread_id, "user", question.strip())
                    ledger.add_sensemaking_turn(
                        thread_id, "assistant", manual.strip(),
                        evidence_card_ids=[str(c.get("card_id", "")) for c in cards],
                        generation_mode="manual_external_llm",
                    )
                    _update_thread_current_state(
                        thread_kind="sensemaking", thread_id=thread_id, title=str(thread.get("title", "")),
                        current_question=question.strip(), model=model, use_ollama=use_ollama,
                        conversation=ledger.sensemaking_turns(thread_id), reports=[], generation_mode="manual_external_llm",
                    )
                    st.rerun()
            else:
                st.caption("먼저 추가 질문·주장을 입력하면 현재 Thread와 관련 지식카드를 포함한 프롬프트를 만듭니다.")

    st.divider()
    st.markdown("### 이 대화에서 발전시키기")
    c1, c2, c3 = st.columns(3)
    if c1.button("Research Question으로 승격", key=f"sm-rq-{thread_id}", disabled=not latest_user):
        rq = ledger.create_research_question_thread(
            question=str(latest_user.get("content", "")), source_type="sensemaking",
            rationale=f"Sensemaking Thread '{thread['title']}'에서 연구적으로 더 검토할 질문으로 승격됨",
            research_context=str(latest_assistant.get("content", ""))[:1800] if latest_assistant else "",
            source_payload={"sensemaking_thread_id": thread_id}, status="interested",
        )
        ledger.link_sensemaking_to_rq(thread_id, str(rq["rq_id"]))
        inherited = ledger.thread_current_state("sensemaking", thread_id)
        if inherited:
            ledger.save_thread_current_state(
                thread_kind="research_question", thread_id=str(rq["rq_id"]),
                current_question=str(rq.get("question", "")), body_text=str(inherited.get("body_text", "")),
                generation_mode="inherited_from_sensemaking",
            )
        st.session_state[f"sm-promoted-rq-{thread_id}"] = {"rq_id": str(rq["rq_id"]), "question": str(rq.get("question", ""))}
        st.rerun()
    if c2.button("지식카드 후보 만들기", key=f"sm-card-{thread_id}", disabled=not (latest_user and latest_assistant)):
        draft = llm_draft(knowledge_card_candidate_prompt(thread_title=str(thread["title"]), latest_question=str(latest_user.get("content", "")), latest_answer=str(latest_assistant.get("content", ""))), model, use_ollama, profile="knowledge_card") or ""
        card = parse_sensemaking_card_candidate(draft, thread_title=str(thread["title"]))
        if not card:
            st.warning("지식카드 후보를 만들지 못했습니다.")
        else:
            case_id = ledger.create_case("research", f"Sensemaking card: {thread['title']}")
            ledger.record(case_id, "decision_request", "m2", ["researcher"], "knowledge_card", {
                "title": f"Sensemaking 지식카드 후보: {card['title']}", "card": card,
                "normalization": "sensemaking_candidate", "warnings": ["외부 주장/대화에서 생성된 후보입니다. 출처와 근거를 확인한 뒤 승인하세요."],
                "next_action": "승인 시 semantic memory에 저장하고 M2 새 정보로 통지",
            }, subject_id=str(card["card_id"]))
            st.session_state[f"sm-last-card-candidate-{thread_id}"] = card
            st.success("지식카드 후보를 만들고 연구자 검토·승인함에 보냈습니다. 아래에서 후보 내용을 확인하세요.")
    candidate_preview = st.session_state.get(f"sm-last-card-candidate-{thread_id}")
    if candidate_preview:
        _render_knowledge_card_candidate_preview(candidate_preview)
        st.info(
            "다음 단계 · ① 연구위원 데스크의 연구자 검토·승인함에서 이 후보를 확인·승인하세요. "
            "② 승인되면 새 승인 지식카드가 M2의 ‘M1 새 지식 기반’ 입력으로 들어가며, 그곳에서 연구질문 생성·보강에 사용할 수 있습니다."
        )
        nav1, clear1 = st.columns([1.4, 1])
        if nav1.button("연구위원 데스크 승인함으로 이동", key=f"sm-go-approval-{thread_id}"):
            st.session_state["_navigate_workspace"] = "연구위원 데스크"
            st.rerun()
        if clear1.button("후보 미리보기 닫기", key=f"sm-clear-candidate-{thread_id}"):
            st.session_state.pop(f"sm-last-card-candidate-{thread_id}", None)
            st.rerun()

    promoted = st.session_state.get(f"sm-promoted-rq-{thread_id}")
    if promoted:
        st.success(f"Research Question Thread로 승격했습니다 · {promoted.get('question','')}")
        st.info(
            "다음 단계 · M2의 ‘연구자 직접 질문’ 페이지에서 이 질문 Thread를 열어 현재 지식으로 M2 Review를 만들고, "
            "필요하면 M1 문헌 보강 Intent로 이어가세요."
        )
        nav2, clear2 = st.columns([1.4, 1])
        if nav2.button("M2 연구자 질문으로 이동", key=f"sm-go-rq-{thread_id}"):
            st.session_state["m2-work-page"] = "연구자 직접 질문"
            st.session_state["m2-selected-researcher-thread-id"] = str(promoted.get("rq_id", ""))
            st.session_state["_navigate_workspace"] = "M2 · 지식 기반 자문"
            st.rerun()
        if clear2.button("안내 닫기", key=f"sm-clear-rq-guide-{thread_id}"):
            st.session_state.pop(f"sm-promoted-rq-{thread_id}", None)
            st.rerun()

    if c3.button("Thread 보관", key=f"sm-archive-{thread_id}"):
        ledger.archive_sensemaking_thread(thread_id, True)
        st.session_state.pop("sensemaking-selected-thread", None)
        st.rerun()


def sensemaking_screen(model: str, use_ollama: bool, semantic: bool, embedding_model: str) -> None:
    st.header("Research Sensemaking")
    st.caption(ui_text("외부에서 접한 주장·사례·궁금증을 기존 지식과 빠르게 대조하고, 필요할 때만 10~20편의 문헌을 확인하며 대화를 이어갑니다. 충분히 중요한 질문은 정식 Research Question Thread로 승격할 수 있습니다.", "Quickly compare external claims, cases, and questions against accumulated knowledge, optionally check 10–20 papers, and continue the dialogue. Promote sufficiently important questions to a formal Research Question Thread."))
    left, right = st.columns([0.32, 0.68])
    with left:
        st.markdown(ui_text("### Sensemaking Threads", "### Sensemaking Threads"))
        with st.expander(ui_text("+ 새 Thread", "+ New Thread"), expanded=not bool(ledger.sensemaking_threads())):
            # A form with clear_on_submit prevents the creation inputs from
            # lingering after the new thread has been opened on the right.
            with st.form("sm-new-thread-form", clear_on_submit=True):
                title = st.text_input(ui_text("제목", "Title"), placeholder=ui_text("예: Upstream Quality와 Shift-left의 유사성", "e.g., Similarities between Upstream Quality and Shift-left"))
                first = st.text_area(ui_text("처음 궁금한 주장·사실·사례", "Initial claim, fact, case, or question"), height=130)
                create = st.form_submit_button(ui_text("Thread 시작", "Start Thread"), type="primary")
                if create:
                    if not first.strip():
                        st.warning(ui_text("처음 궁금한 주장·사실·사례를 입력해 주세요.", "Enter an initial claim, fact, case, or question."))
                    else:
                        item = ledger.create_sensemaking_thread(title, first)
                        st.session_state["sensemaking-selected-thread"] = item["thread_id"]
                        st.rerun()
        threads = ledger.sensemaking_threads(limit=100)
        for idx, item in enumerate(threads, start=1):
            selected = st.session_state.get("sensemaking-selected-thread") == item["thread_id"]
            with st.container(border=True):
                st.caption(f"THREAD {idx:02d}" + (ui_text(" · 현재 열림", " · Open") if selected else ""))
                st.markdown(f"**{item['title']}**")
                turns = ledger.sensemaking_turns(str(item["thread_id"]))
                latest = next((t for t in reversed(turns) if t.get("role") == "assistant"), None)
                if latest:
                    # Thread list shows only the latest quick-interpretation content, in compact text.
                    st.caption(_sensemaking_quick_preview(str(latest.get("content", "")), 150))
                st.caption(f"대화 {len(turns)}턴 · {_fmt_local_time(item.get('updated_at'))}" + (ui_text(" · RQ 연결", " · Linked to RQ") if item.get("linked_rq_id") else ""))
                if st.button(ui_text("Thread 열기", "Open Thread") if not selected else ui_text("현재 Thread", "Current Thread"), key=f"sm-open-{item['thread_id']}", disabled=selected, use_container_width=True):
                    st.session_state["sensemaking-selected-thread"] = item["thread_id"]
                    st.rerun()
    with right:
        selected_id = st.session_state.get("sensemaking-selected-thread")
        if selected_id:
            _render_sensemaking_thread(str(selected_id), model, use_ollama, semantic, embedding_model)
        else:
            st.info(ui_text("왼쪽에서 기존 Thread를 열거나 새 Sensemaking Thread를 시작하세요.", "Open an existing thread on the left or start a new Sensemaking Thread."))


def _paper_project_for_research_question(projects: list[dict[str, Any]], rq_id: str) -> dict[str, Any] | None:
    """Return the paper project already created from an RQ, if any."""
    for project in projects:
        for event in project.get("events", []):
            payload = event.get("payload") or {}
            if event.get("event_type") == "project_origin" and str(payload.get("source_rq_id")) == rq_id:
                return project
    return None


def _create_paper_project_from_research_question(rq: dict[str, Any], title: str) -> dict[str, Any]:
    """Create a Short Paper project while preserving the RQ's evidence and derivation context."""
    rq_id = str(rq.get("rq_id", ""))
    source_card_ids = list(dict.fromkeys(str(item) for item in rq.get("source_card_ids", []) if str(item)))
    thread = ledger.research_question_thread(rq_id) or rq
    project = ledger.create_paper_project(
        title=title.strip(), research_question=str(rq.get("question", "")).strip(),
        origin_type="research_question", origin_ids=source_card_ids,
    )
    ledger.add_paper_project_event(project["project_id"], "project_origin", {
        "source_rq_id": rq_id,
        "source_type": str(thread.get("source_type", "m1_knowledge")),
        "rationale": str(rq.get("rationale", "")),
        "research_context": str(rq.get("research_context", "")),
        "gap_or_tension": str(rq.get("gap_or_tension", "")),
        "exploration_need": str(rq.get("exploration_need", "")),
        "source_card_ids": source_card_ids,
        "source_payload": thread.get("source_payload") or {},
    })
    return ledger.paper_project(project["project_id"]) or project


PAPER_EVIDENCE_SOURCE_LABELS = {
    "inherited_rq": "기존 질문 연결",
    "hybrid_search": "키워드·임베딩",
    "ontology_same_type": "동일 타입 확장",
    "ontology_related_type": "관계 타입 확장",
}


def _discover_paper_evidence(
    title: str, research_question: str, inherited_ids: list[str],
    semantic: bool, embedding_model: str,
) -> tuple[str, list[dict[str, Any]]]:
    query = paper_evidence_query(title, research_question)
    cards = memory.all()
    search_hits = search_knowledge(query, semantic, embedding_model, limit=12)
    hit_rows = [
        {"card_id": hit.card["card_id"], "score": hit.score, "method": hit.method}
        for hit in search_hits
    ]
    seed_ids = list(dict.fromkeys([*inherited_ids, *(row["card_id"] for row in hit_rows)]))
    seed_types = {card_id: ledger.ontology_types_for_card(card_id) for card_id in seed_ids}
    seed_type_ids = {
        str(item["type_id"])
        for rows in seed_types.values() for item in rows if item.get("type_id")
    }
    type_rows = ledger.ontology_types()
    type_names = {
        str(item["type_id"]): str(item.get("name", item["type_id"])) for item in type_rows
    }
    relations = ledger.ontology_type_relations()
    related_type_ids = set(seed_type_ids)
    for relation in relations:
        source_id = str(relation.get("source_type_id", ""))
        target_id = str(relation.get("target_type_id", ""))
        if source_id in seed_type_ids and target_id:
            related_type_ids.add(target_id)
        if target_id in seed_type_ids and source_id:
            related_type_ids.add(source_id)
    card_ids_by_type = {
        type_id: ledger.ontology_card_ids(type_id) for type_id in related_type_ids
    }
    candidates = assemble_paper_evidence_candidates(
        query=query,
        cards=cards,
        search_hits=hit_rows,
        inherited_card_ids=inherited_ids,
        seed_type_ids=seed_type_ids,
        type_names=type_names,
        type_relations=relations,
        card_ids_by_type=card_ids_by_type,
        max_candidates=24,
    )
    for candidate in candidates:
        candidate["type_names"] = [
            str(item.get("name", ""))
            for item in ledger.ontology_types_for_card(candidate["card_id"])
            if item.get("name")
        ]
    return query, candidates


def _render_paper_evidence_selector(
    *,
    title: str,
    research_question: str,
    inherited_ids: list[str],
    card_by_id: dict[str, dict[str, Any]],
    semantic: bool,
    embedding_model: str,
    key_prefix: str,
) -> list[str]:
    result_key, selected_key = f"{key_prefix}-results", f"{key_prefix}-selected"
    disabled = not title.strip() or not research_question.strip()
    if st.button("관련 지식카드 자동 탐색", disabled=disabled, key=f"{key_prefix}-discover"):
        with st.spinner("제목·연구질문 검색 후 온톨로지 1-hop 관계를 확장하고 있습니다."):
            query, candidates = _discover_paper_evidence(
                title, research_question, inherited_ids, semantic, embedding_model,
            )
        st.session_state[result_key] = {"query": query, "candidates": candidates}
        st.session_state[selected_key] = [item["card_id"] for item in candidates]
        st.rerun()
    result = st.session_state.get(result_key) or {}
    candidates = list(result.get("candidates") or [])
    current_query = paper_evidence_query(title, research_question)
    if candidates and result.get("query") != current_query:
        st.warning("제목 또는 연구질문이 변경되었습니다. 관련 지식카드를 다시 탐색해 주세요.")
    if not candidates:
        if inherited_ids:
            st.caption(
                f"기존 연구질문 연결 카드 {len(inherited_ids)}건이 있습니다. 버튼을 누르면 "
                "제목·질문 검색과 온톨로지 확장을 함께 수행합니다."
            )
        else:
            st.caption(
                "제목과 연구질문을 입력한 뒤 버튼을 누르면 키워드·임베딩 검색과 "
                "온톨로지 1-hop 확장을 수행합니다."
            )
        return []
    candidate_by_id = {item["card_id"]: item for item in candidates}
    options = [item["card_id"] for item in candidates if item["card_id"] in card_by_id]
    selected = st.multiselect(
        "논문 프로젝트 출발 근거 카드",
        options,
        key=selected_key,
        format_func=lambda card_id: (
            f"[{PAPER_EVIDENCE_SOURCE_LABELS.get(candidate_by_id[card_id]['source'], candidate_by_id[card_id]['source'])}] "
            f"{card_by_id.get(card_id, {}).get('title') or card_id}"
        ),
    )
    counts: dict[str, int] = {}
    for item in candidates:
        label = PAPER_EVIDENCE_SOURCE_LABELS.get(item["source"], item["source"])
        counts[label] = counts.get(label, 0) + 1
    st.caption(" · ".join(f"{label} {count}건" for label, count in counts.items()))
    with st.expander("탐색 근거·온톨로지 경로 확인", expanded=False):
        rows = []
        for item in candidates:
            card = card_by_id.get(item["card_id"], {})
            rows.append({
                "선택": "✓" if item["card_id"] in selected else "",
                "카드": card.get("title") or item["card_id"],
                "유입 경로": PAPER_EVIDENCE_SOURCE_LABELS.get(item["source"], item["source"]),
                "타입": ", ".join(item.get("type_names", [])),
                "온톨로지 경로": " | ".join(item.get("ontology_paths", [])),
            })
        st.dataframe(rows, use_container_width=True, hide_index=True)
    return list(selected)


def _record_paper_evidence_selection(project_id: str, key_prefix: str, selected_ids: list[str]) -> None:
    result = st.session_state.get(f"{key_prefix}-results") or {}
    if not result:
        return
    selected_set = set(selected_ids)
    candidates = [
        {
            "card_id": item.get("card_id", ""),
            "source": item.get("source", ""),
            "origins": item.get("origins", []),
            "ontology_paths": item.get("ontology_paths", []),
            "selected": item.get("card_id") in selected_set,
        }
        for item in result.get("candidates", [])
    ]
    ledger.add_paper_project_event(project_id, "evidence_selection", {
        "query": result.get("query", ""),
        "selected_card_ids": selected_ids,
        "candidates": candidates,
    })


def _paper_diagnostic_rows(
    manuscript: dict[str, Any], card_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    labels = {row["마크업"]: row["표시"] for row in annotation_legend()}
    rows = []
    for sentence in manuscript_sentences(manuscript):
        evidence_ids = [str(item) for item in sentence.get("evidence_card_ids", [])]
        open_annotations = [
            item for item in sentence.get("annotations", []) if item.get("status") == "open"
        ]
        rows.append({
            "문장 ID": sentence.get("sentence_id", ""),
            "문장 내용": sentence.get("text", ""),
            "역할": sentence.get("role", ""),
            "연결 근거": ", ".join(
                str((card_by_id.get(card_id) or {}).get("title") or card_id)
                for card_id in evidence_ids
            ) or "없음",
            "보완점": ", ".join(labels.get(str(item.get("type", "")), str(item.get("type", ""))) for item in open_annotations) or "완료",
            "진단 내용": " | ".join(str(item.get("comment", "")) for item in open_annotations if item.get("comment")) or "-",
        })
    return rows


def _apply_targeted_paper_revision(
    response: str, manuscript: dict[str, Any], *, valid_card_ids: set[str],
    version: int, sentence_id: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Apply one-sentence revision with a safe fallback for older local modules."""
    try:
        return apply_revisions(
            response, manuscript, valid_card_ids=valid_card_ids, version=version,
            allowed_sentence_ids={sentence_id},
        )
    except TypeError as error:
        if "allowed_sentence_ids" not in str(error):
            raise
        revised, diff = apply_revisions(
            response, manuscript, valid_card_ids=valid_card_ids, version=version,
        )
        unexpected = [item for item in diff if str(item.get("sentence_id", "")) != sentence_id]
        if unexpected:
            raise ValueError("선택한 문장 이외의 수정이 포함되어 반영하지 않았습니다.")
        return revised, diff


def _matches_revision_origin(value: dict[str, Any], project_id: str, todo_id: str) -> bool:
    return any(
        str(link.get("origin_type")) == "paper_writing"
        and str(link.get("origin_id")) == project_id
        and str(link.get("origin_sub_id")) == todo_id
        for link in normalize_origin_links(value.get("origin_links", []))
    )


def _revision_todo_assets(
    project_id: str, todo_id: str, cards: list[dict[str, Any]], papers: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Find M1 outputs that preserve the exact paper-project/To-do lineage."""
    todo_cards = [card for card in cards if _matches_revision_origin(card, project_id, todo_id)]
    todo_papers = [paper for paper in papers if _matches_revision_origin(paper, project_id, todo_id)]
    paper_ids_from_cards = {
        str((card.get("provenance") or {}).get("paper_id") or "") for card in todo_cards
    }
    known_paper_ids = {str(paper.get("paper_id") or "") for paper in todo_papers}
    todo_papers.extend(
        paper for paper in papers
        if str(paper.get("paper_id") or "") in paper_ids_from_cards
        and str(paper.get("paper_id") or "") not in known_paper_ids
    )
    return todo_cards, todo_papers


def _latest_revision_evidence_link(events: list[dict[str, Any]], todo_id: str) -> dict[str, Any]:
    return next((
        event.get("payload") or {} for event in reversed(events)
        if event.get("event_type") == "revision_todo_evidence_link"
        and str((event.get("payload") or {}).get("todo_id") or "") == todo_id
    ), {})


def _paper_project_references(project_id: str) -> list[dict[str, Any]]:
    return [
        reference for reference in ledger.literature_references(limit=1000)
        if str((reference.get("paper") or {}).get("paper_project_id") or "") == project_id
    ]


def _reference_display(reference: dict[str, Any]) -> str:
    paper = reference.get("paper") or {}
    authors = ", ".join(str(item) for item in paper.get("authors", []) if str(item).strip())
    year = str(paper.get("publication_year") or paper.get("published") or "")[:4]
    prefix = "; ".join(part for part in [authors, year] if part)
    return f"{prefix + '. ' if prefix else ''}{paper.get('title') or '제목 없음'}"


def _revision_todos_with_events(manuscript: dict[str, Any], events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for event in events:
        if event.get("event_type") != "revision_todo_update":
            continue
        payload = event.get("payload") or {}
        if payload.get("todo_id"):
            latest[str(payload["todo_id"])] = payload
    result = []
    terminal_statuses={"resolved","obsolete","merged","split"}
    active_statuses={"open","ready","revised_pending_review","modified","researcher_review","reopened"}
    manuscript_ids=set()
    for item in revision_todos(manuscript):
        manuscript_ids.add(str(item["todo_id"]))
        update = latest.get(item["todo_id"], {})
        update_status = str(update.get("status") or "")
        if update_status in terminal_statuses:
            continue
        if update_status in active_statuses:
            item["status"] = update_status
        for field in ("sentence_id","sentence_text","label","priority","recommended_action","problem","search_guide","completion_criteria"):
            if update.get(field):item[field]=update[field]
        item["latest_update"] = update
        result.append(item)
    for todo_id,update in latest.items():
        status=str(update.get("status") or "")
        if todo_id in manuscript_ids or status not in active_statuses:continue
        result.append({**update,"todo_id":todo_id,"status":status,"latest_update":update})
    order={"P0":0,"P1":1,"P2":2}
    result.sort(key=lambda item:(order.get(str(item.get("priority") or "P1"),1),str(item.get("sentence_id") or ""),str(item.get("todo_id") or "")))
    return result


def _revision_todo_history_with_events(
    manuscript: dict[str, Any], events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return active and completed To-dos as one durable project checklist."""
    latest_updates: dict[str, dict[str, Any]] = {}
    for event in events:
        if event.get("event_type") == "revision_todo_update":
            payload = event.get("payload") or {}
            if payload.get("todo_id"):
                latest_updates[str(payload["todo_id"])] = payload
    active = _revision_todos_with_events(manuscript, events)
    return revision_todo_timeline(active, list(latest_updates.values()))


def _record_resolved_revision_todos(
    project_id: str, before_todos: list[dict[str, Any]], annotations: list[dict[str, Any]], source: str,
) -> None:
    remaining_ids = {str(item.get("annotation_id", "")) for item in annotations}
    for todo in before_todos:
        if todo["todo_id"] and todo["todo_id"] not in remaining_ids:
            ledger.add_paper_project_event(project_id, "revision_todo_update", {
                **todo, "status": "resolved", "resolution_source": source,
            })


def _review_result_payload(
    *, source: str, before_todos: list[dict[str, Any]], annotations: list[dict[str, Any]],
    reviewed_from_version: int, result_version: int, response_text: str = "",
) -> dict[str, Any]:
    type_counts: dict[str, int] = {}
    for item in annotations:
        kind = str(item.get("type", ""))
        type_counts[kind] = type_counts.get(kind, 0) + 1
    changes = review_change_set(before_todos, annotations)
    return {
        "source": source,
        "reviewed_from_version": reviewed_from_version,
        "result_version": result_version,
        "todo_count": len(annotations),
        "retained_count": len(changes["retained_todos"]),
        "new_count": len(changes["new_todos"]),
        "resolved_count": len(changes["resolved_todos"]),
        **changes,
        "type_counts": type_counts,
        "response_digest": hashlib.sha256(response_text.strip().encode("utf-8")).hexdigest() if response_text.strip() else "",
    }


def _render_review_result(payload: dict[str, Any], *, latest: bool = False) -> None:
    source_label = {
        "external_llm": "외부 LLM", "internal_llm": "내부 LLM",
        "internal_llm_recheck": "내부 LLM 재검증",
    }.get(str(payload.get("source", "")), str(payload.get("source", "")))
    prefix = "최근 검증 반영" if latest else "검증 반영 완료"
    st.success(
        f"{prefix} · {source_label} · 원고 v{payload.get('reviewed_from_version','?')} → "
        f"v{payload.get('result_version','?')}"
    )
    c1,c2,c3,c4=st.columns(4)
    c1.metric("현재 To-do",int(payload.get("todo_count",0)))
    c2.metric("유지",int(payload.get("retained_count",0)))
    c3.metric("신규",int(payload.get("new_count",0)))
    c4.metric("해결",int(payload.get("resolved_count",0)))
    counts=payload.get("type_counts") or {}
    if counts:
        labels={row["마크업"]:row["표시"] for row in annotation_legend()}
        st.caption("반영된 마크업 · " + " · ".join(f"{labels.get(kind,kind)} {count}건" for kind,count in counts.items()))
    changes=[]
    for key in ("new_todos", "resolved_todos", "retained_todos"):
        changes.extend(payload.get(key) or [])
    if changes:
        st.dataframe([{
            "변경": item.get("change", ""), "To-do": item.get("todo_id", ""),
            "문장": item.get("sentence_id", ""), "보완점": item.get("label", ""),
            "판정 내용": item.get("problem", ""),
        } for item in changes], use_container_width=True, hide_index=True)
    elif any(key in payload for key in ("new_todos", "resolved_todos", "retained_todos")):
        st.caption("이번 검증에서 추가·유지·해결된 To-do가 없습니다.")


def _render_revision_todo_progress(manuscript: dict[str, Any], events: list[dict[str, Any]]) -> None:
    timeline = _revision_todo_history_with_events(manuscript, events)
    total = len(timeline)
    closed = sum(1 for item in timeline if item.get("status") in {"resolved","obsolete","merged","split"})
    progress = closed / total if total else 1.0
    st.progress(progress, text=f"현재 Revision 백로그 정리 · {closed}/{total} 처리 ({progress:.0%})")
    if not timeline:
        return
    with st.expander("전체 To-do 체크리스트", expanded=True):
        status_labels = {
            "open": "진단됨", "ready": "보완 준비", "revised_pending_review": "재검증 대기",
            "modified":"내용 수정","researcher_review":"연구자 판단","reopened":"재개",
            "resolved": "해결", "obsolete":"불필요", "merged":"통합", "split":"분리",
        }
        for item in timeline:
            label = f"[{item.get('priority','P1')}] {item.get('label','')} · {item.get('sentence_id','')}"
            status = status_labels.get(str(item.get("status", "")), str(item.get("status", "")))
            if item.get("status") in {"resolved","obsolete","merged","split"}:
                st.markdown(f"- [x] ~~{label}~~ — **{status}**")
            else:
                st.markdown(f"- [ ] {label} — **{status}**")


def _render_manuscript_revision_history(
    version_events: list[dict[str, Any]], events: list[dict[str, Any]],
) -> None:
    revisions=[event for event in version_events if (event.get("payload") or {}).get("diff")]
    st.divider();st.subheader("문장 변경 이력")
    if not revisions:
        st.caption("아직 문장 리비전 이력이 없습니다. Revision To-do에서 문장을 수정하면 이전·수정 문장이 여기에 기록됩니다.")
        return
    todo_updates=[
        event for event in events
        if event.get("event_type")=="revision_todo_update"
        and (event.get("payload") or {}).get("status")=="revised_pending_review"
    ]
    source_labels={
        "targeted_revision":"내부 LLM 문장 리비전","external_revision":"외부 LLM 문장 리비전",
        "todo_full_revision":"내부 LLM 전체 원고 영향 리비전",
        "external_todo_full_revision":"외부 LLM 전체 원고 영향 리비전",
        "appendix_refresh":"내부 LLM Appendix 후보 갱신",
        "external_appendix_refresh":"외부 LLM Appendix 후보 갱신",
        "researcher_approved_resolution":"연구자 확인 후 To-do 해결 반영",
    }
    st.caption(f"총 {sum(len((event.get('payload') or {}).get('diff') or []) for event in revisions)}개 본문·Appendix 변경 · 최신 변경부터 표시")
    for index,event in enumerate(reversed(revisions)):
        payload=event.get("payload") or {}; manuscript_at_version=payload.get("manuscript") or {}
        version=manuscript_at_version.get("version","?");diffs=list(payload.get("diff") or [])
        sentence_ids={str(item.get("sentence_id","")) for item in diffs}
        matching_update=next((
            update.get("payload") or {} for update in reversed(todo_updates)
            if str(update.get("created_at", ""))<=str(event.get("created_at", ""))
            and str((update.get("payload") or {}).get("sentence_id", "")) in sentence_ids
        ),{})
        todo_label=str(payload.get("todo_label") or matching_update.get("label") or "")
        todo_id=str(payload.get("todo_id") or matching_update.get("todo_id") or "")
        source=source_labels.get(str(payload.get("source","")),str(payload.get("source", "리비전")))
        heading=f"v{version} · {source} · {len(diffs)}개 문장"
        if todo_label:heading+=f" · {todo_label}"
        with st.expander(heading,expanded=index==0):
            st.caption(f"{event.get('created_at','')}" + (f" · To-do `{todo_id}`" if todo_id else ""))
            for diff in diffs:
                scope_label="Appendix 주장" if diff.get("change_scope")=="appendix" else "문장"
                st.markdown(f"**{scope_label} `{diff.get('sentence_id','')}`**" + (f" · {diff.get('reason')}" if diff.get("reason") else ""))
                before_col,after_col=st.columns(2)
                with before_col:
                    st.caption("이전 내용")
                    st.markdown(f"> {str(diff.get('before','')).replace(chr(10), ' ')}")
                with after_col:
                    st.caption("수정 내용")
                    st.markdown(f"> {str(diff.get('after','')).replace(chr(10), ' ')}")
                st.divider()


def _render_revision_todo_reconciliation(
    *, project_id: str, project: dict[str,Any], manuscript: dict[str,Any],
    events: list[dict[str,Any]], version_events: list[dict[str,Any]],
    model: str, use_ollama: bool,
) -> None:
    """Reconcile the backlog after a manuscript-changing revision."""
    if len(version_events)<2:return
    latest_version_event=version_events[-1];latest_payload=latest_version_event.get("payload") or {}
    revision_diff=list(latest_payload.get("diff") or [])
    if not revision_diff:return
    previous_manuscript=(version_events[-2].get("payload") or {}).get("manuscript") or {}
    from_version=int(previous_manuscript.get("version") or 0);to_version=int(manuscript.get("version") or 0)
    before_todos=revision_todos(previous_manuscript)
    if not before_todos:return
    applied_event=next((event for event in reversed(events)
        if event.get("event_type")=="revision_todo_reconciliation_applied"
        and int((event.get("payload") or {}).get("to_version") or -1)==to_version),None)
    if applied_event:
        report=(applied_event.get("payload") or {}).get("report") or {}
        with st.expander(f"최근 Revision {from_version} → {to_version} To-do 정리 결과",expanded=True):
            if report.get("summary"):st.info(report["summary"])
            counts=report.get("counts") or {}
            columns=st.columns(5)
            for column,(label,key) in zip(columns,[("해결","resolved"),("유지·수정","active"),("불필요","obsolete"),("통합·분리","restructured"),("신규","new")]):
                column.metric(label,int(counts.get(key,0)))
            if report.get("revision_achievements"):
                st.markdown("**이번 리비전에서 해소한 내용**")
                for item in report["revision_achievements"]:st.markdown(f"- {item}")
            if report.get("next_revision_recommendations"):
                st.markdown("**다음 리비전 제안**")
                for index,item in enumerate(report["next_revision_recommendations"],1):st.markdown(f"{index}. {item}")
        return

    st.markdown("### 리비전 후 To-do 재정리")
    st.caption(
        f"원고 v{from_version} → v{to_version} 변경으로 기존 To-do가 해결·수정·불필요·통합·분리되었는지 평가하고, "
        "새로 생긴 작업과 다음 리비전 후보를 구성합니다. 논문의 완성 여부를 판정하지 않습니다."
    )
    prompt=short_paper_todo_reconciliation_prompt(project,previous_manuscript,manuscript,before_todos,revision_diff)
    proposal_event=next((event for event in reversed(events)
        if event.get("event_type")=="revision_todo_reconciliation_proposed"
        and int((event.get("payload") or {}).get("to_version") or -1)==to_version),None)
    proposal=(proposal_event.get("payload") or {}).get("proposal") if proposal_event else None
    call_col,external_col=st.columns(2)
    if call_col.button("내부 LLM으로 To-do 변화 평가",type="primary",key=f"todo-reconcile-internal-{project_id}-{to_version}"):
        raw=llm_draft(prompt,model,use_ollama,profile="review") or ""
        try:
            parsed=parse_todo_reconciliation(raw,existing_todos=before_todos,manuscript=manuscript)
            ledger.add_paper_project_event(project_id,"revision_todo_reconciliation_proposed",{
                "from_version":from_version,"to_version":to_version,"source":"internal_llm","proposal":parsed,
            });st.rerun()
        except ValueError as error:st.error(str(error))
    with external_col.expander("외부 LLM으로 To-do 변화 평가",expanded=False):
        st.text_area("To-do 재평가 프롬프트",value=prompt,height=430,key=f"todo-reconcile-prompt-{project_id}-{to_version}")
        external=st.text_area("외부 LLM 재평가 JSON",height=250,key=f"todo-reconcile-response-{project_id}-{to_version}")
        if st.button("외부 재평가 검증·저장",disabled=not external.strip(),key=f"todo-reconcile-save-{project_id}-{to_version}"):
            try:
                parsed=parse_todo_reconciliation(external,existing_todos=before_todos,manuscript=manuscript)
                ledger.add_paper_project_event(project_id,"revision_todo_reconciliation_proposed",{
                    "from_version":from_version,"to_version":to_version,"source":"external_llm","proposal":parsed,
                    "response_digest":hashlib.sha256(external.strip().encode("utf-8")).hexdigest(),
                });st.rerun()
            except ValueError as error:st.error(str(error))
    if not proposal:
        st.info("To-do 변화 평가를 실행하면 연구자가 항목별 상태를 수정하고 다음 백로그에 반영할 수 있습니다.")
        return
    if proposal.get("summary"):st.info(proposal["summary"])
    if proposal.get("revision_achievements"):
        st.markdown("**이번 리비전에서 해소한 내용 제안**")
        for item in proposal["revision_achievements"]:st.markdown(f"- {item}")

    labels={"resolved":"해결","retained":"유지","modified":"수정","obsolete":"불필요","merged":"통합","split":"분리","researcher_review":"연구자 판단"}
    status_options=list(labels)
    reviewed_assessments=[]
    st.markdown("#### 기존 To-do 평가")
    for item in proposal.get("existing_todo_assessments") or []:
        todo_id=str(item.get("todo_id") or "");default_status=str(item.get("status") or "retained")
        with st.container(border=True):
            include=st.checkbox(f"`{todo_id}` 평가 반영",value=True,key=f"reconcile-include-{project_id}-{to_version}-{todo_id}")
            status=st.selectbox("상태",status_options,index=status_options.index(default_status) if default_status in status_options else 1,
                format_func=lambda value:labels[value],key=f"reconcile-status-{project_id}-{to_version}-{todo_id}")
            reason=st.text_area("판단 사유",value=str(item.get("reason") or ""),height=75,key=f"reconcile-reason-{project_id}-{to_version}-{todo_id}")
            problem=st.text_area("다음 작업 내용",value=str(item.get("updated_problem") or ""),height=70,disabled=status not in {"retained","modified","researcher_review"},key=f"reconcile-problem-{project_id}-{to_version}-{todo_id}")
            criterion=st.text_area("완료 기준",value=str(item.get("updated_completion_criteria") or ""),height=70,disabled=status not in {"retained","modified","researcher_review"},key=f"reconcile-criterion-{project_id}-{to_version}-{todo_id}")
            if item.get("evidence"):st.caption("판단 근거 · " + " · ".join(item["evidence"]))
            if include:reviewed_assessments.append({**item,"status":status,"reason":reason.strip(),"updated_problem":problem.strip(),"updated_completion_criteria":criterion.strip()})

    selected_new=[]
    if proposal.get("new_todos"):
        st.markdown("#### 신규·분리 To-do")
        for item in proposal["new_todos"]:
            todo_id=str(item.get("todo_id") or "")
            if st.checkbox(f"[{item.get('priority','P1')}] {item.get('label','신규')} · {item.get('problem','')}",value=True,key=f"reconcile-new-{project_id}-{to_version}-{todo_id}"):
                selected_new.append(item)
            st.caption(f"문장 {item.get('sentence_id','')} · 완료 기준: {item.get('completion_criteria','')}")
    if proposal.get("next_revision_recommendations"):
        st.markdown("#### 다음 Revision 제안")
        for index,item in enumerate(proposal["next_revision_recommendations"],1):st.markdown(f"{index}. {item}")
    if st.button("검토 결과 반영 · 다음 Revision 백로그 구성",type="primary",disabled=not reviewed_assessments,key=f"reconcile-apply-{project_id}-{to_version}"):
        before_by_id={str(item["todo_id"]):item for item in before_todos}
        counts={"resolved":0,"active":0,"obsolete":0,"restructured":0,"new":len(selected_new)}
        for item in reviewed_assessments:
            prior=before_by_id.get(str(item.get("todo_id") or ""),{})
            status=str(item.get("status") or "retained")
            stored_status={"retained":"open","modified":"modified","researcher_review":"researcher_review"}.get(status,status)
            ledger.add_paper_project_event(project_id,"revision_todo_update",{
                **prior,"todo_id":item.get("todo_id"),"sentence_id":item.get("sentence_id") or prior.get("sentence_id",""),
                "status":stored_status,"problem":item.get("updated_problem") or prior.get("problem",""),
                "completion_criteria":item.get("updated_completion_criteria") or prior.get("completion_criteria",""),
                "recommended_action":item.get("updated_recommended_action") or prior.get("recommended_action","researcher_input"),
                "reconciliation_reason":item.get("reason",""),"merged_into":item.get("merged_into",""),
                "reconciled_from_version":from_version,"reconciled_to_version":to_version,
            })
            if status=="resolved":counts["resolved"]+=1
            elif status=="obsolete":counts["obsolete"]+=1
            elif status in {"merged","split"}:counts["restructured"]+=1
            else:counts["active"]+=1
        for item in selected_new:
            ledger.add_paper_project_event(project_id,"revision_todo_update",{
                **item,"status":"open","created_by":"todo_reconciliation","created_at_version":to_version,
            })
        report={
            "summary":proposal.get("summary",""),"counts":counts,
            "revision_achievements":proposal.get("revision_achievements") or [],
            "next_revision_recommendations":proposal.get("next_revision_recommendations") or [],
            "assessments":reviewed_assessments,"new_todos":selected_new,
        }
        ledger.add_paper_project_event(project_id,"revision_todo_reconciliation_applied",{
            "from_version":from_version,"to_version":to_version,"source":(proposal_event.get("payload") or {}).get("source",""),"report":report,
        });st.rerun()


def _request_paper_todo_literature_intent(
    project_id: str, project: dict[str, Any], todo: dict[str, Any], candidate: dict[str, Any],
) -> tuple[str, str]:
    intent_id=f"intent-{uuid.uuid4().hex[:12]}"
    priority={"P0":"높음","P1":"보통","P2":"낮음"}.get(str(todo.get("priority","P1")),"보통")
    context="\n".join(filter(None,[
        f"Paper project: {project.get('title','')}",
        f"Research question: {project.get('research_question','')}",
        f"Revision sentence ({todo.get('sentence_id','')}): {todo.get('sentence_text','')}",
        f"Revision issue: {todo.get('problem','')}",
        str(candidate.get("research_context", "")),
    ]))
    intent=CurationIntent(
        intent_id=intent_id,
        title=str(candidate.get("title") or f"논문 보완 탐색 · {todo.get('label','')}")[:120],
        purpose=f"Revision To-do {todo.get('todo_id','')}의 완료 기준을 충족하기 위한 출처 기반 근거를 수집한다.",
        question=str(candidate.get("target") or todo.get("search_guide") or todo.get("sentence_text")),
        research_context=context,
        labels=[],priority=priority,
        expected_evidence=str(candidate.get("expected_evidence") or todo.get("search_guide") or "관련 선행연구의 방법, 결과, 조건, 한계 및 반대 근거"),
        completion_condition=str(candidate.get("completion_condition") or todo.get("completion_criteria") or "출처가 확인된 탐색 결과를 Revision To-do에 보고한다."),
        execution_mode="manual",created_by="m2",
        origin_links=[{
            "origin_type":"paper_writing","origin_id":project_id,
            "origin_sub_id":str(todo.get("todo_id", "")),
            "label":str(project.get("title") or "논문 프로젝트"),"source_card_ids":[],
        }],
    )
    case_id=ledger.create_case("research",f"논문 Revision 문헌탐색 · {todo.get('label','')}")
    request_id=ledger.record(
        case_id,"decision_request","m2",["researcher"],"curation_intent",
        {"title":f"M1 탐색 Intent 승인: {intent.title}","intent":intent.model_dump(mode="json"),
         "next_action":"승인 시 M1 문헌탐색 작업 큐에 등록",
         "paper_project_id":project_id,"revision_todo_id":todo.get("todo_id","")},
        subject_id=intent_id,
    )
    ledger.add_paper_project_event(project_id,"revision_literature_intent",{
        "todo_id":todo.get("todo_id", ""),"candidate_id":candidate.get("candidate_id", ""),
        "intent_id":intent_id,"request_id":request_id,"status":"approval_requested",
        "title":intent.title,"queue_title":intent.title,"target":intent.question,
        "next_action":"연구자 홈의 검토·승인함에서 승인하면 동일 제목으로 M1 연구 Intent 탐색 작업 큐에 등록",
    })
    return intent_id,request_id


def _queue_paper_todo_literature_intent(
    project_id: str, project: dict[str,Any], todo: dict[str,Any], candidate: dict[str,Any],
) -> dict[str,Any]:
    """Researcher-confirmed M2 action: create, approve, and place an Intent in the M1 queue."""
    intent_id,request_id=_request_paper_todo_literature_intent(project_id,project,todo,candidate)
    request=ledger.phenomenon(request_id) or {}
    if request.get("status")=="proposed":
        ledger.transition(request_id,"proposed","approved")
        ledger.add_decision(request_id,"approved","M2 Revision To-do 화면에서 연구자가 확인 후 M1 큐 등록")
    intent=dict((request.get("payload") or {}).get("intent") or {})
    profile=ledger.create_search_profile(intent)
    ledger.add_paper_project_event(project_id,"revision_literature_queued",{
        "todo_id":todo.get("todo_id",""),"candidate_id":candidate.get("candidate_id",""),
        "intent_id":intent_id,"request_id":request_id,"profile_id":profile.get("profile_id",""),
        "queue_title":profile.get("title") or intent.get("title") or "M1 탐색 Intent","status":"queued",
    })
    return profile


def _approve_existing_revision_intent(event: dict[str,Any]) -> dict[str,Any] | None:
    payload=event.get("payload") or {};request_id=str(payload.get("request_id") or "")
    request=ledger.phenomenon(request_id) or {}
    if request.get("status") in {"deferred","rejected"}:return None
    if request.get("status")=="proposed":
        ledger.transition(request_id,"proposed","approved")
        ledger.add_decision(request_id,"approved","M2 Revision To-do 화면에서 연구자가 확인 후 M1 큐 등록")
    intent=dict((request.get("payload") or {}).get("intent") or {})
    if not intent:return None
    return ledger.create_search_profile(intent)


@st.cache_data(show_spinner=False, ttl=3600)
def _revision_paper_source(
    paper_id: str, title: str, pdf_path: str, source_url: str,
) -> dict[str,Any]:
    """Load a selected shelf paper's extracted source text for a Revision To-do prompt."""
    paper={"paper_id":paper_id,"title":title,"pdf_path":pdf_path,"source_url":source_url}
    try:
        path=Path(pdf_path) if pdf_path and Path(pdf_path).exists() else None
        if path:
            upload=document_from_shelf_path(str(path));source_kind="등록 PDF"
        elif source_url:
            upload=document_from_source_url(paper);source_kind="원문 URL"
        else:
            return {"source_text":"","source_status":"원문 없음 · PDF 또는 원문 URL을 서재함에 등록해야 합니다.","source_chars":0}
        document=extract_document(upload,max_pages=60,chars_per_page=8000,cache_dir=EXTRACTION_CACHE)
        source_text=extracted_document_text(document).strip()
        truncated=document.original_page_count>len(document.pages) or any(page.truncated for page in document.pages)
        status=(
            f"{source_kind} · {len(document.pages)}/{document.original_page_count}개 구간 추출"
            + (" · 추출 한도에 의해 일부 생략" if truncated else " · 전체 추출")
        )
        return {"source_text":source_text,"source_status":status,"source_chars":len(source_text)}
    except Exception as error:
        return {"source_text":"","source_status":f"원문 추출 실패 · {error}","source_chars":0}


def _render_revision_todo_workbench(
    *, project_id: str, project: dict[str,Any], manuscript: dict[str,Any], todo: dict[str,Any],
    events: list[dict[str,Any]], cards: list[dict[str,Any]], model: str, use_ollama: bool,
) -> None:
    """Plan and collect literature, author input, and empirical artifacts for one To-do."""
    todo_id=str(todo.get("todo_id") or "");version=int(manuscript.get("version") or 0)
    all_papers=ledger.shelf_papers();lineage_cards,lineage_papers=_revision_todo_assets(project_id,todo_id,cards,all_papers)
    link=_latest_revision_evidence_link(events,todo_id)
    card_by_id={str(item.get("card_id") or ""):item for item in cards}
    paper_by_id={str(item.get("paper_id") or ""):item for item in all_papers}
    linked_cards=list(lineage_cards)
    for card_id in link.get("card_ids") or []:
        if str(card_id) in card_by_id and card_by_id[str(card_id)] not in linked_cards:linked_cards.append(card_by_id[str(card_id)])
    linked_papers=list(lineage_papers)
    for paper_id in link.get("paper_ids") or []:
        if str(paper_id) in paper_by_id and paper_by_id[str(paper_id)] not in linked_papers:linked_papers.append(paper_by_id[str(paper_id)])
    artifacts=[event.get("payload") or {} for event in events
        if event.get("event_type")=="revision_research_artifact"
        and str((event.get("payload") or {}).get("todo_id") or "")==todo_id]
    confirmed_plan_event=next((event for event in reversed(events)
        if event.get("event_type")=="revision_todo_plan_confirmed"
        and str((event.get("payload") or {}).get("todo_id") or "")==todo_id
        and int((event.get("payload") or {}).get("manuscript_version") or -1)==version),None)
    proposed_plan_event=next((event for event in reversed(events)
        if event.get("event_type")=="revision_todo_plan_proposed"
        and str((event.get("payload") or {}).get("todo_id") or "")==todo_id
        and int((event.get("payload") or {}).get("manuscript_version") or -1)==version),None)
    plan=(confirmed_plan_event.get("payload") or {}).get("plan") if confirmed_plan_event else None
    if not plan and proposed_plan_event:plan=(proposed_plan_event.get("payload") or {}).get("plan")
    resolution_exists=any(event.get("event_type") in {"revision_resolution_proposal","revision_resolution_applied"}
        and str((event.get("payload") or {}).get("todo_id") or "")==todo_id for event in events)
    terminal=str((todo.get("latest_update") or {}).get("status") or todo.get("status") or "") in {"resolved","obsolete","merged","split"}
    plan_done=bool(confirmed_plan_event);evidence_done=bool(linked_cards or linked_papers or artifacts)
    steps=[("문제·완료 기준",True),("해결 계획",plan_done),("근거·연구 결과",evidence_done),("원고 수정",resolution_exists),("해소 평가",terminal)]
    completed=sum(1 for _,done in steps if done)
    st.progress(completed/len(steps),text=f"To-do 작업 진행 · {completed}/{len(steps)}")
    st.caption(" · ".join(("✓ " if done else "○ ")+label for label,done in steps))

    overview_tab,plan_tab,evidence_tab,next_tab=st.tabs(["문제·완료 기준","해결 계획","근거·연구 결과","원고 수정·해소 평가"])
    with overview_tab:
        st.markdown(f"**{todo.get('priority','P1')} · {todo.get('label','')}** · `{todo.get('sentence_id','')}`")
        st.markdown(f"> {todo.get('sentence_text','')}")
        st.markdown(f"**문제**  \n{todo.get('problem','')}")
        st.markdown(f"**완료 기준**  \n{todo.get('completion_criteria','')}")
        if todo.get("question_for_researcher"):st.markdown(f"**연구자 질문**  \n{todo.get('question_for_researcher','')}")
    with plan_tab:
        prompt=short_paper_resolution_plan_prompt(project,manuscript,todo,linked_cards,linked_papers)
        if st.button("내부 LLM으로 해결 계획 제안",type="primary",key=f"todo-plan-internal-{project_id}-{todo_id}-{version}"):
            raw=llm_draft(prompt,model,use_ollama,profile="review") or ""
            try:
                parsed=parse_revision_resolution_plan(raw,todo)
                ledger.add_paper_project_event(project_id,"revision_todo_plan_proposed",{"todo_id":todo_id,"manuscript_version":version,"source":"internal_llm","plan":parsed});st.rerun()
            except ValueError as error:st.error(str(error))
        with st.expander("외부 LLM으로 해결 계획 제안",expanded=False):
            st.text_area("해결 계획 프롬프트",value=prompt,height=420,key=f"todo-plan-prompt-{project_id}-{todo_id}-{version}")
            response=st.text_area("외부 LLM 계획 JSON",height=230,key=f"todo-plan-response-{project_id}-{todo_id}-{version}")
            if st.button("외부 계획 검증·저장",disabled=not response.strip(),key=f"todo-plan-external-{project_id}-{todo_id}-{version}"):
                try:
                    parsed=parse_revision_resolution_plan(response,todo)
                    ledger.add_paper_project_event(project_id,"revision_todo_plan_proposed",{"todo_id":todo_id,"manuscript_version":version,"source":"external_llm","plan":parsed});st.rerun()
                except ValueError as error:st.error(str(error))
        if plan:
            st.markdown(f"**해결 전략**  \n{plan.get('strategy_summary','')}")
            st.caption(f"원고 반영 대상 · {plan.get('revision_target','')}")
            st.dataframe([{
                "유형":item.get("type",""),"작업":item.get("title",""),"목적":item.get("purpose",""),
                "산출물":" · ".join(item.get("expected_artifacts") or []),"완료 기준":item.get("completion_condition",""),
            } for item in plan.get("actions") or []],use_container_width=True,hide_index=True)
            if plan.get("risks"):st.warning("위험·확인사항 · " + " · ".join(plan["risks"]))
            editable_plan=st.text_area("계획 JSON 수정",value=json.dumps(plan,ensure_ascii=False,indent=2),height=320,key=f"todo-plan-edit-{project_id}-{todo_id}-{version}")
            if st.button("연구자 계획 확정",type="primary",key=f"todo-plan-confirm-{project_id}-{todo_id}-{version}"):
                try:
                    confirmed=parse_revision_resolution_plan(editable_plan,todo)
                    ledger.add_paper_project_event(project_id,"revision_todo_plan_confirmed",{"todo_id":todo_id,"manuscript_version":version,"source":"researcher","plan":confirmed});st.rerun()
                except ValueError as error:st.error(str(error))
    with evidence_tab:
        c1,c2,c3=st.columns(3);c1.metric("연결 논문",len(linked_papers));c2.metric("지식카드",len(linked_cards));c3.metric("연구 산출물",len(artifacts))
        if linked_papers:
            with st.expander("M1·서재함 연결 논문",expanded=False):
                for item in linked_papers:st.markdown(f"- **{item.get('title','')}** · `{item.get('paper_id','')}`")
        if linked_cards:
            with st.expander("연결 지식카드",expanded=False):
                for item in linked_cards:st.markdown(f"- **{item.get('title','')}** — {item.get('claim','')}")
        if artifacts:
            st.markdown("**등록된 연구자 산출물**")
            for item in artifacts:
                with st.expander(f"{item.get('artifact_type','')} · {item.get('title','')}",expanded=False):
                    st.caption(f"상태 · {item.get('status','')} · {item.get('created_at','')}")
                    st.markdown(f"**방법**  \n{item.get('method','')}")
                    st.markdown(f"**관찰 결과**  \n{item.get('observed_result','')}")
                    if item.get("interpretation"):st.markdown(f"**해석**  \n{item.get('interpretation','')}")
                    if item.get("limitations"):st.warning("한계 · "+str(item.get("limitations")))
        with st.expander("실험·Survey·사례·Trace 결과 등록",expanded=not bool(artifacts)):
            with st.form(f"research-artifact-{project_id}-{todo_id}"):
                artifact_type=st.selectbox("산출물 유형",["EXPERIMENT","SURVEY","CASE_ANALYSIS","DATA_ANALYSIS","TRACE_REVIEW","RESEARCHER_DECISION"])
                artifact_title=st.text_input("결과 제목")
                method=st.text_area("수행 방법",height=90)
                observed=st.text_area("실제 관찰 결과",height=120,help="계획이나 예상 결과가 아니라 실제로 수행·관찰한 내용을 입력합니다.")
                interpretation=st.text_area("연구자 해석",height=90)
                limitations=st.text_area("한계·유효성 위협",height=80)
                source_ref=st.text_input("파일명 또는 URL",placeholder="예: traces/run-2026-09.csv 또는 https://...")
                confirmed=st.checkbox("실제로 수행한 결과이며 연구자가 내용을 확인했습니다.")
                save_artifact=st.form_submit_button("연구 산출물 등록")
            if save_artifact:
                if not artifact_title.strip() or not method.strip() or not observed.strip():st.error("제목, 수행 방법, 실제 관찰 결과를 입력하세요.")
                else:
                    ledger.add_paper_project_event(project_id,"revision_research_artifact",{
                        "artifact_id":f"artifact-{uuid.uuid4().hex[:10]}","todo_id":todo_id,"artifact_type":artifact_type,
                        "title":artifact_title.strip(),"method":method.strip(),"observed_result":observed.strip(),
                        "interpretation":interpretation.strip(),"limitations":limitations.strip(),"source_ref":source_ref.strip(),
                        "status":"researcher_confirmed" if confirmed else "researcher_draft","manuscript_version":version,
                    });st.rerun()
        st.info("문헌·카드 선택과 M1 탐색은 아래 기존 해결 흐름에서 계속 수행할 수 있습니다. 등록한 연구 산출물은 문장 해결 프롬프트에 함께 전달됩니다.")
    with next_tab:
        st.info("아래 ‘처리 방식’에서 단일 또는 그룹 해결을 선택하면, 확정 계획과 수집 근거를 바탕으로 원고 수정 및 해소 평가를 계속합니다.")
        if plan:
            for item in plan.get("actions") or []:
                kind=str(item.get("type") or "")
                done=(kind in {"M1_LITERATURE_SEARCH","KNOWLEDGE_CARD_REVIEW"} and bool(linked_cards or linked_papers)) or (kind in {"EXPERIMENT","SURVEY","CASE_ANALYSIS","DATA_ANALYSIS","TRACE_REVIEW","RESEARCHER_DECISION"} and bool(artifacts)) or (kind=="WRITING_ONLY" and resolution_exists)
                st.markdown(f"- [{'x' if done else ' '}] **{item.get('title','')}** — {item.get('completion_condition','')}")


def _render_revision_group_workflow(
    *, project_id: str, project: dict[str,Any], manuscript: dict[str,Any],
    todos: list[dict[str,Any]], events: list[dict[str,Any]], cards: list[dict[str,Any]],
    valid_card_ids: set[str], version_events: list[dict[str,Any]], model: str, use_ollama: bool,
) -> None:
    """Plan, confirm, resolve, and selectively apply a bounded group of related To-dos."""
    manuscript_version=int(manuscript.get("version") or 0);todo_by_id={str(item["todo_id"]):item for item in todos}
    st.markdown("### A. LLM 유사 To-do 그룹 제안")
    st.caption("이 단계에는 논문 원문을 보내지 않습니다. 현재 숏페이퍼와 활성 To-do만으로 2~4개 묶음을 제안합니다.")
    grouping_prompt=short_paper_todo_grouping_prompt(project,manuscript,todos)
    current_plan_event=next((event for event in reversed(events) if event.get("event_type")=="revision_todo_group_plan" and int((event.get("payload") or {}).get("from_version") or -1)==manuscript_version),None)
    plan=(current_plan_event or {}).get("payload",{}).get("plan") if current_plan_event else None
    plan_col,manual_col=st.columns(2)
    if plan_col.button("내부 LLM으로 그룹 제안",type="primary",key=f"m2-group-plan-{project_id}"):
        raw=llm_draft(grouping_prompt,model,use_ollama,profile="writing") or ""
        try:
            parsed=parse_todo_group_plan(raw,todos)
            ledger.add_paper_project_event(project_id,"revision_todo_group_plan",{"from_version":manuscript_version,"source":"internal_llm","plan":parsed})
            st.rerun()
        except ValueError as error:st.error(str(error))
    with manual_col.expander("외부 LLM 그룹 제안"):
        group_prompt_version=hashlib.sha256(grouping_prompt.encode("utf-8")).hexdigest()[:12]
        st.text_area("그룹 제안 프롬프트",value=grouping_prompt,height=360,key=f"m2-group-plan-prompt-{project_id}-{group_prompt_version}")
        external_plan=st.text_area("외부 LLM 그룹 제안 JSON",height=220,key=f"m2-group-plan-response-{project_id}")
        if st.button("외부 그룹 제안 저장",disabled=not external_plan.strip(),key=f"m2-group-plan-save-{project_id}"):
            try:
                parsed=parse_todo_group_plan(external_plan,todos)
                ledger.add_paper_project_event(project_id,"revision_todo_group_plan",{"from_version":manuscript_version,"source":"external_llm","plan":parsed})
                st.rerun()
            except ValueError as error:st.error(str(error))
    if not plan:
        st.info("그룹 제안을 생성하면 연구자가 구성원을 조정하고 확정할 수 있습니다.");return
    groups=list(plan.get("groups") or [])
    if not groups:
        st.info("함께 처리할 만큼 유사한 To-do가 없습니다. 단일 To-do 처리 모드를 사용하세요.");return
    for item in groups:
        labels=[todo_by_id[todo_id].get("label",todo_id) for todo_id in item.get("todo_ids",[]) if todo_id in todo_by_id]
        st.markdown(f"- **{item.get('title','')}** · {', '.join(labels)} · {item.get('reason','')}")

    st.markdown("### B. 연구자 그룹 검토·확정")
    proposed_group_id=st.selectbox("검토할 제안 그룹",[str(item["group_id"]) for item in groups],key=f"m2-group-select-{project_id}",format_func=lambda group_id:next(str(item.get("title") or group_id) for item in groups if str(item.get("group_id"))==group_id))
    proposed=next(item for item in groups if str(item.get("group_id"))==proposed_group_id)
    confirmed_event=next((event for event in reversed(events) if event.get("event_type")=="revision_todo_group_confirmed" and int((event.get("payload") or {}).get("from_version") or -1)==manuscript_version and str((event.get("payload") or {}).get("source_group_id") or "")==proposed_group_id),None)
    confirmed=(confirmed_event or {}).get("payload") or {}
    group_title=st.text_input("그룹 제목",value=str(confirmed.get("title") or proposed.get("title") or ""),key=f"m2-group-title-{project_id}-{proposed_group_id}")
    default_members=[todo_id for todo_id in (confirmed.get("todo_ids") or proposed.get("todo_ids") or []) if todo_id in todo_by_id]
    group_todo_ids=st.multiselect("함께 처리할 To-do · 2~4개",list(todo_by_id),default=default_members,max_selections=4,key=f"m2-group-members-{project_id}-{proposed_group_id}",format_func=lambda todo_id:f"[{todo_by_id[todo_id]['priority']}] {todo_by_id[todo_id]['label']} · {todo_by_id[todo_id]['sentence_text'][:65]}")
    group_reason=st.text_area("함께 처리하는 이유",value=str(confirmed.get("reason") or proposed.get("reason") or ""),height=90,key=f"m2-group-reason-{project_id}-{proposed_group_id}")
    shared_need=st.text_area("공통 근거·추론 필요",value=str(confirmed.get("shared_evidence_need") or proposed.get("shared_evidence_need") or ""),height=90,key=f"m2-group-need-{project_id}-{proposed_group_id}")
    group_valid=2<=len(group_todo_ids)<=4 and bool(group_title.strip() and group_reason.strip())
    if st.button("이 구성으로 그룹 확정",disabled=not group_valid,type="primary",key=f"m2-group-confirm-{project_id}-{proposed_group_id}"):
        payload={"from_version":manuscript_version,"source_group_id":proposed_group_id,"group_id":str(confirmed.get("group_id") or f"{proposed_group_id}-confirmed"),"title":group_title.strip(),"reason":group_reason.strip(),"shared_evidence_need":shared_need.strip(),"todo_ids":group_todo_ids,"shared_search":proposed.get("shared_search") or {}}
        ledger.add_paper_project_event(project_id,"revision_todo_group_confirmed",payload);st.rerun()
    if not confirmed:
        st.info("연구자가 그룹을 확정해야 공통 자료 선택과 그룹 리비전을 시작할 수 있습니다.");return
    confirmed_ids=[todo_id for todo_id in confirmed.get("todo_ids",[]) if todo_id in todo_by_id]
    if len(confirmed_ids)<2:
        st.warning("현재 활성 To-do가 줄어 그룹 구성이 유효하지 않습니다. 그룹을 다시 확정하세요.");return
    group={key:confirmed.get(key) for key in ("group_id","title","reason","shared_evidence_need","shared_search")};group_todos=[todo_by_id[todo_id] for todo_id in confirmed_ids]
    st.success(f"확정 그룹 · **{group.get('title','')}** · To-do {len(group_todos)}건")

    st.markdown("### C. 공통 M1 탐색 · 자료 선택")
    shared_search=dict(group.get("shared_search") or {})
    group_queue_events=[event for event in events if event.get("event_type") in {"revision_literature_intent","revision_literature_queued"} and str((event.get("payload") or {}).get("todo_id") or "")==str(group.get("group_id") or "")]
    latest_queue_event=group_queue_events[-1] if group_queue_events else None
    queue_payload=(latest_queue_event or {}).get("payload") or {};intent_id=str(queue_payload.get("intent_id") or "")
    queue_profile=next((item for item in ledger.search_profiles(include_deleted=True) if str(item.get("intent_id") or "")==intent_id),None)
    with st.expander("그룹 공통 M1 문헌탐색",expanded=not bool(queue_profile)):
        if queue_profile:st.success(f"M1 탐색 큐 등록 완료 · **{queue_profile.get('title','')}**")
        search_title=st.text_input("공통 탐색 제목",value=str(shared_search.get("title") or f"{group.get('title','')} 공통 근거 탐색"),key=f"m2-group-search-title-{project_id}-{group.get('group_id')}")
        search_target=st.text_area("공통 탐색 대상",value=str(shared_search.get("target") or shared_need),height=80,key=f"m2-group-search-target-{project_id}-{group.get('group_id')}")
        search_context=st.text_area("공통 탐색 맥락",value=str(shared_search.get("research_context") or f"논문: {project.get('title','')}\n그룹 To-do: {', '.join(confirmed_ids)}\n{group_reason}"),height=100,key=f"m2-group-search-context-{project_id}-{group.get('group_id')}")
        search_evidence=st.text_area("기대 근거",value=str(shared_search.get("expected_evidence") or shared_need),height=75,key=f"m2-group-search-evidence-{project_id}-{group.get('group_id')}")
        search_completion=st.text_area("완료 조건",value=str(shared_search.get("completion_condition") or "선택한 각 To-do의 완료 기준을 판단할 공통 근거가 확보됨"),height=75,key=f"m2-group-search-completion-{project_id}-{group.get('group_id')}")
        candidate={"candidate_id":f"{group.get('group_id')}-shared","title":search_title.strip(),"target":search_target.strip(),"research_context":search_context.strip(),"expected_evidence":search_evidence.strip(),"completion_condition":search_completion.strip()}
        candidate_valid=all(len(str(candidate.get(field) or ""))>=3 for field in ("title","target","research_context","expected_evidence","completion_condition"))
        if st.button("그룹 공통 탐색을 M1 큐에 등록",disabled=bool(queue_profile) or not candidate_valid,key=f"m2-group-search-queue-{project_id}-{group.get('group_id')}"):
            synthetic_todo={"todo_id":group.get("group_id"),"label":group.get("title"),"sentence_text":" / ".join(todo.get("sentence_text","") for todo in group_todos),"problem":group.get("reason"),"completion_criteria":candidate["completion_condition"]}
            profile=_queue_paper_todo_literature_intent(project_id,project,synthetic_todo,candidate)
            ledger.add_paper_project_event(project_id,"revision_todo_group_search_queued",{"from_version":manuscript_version,"group_id":group.get("group_id"),"todo_ids":confirmed_ids,"profile_id":profile.get("profile_id",""),"queue_title":profile.get("title","")});st.rerun()

    cutoff=str((latest_queue_event or {}).get("created_at") or "");all_papers=ledger.shelf_papers()
    recent_papers=([paper for paper in all_papers if str(paper.get("created_at") or paper.get("updated_at") or "")>=cutoff][:30] if cutoff else all_papers[:12])
    ordered_cards=sorted(cards,key=lambda item:str(item.get("approved_at") or item.get("updated_at") or item.get("created_at") or ""),reverse=True)
    recent_cards=([card for card in ordered_cards if str(card.get("approved_at") or card.get("updated_at") or card.get("created_at") or "")>=cutoff][:40] if cutoff else ordered_cards[:12])
    latest_resolution_event=next((event for event in reversed(events) if event.get("event_type")=="revision_group_resolution_proposal" and str((event.get("payload") or {}).get("group_id") or "")==str(group.get("group_id") or "")),None)
    prior=((latest_resolution_event or {}).get("payload") or {}) if latest_resolution_event and int(((latest_resolution_event or {}).get("payload") or {}).get("from_version") or -1)==manuscript_version else {}
    asset_query=st.text_input("그룹 공통 논문·지식카드 검색",key=f"m2-group-asset-search-{project_id}-{group.get('group_id')}",placeholder="논문 제목, 저자, 주장, 개념, 조건")
    search_papers,search_cards=search_revision_assets(asset_query,all_papers,cards,limit=30)
    if asset_query.strip():
        result_col_papers,result_col_cards=st.columns(2)
        with result_col_papers:
            if search_papers:
                with st.expander(f"서재함 논문 검색 결과 · {len(search_papers)}편",expanded=True):
                    for paper in search_papers:
                        st.markdown(f"- **{paper.get('title','')}** · {paper.get('publication_year') or '연도 확인 필요'} · `{paper.get('paper_id','')}`")
            else:
                st.info("일치하는 서재함 논문이 없습니다.")
        with result_col_cards:
            if search_cards:
                with st.expander(f"승인 지식카드 검색 결과 · {len(search_cards)}건",expanded=True):
                    for card in search_cards:
                        st.markdown(f"- **{card.get('title','')}**  \n{card.get('claim','')}  \n`{card.get('card_id','')}`")
            else:
                st.info("일치하는 승인 지식카드가 없습니다.")
        st.caption("검색 결과는 아래 선택 목록에 추가됩니다. 실제 그룹 해결에 사용할 자료를 선택하세요.")
    paper_options={str(item["paper_id"]):item for item in recent_papers};card_options={str(item["card_id"]):item for item in recent_cards}
    for item in search_papers:paper_options.setdefault(str(item["paper_id"]),item)
    for item in search_cards:card_options.setdefault(str(item["card_id"]),item)
    all_paper_by_id={str(item.get("paper_id") or ""):item for item in all_papers};all_card_by_id={str(item.get("card_id") or ""):item for item in cards}
    for paper_id in prior.get("selected_paper_ids",[]):
        if paper_id in all_paper_by_id:paper_options.setdefault(paper_id,all_paper_by_id[paper_id])
    for card_id in prior.get("selected_card_ids",[]):
        if card_id in all_card_by_id:card_options.setdefault(card_id,all_card_by_id[card_id])
    recent_paper_ids={str(item.get("paper_id") or "") for item in recent_papers};recent_card_ids={str(item.get("card_id") or "") for item in recent_cards}
    search_paper_ids={str(item.get("paper_id") or "") for item in search_papers};search_card_ids={str(item.get("card_id") or "") for item in search_cards}
    def _group_asset_source(value: str, *, paper: bool) -> str:
        if value in (search_paper_ids if paper else search_card_ids):return "검색 결과"
        if value in (recent_paper_ids if paper else recent_card_ids):return "최신"
        return "이전 선택"
    selected_paper_ids=st.multiselect("그룹 전체에 사용할 논문",list(paper_options),default=[value for value in prior.get("selected_paper_ids",[]) if value in paper_options],format_func=lambda value:f"[{_group_asset_source(value,paper=True)}] {paper_options[value].get('title',value)}",key=f"m2-group-papers-{project_id}-{group.get('group_id')}")
    selected_card_ids=st.multiselect("그룹 전체에 사용할 지식카드",list(card_options),default=[value for value in prior.get("selected_card_ids",[]) if value in card_options],format_func=lambda value:f"[{_group_asset_source(value,paper=False)}] {card_options[value].get('title',value)}",key=f"m2-group-cards-{project_id}-{group.get('group_id')}")
    connection_note=st.text_area("선택 자료가 그룹 To-do들을 해결하는 방식",value=str(prior.get("connection_note") or ""),height=110,key=f"m2-group-note-{project_id}-{group.get('group_id')}")
    selected_papers=[]
    for paper_id in selected_paper_ids:
        paper=paper_options[paper_id];selected_papers.append({**paper,**_revision_paper_source(paper_id,str(paper.get("title") or ""),str(paper.get("pdf_path") or ""),str(paper.get("source_url") or ""))})
    selected_cards=[card_options[card_id] for card_id in selected_card_ids]
    if selected_papers:
        with st.expander("그룹 프롬프트 원문 포함 상태",expanded=True):
            for paper in selected_papers:
                st.markdown(f"- {'✅' if paper.get('source_text') else '⚠️'} **{paper.get('title','')}** · {paper.get('source_status','')} · {int(paper.get('source_chars') or 0):,}자")
    include_full=st.checkbox("전체 숏페이퍼도 그룹 일관성 확인용으로 포함",value=False,key=f"m2-group-full-{project_id}-{group.get('group_id')}")
    resolution_prompt=short_paper_group_resolution_prompt(project,manuscript,group,group_todos,selected_papers,selected_cards,connection_note,include_full_manuscript=include_full)
    usable_paper_ids={str(item.get("paper_id") or "") for item in selected_papers if item.get("source_text")};can_request=bool((selected_card_ids or usable_paper_ids) and connection_note.strip())
    request_col,external_col=st.columns(2)
    if request_col.button("내부 LLM에 그룹 해결 요청",type="primary",disabled=not can_request,key=f"m2-group-resolve-{project_id}-{group.get('group_id')}"):
        raw=llm_draft(resolution_prompt,model,use_ollama,profile="writing") or ""
        try:
            result=parse_group_resolution(raw,group,group_todos,valid_card_ids=set(selected_card_ids),valid_paper_ids=usable_paper_ids)
            ledger.add_paper_project_event(project_id,"revision_group_resolution_proposal",{"group_id":group.get("group_id"),"todo_ids":confirmed_ids,"from_version":manuscript_version,"source":"internal_llm","proposal":result,"selected_paper_ids":selected_paper_ids,"selected_card_ids":selected_card_ids,"connection_note":connection_note.strip()});st.rerun()
        except ValueError as error:st.error(str(error))
    with external_col.expander("외부 LLM 그룹 해결"):
        resolution_digest=hashlib.sha256(resolution_prompt.encode("utf-8")).hexdigest()[:12]
        st.caption(f"To-do {len(group_todos)}건 · 원문 {len(usable_paper_ids)}편 · {len(resolution_prompt):,}자")
        st.text_area("그룹 해결 프롬프트",value=resolution_prompt,height=480,key=f"m2-group-resolution-prompt-{project_id}-{group.get('group_id')}-{resolution_digest}")
        external_result=st.text_area("외부 LLM 그룹 해결 JSON",height=240,key=f"m2-group-resolution-response-{project_id}-{group.get('group_id')}")
        if not can_request:
            st.caption("새 해결 프롬프트를 실행하려면 자료 선택과 연결 설명이 필요합니다. 이미 외부 LLM 응답을 받은 경우에는 JSON을 붙여넣어 검증·저장할 수 있습니다.")
        if st.button("외부 그룹 해결 제안 저장",disabled=not external_result.strip(),key=f"m2-group-resolution-save-{project_id}-{group.get('group_id')}"):
            try:
                result=parse_group_resolution(external_result,group,group_todos,valid_card_ids=set(selected_card_ids),valid_paper_ids=usable_paper_ids)
                ledger.add_paper_project_event(project_id,"revision_group_resolution_proposal",{"group_id":group.get("group_id"),"todo_ids":confirmed_ids,"from_version":manuscript_version,"source":"external_llm","proposal":result,"selected_paper_ids":selected_paper_ids,"selected_card_ids":selected_card_ids,"connection_note":connection_note.strip()});st.rerun()
            except ValueError as error:st.error(str(error))

    st.markdown("### D. To-do별 판정 · 선택 적용")
    if not latest_resolution_event:
        st.info("그룹 해결 요청을 실행하면 각 To-do의 독립 판정과 변경 전후가 표시됩니다.");return
    result_payload=(latest_resolution_event.get("payload") or {});group_result=result_payload.get("proposal") or {}
    stale=int(result_payload.get("from_version") or -1)!=manuscript_version
    consistency=group_result.get("cross_todo_consistency") or {}
    if consistency.get("summary"):st.info(str(consistency.get("summary")))
    for conflict in consistency.get("conflicts") or []:st.warning(f"그룹 충돌 · {conflict}")
    resolved_ids=[]
    for item in group_result.get("todo_results") or []:
        todo_id=str(item.get("todo_id") or "");todo=todo_by_id.get(todo_id,{})
        with st.container(border=True):
            if item.get("verdict")=="resolved":resolved_ids.append(todo_id);st.success(f"해결 가능 · {todo.get('label',todo_id)}")
            else:st.warning(f"추가 보완 필요 · {todo.get('label',todo_id)}")
            st.write(item.get("reason") or "")
            if item.get("verdict")!="resolved" and (item.get("next_search") or {}).get("title"):
                st.caption(f"다음 M1 탐색: {(item.get('next_search') or {}).get('title')}")
    apply_ids=st.multiselect("이번 원고 버전에 반영할 해결된 To-do",resolved_ids,default=resolved_ids,key=f"m2-group-apply-ids-{project_id}-{group.get('group_id')}",format_func=lambda value:todo_by_id.get(value,{}).get("label",value))
    try:
        revisions,appendix_updates=selected_group_revisions(group_result,set(apply_ids))
        proposed_manuscript,diffs=apply_revisions(json.dumps({"revisions":revisions,"appendix_updates":appendix_updates},ensure_ascii=False),manuscript,valid_card_ids=valid_card_ids,version=len(version_events)+1)
        merge_error=""
    except ValueError as error:
        proposed_manuscript,diffs=manuscript,[];merge_error=str(error);st.error(merge_error)
    for diff in diffs:
        st.markdown(f"**{'Appendix' if diff.get('change_scope')=='appendix' else '문장 '+str(diff.get('sentence_id',''))}** · {diff.get('reason','')}")
        before_col,after_col=st.columns(2);before_col.caption("이전");before_col.markdown(f"> {diff.get('before','')}");after_col.caption("제안");after_col.markdown(f"> {diff.get('after','')}")
    suggested_refs=list(dict.fromkeys(paper_id for item in group_result.get("todo_results") or [] if item.get("todo_id") in apply_ids for paper_id in item.get("suggested_reference_paper_ids") or [] if paper_id in all_paper_by_id))
    reference_ids=st.multiselect("References에 포함할 선택 논문",[paper_id for paper_id in selected_paper_ids if paper_id in all_paper_by_id],default=suggested_refs,key=f"m2-group-reference-ids-{project_id}-{group.get('group_id')}",format_func=lambda value:all_paper_by_id[value].get("title",value))
    if stale:st.error("그룹 제안 이후 원고 버전이 변경되었습니다. 현재 원고에서 그룹 해결 요청을 다시 실행하세요.")
    if st.button("선택 To-do 변경 반영 · 완료 처리",type="primary",disabled=stale or bool(merge_error) or not apply_ids or not diffs,key=f"m2-group-apply-{project_id}-{group.get('group_id')}"):
        saved_reference_ids=[]
        for paper_id in reference_ids:
            paper=all_paper_by_id[paper_id];reference=ledger.upsert_literature_reference(topic=f"{project.get('title','논문 프로젝트')} · References",research_context=f"Revision 그룹 {group.get('title','')}: {', '.join(apply_ids)}",paper={**paper,"paper_project_id":project_id,"revision_todo_group_id":group.get("group_id")},labels=list(paper.get("labels",[])) or build_paper_labels(paper),status="selected");saved_reference_ids.append(str(reference.get("reference_id") or ""))
        ledger.add_paper_project_event(project_id,"manuscript_version",{"manuscript":proposed_manuscript,"source":"researcher_approved_group_resolution","diff":diffs,"group_id":group.get("group_id"),"todo_ids":apply_ids})
        ledger.add_paper_project_event(project_id,"revision_group_resolution_applied",{"group_id":group.get("group_id"),"todo_ids":apply_ids,"from_version":manuscript_version,"to_version":proposed_manuscript.get("version"),"paper_ids":selected_paper_ids,"card_ids":selected_card_ids,"reference_ids":[value for value in saved_reference_ids if value]})
        for todo_id in apply_ids:
            todo=todo_by_id[todo_id];ledger.add_paper_project_event(project_id,"revision_todo_update",{**{key:value for key,value in todo.items() if key!="latest_update"},"status":"resolved","resolution_source":"researcher_approved_group_resolution","group_id":group.get("group_id"),"selected_card_ids":selected_card_ids,"selected_paper_ids":selected_paper_ids})
        st.rerun()


def _render_revision_resolution_workflow(
    *, project_id: str, project: dict[str,Any], manuscript: dict[str,Any],
    todos: list[dict[str,Any]], events: list[dict[str,Any]], cards: list[dict[str,Any]],
    card_by_id: dict[str,dict[str,Any]], valid_card_ids: set[str], version_events: list[dict[str,Any]],
    model: str, use_ollama: bool,
) -> None:
    st.dataframe([{
        "우선순위":item["priority"],"상태":{"open":"대기","ready":"자료 선택","revised_pending_review":"해결안 검토","modified":"내용 수정","researcher_review":"연구자 판단","reopened":"재개"}.get(item["status"],item["status"]),
        "문장 ID":item["sentence_id"],"대상 문장":item["sentence_text"],
        "보완점":item["label"],"해야 할 일":item["problem"],"완료 기준":item["completion_criteria"],
    } for item in todos],use_container_width=True,hide_index=True,column_config={
        "대상 문장":st.column_config.TextColumn(width="large"),"해야 할 일":st.column_config.TextColumn(width="large"),
        "완료 기준":st.column_config.TextColumn(width="large"),
    })
    todo_ids=[item["todo_id"] for item in todos]
    workspace_todo_id=st.selectbox(
        "작업할 Revision To-do",todo_ids,key=f"m2-workbench-todo-{project_id}",
        format_func=lambda value:next(f"[{item['priority']}] {item['label']} · {item['sentence_text'][:75]}" for item in todos if item["todo_id"]==value),
    )
    workspace_todo=next(item for item in todos if item["todo_id"]==workspace_todo_id)
    _render_revision_todo_workbench(
        project_id=project_id,project=project,manuscript=manuscript,todo=workspace_todo,
        events=events,cards=cards,model=model,use_ollama=use_ollama,
    )
    st.divider();st.markdown("### 원고 수정·해소 평가 실행")
    modes=["유사 To-do 그룹 처리","단일 To-do 처리"] if len(todos)>=2 else ["단일 To-do 처리"]
    mode=st.radio("처리 방식",modes,horizontal=True,key=f"m2-revision-mode-{project_id}")
    if mode=="유사 To-do 그룹 처리":
        _render_revision_group_workflow(
            project_id=project_id,project=project,manuscript=manuscript,todos=todos,events=events,
            cards=cards,valid_card_ids=valid_card_ids,version_events=version_events,model=model,use_ollama=use_ollama,
        )
        return
    todo_id=st.selectbox("수행할 To-do",todo_ids,index=todo_ids.index(workspace_todo_id),key=f"m2-resolution-todo-{project_id}",format_func=lambda value:next(f"[{item['priority']}] {item['label']} · {item['sentence_text'][:75]}" for item in todos if item["todo_id"]==value))
    todo=next(item for item in todos if item["todo_id"]==todo_id)
    with st.container(border=True):
        st.markdown(f"**{todo['priority']} · {todo['label']}** · `{todo['sentence_id']}`")
        st.write(todo["sentence_text"]);st.markdown(f"**문제 설명**  \n{todo['problem']}")
        st.markdown(f"**완료 기준**  \n{todo['completion_criteria']}")

    intent_events=[
        event for event in events
        if event.get("event_type") in {"revision_literature_intent","revision_literature_queued"}
        and str((event.get("payload") or {}).get("todo_id") or "")==todo_id
    ]
    latest_intent_event=intent_events[-1] if intent_events else None
    latest_intent_payload=(latest_intent_event or {}).get("payload") or {}
    intent_id=str(latest_intent_payload.get("intent_id") or "")
    profile=next((item for item in ledger.search_profiles(include_deleted=True) if str(item.get("intent_id") or "")==intent_id),None)

    st.divider();st.markdown("### ① M1 문헌탐색 큐 등록")
    if profile:
        st.success(f"Done · M1 탐색 큐 등록 완료 · **{profile.get('title','')}**")
        st.caption("큐 등록으로 1단계가 완료됩니다. M1 실행과 서재함·지식카드 등록은 이후 진행될 수 있습니다.")
    else:
        candidates=list(todo.get("literature_search_candidates") or [])
        candidate=dict(candidates[0]) if candidates else {
            "candidate_id":f"{todo_id}-initial","title":f"{todo.get('label','')}의 근거와 적용 조건",
            "target":todo.get("search_guide") or todo.get("sentence_text"),
            "research_context":f"논문: {project.get('title','')}\n대상 문장: {todo.get('sentence_text','')}\n보완점: {todo.get('problem','')}",
            "expected_evidence":todo.get("search_guide",""),"completion_condition":todo.get("completion_criteria",""),
        }
        title=st.text_input("탐색 제목",value=str(candidate.get("title") or ""),key=f"m2-flow-title-{project_id}-{todo_id}")
        target=st.text_area("탐색 대상",value=str(candidate.get("target") or ""),height=85,key=f"m2-flow-target-{project_id}-{todo_id}")
        context=st.text_area("연구 맥락",value=str(candidate.get("research_context") or ""),height=105,key=f"m2-flow-context-{project_id}-{todo_id}")
        expected=st.text_area("기대 근거",value=str(candidate.get("expected_evidence") or ""),height=80,key=f"m2-flow-evidence-{project_id}-{todo_id}")
        completion=st.text_area("탐색 완료 조건",value=str(candidate.get("completion_condition") or ""),height=80,key=f"m2-flow-completion-{project_id}-{todo_id}")
        edited={**candidate,"title":title.strip(),"target":target.strip(),"research_context":context.strip(),"expected_evidence":expected.strip(),"completion_condition":completion.strip()}
        valid=all(len(str(edited.get(field) or ""))>=3 for field in ("title","target","research_context","expected_evidence","completion_condition"))
        if latest_intent_event and latest_intent_event.get("event_type")=="revision_literature_intent":
            st.info("기존 승인 요청이 있습니다. 아래 버튼으로 연구자가 확인하고 바로 M1 큐에 등록할 수 있습니다.")
            if st.button("확인하고 M1 탐색 큐에 등록",type="primary",key=f"m2-flow-approve-{project_id}-{todo_id}"):
                approved_profile=_approve_existing_revision_intent(latest_intent_event)
                if approved_profile:
                    ledger.add_paper_project_event(project_id,"revision_literature_queued",{
                        **latest_intent_payload,"profile_id":approved_profile.get("profile_id",""),"queue_title":approved_profile.get("title",""),"status":"queued",
                    });st.rerun()
        elif st.button("확인하고 M1 탐색 큐에 등록",type="primary",disabled=not valid,key=f"m2-flow-queue-{project_id}-{todo_id}"):
            _queue_paper_todo_literature_intent(project_id,project,todo,edited);st.rerun()

    st.divider();st.markdown("### ② 최신 원문·지식카드 선택 및 해결 요청")
    if not profile:
        st.info("1단계에서 M1 탐색 큐 등록을 완료하면 자료 선택과 해결 요청이 활성화됩니다.")
        return
    cutoff=str((latest_intent_event or {}).get("created_at") or "")
    all_papers=ledger.shelf_papers()
    recent_papers=[paper for paper in all_papers if not cutoff or str(paper.get("created_at") or paper.get("updated_at") or "")>=cutoff][:30]
    recent_cards=sorted([
        card for card in cards if not cutoff or str(card.get("approved_at") or card.get("updated_at") or card.get("created_at") or "")>=cutoff
    ],key=lambda item:str(item.get("approved_at") or item.get("updated_at") or item.get("created_at") or ""),reverse=True)[:40]
    if recent_papers:
        with st.expander(f"큐 등록 이후 서재함 논문 · 최신순 {len(recent_papers)}편",expanded=True):
            for paper in recent_papers:
                st.markdown(f"- **{paper.get('title','')}** · {paper.get('publication_year') or '연도 확인 필요'} · `{paper.get('paper_id','')}`")
    if recent_cards:
        with st.expander(f"큐 등록 이후 승인 지식카드 · 최신순 {len(recent_cards)}건",expanded=True):
            for card in recent_cards:
                st.markdown(f"- **{card.get('title','')}** · {card.get('claim','')} · `{card.get('card_id','')}`")
    if not recent_papers and not recent_cards:
        st.warning("마지막 큐 등록 이후 새로 등록된 서재함 논문이나 승인 지식카드가 없습니다. M1 탐색·논문 읽기·카드 등록을 먼저 완료하세요.")
    latest_proposal_event=next((event for event in reversed(events) if event.get("event_type")=="revision_resolution_proposal" and str((event.get("payload") or {}).get("todo_id") or "")==todo_id),None)
    proposal_is_current_cycle=bool(
        latest_proposal_event and (
            not latest_intent_event
            or str(latest_proposal_event.get("created_at") or "")>=str(latest_intent_event.get("created_at") or "")
        )
    )
    if proposal_is_current_cycle:
        st.success("Done · 논문·지식카드 선택과 To-do 해결 요청이 저장되었습니다. 필요하면 자료나 설명을 바꿔 다시 요청할 수 있습니다.")
    prior_payload=((latest_proposal_event or {}).get("payload") or {}) if proposal_is_current_cycle else {}

    st.markdown("#### 기존 서재함·지식카드 검색")
    asset_query=st.text_input(
        "논문·지식카드 검색",
        key=f"m2-flow-asset-search-{project_id}-{todo_id}",
        placeholder="논문 제목, 저자, 주장, 개념, 조건을 입력하세요.",
        help="전체 서재함과 승인 지식카드를 함께 검색합니다. 최근 자료는 검색 여부와 무관하게 기본 후보로 유지됩니다.",
    )
    search_papers,search_cards=search_revision_assets(asset_query,all_papers,cards,limit=30)
    if asset_query.strip():
        if search_papers:
            with st.expander(f"기존 서재함 검색 결과 {len(search_papers)}편",expanded=True):
                for paper in search_papers:
                    st.markdown(f"- **{paper.get('title','')}** · {paper.get('publication_year') or '연도 확인 필요'} · `{paper.get('paper_id','')}`")
        if search_cards:
            with st.expander(f"기존 승인 지식카드 검색 결과 {len(search_cards)}건",expanded=True):
                for card in search_cards:
                    st.markdown(f"- **{card.get('title','')}** · {card.get('claim','')} · `{card.get('card_id','')}`")
        if not search_papers and not search_cards:
            st.info("일치하는 서재함 논문이나 승인 지식카드가 없습니다. 제목·저자·핵심 주장·개념으로 다시 검색해 보세요.")

    recent_paper_ids={str(item.get("paper_id") or "") for item in recent_papers}
    recent_card_ids={str(item.get("card_id") or "") for item in recent_cards}
    paper_by_id={str(item["paper_id"]):item for item in recent_papers}
    card_by_id={str(item["card_id"]):item for item in recent_cards}
    for item in search_papers:paper_by_id.setdefault(str(item["paper_id"]),item)
    for item in search_cards:card_by_id.setdefault(str(item["card_id"]),item)
    all_paper_by_id={str(item.get("paper_id") or ""):item for item in all_papers}
    all_card_by_id={str(item.get("card_id") or ""):item for item in cards}
    for paper_id in prior_payload.get("selected_paper_ids",[]):
        if paper_id in all_paper_by_id:paper_by_id.setdefault(paper_id,all_paper_by_id[paper_id])
    for card_id in prior_payload.get("selected_card_ids",[]):
        if card_id in all_card_by_id:card_by_id.setdefault(card_id,all_card_by_id[card_id])
    st.caption("후보 출처: 신규 = 마지막 M1 큐 등록 이후 생성 · 기존 검색 = 전체 서재함/승인 카드 검색")
    selected_paper_ids=st.multiselect(
        "이 To-do를 처리할 논문",list(paper_by_id),
        default=[pid for pid in prior_payload.get("selected_paper_ids",[]) if pid in paper_by_id],
        format_func=lambda pid:f"[{'신규' if pid in recent_paper_ids else '기존 검색'}] {paper_by_id[pid].get('title',pid)}",
        key=f"m2-flow-papers-{project_id}-{todo_id}",
    )
    selected_card_ids=st.multiselect(
        "이 To-do를 처리할 지식카드",list(card_by_id),
        default=[cid for cid in prior_payload.get("selected_card_ids",[]) if cid in card_by_id],
        format_func=lambda cid:f"[{'신규' if cid in recent_card_ids else '기존 검색'}] {card_by_id[cid].get('title',cid)}",
        key=f"m2-flow-cards-{project_id}-{todo_id}",
    )
    connection_note=st.text_area("선택한 논문·지식카드가 To-do 해결에 연결되는 이유",value=str(prior_payload.get("connection_note") or ""),height=120,key=f"m2-flow-note-{project_id}-{todo_id}",placeholder="어떤 주장·조건·반례를 보완하며 대상 문장을 어떻게 바꿔야 하는지 간단히 적습니다.")
    selected_papers=[]
    for paper_id in selected_paper_ids:
        paper=paper_by_id[paper_id]
        source=_revision_paper_source(
            paper_id,str(paper.get("title") or ""),str(paper.get("pdf_path") or ""),str(paper.get("source_url") or ""),
        )
        selected_papers.append({**paper,**source})
    selected_cards=[card_by_id[cid] for cid in selected_card_ids]
    if selected_papers:
        with st.expander("선택 논문 원문 포함 상태",expanded=True):
            for paper in selected_papers:
                icon="✅" if paper.get("source_text") else "⚠️"
                st.markdown(f"- {icon} **{paper.get('title','')}** · {paper.get('source_status','')} · {int(paper.get('source_chars') or 0):,}자")
            st.caption("원문을 읽지 못한 논문의 제목·초록은 참고 정보로만 전달되며 해결 근거로 인정하지 않습니다.")
    include_full_manuscript=st.checkbox(
        "전체 2페이지 숏페이퍼도 일관성 확인용으로 포함",
        value=False,key=f"m2-flow-full-manuscript-{project_id}-{todo_id}",
        help="기본 프롬프트에는 현재 To-do의 대상 문장과 해당 문단만 들어갑니다. 논문 전체에 미치는 영향까지 확인할 때만 선택하세요.",
    )
    research_artifacts=[event.get("payload") or {} for event in events
        if event.get("event_type")=="revision_research_artifact"
        and str((event.get("payload") or {}).get("todo_id") or "")==todo_id]
    proposal_prompt=short_paper_resolution_proposal_prompt(
        project,manuscript,todo,selected_papers,selected_cards,connection_note,
        include_full_manuscript=include_full_manuscript,research_artifacts=research_artifacts,
    )
    usable_paper_ids={str(paper.get("paper_id") or "") for paper in selected_papers if paper.get("source_text")}
    confirmed_artifacts=[item for item in research_artifacts if item.get("status")=="researcher_confirmed"]
    can_request=bool((selected_card_ids or usable_paper_ids or confirmed_artifacts) and connection_note.strip())
    if selected_paper_ids and not usable_paper_ids and not selected_card_ids:
        st.warning("선택 논문에서 원문을 추출하지 못했습니다. 서재함에 PDF 또는 읽을 수 있는 원문 URL을 등록하거나 승인 지식카드를 선택하세요.")
    if research_artifacts:
        st.caption(f"연구자 산출물 {len(research_artifacts)}건이 해결 프롬프트에 포함됩니다. 연구자 확인 완료 항목만 직접 관찰 근거로 사용하도록 지시합니다.")
    if st.button("내부 LLM에 To-do 해결 요청",type="primary",disabled=not can_request,key=f"m2-flow-resolve-{project_id}-{todo_id}"):
        raw=llm_draft(proposal_prompt,model,use_ollama,profile="writing") or ""
        try:
            proposal=parse_resolution_proposal(raw,todo,valid_card_ids=set(selected_card_ids),valid_paper_ids=usable_paper_ids)
            ledger.add_paper_project_event(project_id,"revision_resolution_proposal",{
                "todo_id":todo_id,"from_version":manuscript.get("version"),"source":"internal_llm","proposal":proposal,
                "selected_paper_ids":selected_paper_ids,"selected_card_ids":selected_card_ids,"connection_note":connection_note.strip(),
            })
            ledger.add_paper_project_event(project_id,"revision_todo_update",{
                **{key:value for key,value in todo.items() if key!="latest_update"},"status":"ready",
                "selected_card_ids":selected_card_ids,"selected_paper_ids":selected_paper_ids,
                "researcher_response":connection_note.strip(),
            });st.rerun()
        except ValueError as error:st.error(str(error))
    with st.expander("외부 LLM으로 To-do 해결 요청"):
        prompt_version=hashlib.sha256(proposal_prompt.encode("utf-8")).hexdigest()[:12]
        st.caption(f"현재 To-do 중심 프롬프트 · 선택 원문 {len(usable_paper_ids)}편 · {len(proposal_prompt):,}자")
        st.text_area("해결 요청 프롬프트",value=proposal_prompt,height=520,key=f"m2-flow-prompt-{project_id}-{todo_id}-{prompt_version}")
        external=st.text_area("외부 LLM 해결 제안 JSON",height=260,key=f"m2-flow-response-{project_id}-{todo_id}")
        if st.button("외부 해결 제안 저장",disabled=not can_request or not external.strip(),key=f"m2-flow-apply-proposal-{project_id}-{todo_id}"):
            try:
                proposal=parse_resolution_proposal(external,todo,valid_card_ids=set(selected_card_ids),valid_paper_ids=usable_paper_ids)
                ledger.add_paper_project_event(project_id,"revision_resolution_proposal",{
                    "todo_id":todo_id,"from_version":manuscript.get("version"),"source":"external_llm","proposal":proposal,
                    "selected_paper_ids":selected_paper_ids,"selected_card_ids":selected_card_ids,"connection_note":connection_note.strip(),
                })
                ledger.add_paper_project_event(project_id,"revision_todo_update",{
                    **{key:value for key,value in todo.items() if key!="latest_update"},"status":"ready",
                    "selected_card_ids":selected_card_ids,"selected_paper_ids":selected_paper_ids,
                    "researcher_response":connection_note.strip(),
                });st.rerun()
            except ValueError as error:st.error(str(error))

    st.divider();st.markdown("### ③ 해결 확인 · 변경 검토 · 원고 반영")
    if not latest_proposal_event:
        st.caption("2단계의 해결 요청 결과가 생성되면 변경 전후 검토가 표시됩니다.")
    else:
        payload=latest_proposal_event.get("payload") or {};proposal=payload.get("proposal") or {}
        newer_search_cycle=bool(latest_intent_event and str(latest_intent_event.get("created_at") or "")>str(latest_proposal_event.get("created_at") or ""))
        stale=int(payload.get("from_version") or -1)!=int(manuscript.get("version") or -2) or newer_search_cycle
        if proposal.get("verdict")=="resolved":st.success(f"LLM 해결 제안 · 완료 기준 충족 가능 · {proposal.get('reason','')}")
        else:st.warning(f"LLM 판단 · 추가 근거 필요 · {proposal.get('reason','')}")
        try:
            proposed_manuscript,proposal_diff=apply_revisions(json.dumps({"revisions":proposal.get("revisions",[]),"appendix_updates":proposal.get("appendix_updates",[])},ensure_ascii=False),manuscript,valid_card_ids=valid_card_ids,version=len(version_events)+1)
        except ValueError:
            proposed_manuscript,proposal_diff=manuscript,[]
        for diff in proposal_diff:
            label="Appendix" if diff.get("change_scope")=="appendix" else f"문장 {diff.get('sentence_id','')}"
            st.markdown(f"**{label}** · {diff.get('reason','')}")
            before_col,after_col=st.columns(2);before_col.caption("이전");before_col.markdown(f"> {diff.get('before','')}");after_col.caption("제안");after_col.markdown(f"> {diff.get('after','')}")
        selected_source_papers={str(paper.get("paper_id")):paper for paper in ledger.shelf_papers() if str(paper.get("paper_id")) in payload.get("selected_paper_ids",[])}
        suggested_refs=[pid for pid in proposal.get("suggested_reference_paper_ids",[]) if pid in selected_source_papers]
        reference_ids=st.multiselect("원고 References에 포함할 논문",list(selected_source_papers),default=suggested_refs,format_func=lambda pid:selected_source_papers[pid].get("title",pid),key=f"m2-flow-refs-{project_id}-{todo_id}")
        if stale:st.error("이 해결 제안 이후 원고가 변경되었거나 추가 문헌탐색 주기가 시작되었습니다. 2단계에서 최신 자료와 현재 원고 기준으로 해결 요청을 다시 실행하세요.")
        can_apply=proposal.get("verdict")=="resolved" and bool(proposal_diff) and not stale
        if st.button("변경 확인 · 원고 반영 · To-do 완료",type="primary",disabled=not can_apply,key=f"m2-flow-complete-{project_id}-{todo_id}"):
            saved_reference_ids=[]
            for paper_id in reference_ids:
                paper=selected_source_papers[paper_id]
                reference=ledger.upsert_literature_reference(topic=f"{project.get('title','논문 프로젝트')} · References",research_context=f"Revision To-do {todo_id}: {todo.get('problem','')}",paper={**paper,"paper_project_id":project_id,"revision_todo_id":todo_id},labels=list(paper.get("labels",[])) or build_paper_labels(paper),status="selected")
                saved_reference_ids.append(str(reference.get("reference_id") or ""))
            ledger.add_paper_project_event(project_id,"manuscript_version",{"manuscript":proposed_manuscript,"source":"researcher_approved_resolution","diff":proposal_diff,"todo_id":todo_id,"todo_label":todo.get("label",""),"todo_priority":todo.get("priority","")})
            ledger.add_paper_project_event(project_id,"revision_resolution_applied",{"todo_id":todo_id,"from_version":payload.get("from_version"),"to_version":proposed_manuscript.get("version"),"paper_ids":payload.get("selected_paper_ids",[]),"card_ids":payload.get("selected_card_ids",[]),"reference_ids":[item for item in saved_reference_ids if item],"connection_note":payload.get("connection_note","")})
            ledger.add_paper_project_event(project_id,"revision_todo_update",{**{key:value for key,value in todo.items() if key!="latest_update"},"status":"resolved","resolution_source":"researcher_approved_resolution","selected_card_ids":payload.get("selected_card_ids",[])})
            st.rerun()

    st.divider();st.markdown("### ④ 해결 어려움 · 추가 문헌탐색")
    if not latest_proposal_event:
        st.caption("해결 요청 결과가 부족하다고 판단되면 다음 탐색 제목과 맥락을 검토해 다시 M1 큐에 넣습니다.")
        return
    payload=latest_proposal_event.get("payload") or {};proposal=payload.get("proposal") or {};next_search=proposal.get("next_search") or {}
    followup_already_queued=bool(latest_intent_event and str(latest_intent_event.get("created_at") or "")>str(latest_proposal_event.get("created_at") or ""))
    if followup_already_queued:
        st.success(f"Done · 추가 문헌탐색 큐 등록 완료 · **{(profile or {}).get('title') or latest_intent_payload.get('queue_title','')}**")
    next_title=st.text_input("추가 탐색 제목",value=str(next_search.get("title") or f"{todo.get('label','')} 추가 근거와 반례"),key=f"m2-follow-title-{project_id}-{todo_id}")
    next_target=st.text_area("추가 탐색 대상",value=str(next_search.get("target") or todo.get("search_guide") or ""),height=85,key=f"m2-follow-target-{project_id}-{todo_id}")
    next_context=st.text_area("추가 탐색 맥락",value=str(next_search.get("research_context") or f"이전 해결 요청으로 남은 공백: {proposal.get('reason','')}"),height=105,key=f"m2-follow-context-{project_id}-{todo_id}")
    next_evidence=st.text_area("추가로 필요한 근거",value=str(next_search.get("expected_evidence") or todo.get("search_guide") or ""),height=80,key=f"m2-follow-evidence-{project_id}-{todo_id}")
    next_completion=st.text_area("재검토 가능 조건",value=str(next_search.get("completion_condition") or todo.get("completion_criteria") or ""),height=80,key=f"m2-follow-completion-{project_id}-{todo_id}")
    followup={"candidate_id":f"{todo_id}-followup-{uuid.uuid4().hex[:6]}","title":next_title.strip(),"target":next_target.strip(),"research_context":next_context.strip(),"expected_evidence":next_evidence.strip(),"completion_condition":next_completion.strip()}
    followup_valid=all(len(str(followup.get(field) or ""))>=3 for field in ("title","target","research_context","expected_evidence","completion_condition"))
    if st.button("확인하고 추가 문헌탐색 큐에 등록",type="primary",disabled=not followup_valid or followup_already_queued,key=f"m2-follow-queue-{project_id}-{todo_id}"):
        _queue_paper_todo_literature_intent(project_id,project,todo,followup);st.rerun()


def _record_paper_todo_verification(
    project_id: str, todo: dict[str, Any], result: dict[str, Any], *, source: str,
    response_text: str = "",
) -> None:
    durable_todo={key:value for key,value in todo.items() if key!="latest_update"}
    verdict = str(result.get("verdict") or "needs_more_work")
    next_action = str(result.get("next_action") or "revise_again")
    action_map = {
        "researcher_input":"researcher_input", "knowledge_search":"knowledge_search",
        "literature_search":"literature_search", "revise_again":"researcher_input",
    }
    payload = {
        **result, "source":source,
        "response_digest":hashlib.sha256(response_text.strip().encode("utf-8")).hexdigest() if response_text.strip() else "",
    }
    ledger.add_paper_project_event(project_id,"revision_todo_verification",payload)
    ledger.add_paper_project_event(project_id,"revision_todo_update",{
        **durable_todo,
        "status":"resolved" if verdict=="resolved" else "open",
        "resolution_source":source if verdict=="resolved" else "",
        "recommended_action":action_map.get(next_action,str(todo.get("recommended_action") or "researcher_input")),
        "problem":str(result.get("remaining_gap") or todo.get("problem") or ""),
        "verification":payload,
    })


def _render_paper_todo_verification_result(payload: dict[str, Any]) -> None:
    resolved = payload.get("verdict") == "resolved"
    (st.success if resolved else st.warning)(
        f"{'해결 완료' if resolved else '추가 보완 필요'} · To-do `{payload.get('todo_id','')}` · "
        f"문장 `{payload.get('sentence_id','')}`"
    )
    if payload.get("reason"):
        st.write(payload["reason"])
    checks = payload.get("criteria_checks") or []
    if checks:
        st.dataframe([{
            "충족": "✓" if item.get("satisfied") else "✗",
            "검증 기준": item.get("criterion", ""), "판단 근거": item.get("evidence", ""),
        } for item in checks],use_container_width=True,hide_index=True)
    if payload.get("remaining_gap"):
        st.markdown(f"**남은 보완점**  \n{payload['remaining_gap']}")
    action_labels={
        "close":"To-do 종료","researcher_input":"연구자 입력 보완",
        "knowledge_search":"지식카드 보완","literature_search":"M1 문헌탐색",
        "revise_again":"문장 재수정",
    }
    st.caption(f"다음 행동 · {action_labels.get(str(payload.get('next_action','')),payload.get('next_action',''))}")


def _render_short_paper_milestone(
    *, project_id: str, project: dict[str, Any], events: list[dict[str, Any]],
    manuscript: dict[str, Any] | None, card_by_id: dict[str, dict[str, Any]],
    references: list[dict[str, Any]], todos: list[dict[str, Any]],
    model: str, use_ollama: bool,
) -> None:
    """Edit the writing contract, run callbacks, and freeze/reopen K_short."""
    frozen=frozen_milestone(events)
    spec_payload=latest_event_payload(events,"writing_spec_updated")
    spec=normalize_writing_spec((spec_payload or {}).get("spec") or spec_payload,project)
    if frozen:
        st.success(
            f"Short Paper 기준 버전 채택 · 원고 v{frozen.get('manuscript_version','?')} · "
            f"명세 S{frozen.get('spec_version','?')}"
        )
        st.caption(f"기준 버전 ID · {frozen.get('milestone_id','')} · 연구자 채택 {frozen.get('confirmed_at','')}")
        st.info("기준 버전 채택은 논문 완성 판정이 아닙니다. 이번 리비전의 상태와 남은 작업을 보존한 스냅샷이며, 다음 Revision을 시작해 계속 수정할 수 있습니다.")
        if frozen.get("researcher_comment"):st.markdown(f"**연구자 승인 의견**  \n{frozen['researcher_comment']}")
        validation=frozen.get("validation") or {}
        if validation.get("checks"):
            st.dataframe([{
                "결과":"PASS" if item.get("passed") else "FAIL",
                "검증":item.get("label",""),"수준":item.get("severity",""),"내용":item.get("detail",""),
            } for item in validation["checks"]],use_container_width=True,hide_index=True)
        reopen_reason=st.text_area("다음 Revision 목표",key=f"short-paper-unfreeze-reason-{project_id}",placeholder="예: 새 반례를 검토하여 중심 주장의 적용 범위를 조정한다.")
        if st.button("다음 Revision 시작",disabled=not reopen_reason.strip(),key=f"short-paper-unfreeze-{project_id}"):
            ledger.add_paper_project_event(project_id,"short_paper_unfrozen",{
                "milestone_id":frozen.get("milestone_id",""),
                "from_manuscript_version":frozen.get("manuscript_version",0),
                "reason":reopen_reason.strip(),
            });st.rerun()
        return

    st.markdown("#### 1. 최소 집필 명세")
    st.caption("초안 생성과 검토의 기준을 연구자가 먼저 정합니다. LLM은 이 명세를 임의로 변경하거나 미결정 사항을 대신 결정하지 않습니다.")
    proposal=latest_event_payload(events,"paper_proposal_confirmed") or {
        "initial":{"title":project.get("title",""),"research_question":project.get("research_question","")},
        "final_title":project.get("title",""),"final_research_question":project.get("research_question",""),
        "research_context":"","review":{},"review_card_ids":list(project.get("origin_ids") or []),
        "review_paper_ids":[],
    }
    guidance_card_ids=list(dict.fromkeys([
        *[str(value) for value in project.get("origin_ids") or []],
        *[str(value) for value in proposal.get("review_card_ids") or []],
        *(
            [str(card_id) for sentence in manuscript_sentences(manuscript) for card_id in sentence.get("evidence_card_ids") or []]
            if manuscript else []
        ),
    ]))
    guidance_cards=[card_by_id[value] for value in guidance_card_ids if value in card_by_id]
    guidance_papers=[]
    for paper_id in proposal.get("review_paper_ids") or []:
        paper=ledger.shelf_paper(str(paper_id))
        if not paper:continue
        analysis=ledger.paper_analysis(str(paper_id)) or {}
        guidance_papers.append({**paper,"analysis_summary":str(analysis.get("summary") or "")})
    guidance_input={
        "title":project.get("title",""),"research_question":project.get("research_question",""),
        "proposal_event":proposal,"card_ids":guidance_card_ids,
        "paper_ids":[str(item.get("paper_id") or "") for item in guidance_papers],
    }
    guidance_input_digest=hashlib.sha256(
        json.dumps(guidance_input,ensure_ascii=False,sort_keys=True).encode("utf-8")
    ).hexdigest()
    try:
        guidance_prompt=writing_spec_guidance_prompt(
            project=project,proposal=proposal,current_spec=spec,cards=guidance_cards,papers=guidance_papers,
        );guidance_prompt_error=""
    except Exception as error:
        guidance_prompt="";guidance_prompt_error=str(error)

    with st.container(border=True):
        st.markdown("##### LLM 집필 명세 가이드")
        st.caption(
            "확정 제목·연구질문, 이전 질문, 논문 기획 검토, 선택 지식카드와 서재함 논문을 바탕으로 "
            "중심 주장 후보와 집필 규칙을 제안합니다. 제안 적용 후 연구자가 아래 양식에서 수정할 수 있습니다."
        )
        if guidance_prompt_error:st.error(f"집필 명세 가이드 프롬프트 준비에 실패했습니다: {guidance_prompt_error}")
        if st.button("내부 LLM 호출 · 집필 명세 제안",type="primary",disabled=not guidance_prompt,key=f"writing-spec-guide-internal-{project_id}"):
            raw=llm_draft(guidance_prompt,model,use_ollama,profile="review") or ""
            try:
                guidance=parse_writing_spec_guidance(raw,valid_card_ids=set(card_by_id))
                ledger.add_paper_project_event(project_id,"writing_spec_guidance",{
                    "source":"internal_llm","input_digest":guidance_input_digest,"guidance":guidance,
                });st.rerun()
            except ValueError as error:st.error(str(error))
        with st.expander("외부 LLM용 집필 명세 프롬프트",expanded=False):
            st.text_area("외부 LLM에 보낼 편집 가능한 프롬프트",value=guidance_prompt,height=440,disabled=not guidance_prompt,key=f"writing-spec-guide-prompt-{project_id}-{guidance_input_digest[:10]}")
            external_guidance=st.text_area("외부 LLM JSON 응답",height=260,key=f"writing-spec-guide-response-{project_id}-{guidance_input_digest[:10]}")
            if st.button("외부 집필 명세 제안 검증·반영",disabled=not guidance_prompt or not external_guidance.strip(),key=f"writing-spec-guide-external-{project_id}"):
                try:
                    guidance=parse_writing_spec_guidance(external_guidance,valid_card_ids=set(card_by_id))
                    ledger.add_paper_project_event(project_id,"writing_spec_guidance",{
                        "source":"external_llm","input_digest":guidance_input_digest,"guidance":guidance,
                        "response_digest":hashlib.sha256(external_guidance.strip().encode("utf-8")).hexdigest(),
                    });st.rerun()
                except ValueError as error:st.error(str(error))

        guidance_payload=latest_event_payload(events,"writing_spec_guidance") or {}
        guidance=dict(guidance_payload.get("guidance") or {})
        guidance_current=bool(guidance and guidance_payload.get("input_digest")==guidance_input_digest)
        if guidance:
            if not guidance_current:st.warning("제안 생성 후 논문 기획, 근거 또는 집필 명세가 변경되었습니다. LLM 가이드를 다시 실행하세요.")
            if guidance.get("guidance_summary"):st.markdown(f"**제안 근거**  \n{guidance['guidance_summary']}")
            claim_by_id={str(item["claim_id"]):item for item in guidance.get("central_claim_candidates") or []}
            if claim_by_id:
                claim_ids=list(claim_by_id);recommended=str(guidance.get("recommended_claim_id") or claim_ids[0])
                guidance_digest=hashlib.sha256(json.dumps(guidance,ensure_ascii=False,sort_keys=True).encode("utf-8")).hexdigest()[:10]
                selected_claim_id=st.radio(
                    "중심 주장 후보",claim_ids,index=claim_ids.index(recommended) if recommended in claim_ids else 0,
                    key=f"writing-spec-claim-{project_id}-{guidance_digest}",
                    format_func=lambda value:claim_by_id[value]["claim"],
                )
                selected_claim=claim_by_id[selected_claim_id]
                if selected_claim.get("rationale"):st.caption("제안 이유 · " + selected_claim["rationale"])
                if selected_claim.get("evidence_card_ids"):st.caption("연결 가능 근거 · " + ", ".join(selected_claim["evidence_card_ids"]))
                if selected_claim.get("risk_or_condition"):st.warning("조건·위험 · " + selected_claim["risk_or_condition"])
                design_by_id={str(item["design_id"]):item for item in guidance.get("research_design_candidates") or []}
                selected_design={}
                if design_by_id:
                    compatible={
                        key:value for key,value in design_by_id.items()
                        if not value.get("compatible_claim_ids") or selected_claim_id in value.get("compatible_claim_ids",[])
                    } or design_by_id
                    design_ids=list(compatible)
                    recommended_design=str(guidance.get("recommended_design_id") or design_ids[0])
                    selected_design_id=st.radio(
                        "논문 입증 방법 후보",design_ids,
                        index=design_ids.index(recommended_design) if recommended_design in design_ids else 0,
                        key=f"writing-spec-design-{project_id}-{guidance_digest}-{selected_claim_id}",
                        format_func=lambda value:compatible[value]["method"],
                    )
                    selected_design=compatible[selected_design_id]
                    if selected_design.get("rationale"):st.caption("방법 선택 이유 · " + selected_design["rationale"])
                    for plan in selected_design.get("verification_plan") or []:
                        with st.expander(f"검증 계획 · {plan.get('research_question') or plan.get('claim_to_verify') or plan.get('plan_id','')}",expanded=True):
                            st.markdown(f"**검증 주장**  \n{plan.get('claim_to_verify','')}")
                            st.markdown(f"**방법·비교 기준**  \n{plan.get('method','')} · Baseline: {plan.get('baseline') or '연구자 확인 필요'}")
                            st.caption(f"분석 단위 · {plan.get('unit_of_analysis') or '연구자 확인 필요'}")
                            if plan.get("required_data"):st.markdown("**필요 데이터·산출물** · " + " · ".join(plan["required_data"]))
                            if plan.get("metrics"):
                                st.dataframe([{
                                    "지표":metric.get("name",""),"조작적 정의":metric.get("definition",""),
                                    "기록 예시":metric.get("example_record","") or "예시·미수행: [관측값 입력]",
                                } for metric in plan["metrics"]],use_container_width=True,hide_index=True)
                            st.markdown(f"**분석 방법**  \n{plan.get('analysis_method','')}")
                            st.markdown(f"**지지 기준**  \n{plan.get('success_criteria','')}")
                            st.markdown(f"**반증 조건**  \n{plan.get('falsification_condition','')}")
                            if plan.get("validity_threats"):st.warning("유효성 위협 · " + " · ".join(plan["validity_threats"]))
                    if selected_design.get("execution_guide"):
                        st.markdown("**실행 가이드**")
                        for index,item in enumerate(selected_design["execution_guide"],1):st.markdown(f"{index}. {item}")
                    if selected_design.get("result_recording_plan"):
                        st.markdown("**결과 기록 양식 예시**")
                        st.dataframe([{
                            "산출물":item.get("artifact",""),"필드":" · ".join(item.get("fields") or []),
                            "예시·상태":item.get("example_row","") or "예시·미수행: [실제 결과 입력 전]",
                        } for item in selected_design["result_recording_plan"]],use_container_width=True,hide_index=True)
                if guidance.get("literature_search_candidates"):
                    with st.expander(f"입증 설계를 위한 M1 탐색 후보 · {len(guidance['literature_search_candidates'])}건",expanded=False):
                        for item in guidance["literature_search_candidates"]:
                            st.markdown(f"- **{item.get('title','')}** — {item.get('target','')}")
                            st.caption(f"필요 근거 · {item.get('expected_evidence','')} · 완료 조건 · {item.get('completion_condition','')}")
                st.markdown("**필수 섹션 제안** · " + " · ".join(guidance.get("required_section_terms") or []))
                if guidance.get("writing_rules"):
                    st.markdown("**작성 규칙 제안**")
                    for item in guidance["writing_rules"]:st.markdown(f"- {item}")
                if guidance.get("open_decisions"):
                    st.markdown("**연구자 결정 제안**")
                    for item in guidance["open_decisions"]:st.markdown(f"- {item}")
                if st.button("선택한 제안을 집필 명세 초안으로 적용",disabled=not guidance_current,key=f"writing-spec-guide-apply-{project_id}-{guidance_digest}"):
                    next_spec=normalize_writing_spec({
                        **spec,"spec_version":int(spec.get("spec_version") or 0)+1,
                        "audience":guidance.get("audience") or spec.get("audience",""),
                        "central_claim":selected_claim["claim"],
                        "target_min_chars":guidance.get("target_min_chars") or spec.get("target_min_chars",4000),
                        "target_max_chars":guidance.get("target_max_chars") or spec.get("target_max_chars",9000),
                        "required_section_terms":guidance.get("required_section_terms") or spec.get("required_section_terms",[]),
                        "required_card_ids":list(dict.fromkeys([*list(spec.get("required_card_ids") or []),*selected_claim.get("evidence_card_ids",[])])),
                        "writing_rules":guidance.get("writing_rules") or spec.get("writing_rules",[]),
                        "open_decisions":guidance.get("open_decisions") or spec.get("open_decisions",[]),
                        "research_method":selected_design.get("method","") if selected_design else spec.get("research_method",""),
                        "research_method_rationale":selected_design.get("rationale","") if selected_design else spec.get("research_method_rationale",""),
                        "verification_plan":selected_design.get("verification_plan",[]) if selected_design else spec.get("verification_plan",[]),
                        "execution_guide":selected_design.get("execution_guide",[]) if selected_design else spec.get("execution_guide",[]),
                        "result_recording_plan":selected_design.get("result_recording_plan",[]) if selected_design else spec.get("result_recording_plan",[]),
                        "literature_search_candidates":guidance.get("literature_search_candidates",[]) or spec.get("literature_search_candidates",[]),
                    },project)
                    ledger.add_paper_project_event(project_id,"writing_spec_updated",{
                        "spec":next_spec,"source":"llm_guidance_applied",
                        "guidance_source":guidance_payload.get("source",""),"selected_claim_id":selected_claim_id,
                    });st.rerun()

    with st.expander("입력 가이드와 예시 보기",expanded=True):
        st.markdown(
            """
- **대상 독자**: 이 논문이 주로 설명하려는 독자입니다. 예: `에이전트·소프트웨어 공학 연구자와 제조 현장 엔지니어`
- **중심 주장**: 연구질문에 대해 논문 전체가 입증하거나 논증할 한 문장입니다. 예: `고위험 산업 에이전트의 자율성은 모델 역량이 아니라 검증 가능한 직무 경계에 따라 조정되어야 한다.`
- **필수 섹션 구성**: 섹션별 주제 키워드가 아니라, 원고에 반드시 존재해야 할 **섹션 제목 또는 제목 포함 단어**입니다. 예: `서론`, `제안 방법`, `논의`, `결론`
- **작성 규칙**: 이번 논문의 논증과 표현에서 지켜야 할 프로젝트별 원칙입니다. 예: `정량 수치는 출처가 확인된 경우에만 사용한다.`, `APF와 AJD는 최초 사용 시 정의한다.`
- **입증 방법**: 중심 주장을 어떤 실험·사례 비교·Survey·추적 분석으로 확인할지 정합니다. 아직 실행하지 않았다면 결과 대신 `예시·미수행` 기록 양식만 둡니다.
- **연구자 미결정 사항**: LLM이 대신 결정하면 안 되는 범위·용어·대조군·주장 강도 등의 선택입니다. 예: `연구 대상을 고위험 제조업으로 한정할지 결정`, `불가능하다는 표현을 완화할지 결정`
"""
        )
    spec_widget_key=f"{project_id}-s{int(spec.get('spec_version') or 0)}"
    with st.form(f"short-paper-spec-{project_id}"):
        audience=st.text_input("대상 독자",value=spec.get("audience",""),placeholder="예: 에이전트·소프트웨어 공학 연구자와 제조 현장 엔지니어",key=f"spec-audience-{spec_widget_key}")
        central_claim=st.text_area("중심 주장",value=spec.get("central_claim",""),height=90,placeholder="예: 고위험 산업 에이전트의 자율성은 검증 가능한 직무 경계에 따라 조정되어야 한다.",key=f"spec-claim-{spec_widget_key}")
        research_method=st.text_input("입증 방법",value=spec.get("research_method",""),placeholder="예: 비교 사례연구 + 실행 Trace 분석",key=f"spec-method-{spec_widget_key}")
        research_method_rationale=st.text_area("입증 방법 선택 이유",value=spec.get("research_method_rationale",""),height=80,key=f"spec-method-rationale-{spec_widget_key}")
        verification_plan_json=st.text_area(
            "RQ별 검증 계획 · JSON",value=json.dumps(spec.get("verification_plan") or [],ensure_ascii=False,indent=2),height=220,
            help="LLM 제안을 적용하면 비교 기준·분석 단위·데이터·지표·반증 조건이 자동으로 채워집니다. 실제 결과가 아니라 계획입니다.",
            key=f"spec-verification-{spec_widget_key}",
        )
        execution_guide=st.text_area("실험·Survey 실행 가이드 · 한 줄에 하나",value="\n".join(spec.get("execution_guide") or []),height=130,key=f"spec-execution-{spec_widget_key}")
        result_plan_json=st.text_area(
            "결과 기록 계획 · JSON",value=json.dumps(spec.get("result_recording_plan") or [],ensure_ascii=False,indent=2),height=180,
            help="예시는 반드시 ‘예시·미수행’으로 유지하고, 연구 수행 후 실제 관찰값으로 교체합니다.",key=f"spec-result-plan-{spec_widget_key}",
        )
        min_col,max_col=st.columns(2)
        minimum=min_col.number_input("최소 글자 수",min_value=100,max_value=50000,value=int(spec.get("target_min_chars") or 4000),step=100,key=f"spec-min-{spec_widget_key}")
        maximum=max_col.number_input("최대 글자 수",min_value=100,max_value=50000,value=int(spec.get("target_max_chars") or 9000),step=100,key=f"spec-max-{spec_widget_key}")
        required_sections=st.text_area(
            "필수 섹션 구성 · 한 줄에 하나",value="\n".join(spec.get("required_section_terms") or []),height=120,
            placeholder="서론\n연구질문\n제안 방법\n논의\n결론",
            help="입력한 단어가 포함된 섹션 제목이 원고에 있는지 자동 검사합니다. 섹션별 본문 키워드를 입력하는 항목은 아닙니다.",
            key=f"spec-sections-{spec_widget_key}",
        )
        required_cards=st.multiselect(
            "반드시 본문 또는 Appendix에 반영할 지식카드",list(card_by_id),
            default=[card_id for card_id in spec.get("required_card_ids") or [] if card_id in card_by_id],
            format_func=lambda card_id:card_by_id[card_id].get("title",card_id),
            key=f"spec-cards-{spec_widget_key}",
        )
        writing_rules=st.text_area(
            "이번 논문의 작성 규칙 · 한 줄에 하나",value="\n".join(spec.get("writing_rules") or []),height=130,
            placeholder="산업 현장 전체로 일반화하지 말고 고위험 공정으로 범위를 제한한다.\n정량 수치는 출처가 확인된 경우에만 사용한다.\n수식 다음에는 변수의 의미를 설명한다.",
            help="일반 문법이 아니라, 이번 논문에서 LLM과 연구자가 지켜야 할 논증·근거·표현 원칙입니다.",
            key=f"spec-rules-{spec_widget_key}",
        )
        open_decisions=st.text_area(
            "마일스톤 확정 전 연구자가 결정할 항목 · 한 줄에 하나",value="\n".join(spec.get("open_decisions") or []),height=130,
            placeholder="연구 대상을 고위험 제조업으로 한정할지 결정\n기존 워크플로우 에이전트를 대조군으로 사용할지 결정\n'불가능하다'는 표현을 완화할지 결정",
            help="남아 있는 항목은 자동 구조·근거 무결성 검사에서 마일스톤 확정을 차단합니다.",
            key=f"spec-decisions-{spec_widget_key}",
        )
        save_spec=st.form_submit_button("집필 명세 저장")
    if save_spec:
        try:
            verification_plan=json.loads(verification_plan_json or "[]")
            result_recording_plan=json.loads(result_plan_json or "[]")
            if not isinstance(verification_plan,list) or not isinstance(result_recording_plan,list):raise ValueError("검증 계획과 결과 기록 계획은 JSON 배열이어야 합니다.")
            next_spec=normalize_writing_spec({
                **spec,"spec_version":int(spec.get("spec_version") or 0)+1,
                "audience":audience,"central_claim":central_claim,
                "research_method":research_method,"research_method_rationale":research_method_rationale,
                "verification_plan":verification_plan,"execution_guide":execution_guide.splitlines(),
                "result_recording_plan":result_recording_plan,
                "target_min_chars":int(minimum),"target_max_chars":int(maximum),
                "required_section_terms":required_sections.splitlines(),
                "required_card_ids":required_cards,"writing_rules":writing_rules.splitlines(),
                "open_decisions":open_decisions.splitlines(),
            },project)
            ledger.add_paper_project_event(project_id,"writing_spec_updated",{"spec":next_spec,"source":"researcher"})
            st.rerun()
        except (json.JSONDecodeError,ValueError) as error:st.error(f"집필 명세를 저장하지 못했습니다: {error}")

    st.markdown("#### 2. 자동 구조·근거 무결성 검사")
    st.caption("LLM의 학술적 판단이 아니라, 프로그램이 원고 구조·분량·ID·근거 연결·References·P0 To-do·미결정 사항을 동일한 규칙으로 확인합니다.")
    if not manuscript:
        st.info("원고 초안을 생성하면 분량·섹션·근거·To-do 검증을 실행할 수 있습니다.")
        return
    if st.button("현재 원고 자동 검사 실행",type="primary",key=f"short-paper-validate-{project_id}"):
        result=validate_short_paper(
            manuscript,spec,valid_card_ids=set(card_by_id),
            valid_reference_paper_ids={
                str((item.get("paper") or {}).get("paper_id") or "")
                for item in references if (item.get("paper") or {}).get("paper_id")
            },
            todos=todos,reference_count=len(references),
        )
        ledger.add_paper_project_event(project_id,"short_paper_validation",result);st.rerun()
    latest_validation=latest_event_payload(events,"short_paper_validation") or {}
    current_result=validate_short_paper(
        manuscript,spec,valid_card_ids=set(card_by_id),
        valid_reference_paper_ids={
            str((item.get("paper") or {}).get("paper_id") or "")
            for item in references if (item.get("paper") or {}).get("paper_id")
        },
        todos=todos,reference_count=len(references),
    )
    current_validation=bool(
        latest_validation
        and latest_validation.get("manuscript_hash")==current_result.get("manuscript_hash")
        and latest_validation.get("spec_hash")==current_result.get("spec_hash")
    )
    if latest_validation:
        if not current_validation:st.warning("최근 검사 이후 원고 또는 집필 명세가 변경되었습니다. 자동 검사를 다시 실행하세요.")
        (st.success if latest_validation.get("passed") and current_validation else st.warning)(
            f"검증 {'PASS' if latest_validation.get('passed') and current_validation else '보완 필요'} · "
            f"Blocker {latest_validation.get('blocker_count',0)} · Warning {latest_validation.get('warning_count',0)}"
        )
        checks=latest_validation.get("checks") or []
        passed_checks=[item for item in checks if item.get("passed")]
        blocker_checks=[item for item in checks if not item.get("passed") and item.get("severity")=="blocker"]
        warning_checks=[item for item in checks if not item.get("passed") and item.get("severity")=="warning"]
        summary_cols=st.columns(3)
        summary_cols[0].metric("통과",len(passed_checks))
        summary_cols[1].metric("반드시 보완",len(blocker_checks))
        summary_cols[2].metric("확인 권고",len(warning_checks))
        if blocker_checks or warning_checks:
            st.markdown("##### 남은 보완 작업")
            st.dataframe([{
                "우선순위":"마일스톤 확정 차단" if item.get("severity")=="blocker" else "확인 권고",
                "보완 항목":item.get("label",""),"현재 상태":item.get("detail",""),
            } for item in [*blocker_checks,*warning_checks]],use_container_width=True,hide_index=True)
        with st.expander(f"통과한 자동 검증 · {len(passed_checks)}건",expanded=False):
            st.dataframe([{
                "검증":item.get("label",""),"확인 내용":item.get("detail",""),
            } for item in passed_checks],use_container_width=True,hide_index=True)

    st.markdown("#### 3. 이번 Revision 종료 · 기준 버전 채택")
    st.caption("출판 승인이나 논문 완성 판정이 아닙니다. 이번 리비전에서 해소한 내용과 남은 작업을 확인하고 현재 버전을 스냅샷으로 남깁니다.")
    confirm_ready=bool(manuscript)
    if latest_validation and not (latest_validation.get("passed") and current_validation):
        st.warning("자동 검사에서 남은 항목이 있습니다. 기준 버전은 채택할 수 있지만, 해당 항목은 다음 Revision 백로그로 이월해야 합니다.")
    confirm_argument=st.checkbox("이번 Revision에서 변경·해소한 내용을 확인했습니다.",disabled=not confirm_ready,key=f"short-confirm-argument-{project_id}")
    confirm_scope=st.checkbox("미해결·불필요·수정·신규 To-do의 정리 상태를 확인했습니다.",disabled=not confirm_ready,key=f"short-confirm-scope-{project_id}")
    confirm_evidence=st.checkbox("다음 Revision에서 이어갈 작업과 근거·References 상태를 확인했습니다.",disabled=not confirm_ready,key=f"short-confirm-evidence-{project_id}")
    researcher_comment=st.text_area("이번 Revision 요약·다음 목표",disabled=not confirm_ready,key=f"short-confirm-comment-{project_id}",height=90)
    can_freeze=bool(confirm_ready and confirm_argument and confirm_scope and confirm_evidence)
    if st.button("이번 Revision 종료 · 기준 버전으로 채택",type="primary",disabled=not can_freeze,key=f"short-paper-freeze-{project_id}"):
        evidence_ids=list(dict.fromkeys(
            [
                str(card_id) for sentence in manuscript_sentences(manuscript)
                for card_id in sentence.get("evidence_card_ids") or []
            ] + [
                str(card_id) for item in manuscript.get("appendix_claims") or []
                for card_id in item.get("evidence_card_ids") or []
            ]
        ))
        milestone_id=f"kshort-{project_id}-{latest_validation.get('manuscript_hash','')[:10]}"
        ledger.add_paper_project_event(project_id,"short_paper_frozen",{
            "milestone_id":milestone_id,
            "manuscript_version":manuscript.get("version",0),
            "manuscript_hash":latest_validation.get("manuscript_hash",""),
            "spec_version":spec.get("spec_version",0),"spec_hash":latest_validation.get("spec_hash",""),
            "spec":spec,"manuscript":manuscript,"validation":latest_validation,
            "evidence_card_ids":evidence_ids,
            "reference_ids":[str(item.get("reference_id") or "") for item in references if item.get("reference_id")],
            "reference_paper_ids":[
                str((item.get("paper") or {}).get("paper_id") or "")
                for item in references if (item.get("paper") or {}).get("paper_id")
            ],
            "researcher_confirm":{
                "argument":True,"scope":True,"evidence":True,
            },
            "researcher_comment":researcher_comment.strip(),"confirmed_at":datetime.now(UTC).isoformat(timespec="seconds"),
        });st.rerun()


def _paper_project_overview(project: dict[str, Any]) -> dict[str, Any]:
    events=list(project.get("events") or [])
    versions=[event for event in events if event.get("event_type")=="manuscript_version"]
    manuscript=(versions[-1].get("payload") or {}).get("manuscript") if versions else None
    todos=_revision_todos_with_events(manuscript,events) if manuscript else []
    return {
        "versions":len(versions),
        "open_todos":sum(1 for item in todos if item.get("status")!="resolved"),
        "has_draft":bool(manuscript),
        "workflow_stage":project_workflow_stage(project),
    }


def _render_paper_proposal_assessment(review: dict[str, Any]) -> None:
    status_labels={
        "grounded_internal":"내부 근거 확인","model_prior":"LLM 사전지식·잠정",
        "verification_required":"M1 확인 필요","researcher_decision":"연구자 결정",
    }
    if review.get("planning_summary"):
        st.markdown(f"**기획 검토 요약**  \n{review['planning_summary']}")
    sections=[
        ("내부 지식·문헌 기반 유사성","internal_similarity_assessment"),
        ("외부 연구 지형에 대한 잠정 의견","external_landscape_assessment"),
        ("신규성·차별성 위험","novelty_risks"),
    ]
    for label,key in sections:
        rows=review.get(key) or []
        if not rows:continue
        st.markdown(f"##### {label}")
        st.dataframe([{
            "판단 상태":status_labels.get(str(item.get("evidence_status","")),item.get("evidence_status","")),
            "검토 대상":item.get("subject",""),"의견":item.get("assessment",""),
            "내부 근거":", ".join(item.get("source_ids") or []) or "-",
            "추가 확인":item.get("verification_need","") or "-",
        } for item in rows],use_container_width=True,hide_index=True)
    decisions=review.get("researcher_decisions") or []
    if decisions:
        st.markdown("##### 연구자가 결정할 사항")
        for item in decisions:
            options=" / ".join(item.get("options") or [])
            st.markdown(f"- **{item.get('question','')}**" + (f" · {options}" if options else ""))
            if item.get("reason"):st.caption(item["reason"])


def _request_paper_proposal_literature_intent(
    project: dict[str, Any], candidate: dict[str, Any],
) -> tuple[str,str]:
    """Create a researcher approval request for proposal-stage novelty checking."""
    project_id=str(project.get("project_id") or "");intent_id=f"intent-{uuid.uuid4().hex[:12]}"
    intent=CurationIntent(
        intent_id=intent_id,title=str(candidate.get("title") or "논문 기획 유사연구 검증")[:120],
        purpose="논문 제목과 연구질문의 유사연구 중복 가능성 및 잠정 기여를 실제 문헌으로 검증한다.",
        question=str(candidate.get("target") or project.get("research_question") or ""),
        research_context="\n".join(filter(None,[
            f"Paper project: {project.get('title','')}",f"Research question: {project.get('research_question','')}",
            str(candidate.get("research_context") or ""),
        ])),labels=[],priority="높음",
        expected_evidence=str(candidate.get("expected_evidence") or "가장 가까운 선행연구의 문제, 방법, 결과, 조건과 본 연구의 차이"),
        completion_condition=str(candidate.get("completion_condition") or "가장 가까운 연구와의 공통점·차이·중복 위험을 출처와 함께 보고한다."),
        execution_mode="manual",created_by="m2",
        origin_links=[{
            "origin_type":"paper_writing","origin_id":project_id,"origin_sub_id":"paper_proposal",
            "label":str(project.get("title") or "논문 기획"),"source_card_ids":list(project.get("origin_ids") or []),
        }],
    )
    case_id=ledger.create_case("research","논문 기획 유사연구 검증")
    request_id=ledger.record(
        case_id,"decision_request","m2",["researcher"],"curation_intent",
        {"title":f"M1 탐색 Intent 승인: {intent.title}","intent":intent.model_dump(mode="json"),
         "next_action":"승인 시 M1 문헌탐색 작업 큐에 등록","paper_project_id":project_id,
         "proposal_candidate_id":candidate.get("candidate_id","")},subject_id=intent_id,
    )
    ledger.add_paper_project_event(project_id,"paper_proposal_literature_intent",{
        "candidate_id":candidate.get("candidate_id",""),"intent_id":intent_id,"request_id":request_id,
        "status":"approval_requested","title":intent.title,"target":intent.question,
    })
    return intent_id,request_id


def render_paper_project_hub(model: str, use_ollama: bool, semantic: bool, embedding_model: str) -> None:
    """Manage the project portfolio and a proposal-reviewed creation wizard."""
    st.header("M2 · 논문 프로젝트")
    st.caption("프로젝트를 먼저 구성한 뒤, 별도 논문 작업실에서 문헌 탐색·근거 보완·리비전을 반복합니다.")
    projects=ledger.paper_projects();cards=memory.all();card_by_id={str(card["card_id"]):card for card in cards}
    active_count=sum(1 for project in projects if project.get("status")=="active")
    completed_count=sum(1 for project in projects if project.get("status")=="completed")
    c1,c2,c3=st.columns(3);c1.metric("전체 프로젝트",len(projects));c2.metric("진행 중",active_count);c3.metric("완료",completed_count)

    st.markdown("### 프로젝트 목록")
    status_filter=st.radio(
        "상태",["all","active","completed"],index=0,horizontal=True,key="paper-project-status-filter",
        format_func={"all":"전체","active":"진행 중","completed":"완료"}.get,
    )
    visible=[project for project in projects if status_filter=="all" or project.get("status")==status_filter]
    if not visible:st.info("조건에 맞는 논문 프로젝트가 없습니다.")
    for project in visible:
        overview=_paper_project_overview(project);status=str(project.get("status") or "active")
        with st.container(border=True):
            title_col,state_col=st.columns([5,1])
            title_col.markdown(f"**{project.get('title','제목 없음')}**")
            state_col.markdown("✅ 완료" if status=="completed" else "🟠 진행 중")
            st.write(project.get("research_question") or "연구질문 없음")
            st.caption(
                f"{WORKFLOW_STAGE_LABELS.get(overview['workflow_stage'],overview['workflow_stage'])} · "
                f"원고 버전 {overview['versions']} · 열린 To-do {overview['open_todos']} · "
                f"최근 변경 {_fmt_local_time(project.get('updated_at'))}"
            )
            open_col,status_col=st.columns([3,2])
            if open_col.button("논문 작업실 열기",type="primary",key=f"paper-project-open-{project['project_id']}",use_container_width=True):
                st.session_state["m2-paper-project-id"]=project["project_id"]
                st.session_state["_navigate_workspace"]="논문 작업실"
                st.rerun()
            next_status="active" if status=="completed" else "completed"
            if status_col.button("작업 재개" if status=="completed" else "완료로 표시",key=f"paper-project-status-{project['project_id']}",use_container_width=True):
                ledger.update_paper_project_status(project["project_id"],next_status);st.rerun()

    st.divider();st.markdown("### 새 논문 프로젝트 만들기")
    step=int(st.session_state.get("paper-project-wizard-step",1))
    st.progress(step/3,text=f"{step}/3 · " + {1:"출발점 선택",2:"논문 기획 LLM 검토",3:"근거 선택·프로젝트 생성"}[step])
    draft=dict(st.session_state.get("paper-project-wizard") or {})
    if step==1:
        source_mode=st.radio("프로젝트 출발점",["existing_rq","researcher_input"],horizontal=True,key="paper-project-source-mode",format_func={"existing_rq":"기존 연구질문","researcher_input":"연구자 직접 입력"}.get)
        if source_mode=="existing_rq":
            questions=[item for item in ledger.research_question_backlog(limit=500) if item.get("status")!="rejected"]
            if not questions:
                st.info("선택할 연구질문이 없습니다. 아래에서 새 지식으로 질문 후보를 만들거나 직접 입력을 선택하세요.")
            else:
                rq_ids=[str(item["rq_id"]) for item in questions]
                rq_id=st.selectbox("연구질문",rq_ids,key="paper-project-wizard-rq",format_func=lambda value:next(str(item.get("question") or value) for item in questions if str(item["rq_id"])==value))
                rq=next(item for item in questions if str(item["rq_id"])==rq_id)
                st.markdown(f"**연구질문**  \n{rq.get('question','')}")
                if rq.get("research_context"):st.markdown(f"**연구 맥락**  \n{rq['research_context']}")
                st.caption(f"연결 지식카드 {len(rq.get('source_card_ids') or [])}건")
                if st.button("다음 · 제목과 연구질문 확인",type="primary",key="paper-project-step1-rq"):
                    lineage=cards_for_origin(card_by_id.values(),[rq_id])
                    inherited=list(dict.fromkeys([*list(rq.get("source_card_ids") or []),*[card["card_id"] for card in lineage]]))
                    for key in ("paper-proposal-title","paper-proposal-rq","paper-proposal-context","paper-proposal-cards","paper-proposal-papers","paper-proposal-external","paper-proposal-prompt","paper-proposal-candidate","paper-proposal-searches"):
                        st.session_state.pop(key,None)
                    st.session_state["paper-project-wizard"]={"source_mode":"existing_rq","source_rq_id":rq_id,"title":str(rq.get("question") or "")[:80],"research_question":str(rq.get("question") or ""),"research_context":str(rq.get("research_context") or ""),"inherited_ids":inherited}
                    st.session_state["paper-project-wizard-step"]=2;st.rerun()
        else:
            direct_title=st.text_input("가제",key="paper-project-direct-title")
            direct_rq=st.text_area("확정 연구질문",key="paper-project-direct-rq",height=120)
            if st.button("다음 · 제목과 연구질문 확인",type="primary",disabled=not direct_title.strip() or not direct_rq.strip(),key="paper-project-step1-direct"):
                for key in ("paper-proposal-title","paper-proposal-rq","paper-proposal-context","paper-proposal-cards","paper-proposal-papers","paper-proposal-external","paper-proposal-prompt","paper-proposal-candidate","paper-proposal-searches"):
                    st.session_state.pop(key,None)
                st.session_state["paper-project-wizard"]={"source_mode":"researcher_input","source_rq_id":"","title":direct_title.strip(),"research_question":direct_rq.strip(),"research_context":"","inherited_ids":[]}
                st.session_state["paper-project-wizard-step"]=2;st.rerun()
        with st.expander("새 지식에서 연구질문 후보 만들기",expanded=False):
            st.caption("최근 지식카드에서 질문 후보를 만든 뒤 1단계의 ‘기존 연구질문’에서 선택합니다.")
            _render_m1_new_information(model,use_ollama,semantic,embedding_model)
    elif step==2:
        st.markdown("#### 1. 검토할 제목·연구질문")
        title=st.text_input("현재 가제",value=str(draft.get("title") or ""),key="paper-proposal-title")
        research_question=st.text_area("현재 연구질문",value=str(draft.get("research_question") or ""),height=110,key="paper-proposal-rq")
        research_context=st.text_area("연구 동기·맥락",value=str(draft.get("research_context") or ""),height=100,key="paper-proposal-context",placeholder="왜 이 문제가 중요하며, 어떤 대상·환경·한계를 다루려는지 적습니다.")

        query=" ".join(filter(None,[title,research_question,research_context]))
        shelf_papers=[]
        for paper in ledger.shelf_papers():
            analysis=ledger.paper_analysis(str(paper.get("paper_id") or "")) or {}
            shelf_papers.append({**paper,"analysis_summary":str(analysis.get("summary") or "")})
        matched_papers,matched_cards=search_revision_assets(query,shelf_papers,cards,limit=12)
        inherited_ids=[str(value) for value in draft.get("inherited_ids") or [] if str(value) in card_by_id]
        default_card_ids=list(dict.fromkeys([*inherited_ids,*[str(item.get("card_id")) for item in matched_cards[:8]]]))
        paper_by_id={str(item.get("paper_id")):item for item in shelf_papers if item.get("paper_id")}
        default_paper_ids=[str(item.get("paper_id")) for item in matched_papers[:6] if item.get("paper_id")]

        st.markdown("#### 2. 내부 검토 근거 선택")
        st.caption("현재 제목과 연구질문으로 관련 후보를 찾았습니다. 지식카드는 승인된 주장으로, 서재함 논문은 저장된 초록·M1 분석 범위에서 검토합니다.")
        proposal_card_ids=st.multiselect(
            "검토할 승인 지식카드",list(card_by_id),default=default_card_ids,key="paper-proposal-cards",
            format_func=lambda value:card_by_id[value].get("title",value),
        )
        proposal_paper_ids=st.multiselect(
            "검토할 서재함 논문",list(paper_by_id),default=default_paper_ids,key="paper-proposal-papers",
            format_func=lambda value:paper_by_id[value].get("title",value),
        )
        selected_cards=[card_by_id[value] for value in proposal_card_ids if value in card_by_id]
        selected_papers=[paper_by_id[value] for value in proposal_paper_ids if value in paper_by_id]
        try:
            proposal_prompt=paper_proposal_prompt(
                title=title,research_question=research_question,research_context=research_context,
                cards=selected_cards,papers=selected_papers,
            )
            proposal_prompt_error=""
        except Exception as error:
            proposal_prompt="";proposal_prompt_error=str(error)
        proposal_input={
            "title":title.strip(),"research_question":research_question.strip(),
            "research_context":research_context.strip(),"card_ids":proposal_card_ids,"paper_ids":proposal_paper_ids,
        }
        proposal_input_digest=hashlib.sha256(
            json.dumps(proposal_input,ensure_ascii=False,sort_keys=True).encode("utf-8")
        ).hexdigest()[:10]

        st.markdown("#### 3. LLM 논문 기획·유사연구 검토")
        st.caption("외부 연구 지형 의견은 LLM 사전지식에 따른 잠정 판단으로 표시되며, 실제 신규성 판단은 M1 탐색으로 검증합니다.")
        if proposal_prompt_error:
            st.error(f"기획 검토 프롬프트 준비에 실패했습니다: {proposal_prompt_error}")
        st.caption("내부 LLM을 호출하거나, 아래 프롬프트를 외부 LLM에 보내고 JSON 응답을 붙여넣을 수 있습니다.")
        if st.button("내부 LLM 호출 · 논문 기획 검토",type="primary",disabled=not proposal_prompt or not title.strip() or not research_question.strip(),key="paper-proposal-internal"):
            raw=llm_draft(proposal_prompt,model,use_ollama,profile="review") or ""
            try:
                review=parse_paper_proposal(raw)
                st.session_state["paper-project-wizard"]={**draft,"proposal_review":review,"proposal_review_source":"internal_llm","proposal_review_input":proposal_input}
                st.rerun()
            except ValueError as error:st.error(str(error))
        with st.expander("외부 LLM용 프롬프트 복사·응답 반영",expanded=True):
            st.text_area("외부 LLM에 보낼 편집 가능한 프롬프트",value=proposal_prompt,height=440,disabled=not proposal_prompt,key=f"paper-proposal-prompt-{proposal_input_digest}")
            external_proposal=st.text_area("외부 LLM JSON 응답",height=280,key=f"paper-proposal-external-{proposal_input_digest}")
            if st.button("외부 기획 검토 응답 검증·반영",disabled=not proposal_prompt or not external_proposal.strip(),key="paper-proposal-external-apply"):
                try:
                    review=parse_paper_proposal(external_proposal)
                    st.session_state["paper-project-wizard"]={**draft,"proposal_review":review,"proposal_review_source":"external_llm","proposal_review_input":proposal_input}
                    st.rerun()
                except ValueError as error:st.error(str(error))

        review=dict(draft.get("proposal_review") or {})
        reviewed_input=dict(draft.get("proposal_review_input") or {})
        review_current=bool(review and reviewed_input==proposal_input)
        final_title=title.strip();final_rq=research_question.strip();selected_search_ids=[];selected_candidate_id=""
        if review:
            if not review_current:st.warning("검토 이후 제목·질문·맥락 또는 선택 근거가 변경되었습니다. LLM 기획 검토를 다시 실행하세요.")
            st.markdown("#### 4. 검토 결과와 최종안 선택")
            _render_paper_proposal_assessment(review)
            candidates=review.get("candidate_pairs") or []
            candidate_by_id={str(item["candidate_id"]):item for item in candidates}
            candidate_ids=list(candidate_by_id)
            recommended=str(review.get("recommended_candidate_id") or candidate_ids[0])
            review_digest=hashlib.sha256(
                json.dumps(review,ensure_ascii=False,sort_keys=True).encode("utf-8")
            ).hexdigest()[:10]
            selected_candidate_id=st.radio(
                "제목·연구질문 조합",candidate_ids,index=candidate_ids.index(recommended) if recommended in candidate_ids else 0,
                key=f"paper-proposal-candidate-{review_digest}",
                format_func=lambda value:f"{candidate_by_id[value]['title']} — {candidate_by_id[value]['research_question']}",
            )
            selected_candidate=candidate_by_id[selected_candidate_id]
            if selected_candidate.get("rationale"):st.caption(selected_candidate["rationale"])
            final_title=st.text_input("최종 논문 제목",value=selected_candidate["title"],key=f"paper-proposal-final-title-{review_digest}-{selected_candidate_id}")
            final_rq=st.text_area("최종 연구질문",value=selected_candidate["research_question"],height=105,key=f"paper-proposal-final-rq-{review_digest}-{selected_candidate_id}")
            searches=review.get("m1_verification_candidates") or []
            search_by_id={str(item["candidate_id"]):item for item in searches}
            if search_by_id:
                selected_search_ids=st.multiselect(
                    "프로젝트 생성 후 연구자 승인함에 보낼 M1 유사연구 검증 후보",list(search_by_id),
                    default=list(search_by_id),key=f"paper-proposal-searches-{review_digest}",
                    format_func=lambda value:search_by_id[value].get("title",value),
                )
                with st.expander("M1 검증 후보 내용",expanded=False):
                    for value in selected_search_ids:
                        item=search_by_id[value];st.markdown(f"**{item.get('title','')}**")
                        st.write(item.get("target",""));st.caption(item.get("completion_condition",""))

        back_col,next_col=st.columns(2)
        if back_col.button("이전",key="paper-project-step2-back",use_container_width=True):
            st.session_state["paper-project-wizard-step"]=1;st.rerun()
        if next_col.button("제목·연구질문 확정 · 근거 선택으로",type="primary",disabled=not review_current or not final_title.strip() or not final_rq.strip(),key="paper-project-step2-next",use_container_width=True):
            selected_searches=[item for item in review.get("m1_verification_candidates") or [] if str(item.get("candidate_id")) in selected_search_ids]
            st.session_state["paper-project-wizard"]={
                **draft,"title":final_title.strip(),"research_question":final_rq.strip(),
                "research_context":research_context.strip(),"proposal_review":review,
                "proposal_review_source":draft.get("proposal_review_source",""),
                "proposal_review_input":reviewed_input,"proposal_card_ids":proposal_card_ids,
                "proposal_paper_ids":proposal_paper_ids,"proposal_selected_candidate_id":selected_candidate_id,
                "proposal_m1_candidates":selected_searches,
            }
            st.session_state["paper-project-wizard-step"]=3;st.rerun()
    else:
        st.markdown(f"**{draft.get('title','')}**")
        st.write(draft.get("research_question") or "")
        evidence_key=f"paper-project-wizard-evidence-{draft.get('source_rq_id') or 'direct'}"
        selected=_render_paper_evidence_selector(
            title=str(draft.get("title") or ""),research_question=str(draft.get("research_question") or ""),
            inherited_ids=list(draft.get("inherited_ids") or []),card_by_id=card_by_id,
            semantic=semantic,embedding_model=embedding_model,key_prefix=evidence_key,
        )
        st.caption("카드를 선택하지 않아도 프로젝트를 만들 수 있으며, 작업실에서 M1 탐색을 통해 보완할 수 있습니다.")
        back_col,create_col=st.columns(2)
        if back_col.button("이전",key="paper-project-step3-back",use_container_width=True):
            st.session_state["paper-project-wizard-step"]=2;st.rerun()
        if create_col.button("프로젝트 생성 후 작업실 열기",type="primary",key="paper-project-create",use_container_width=True):
            final_ids=selected or list(draft.get("inherited_ids") or [])
            if draft.get("source_rq_id"):
                rq=next((item for item in ledger.research_question_backlog(limit=500) if str(item.get("rq_id"))==str(draft["source_rq_id"])),None)
                if not rq:st.error("선택한 연구질문을 찾을 수 없습니다.");return
                source={**rq,"question":str(draft.get("research_question") or rq.get("question") or ""),"source_card_ids":final_ids}
                project=_create_paper_project_from_research_question(source,str(draft.get("title") or ""))
            else:
                project=ledger.create_paper_project(title=str(draft.get("title") or ""),research_question=str(draft.get("research_question") or ""),origin_type="knowledge_cards" if final_ids else "researcher_input",origin_ids=final_ids)
            _record_paper_evidence_selection(project["project_id"],evidence_key,final_ids)
            if draft.get("proposal_review"):
                ledger.add_paper_project_event(project["project_id"],"paper_proposal_confirmed",{
                    "source":draft.get("proposal_review_source",""),
                    "initial":draft.get("proposal_review_input") or {},
                    "final_title":project.get("title") or draft.get("title",""),
                    "final_research_question":project.get("research_question") or draft.get("research_question",""),
                    "research_context":draft.get("research_context",""),
                    "review":draft.get("proposal_review") or {},
                    "selected_candidate_id":draft.get("proposal_selected_candidate_id",""),
                    "review_card_ids":draft.get("proposal_card_ids") or [],
                    "review_paper_ids":draft.get("proposal_paper_ids") or [],
                })
                for candidate in draft.get("proposal_m1_candidates") or []:
                    _request_paper_proposal_literature_intent(project,candidate)
            st.session_state["m2-paper-project-id"]=project["project_id"]
            st.session_state["paper-project-wizard-step"]=1
            st.session_state["paper-project-wizard"]={}
            st.session_state["_navigate_workspace"]="논문 작업실";st.rerun()


def _short_paper_validation_state(
    manuscript: dict[str, Any] | None, spec: dict[str, Any], card_by_id: dict[str, dict[str, Any]],
    references: list[dict[str, Any]], todos: list[dict[str, Any]], events: list[dict[str, Any]],
) -> tuple[dict[str, Any], bool]:
    if not manuscript:
        return {}, False
    result=validate_short_paper(
        manuscript,spec,valid_card_ids=set(card_by_id),
        valid_reference_paper_ids={
            str((item.get("paper") or {}).get("paper_id") or "")
            for item in references if (item.get("paper") or {}).get("paper_id")
        },
        todos=todos,reference_count=len(references),
    )
    saved=latest_event_payload(events,"short_paper_validation") or {}
    current=bool(
        saved and saved.get("manuscript_hash")==result.get("manuscript_hash")
        and saved.get("spec_hash")==result.get("spec_hash")
    )
    return result,current


def _render_short_paper_stage_header(
    *, project_id: str, project: dict[str, Any], events: list[dict[str, Any]],
    manuscript: dict[str, Any] | None, spec: dict[str, Any], evidence: list[dict[str, Any]],
    card_by_id: dict[str, dict[str, Any]], references: list[dict[str, Any]],
    todos: list[dict[str, Any]],
) -> str:
    """Render the workspace-level stage navigator and return the selected view."""
    active_freeze=frozen_milestone(events)
    validation,current_validation=_short_paper_validation_state(
        manuscript,spec,card_by_id,references,todos,events,
    )
    spec_ready=bool(int(spec.get("spec_version") or 0)>0 and spec.get("central_claim"))
    evidence_ready=bool(evidence)
    draft_ready=bool(manuscript)
    review_complete=any(item.get("event_type")=="revision_review" for item in events)
    revision_ready=bool(manuscript and review_complete and not todos)
    validation_ready=bool(validation.get("passed") and current_validation)
    steps=[
        ("연구질문·근거",bool(project.get("research_question")) and evidence_ready),
        ("집필 명세",spec_ready),
        ("초안 작성",draft_ready),
        ("근거 보완·리비전",revision_ready),
        ("자동 구조·근거 검사",validation_ready),
        ("K_short 마일스톤",bool(active_freeze)),
    ]
    completed=sum(1 for _,done in steps if done)
    if active_freeze:
        next_title="확정된 K_short 확인"
        next_detail="기준본이 편집 잠금 상태입니다. 새 근거나 범위 변경이 있으면 사유를 남기고 마일스톤을 재개할 수 있습니다."
        next_view="milestone"
    elif not spec_ready:
        next_title="최소 집필 명세 확정"
        next_detail="중심 주장, 대상 독자, 분량 예산과 연구자 미결정 항목을 먼저 정합니다."
        next_view="milestone"
    elif not manuscript:
        next_title="2페이지 초안 생성"
        next_detail="확정 연구질문·집필 명세·선택 지식카드로 첫 원고를 만듭니다."
        next_view="writing"
    elif not review_complete:
        next_title="LLM 문장 검증 실행"
        next_detail="초안의 문장별 근거·범위·반론·연구자 결정 필요 여부를 먼저 진단합니다."
        next_view="writing"
    elif todos:
        p0_count=sum(1 for item in todos if item.get("priority")=="P0")
        next_title=f"Revision To-do {len(todos)}건 처리"
        next_detail=f"열린 P0 {p0_count}건을 우선 해결하고 문장별 해결 여부를 검증합니다."
        next_view="writing"
    elif not validation_ready:
        next_title="자동 구조·근거 무결성 검사"
        next_detail="현재 원고의 구조·분량·근거·References 무결성을 검사합니다."
        next_view="milestone"
    else:
        next_title="연구자 승인 및 K_short 확정"
        next_detail="논증, 주장 범위, 근거 연결을 확인하고 Full Paper 확장의 기준본으로 확정합니다."
        next_view="milestone"

    st.markdown("### Short Paper 진행 상태")
    st.progress(completed/len(steps),text=f"{completed}/{len(steps)} 단계 충족")
    for row_start in (0,3):
        stage_cols=st.columns(3)
        for offset in range(3):
            index=row_start+offset;label,done=steps[index]
            with stage_cols[offset]:
                st.markdown(f"{'✅' if done else '○'} **{index+1}. {label}**")

    metric_cols=st.columns(5)
    metric_cols[0].metric("원고",f"v{manuscript.get('version',0)}" if manuscript else "없음")
    metric_cols[1].metric("근거카드",len(evidence))
    p0_todos=[item for item in todos if item.get("priority")=="P0"]
    metric_cols[2].metric("최우선 P0",len(p0_todos))
    metric_cols[3].metric("References",len(references))
    metric_cols[4].metric("검증",("PASS" if validation_ready else "대기"))
    (st.success if active_freeze else st.info)(f"**다음 작업 · {next_title}**  \n{next_detail}")

    state_key=f"paper-workspace-view-{project_id}"
    if state_key not in st.session_state:
        st.session_state[state_key]="overview"
    action_col,alt_col=st.columns([2,1])
    if action_col.button(f"다음 작업 열기 · {next_title}",type="primary",key=f"paper-next-action-{project_id}",use_container_width=True):
        st.session_state[state_key]=next_view;st.rerun()
    if alt_col.button("작업 개요로",key=f"paper-overview-action-{project_id}",use_container_width=True):
        st.session_state[state_key]="overview";st.rerun()
    return st.radio(
        "작업 화면",["overview","writing","milestone","history"],horizontal=True,key=state_key,
        format_func={
            "overview":"프로젝트 개요","writing":"원고 작성·리비전",
            "milestone":"자동 검사·마일스톤 승인","history":"변경 이력",
        }.get,
    )


def _render_short_paper_overview(
    *, project: dict[str, Any], manuscript: dict[str, Any] | None, spec: dict[str, Any],
    evidence: list[dict[str, Any]], references: list[dict[str, Any]], todos: list[dict[str, Any]],
    events: list[dict[str, Any]], origin_event: dict[str, Any] | None,
) -> None:
    st.markdown("## 프로젝트 개요")
    left,right=st.columns([1,1])
    with left:
        with st.container(border=True):
            st.markdown("#### 프로젝트 구성")
            st.markdown(f"**연구질문**  \n{project.get('research_question') or '미정'}")
            st.markdown(f"**중심 주장**  \n{spec.get('central_claim') or '아직 입력하지 않음'}")
            st.caption(f"대상 독자 · {spec.get('audience') or '미정'}")
            st.caption(
                f"분량 예산 · {int(spec.get('target_min_chars') or 0):,}–"
                f"{int(spec.get('target_max_chars') or 0):,}자 · 집필 명세 S{spec.get('spec_version',0)}"
            )
            if spec.get("open_decisions"):
                st.warning("연구자 미결정 · " + " | ".join(spec["open_decisions"]))
    with right:
        with st.container(border=True):
            st.markdown("#### 현재 원고")
            if manuscript:
                st.markdown(f"**{manuscript.get('title') or '제목 없음'}**")
                st.caption(
                    f"원고 v{manuscript.get('version',0)} · 섹션 {len(manuscript.get('sections') or [])}개 · "
                    f"공백 제외 {len(''.join(manuscript_plain_text(manuscript).split())):,}자"
                )
                p0=sum(1 for item in todos if item.get("priority")=="P0")
                st.caption(f"최우선 보완 P0 {p0}건 · References {len(references)}편")
            else:
                st.info("아직 원고가 없습니다. 집필 명세를 확정한 뒤 초안을 생성하세요.")

    evidence_col,todo_col=st.columns([1,1])
    with evidence_col:
        with st.container(border=True):
            st.markdown(f"#### 근거 자산 · 카드 {len(evidence)}건 / 논문 {len(references)}편")
            for card in evidence[:5]:
                st.markdown(f"- **{card.get('title',card.get('card_id',''))}**")
            if len(evidence)>5:st.caption(f"그 외 {len(evidence)-5}건")
            if not evidence:st.warning("관련 지식카드가 없습니다. 초안 생성 전에 근거 탐색을 권장합니다.")
    with todo_col:
        with st.container(border=True):
            p0_todos=[item for item in todos if item.get("priority")=="P0"]
            st.markdown(f"#### 최우선 Revision To-do · P0 {len(p0_todos)}건")
            for todo in p0_todos[:5]:
                st.markdown(f"- **{todo.get('priority','P1')} · {todo.get('label','보완')}** — {todo.get('problem','')}")
            if not p0_todos:st.success("현재 최우선 P0 To-do가 없습니다.")
            if len(todos)>len(p0_todos):st.caption(f"그 외 To-do {len(todos)-len(p0_todos)}건은 ‘원고 작성·리비전’ 화면에서 확인합니다.")

    proposal_intents=[
        event.get("payload") or {} for event in events
        if event.get("event_type")=="paper_proposal_literature_intent"
        and (event.get("payload") or {}).get("status")=="approval_requested"
    ]
    if proposal_intents:
        st.info(
            f"논문 기획 단계에서 M1 유사연구 검증 승인 요청 {len(proposal_intents)}건을 만들었습니다. "
            "연구자 홈의 검토·승인함에서 승인하면 연구 Intent 탐색 작업 큐에 등록됩니다."
        )
        for item in proposal_intents:st.markdown(f"- **{item.get('title','')}** · Intent `{item.get('intent_id','')}`")

    if origin_event:
        origin=origin_event.get("payload") or {}
        with st.expander("출발 연구질문·승계 맥락",expanded=False):
            st.caption(f"Research Question · {origin.get('source_rq_id','')} · {M2_SOURCE_LABELS.get(str(origin.get('source_type','m1_knowledge')),origin.get('source_type','m1_knowledge'))}")
            if origin.get("rationale"):st.markdown(f"**도출 이유**  \n{origin['rationale']}")
            if origin.get("research_context"):st.markdown(f"**연구 맥락**  \n{origin['research_context']}")
            if origin.get("gap_or_tension"):st.markdown(f"**공백·긴장**  \n{origin['gap_or_tension']}")

    recent=[event for event in reversed(events) if event.get("event_type") in {
        "manuscript_version","revision_review","revision_resolution_applied",
        "short_paper_validation","short_paper_frozen","short_paper_unfrozen",
        "paper_proposal_confirmed","paper_proposal_literature_intent",
        "writing_spec_guidance","writing_spec_updated",
        "revision_todo_plan_confirmed","revision_research_artifact","revision_todo_reconciliation_applied",
    }][:5]
    if recent:
        st.markdown("#### 최근 주요 변경")
        st.dataframe([{
            "시각":item.get("created_at",""),
            "변경":item.get("event_type",""),
            "원고 버전":(item.get("payload") or {}).get("manuscript_version",(item.get("payload") or {}).get("to_version","")),
        } for item in recent],use_container_width=True,hide_index=True)


def _render_paper_project_history(events: list[dict[str, Any]]) -> None:
    st.markdown("## 변경 이력")
    if not events:
        st.info("기록된 프로젝트 이벤트가 없습니다.");return
    for event in reversed(events):
        payload=event.get("payload") or {}
        with st.expander(f"{event.get('created_at','')} · {event.get('event_type','')}",expanded=False):
            if payload.get("source"):st.caption(f"source · {payload.get('source')}")
            if event.get("event_type")=="revision_todo_evidence_link":
                st.write(f"To-do `{payload.get('todo_id','')}` · 원문 {len(payload.get('paper_ids') or [])}편 · 지식카드 {len(payload.get('card_ids') or [])}건 · References {len(payload.get('reference_ids') or [])}편")
            elif event.get("event_type")=="revision_literature_queued":
                st.write(f"To-do `{payload.get('todo_id','')}` · M1 탐색 큐 등록 · {payload.get('queue_title','')}")
            elif event.get("event_type")=="revision_resolution_applied":
                st.write(f"To-do `{payload.get('todo_id','')}` 완료 · 원고 v{payload.get('from_version','?')} → v{payload.get('to_version','?')}")
            elif event.get("event_type")=="revision_review":
                st.write(f"원고 v{payload.get('reviewed_from_version','?')} → v{payload.get('result_version','?')} · To-do {payload.get('todo_count',0)}건 · 신규 {payload.get('new_count',0)} · 해결 {payload.get('resolved_count',0)}")
            elif event.get("event_type")=="revision_todo_reconciliation_applied":
                report=payload.get("report") or {};counts=report.get("counts") or {}
                st.write(f"Revision v{payload.get('from_version','?')} → v{payload.get('to_version','?')} To-do 재정리")
                st.caption(f"해결 {counts.get('resolved',0)} · 유지/수정 {counts.get('active',0)} · 불필요 {counts.get('obsolete',0)} · 통합/분리 {counts.get('restructured',0)} · 신규 {counts.get('new',0)}")
                for item in report.get("next_revision_recommendations") or []:st.markdown(f"- 다음 제안 · {item}")
            elif event.get("event_type")=="revision_todo_plan_confirmed":
                plan=payload.get("plan") or {}
                st.write(f"To-do `{payload.get('todo_id','')}` 해결 계획 확정 · 작업 {len(plan.get('actions') or [])}건")
                if plan.get("strategy_summary"):st.markdown(plan["strategy_summary"])
            elif event.get("event_type")=="revision_research_artifact":
                st.write(f"To-do `{payload.get('todo_id','')}` 연구 산출물 · {payload.get('artifact_type','')} · {payload.get('title','')}")
                st.caption(f"상태 {payload.get('status','')} · {payload.get('source_ref','')}")
            elif event.get("event_type")=="paper_proposal_confirmed":
                st.write(f"논문 기획 LLM 검토 확정 · {payload.get('final_title','')}")
                st.markdown(f"**연구질문**  \n{payload.get('final_research_question','')}")
                st.caption(f"검토 카드 {len(payload.get('review_card_ids') or [])}건 · 서재함 논문 {len(payload.get('review_paper_ids') or [])}편")
            elif event.get("event_type")=="paper_proposal_literature_intent":
                st.write(f"논문 기획 M1 검증 승인 요청 · {payload.get('title','')}")
                st.caption(f"Intent `{payload.get('intent_id','')}` · 상태 {payload.get('status','')}")
            elif event.get("event_type")=="writing_spec_guidance":
                guidance=payload.get("guidance") or {}
                st.write(f"LLM 집필 명세 제안 · 중심 주장 후보 {len(guidance.get('central_claim_candidates') or [])}건")
                if guidance.get("guidance_summary"):st.markdown(guidance["guidance_summary"])
            elif event.get("event_type")=="writing_spec_updated":
                saved_spec=payload.get("spec") or payload
                st.write(f"최소 집필 명세 S{saved_spec.get('spec_version','?')} 저장")
                if saved_spec.get("central_claim"):st.markdown(f"**중심 주장**  \n{saved_spec['central_claim']}")
                if payload.get("selected_claim_id"):st.caption(f"LLM 제안 적용 · {payload.get('selected_claim_id')}")
            elif event.get("event_type")=="revision_todo_verification":
                _render_paper_todo_verification_result(payload)
            elif event.get("event_type")=="short_paper_validation":
                st.write(f"자동 구조·근거 무결성 검사 · {'PASS' if payload.get('passed') else '보완 필요'} · Blocker {payload.get('blocker_count',0)} · Warning {payload.get('warning_count',0)}")
            elif event.get("event_type")=="short_paper_frozen":
                st.write(f"K_short 마일스톤 확정 · 원고 v{payload.get('manuscript_version','?')} · {payload.get('milestone_id','')}")
            elif event.get("event_type")=="short_paper_unfrozen":
                st.write(f"K_short 마일스톤 재개 · {payload.get('reason','')}")
            elif event.get("event_type")=="manuscript_version":
                st.write(f"원고 v{(payload.get('manuscript') or {}).get('version','?')}")
            if payload.get("diff"):st.dataframe(payload["diff"],use_container_width=True,hide_index=True)


def render_m2_coauthor_workspace(model: str, use_ollama: bool, semantic: bool, embedding_model: str) -> None:
    st.header("논문 작업실 · Short Paper Co-author")
    st.caption("선택한 프로젝트의 초안을 만들고, M1 문헌 탐색과 지식카드 보완을 거쳐 문장별 리비전을 반복합니다.")
    projects=ledger.paper_projects(); cards=memory.all(); card_by_id={str(c["card_id"]):c for c in cards}; valid_ids=set(card_by_id)
    if not projects:
        st.info("생성된 논문 프로젝트가 없습니다. M2 · 논문 프로젝트 페이지에서 먼저 프로젝트를 만드세요.")
        if st.button("논문 프로젝트 만들기로 이동",type="primary",key="paper-workspace-go-hub"):
            st.session_state["_navigate_workspace"]="M2 · 지식 기반 자문";st.rerun()
        return
    ids=[p["project_id"] for p in projects]; current_id=st.selectbox("프로젝트",ids,format_func=lambda x:next(p["title"] for p in projects if p["project_id"]==x),key="m2-paper-project-id")
    project=ledger.paper_project(current_id) or {}; events=project.get("events",[])
    nav_col,status_col=st.columns([3,2])
    if nav_col.button("프로젝트 목록·생성으로 돌아가기",key=f"paper-workspace-back-{current_id}",use_container_width=True):
        st.session_state["_navigate_workspace"]="M2 · 지식 기반 자문";st.rerun()
    if project.get("status")=="completed":
        if status_col.button("완료 프로젝트 작업 재개",key=f"paper-workspace-resume-{current_id}",use_container_width=True):
            ledger.update_paper_project_status(current_id,"active");st.rerun()
        st.info("완료된 프로젝트입니다. 기존 원고와 이력은 볼 수 있으며, 수정 작업을 계속하려면 ‘작업 재개’를 선택하세요.")
    else:
        if status_col.button("프로젝트 완료로 표시",key=f"paper-workspace-complete-{current_id}",use_container_width=True):
            ledger.update_paper_project_status(current_id,"completed");st.rerun()
    origin_event=next((e for e in reversed(events) if e.get("event_type")=="project_origin"),None)
    lineage_origin_ids=[current_id]
    if origin_event and str((origin_event.get("payload") or {}).get("source_rq_id", "")).strip():
        lineage_origin_ids.append(str((origin_event.get("payload") or {}).get("source_rq_id")))
    lineage_cards=cards_for_origin(card_by_id.values(),lineage_origin_ids)
    evidence_ids=list(dict.fromkeys([*project.get("origin_ids",[]),*[card["card_id"] for card in lineage_cards]]))
    evidence=[card_by_id[x] for x in evidence_ids if x in card_by_id]
    version_events=[e for e in events if e["event_type"]=="manuscript_version"]; manuscript=version_events[-1]["payload"].get("manuscript") if version_events else None
    manuscript_evidence_ids={str(card["card_id"]) for card in evidence}
    if manuscript:
        manuscript_evidence_ids.update(
            str(card_id) for sentence in manuscript_sentences(manuscript)
            for card_id in sentence.get("evidence_card_ids",[]) if str(card_id) in card_by_id
        )
    review_evidence=[card_by_id[card_id] for card_id in manuscript_evidence_ids if card_id in card_by_id]
    project_references=_paper_project_references(current_id)
    spec_payload=latest_event_payload(events,"writing_spec_updated")
    writing_spec=normalize_writing_spec((spec_payload or {}).get("spec") or spec_payload,project)
    project["writing_spec"]=writing_spec
    milestone_todos=_revision_todos_with_events(manuscript,events) if manuscript else []
    active_freeze=frozen_milestone(events)
    workspace_view=_render_short_paper_stage_header(
        project_id=current_id,project=project,events=events,manuscript=manuscript,
        spec=writing_spec,evidence=review_evidence,card_by_id=card_by_id,
        references=project_references,todos=milestone_todos,
    )
    if workspace_view=="overview":
        _render_short_paper_overview(
            project=project,manuscript=manuscript,spec=writing_spec,evidence=review_evidence,
            references=project_references,todos=milestone_todos,events=events,
            origin_event=origin_event,
        )
        if lineage_cards:
            with st.expander(f"문헌탐색 계보로 연결된 지식카드 · {len(lineage_cards)}건",expanded=False):
                for card in lineage_cards:
                    st.markdown(f"- **{card.get('title',card.get('card_id',''))}** · {card.get('claim','')}")
                    render_origin_labels(card)
        return
    if workspace_view=="milestone":
        st.markdown("## " + ("프로젝트 구성 · 최소 집필 명세" if not manuscript else "상태 진단·Revision 종료·기준 버전"))
        _render_short_paper_milestone(
            project_id=current_id,project=project,events=events,manuscript=manuscript,
            card_by_id=card_by_id,references=project_references,todos=milestone_todos,
            model=model,use_ollama=use_ollama,
        )
        return
    if workspace_view=="history":
        _render_paper_project_history(events);return
    if active_freeze:
        st.warning("현재 원고가 이번 Revision의 기준 버전으로 채택되어 있습니다. 계속 수정하려면 ‘상태 진단·Revision 종료·기준 버전’ 화면에서 다음 Revision을 시작하세요.")
        frozen_manuscript=active_freeze.get("manuscript") or manuscript
        if frozen_manuscript:st.markdown(manuscript_markdown(frozen_manuscript,markup=False))
        if project_references:
            st.markdown("## References")
            for index,reference in enumerate(project_references,1):st.markdown(f"{index}. {_reference_display(reference)}")
        return
    st.markdown("## 원고 작성·근거 보완·리비전")
    draft_tab,review_tab,revise_tab=st.tabs(["원고","문장 검증","Revision To-do"])
    with draft_tab:
        st.info(project.get("research_question")); prompt=short_paper_draft_prompt(project,evidence)
        draft_spec_missing=[]
        if int(writing_spec.get("spec_version") or 0)<=0:draft_spec_missing.append("저장된 집필 명세")
        if not writing_spec.get("central_claim"):draft_spec_missing.append("중심 주장")
        if not writing_spec.get("required_section_terms"):draft_spec_missing.append("필수 섹션 구성")
        draft_spec_ready=not draft_spec_missing
        with st.container(border=True):
            st.markdown("#### Draft에 적용되는 최소 집필 명세")
            if draft_spec_ready:
                st.success(f"집필 명세 S{writing_spec.get('spec_version',0)} 적용 준비 완료")
                st.markdown(f"**중심 주장**  \n{writing_spec.get('central_claim','')}")
                st.caption("필수 섹션 · " + " · ".join(writing_spec.get("required_section_terms") or []))
                if writing_spec.get("research_method"):
                    st.markdown(f"**입증 방법**  \n{writing_spec.get('research_method','')}")
                    st.caption(f"RQ별 검증 계획 {len(writing_spec.get('verification_plan') or [])}건 · 결과 기록 양식 {len(writing_spec.get('result_recording_plan') or [])}건")
                if writing_spec.get("writing_rules"):
                    st.caption("작성 규칙 · " + " | ".join(writing_spec["writing_rules"]))
                if writing_spec.get("open_decisions"):
                    st.warning("미결정 사항은 LLM이 결론내리지 않고 문장 마크업과 연구자 질문으로 남깁니다. · " + " | ".join(writing_spec["open_decisions"]))
                st.caption("LLM은 섹션 제목에 필수 구성을 반영하고, 작성 규칙을 제약으로 적용하며, 근거가 부족하면 사실을 만들지 않고 Revision To-do 후보로 표시합니다.")
            else:
                st.warning("최초 Draft 생성 전에 최소 집필 명세를 확정해야 합니다. 누락: " + ", ".join(draft_spec_missing))
                st.caption("상단의 ‘자동 검사·마일스톤 승인’ 화면에서 집필 명세를 저장한 뒤 돌아오세요.")
        c1,c2=st.columns(2)
        if c1.button("내부 LLM 초안 생성",type="primary",disabled=not draft_spec_ready):
            raw=llm_draft(prompt,model,use_ollama,profile="writing") or ""
            try:m=parse_manuscript(raw,valid_card_ids=valid_ids,version=len(version_events)+1); ledger.add_paper_project_event(current_id,"manuscript_version",{"manuscript":m,"source":"internal_draft","diff":[]}); st.rerun()
            except ValueError as error:st.error(str(error))
        with st.expander("외부 LLM으로 초안 생성"):
            st.caption("집필 명세의 값뿐 아니라 각 항목의 적용 방법도 프롬프트에 포함됩니다.")
            st.text_area("편집 가능한 프롬프트",value=prompt,height=420,disabled=not draft_spec_ready,key=f"m2-draft-prompt-{current_id}"); external=st.text_area("외부 LLM JSON 응답",height=260,disabled=not draft_spec_ready,key=f"m2-draft-response-{current_id}")
            if st.button("외부 초안 검증·저장",disabled=not draft_spec_ready or not external.strip()):
                try:m=parse_manuscript(external,valid_card_ids=valid_ids,version=len(version_events)+1); ledger.add_paper_project_event(current_id,"manuscript_version",{"manuscript":m,"source":"external_draft","diff":[]}); st.rerun()
                except ValueError as error:st.error(str(error))
        if manuscript:
            st.markdown(manuscript_markdown(manuscript,markup=True))
            if project_references:
                st.markdown("## References")
                for index, reference in enumerate(project_references, start=1):
                    paper = reference.get("paper") or {}
                    source_url = str(paper.get("source_url") or paper.get("html_url") or paper.get("url") or "").strip()
                    citation = _reference_display(reference)
                    st.markdown(f"{index}. [{citation}]({source_url})" if source_url else f"{index}. {citation}")
            with st.expander("Appendix · Full Paper 확장 후보 갱신",expanded=False):
                appendix_count=len(manuscript.get("appendix_claims") or [])
                st.caption(
                    f"현재 후보 {appendix_count}건 · 2페이지 본문을 늘리지 않고, 빠진 중요 주장을 최대 8건까지 보존합니다. "
                    "본문과 중복되거나 승인 지식카드 근거가 없는 주장은 저장하지 않습니다."
                )
                appendix_task_prompt=short_paper_appendix_prompt(project,manuscript,evidence)
                if st.button("내부 LLM으로 Appendix 후보 갱신",key=f"m2-appendix-refresh-{current_id}"):
                    raw=llm_draft(appendix_task_prompt,model,use_ollama,profile="writing") or ""
                    try:
                        refreshed,diff=apply_appendix_refresh(raw,manuscript,valid_card_ids=valid_ids,version=len(version_events)+1)
                        if not diff:
                            st.info("현재 본문과 근거 기준에서 Appendix 후보 변경이 없습니다.")
                        else:
                            ledger.add_paper_project_event(current_id,"manuscript_version",{
                                "manuscript":refreshed,"source":"appendix_refresh","diff":diff,
                            })
                            st.rerun()
                    except ValueError as error:st.error(str(error))
                st.text_area("Appendix 후보 갱신 프롬프트",value=appendix_task_prompt,height=380,key=f"m2-appendix-prompt-{current_id}")
                appendix_external=st.text_area("외부 LLM Appendix JSON",height=240,key=f"m2-appendix-response-{current_id}")
                if st.button("외부 Appendix 후보 반영",disabled=not appendix_external.strip(),key=f"m2-appendix-apply-{current_id}"):
                    try:
                        refreshed,diff=apply_appendix_refresh(appendix_external,manuscript,valid_card_ids=valid_ids,version=len(version_events)+1)
                        if not diff:
                            st.info("외부 응답에 반영할 Appendix 변경이 없습니다.")
                        else:
                            ledger.add_paper_project_event(current_id,"manuscript_version",{
                                "manuscript":refreshed,"source":"external_appendix_refresh","diff":diff,
                            })
                            st.rerun()
                    except ValueError as error:st.error(str(error))
            _render_manuscript_revision_history(version_events,events)
    with review_tab:
        if not manuscript:st.info("초안을 먼저 생성하세요.")
        else:
            st.info("문장 검증·마크업은 원고의 문제를 진단합니다. Revision To-do는 각 진단을 보완·수정·재검증하는 실행 작업입니다.")
            review_flash=st.session_state.pop(f"m2-review-flash-{current_id}",None)
            latest_review_event=next((event for event in reversed(events) if event.get("event_type")=="revision_review"),None)
            latest_review_payload=(latest_review_event or {}).get("payload") or {}
            current_version_reviewed=(
                bool(latest_review_event)
                and int(latest_review_payload.get("result_version") or -1)==int(manuscript.get("version") or -2)
            )
            if review_flash:
                _render_review_result(review_flash)
                st.info("반영된 항목은 아래 진단표와 `실행 · Revision To-do` 탭에서 확인할 수 있습니다.")
            elif latest_review_event:
                with st.expander("최근 검증 반영 결과",expanded=False):
                    _render_review_result(latest_review_event.get("payload") or {},latest=True)
            with st.expander("마크업 의미 보기", expanded=False):
                st.dataframe(annotation_legend(),use_container_width=True,hide_index=True)
            st.dataframe(
                _paper_diagnostic_rows(manuscript,card_by_id),
                use_container_width=True,hide_index=True,
                column_config={"문장 내용":st.column_config.TextColumn(width="large"),"진단 내용":st.column_config.TextColumn(width="large")},
            )
            todos=_revision_todos_with_events(manuscript,events)
            if todos:
                st.warning(f"열린 Revision To-do {sum(1 for item in todos if item['status']=='open')}건 · 보완 정보 준비 {sum(1 for item in todos if item['status']=='ready')}건 · 수정 후 재검증 대기 {sum(1 for item in todos if item['status']=='revised_pending_review')}건")
            else:
                st.success("현재 열린 보완점이 없습니다.")
            if current_version_reviewed:
                st.info(
                    f"현재 원고 v{manuscript.get('version','?')}의 문장 검증은 반영 완료되어 이 화면은 읽기 전용입니다. "
                    "다음 작업은 `실행 · Revision To-do`에서 수행하세요. 문장이 수정되어 새 버전이 생기면 검증 기능이 다시 활성화됩니다."
                )
            prompt=short_paper_review_prompt(project,manuscript,review_evidence)
            if st.button("내부 LLM 문장 검증 실행",disabled=current_version_reviewed):
                raw=llm_draft(prompt,model,use_ollama,profile="review") or ""
                try:
                    result_version=len(version_events)+1
                    anns=parse_review(raw,manuscript); reviewed=apply_review(manuscript,anns); reviewed["version"]=result_version
                    review_result=_review_result_payload(
                        source="internal_llm",before_todos=todos,annotations=anns,
                        reviewed_from_version=int(manuscript.get("version") or len(version_events)),result_version=result_version,
                    )
                    _record_resolved_revision_todos(current_id,todos,anns,"internal_llm_review")
                    ledger.add_paper_project_event(current_id,"manuscript_version",{"manuscript":reviewed,"source":"llm_review","diff":[]})
                    ledger.add_paper_project_event(current_id,"revision_review",review_result)
                    st.session_state[f"m2-review-flash-{current_id}"]=review_result
                    st.rerun()
                except ValueError as error:st.error(str(error))
            with st.expander("외부 LLM 문장 검증",expanded=False):
                st.text_area("검증 프롬프트",value=prompt,height=350,key=f"m2-review-prompt-{current_id}",disabled=current_version_reviewed)
                response=st.text_area("외부 검증 JSON",height=240,key=f"m2-review-response-{current_id}",disabled=current_version_reviewed)
                response_digest=hashlib.sha256(response.strip().encode("utf-8")).hexdigest() if response.strip() else ""
                already_applied=next((
                    event.get("payload") or {} for event in reversed(events)
                    if event.get("event_type")=="revision_review"
                    and (event.get("payload") or {}).get("source")=="external_llm"
                    and (event.get("payload") or {}).get("response_digest")==response_digest
                ),None) if response_digest else None
                if already_applied:
                    st.success(f"이 외부 검증 응답은 원고 v{already_applied.get('result_version')}에 이미 반영되었습니다.")
                    _render_review_result(already_applied,latest=True)
                    st.caption("같은 응답은 다시 반영할 수 없습니다. 새 검증 JSON을 입력하면 버튼이 다시 활성화됩니다.")
                if st.button("외부 검증 반영",disabled=current_version_reviewed or not response.strip() or bool(already_applied),key=f"m2-apply-external-review-{current_id}"):
                    try:
                        result_version=len(version_events)+1
                        anns=parse_review(response,manuscript); reviewed=apply_review(manuscript,anns); reviewed["version"]=result_version
                        review_result=_review_result_payload(
                            source="external_llm",before_todos=todos,annotations=anns,
                            reviewed_from_version=int(manuscript.get("version") or len(version_events)),result_version=result_version,response_text=response,
                        )
                        _record_resolved_revision_todos(current_id,todos,anns,"external_llm_review")
                        ledger.add_paper_project_event(current_id,"manuscript_version",{"manuscript":reviewed,"source":"external_review","diff":[]})
                        ledger.add_paper_project_event(current_id,"revision_review",review_result)
                        st.session_state[f"m2-review-flash-{current_id}"]=review_result
                        st.rerun()
                    except ValueError as error:st.error(str(error))
    with revise_tab:
        if not manuscript:
            st.info("초안을 먼저 생성하세요.")
        else:
            _render_revision_todo_reconciliation(
                project_id=current_id,project=project,manuscript=manuscript,events=events,
                version_events=version_events,model=model,use_ollama=use_ollama,
            )
            st.divider()
            todos=_revision_todos_with_events(manuscript,events)
            _render_revision_todo_progress(manuscript,events)
            open_count=sum(1 for item in todos if item["status"] in {"open","modified","researcher_review","reopened"})
            ready_count=sum(1 for item in todos if item["status"]=="ready")
            pending_count=sum(1 for item in todos if item["status"]=="revised_pending_review")
            timeline=_revision_todo_history_with_events(manuscript,events)
            closed_count=sum(1 for item in timeline if item.get("status") in {"resolved","obsolete","merged","split"})
            c1,c2,c3,c4=st.columns(4);c1.metric("열린 작업",open_count);c2.metric("보완 준비",ready_count);c3.metric("재검증 대기",pending_count);c4.metric("정리된 이력",closed_count)
            latest_todo_verification=next((
                event for event in reversed(events) if event.get("event_type")=="revision_todo_verification"
            ),None)
            if latest_todo_verification:
                with st.expander("최근 To-do 해결 여부 검증",expanded=False):
                    _render_paper_todo_verification_result(latest_todo_verification.get("payload") or {})
            workflow_rendered=False
            if todos:
                _render_revision_resolution_workflow(
                    project_id=current_id,project=project,manuscript=manuscript,todos=todos,events=events,
                    cards=cards,card_by_id=card_by_id,valid_card_ids=valid_ids,version_events=version_events,
                    model=model,use_ollama=use_ollama,
                )
                workflow_rendered=True
            if not todos:
                st.success("현재 Revision의 열린 To-do가 없습니다. 이는 논문 완성을 의미하지 않습니다.")
                st.caption("현재 버전을 기준본으로 남기거나, 문장 검증과 다음 Revision 제안을 통해 새 작업을 구성할 수 있습니다.")
            elif not workflow_rendered:
                st.dataframe([{
                    "우선순위":item["priority"],"상태":{"open":"Open","ready":"보완 준비","revised_pending_review":"재검증 대기"}.get(item["status"],item["status"]),
                    "문장 ID":item["sentence_id"],"대상 문장":item["sentence_text"],
                    "보완점":item["label"],"해야 할 일":item["problem"],"완료 기준":item["completion_criteria"],
                } for item in todos],use_container_width=True,hide_index=True,column_config={
                    "대상 문장":st.column_config.TextColumn(width="large"),
                    "해야 할 일":st.column_config.TextColumn(width="large"),
                    "완료 기준":st.column_config.TextColumn(width="large"),
                })
                todo_ids=[item["todo_id"] for item in todos]
                todo_id=st.selectbox("수행할 To-do",todo_ids,key=f"m2-todo-{current_id}",format_func=lambda value:next(f"[{item['priority']}] {item['label']} · {item['sentence_text'][:75]}" for item in todos if item["todo_id"]==value))
                todo=next(item for item in todos if item["todo_id"]==todo_id)
                with st.container(border=True):
                    st.markdown(f"**{todo['priority']} · {todo['label']}** · `{todo['sentence_id']}`")
                    st.write(todo["sentence_text"])
                    st.markdown(f"**문제 설명**  \n{todo['problem']}")
                    if todo["question_for_researcher"]:st.markdown(f"**연구자에게 묻는 질문**  \n{todo['question_for_researcher']}")
                    st.markdown(f"**정보 탐색 가이드**  \n{todo['search_guide']}")
                    st.markdown(f"**완료 기준**  \n{todo['completion_criteria']}")
                st.divider()
                st.markdown("### ① 근거·연구자 정보 보완")
                requires_m1=todo.get("recommended_action")=="literature_search"
                if requires_m1:
                    st.warning("이 To-do는 문헌 근거 보완이 선행되어야 합니다. M1 탐색을 완료하고 결과 지식카드를 연결한 뒤 문장을 리비전하세요.")
                else:
                    st.caption("현재 권장 경로는 연구자 입력 또는 기존 지식카드 보완입니다. 추가 문헌이 필요하다고 판단하면 M1 탐색 요청을 사용할 수 있습니다.")
                submitted_event=None;request_status="";m1_run_complete=False
                literature_candidates=list(todo.get("literature_search_candidates") or [])
                if literature_candidates:
                    st.markdown("#### M1 문헌탐색 승인 요청")
                    st.caption("아래 내용이 그대로 M1 탐색 Intent 승인 요청에 전달됩니다.")
                    candidate_ids=[str(item["candidate_id"]) for item in literature_candidates]
                    candidate_id=(candidate_ids[0] if len(candidate_ids)==1 else st.selectbox(
                        "승인 요청 후보 선택",candidate_ids,key=f"m2-lit-candidate-{current_id}-{todo_id}",
                        format_func=lambda value:next(item.get("title") or item.get("target") for item in literature_candidates if str(item["candidate_id"])==value),
                    ))
                    candidate=next(dict(item) for item in literature_candidates if str(item["candidate_id"])==candidate_id)
                    candidate_title=st.text_input("승인 요청 제목",value=str(candidate.get("title","")),key=f"m2-lit-title-{current_id}-{todo_id}-{candidate_id}")
                    candidate_target=st.text_area("탐색 대상",value=str(candidate.get("target","")),height=90,key=f"m2-lit-target-{current_id}-{todo_id}-{candidate_id}")
                    candidate_context=st.text_area("연구 맥락",value=str(candidate.get("research_context","")),height=110,key=f"m2-lit-context-{current_id}-{todo_id}-{candidate_id}")
                    candidate_evidence=st.text_area("기대 근거",value=str(candidate.get("expected_evidence","")),height=90,key=f"m2-lit-evidence-{current_id}-{todo_id}-{candidate_id}")
                    candidate_completion=st.text_area("완료 조건",value=str(candidate.get("completion_condition","")),height=90,key=f"m2-lit-completion-{current_id}-{todo_id}-{candidate_id}")
                    submitted_event=next((event for event in reversed(events) if event.get("event_type")=="revision_literature_intent" and (event.get("payload") or {}).get("todo_id")==todo_id and (event.get("payload") or {}).get("candidate_id")==candidate_id),None)
                    if submitted_event:
                        submitted_payload=submitted_event.get("payload") or {};request=ledger.phenomenon(str(submitted_payload.get("request_id","")))
                        request_status=str((request or {}).get("status") or submitted_payload.get("status") or "approval_requested")
                        intent_id=str(submitted_payload.get("intent_id",""))
                        profile=next((item for item in ledger.search_profiles(include_deleted=True) if str(item.get("intent_id",""))==intent_id),None)
                        runs=ledger.search_runs(str(profile.get("profile_id")),limit=20) if profile else []
                        m1_run_complete=any(str(item.get("status","")) in {"completed","completed_no_candidates"} for item in runs)
                        queue_title=str((profile or {}).get("title") or submitted_payload.get("queue_title") or submitted_payload.get("title") or "M1 탐색 Intent")
                        if m1_run_complete:
                            st.success(f"M1 탐색 수행 완료 · **{queue_title}**")
                        elif profile:
                            st.success(f"M1 연구 Intent 큐 등록 완료 · **{queue_title}**")
                            st.caption(f"Intent `{intent_id}` · M1 > 연구 Intent 탐색 작업 큐에서 실행할 수 있습니다.")
                        elif request_status in {"proposed","approval_requested"}:
                            st.info(f"승인 요청 완료 · M1 승인 대기 목록의 등록 예정 제목은 **{queue_title}**입니다.")
                            st.caption("연구자 홈 > 연구자 검토·승인함에서 승인하면 같은 제목으로 M1 실행 큐에 등록됩니다.")
                        elif request_status=="approved":
                            st.warning(f"**{queue_title}** 승인 완료 · 아직 M1 실행 프로필이 확인되지 않습니다. M1 큐의 ‘새 Intent 등록’ 기능을 확인하세요.")
                        else:
                            status_label={"deferred":"보완 요청","rejected":"반려"}.get(request_status,request_status)
                            st.warning(f"M1 탐색 승인 상태 · {status_label} · **{queue_title}**")
                    edited_candidate={
                        **candidate,"title":candidate_title.strip(),"target":candidate_target.strip(),
                        "research_context":candidate_context.strip(),"expected_evidence":candidate_evidence.strip(),
                        "completion_condition":candidate_completion.strip(),
                    }
                    valid_candidate=all(len(str(edited_candidate.get(field,"")))>=3 for field in ("title","target","research_context","expected_evidence","completion_condition"))
                    if st.button(
                        "이 후보를 M1 탐색 승인 요청으로 보내기",type="primary",
                        disabled=bool(submitted_event) or not valid_candidate,
                        key=f"m2-lit-submit-{current_id}-{todo_id}-{candidate_id}",
                    ):
                        _request_paper_todo_literature_intent(current_id,project,todo,edited_candidate)
                        st.rerun()
                latest_update=todo.get("latest_update") or {}
                linked_evidence=_latest_revision_evidence_link(events,todo_id)
                if link_flash:=st.session_state.pop(f"m2-todo-link-flash-{current_id}-{todo_id}",""):
                    st.success(str(link_flash))
                todo_source_cards,todo_source_papers=_revision_todo_assets(
                    current_id,todo_id,cards,ledger.shelf_papers(),
                )
                if m1_run_complete or todo_source_cards or todo_source_papers:
                    st.markdown("#### M1 탐색 산출물 → To-do 근거 연결")
                    st.caption("이 To-do의 계보가 붙은 원문과 승인 지식카드만 표시합니다. 원문은 References 채택 여부를 별도로 결정할 수 있습니다.")
                    if not todo_source_cards and not todo_source_papers:
                        st.info("탐색은 완료됐지만 아직 이 To-do 계보의 서재함 원문 또는 승인 지식카드가 없습니다. M1에서 논문을 서재함에 추가하고 읽기·지식카드 등록을 완료하세요.")
                    else:
                        todo_paper_by_id={str(item["paper_id"]):item for item in todo_source_papers}
                        todo_card_by_id={str(item["card_id"]):item for item in todo_source_cards}
                        if todo_source_papers:
                            for paper_index,paper in enumerate(todo_source_papers):
                                source_url=str(paper.get("source_url") or "").strip()
                                paper_col,source_col=st.columns([8,1.5])
                                paper_col.markdown(f"- **{paper.get('title','제목 없음')}** · {paper.get('publication_year') or '연도 확인 필요'}")
                                if source_url:
                                    source_col.link_button("원문",source_url,key=f"m2-todo-source-{current_id}-{todo_id}-{paper_index}")
                        selected_paper_ids=st.multiselect(
                            "To-do에 연결할 원문",list(todo_paper_by_id),
                            default=[pid for pid in linked_evidence.get("paper_ids",[]) if pid in todo_paper_by_id],
                            format_func=lambda pid:todo_paper_by_id[pid].get("title",pid),
                            key=f"m2-todo-source-papers-{current_id}-{todo_id}",
                        )
                        selected_lineage_card_ids=st.multiselect(
                            "To-do에 연결할 승인 지식카드",list(todo_card_by_id),
                            default=[cid for cid in linked_evidence.get("card_ids",[]) if cid in todo_card_by_id],
                            format_func=lambda cid:todo_card_by_id[cid].get("title",cid),
                            key=f"m2-todo-source-cards-{current_id}-{todo_id}",
                        )
                        reference_paper_ids=st.multiselect(
                            "논문 References에 채택할 원문",selected_paper_ids,
                            default=[pid for pid in linked_evidence.get("reference_paper_ids",[]) if pid in selected_paper_ids],
                            format_func=lambda pid:todo_paper_by_id[pid].get("title",pid),
                            key=f"m2-todo-reference-papers-{current_id}-{todo_id}",
                            help="선택한 원문만 이 논문 프로젝트의 References에 등록합니다.",
                        )
                        if st.button(
                            "원문·지식카드를 To-do에 연결",type="primary",
                            disabled=not selected_paper_ids and not selected_lineage_card_ids,
                            key=f"m2-todo-link-evidence-{current_id}-{todo_id}",
                        ):
                            reference_ids=[]
                            for paper_id in reference_paper_ids:
                                paper=todo_paper_by_id[paper_id]
                                saved_reference=ledger.upsert_literature_reference(
                                    topic=f"{project.get('title','논문 프로젝트')} · References",
                                    research_context=f"Revision To-do {todo_id}: {todo.get('problem','')}",
                                    paper={**paper,"paper_project_id":current_id,"revision_todo_id":todo_id},
                                    labels=list(paper.get("labels",[])) or build_paper_labels(paper),
                                    status="selected",
                                )
                                reference_ids.append(str(saved_reference.get("reference_id") or ""))
                            merged_card_ids=list(dict.fromkeys([
                                *[str(item) for item in latest_update.get("selected_card_ids",[])],
                                *selected_lineage_card_ids,
                            ]))
                            link_payload={
                                "todo_id":todo_id,"sentence_id":todo.get("sentence_id",""),
                                "paper_ids":selected_paper_ids,"card_ids":selected_lineage_card_ids,
                                "reference_paper_ids":reference_paper_ids,"reference_ids":[item for item in reference_ids if item],
                            }
                            ledger.add_paper_project_event(current_id,"revision_todo_evidence_link",link_payload)
                            ledger.add_paper_project_event(current_id,"revision_todo_update",{
                                **{key:value for key,value in todo.items() if key!="latest_update"},
                                "researcher_response":str(latest_update.get("researcher_response") or ""),
                                "selected_card_ids":merged_card_ids,"status":"ready",
                            })
                            st.session_state[f"m2-todo-link-flash-{current_id}-{todo_id}"]=(
                                f"To-do에 원문 {len(selected_paper_ids)}편과 지식카드 {len(selected_lineage_card_ids)}건을 연결했고, "
                                f"이 중 원문 {len(reference_paper_ids)}편을 References에 등록했습니다."
                            )
                            st.rerun()
                response_key=f"m2-todo-response-{current_id}-{todo_id}"
                researcher_response=st.text_area("연구자의 답변·판단·추가 정보",value=str(latest_update.get("researcher_response", "")),key=response_key,height=130)
                search_context="\n".join(filter(None,[str(project.get("research_question","")),todo["sentence_text"],todo["search_guide"],researcher_response]))
                hits_key=f"m2-todo-hits-{current_id}-{todo_id}"
                if st.button("보완 지식카드 자동 탐색",key=f"m2-todo-search-{current_id}-{todo_id}"):
                    sentence_row=next((row for row in manuscript_sentences(manuscript) if row["sentence_id"]==todo["sentence_id"]),{})
                    _,todo_candidates=_discover_paper_evidence(
                        str(project.get("title","")),search_context,list(sentence_row.get("evidence_card_ids",[])),semantic,embedding_model,
                    )
                    st.session_state[hits_key]=[item["card_id"] for item in todo_candidates[:12]]
                    st.rerun()
                linked_card_ids=[str(cid) for cid in linked_evidence.get("card_ids",[]) if str(cid) in card_by_id]
                hit_ids=list(dict.fromkeys(list(st.session_state.get(hits_key,[]))+[str(cid) for cid in latest_update.get("selected_card_ids",[]) if str(cid) in card_by_id]+[str(card.get("card_id")) for card in todo_source_cards if str(card.get("card_id")) in card_by_id]+linked_card_ids))
                default_added_ids=list(dict.fromkeys([str(cid) for cid in latest_update.get("selected_card_ids",[]) if str(cid) in hit_ids]+[cid for cid in linked_card_ids if cid in hit_ids]))
                added_ids=st.multiselect("추가 근거 카드",hit_ids,default=default_added_ids,format_func=lambda x:card_by_id.get(x,{}).get("title",x),key=f"m2-todo-cards-{current_id}-{todo_id}")
                added=[card_by_id[x] for x in added_ids if x in card_by_id]
                task_payload={**{key:value for key,value in todo.items() if key!="latest_update"},"researcher_response":researcher_response,"selected_card_ids":added_ids}
                if st.button("보완 정보 저장",disabled=not researcher_response.strip() and not added_ids,key=f"m2-todo-save-{current_id}-{todo_id}"):
                    ledger.add_paper_project_event(current_id,"revision_todo_update",{**task_payload,"status":"ready"})
                    st.rerun()
                st.divider()
                st.markdown("### ② 선택 문장 리비전")
                comments=[task_payload]
                revision_prompt=short_paper_revision_prompt(project,manuscript,comments,added)
                literature_ready=not requires_m1 or ((m1_run_complete or bool(todo_source_cards)) and bool(added_ids))
                can_revise=bool(researcher_response.strip() or added_ids) and literature_ready and todo.get("status")!="revised_pending_review"
                if requires_m1 and not literature_ready:
                    st.info("리비전 대기 · M1 탐색 완료와 결과 지식카드 연결이 모두 필요합니다.")
                elif todo.get("status")=="revised_pending_review":
                    st.info("문장 리비전이 반영되었습니다. 아래 해결 여부 검증을 수행하세요.")
                if st.button("내부 LLM으로 해당 문장 리비전",type="primary",disabled=not can_revise,key=f"m2-todo-revise-{current_id}-{todo_id}"):
                    raw=llm_draft(revision_prompt,model,use_ollama,profile="writing") or ""
                    try:
                        revised,diff=_apply_targeted_paper_revision(raw,manuscript,valid_card_ids=valid_ids,version=len(version_events)+1,sentence_id=todo["sentence_id"])
                        if not diff:st.warning("수정된 문장을 찾지 못했습니다. LLM 응답 형식을 확인하세요.")
                        else:
                            ledger.add_paper_project_event(current_id,"researcher_comment",task_payload)
                            ledger.add_paper_project_event(current_id,"revision_todo_update",{**task_payload,"status":"revised_pending_review","diff":diff})
                            ledger.add_paper_project_event(current_id,"manuscript_version",{
                                "manuscript":revised,"source":"targeted_revision","diff":diff,
                                "todo_id":todo_id,"todo_label":todo["label"],"todo_priority":todo["priority"],
                            })
                            st.rerun()
                    except ValueError as error:st.error(str(error))
                with st.expander("외부 LLM으로 선택 문장만 리비전"):
                    st.caption("선택한 To-do의 대상 문장 1개만 수정합니다. 앞뒤 문장은 읽기 전용 문맥이며 논문 전체를 다시 작성하지 않습니다.")
                    st.text_area("선택 문장 리비전 프롬프트",value=revision_prompt,height=380,key=f"m2-revision-prompt-{current_id}-{todo_id}")
                    external_response=st.text_area("외부 리비전 JSON",height=220,key=f"m2-revision-response-{current_id}-{todo_id}")
                    if st.button("외부 리비전 반영",disabled=not can_revise or not external_response.strip(),key=f"m2-external-revise-{current_id}-{todo_id}"):
                        try:
                            revised,diff=_apply_targeted_paper_revision(external_response,manuscript,valid_card_ids=valid_ids,version=len(version_events)+1,sentence_id=todo["sentence_id"])
                            if not diff:st.warning("수정된 문장을 찾지 못했습니다. 외부 LLM 응답 형식을 확인하세요.")
                            else:
                                ledger.add_paper_project_event(current_id,"researcher_comment",task_payload)
                                ledger.add_paper_project_event(current_id,"revision_todo_update",{**task_payload,"status":"revised_pending_review","diff":diff})
                                ledger.add_paper_project_event(current_id,"manuscript_version",{
                                    "manuscript":revised,"source":"external_revision","diff":diff,
                                    "todo_id":todo_id,"todo_label":todo["label"],"todo_priority":todo["priority"],
                                })
                                st.rerun()
                        except ValueError as error:st.error(str(error))
                st.divider()
                st.markdown("### ③ 전체 원고 영향 검토·리비전")
                todo_references=[
                    reference for reference in project_references
                    if str((reference.get("paper") or {}).get("revision_todo_id") or "")==todo_id
                ]
                full_revision_prompt=short_paper_full_revision_prompt(
                    project,manuscript,task_payload,added,todo_references,
                )
                full_revision_version=int(latest_update.get("full_revision_applied_version") or -1)
                full_revision_done=full_revision_version==int(manuscript.get("version") or -2)
                can_full_revise=(
                    todo.get("status")=="revised_pending_review"
                    and bool(added or todo_references) and not full_revision_done
                )
                if todo.get("status")!="revised_pending_review":
                    st.caption("선택 문장 리비전 후, 새 근거가 논문의 다른 문장·결론·용어에도 영향을 주는지 검토합니다.")
                elif full_revision_done:
                    st.success(f"현재 원고 v{manuscript.get('version')}에 전체 원고 영향 검토가 반영되었습니다.")
                elif not added and not todo_references:
                    st.info("전체 원고 영향 검토에는 이 To-do에 연결된 지식카드 또는 References가 필요합니다.")
                if st.button(
                    "내부 LLM으로 전체 원고 영향 리비전",type="primary",disabled=not can_full_revise,
                    key=f"m2-todo-full-revise-{current_id}-{todo_id}",
                ):
                    raw=llm_draft(full_revision_prompt,model,use_ollama,profile="writing") or ""
                    try:
                        revised,diff=apply_revisions(raw,manuscript,valid_card_ids=valid_ids,version=len(version_events)+1)
                        if not diff:
                            current_version=int(manuscript.get("version") or len(version_events))
                            ledger.add_paper_project_event(current_id,"revision_full_impact_review",{
                                "todo_id":todo_id,"source":"internal_llm","result":"no_additional_changes",
                                "manuscript_version":current_version,
                            })
                            ledger.add_paper_project_event(current_id,"revision_todo_update",{
                                **task_payload,"status":"revised_pending_review",
                                "full_revision_applied_version":current_version,"full_revision_diff":[],
                            })
                            st.rerun()
                        else:
                            result_version=len(version_events)+1
                            ledger.add_paper_project_event(current_id,"revision_todo_update",{
                                **task_payload,"status":"revised_pending_review",
                                "full_revision_applied_version":result_version,"full_revision_diff":diff,
                            })
                            ledger.add_paper_project_event(current_id,"manuscript_version",{
                                "manuscript":revised,"source":"todo_full_revision","diff":diff,
                                "todo_id":todo_id,"todo_label":todo["label"],"todo_priority":todo["priority"],
                            })
                            st.rerun()
                    except ValueError as error:st.error(str(error))
                with st.expander("외부 LLM으로 전체 원고 영향 리비전"):
                    st.caption("새 근거가 목표 문장 외의 주장·결론·용어에 미치는 영향만 수정합니다. 문체 개선만을 위한 전체 재작성은 허용하지 않습니다.")
                    st.text_area("전체 원고 영향 리비전 프롬프트",value=full_revision_prompt,height=420,key=f"m2-full-revision-prompt-{current_id}-{todo_id}")
                    full_external_response=st.text_area("외부 전체 리비전 JSON",height=240,key=f"m2-full-revision-response-{current_id}-{todo_id}")
                    if st.button(
                        "외부 전체 리비전 반영",
                        disabled=not can_full_revise or not full_external_response.strip(),
                        key=f"m2-external-full-revise-{current_id}-{todo_id}",
                    ):
                        try:
                            revised,diff=apply_revisions(full_external_response,manuscript,valid_card_ids=valid_ids,version=len(version_events)+1)
                            if not diff:
                                current_version=int(manuscript.get("version") or len(version_events))
                                ledger.add_paper_project_event(current_id,"revision_full_impact_review",{
                                    "todo_id":todo_id,"source":"external_llm","result":"no_additional_changes",
                                    "manuscript_version":current_version,
                                })
                                ledger.add_paper_project_event(current_id,"revision_todo_update",{
                                    **task_payload,"status":"revised_pending_review",
                                    "full_revision_applied_version":current_version,"full_revision_diff":[],
                                })
                                st.rerun()
                            else:
                                result_version=len(version_events)+1
                                ledger.add_paper_project_event(current_id,"revision_todo_update",{
                                    **task_payload,"status":"revised_pending_review",
                                    "full_revision_applied_version":result_version,"full_revision_diff":diff,
                                })
                                ledger.add_paper_project_event(current_id,"manuscript_version",{
                                    "manuscript":revised,"source":"external_todo_full_revision","diff":diff,
                                    "todo_id":todo_id,"todo_label":todo["label"],"todo_priority":todo["priority"],
                                })
                                st.rerun()
                        except ValueError as error:st.error(str(error))
                st.divider()
                st.markdown("### ④ To-do 해결 여부 검증")
                current_sentence=next((row for row in manuscript_sentences(manuscript) if row.get("sentence_id")==todo["sentence_id"]),{})
                linked_ids=list(dict.fromkeys([str(cid) for cid in current_sentence.get("evidence_card_ids",[])]+added_ids))
                verification_cards=[card_by_id[cid] for cid in linked_ids if cid in card_by_id]
                verification_todo=dict(task_payload)
                verification_prompt=short_paper_todo_verification_prompt(project,verification_todo,current_sentence,verification_cards)
                impact_review_required=bool(added or todo_references)
                can_verify=(
                    todo.get("status")=="revised_pending_review"
                    and (not impact_review_required or full_revision_done)
                )
                latest_verification=next((
                    event.get("payload") or {} for event in reversed(events)
                    if event.get("event_type")=="revision_todo_verification"
                    and (event.get("payload") or {}).get("todo_id")==todo_id
                ),None)
                if latest_verification:
                    _render_paper_todo_verification_result(latest_verification)
                if not can_verify:
                    st.caption("문장 리비전과 필요한 전체 원고 영향 검토를 마치면 해결 여부 검증 Task가 활성화됩니다.")
                if st.button("내부 LLM으로 해결 여부 검증",type="primary",disabled=not can_verify,key=f"m2-todo-verify-{current_id}-{todo_id}"):
                    raw=llm_draft(verification_prompt,model,use_ollama,profile="review") or ""
                    try:
                        verification=parse_todo_verification(raw,todo)
                        _record_paper_todo_verification(current_id,verification_todo,verification,source="internal_llm")
                        st.rerun()
                    except ValueError as error:st.error(str(error))
                with st.expander("외부 LLM으로 해결 여부 검증"):
                    st.caption("수정된 한 문장이 원래 문제와 완료 기준을 충족했는지만 판정합니다. 이 Task는 문장을 다시 작성하지 않습니다.")
                    st.text_area("해결 여부 검증 프롬프트",value=verification_prompt,height=360,key=f"m2-verify-prompt-{current_id}-{todo_id}")
                    verification_response=st.text_area("외부 검증 JSON",height=220,key=f"m2-verify-response-{current_id}-{todo_id}")
                    verification_digest=hashlib.sha256(verification_response.strip().encode("utf-8")).hexdigest() if verification_response.strip() else ""
                    verification_applied=next((
                        event.get("payload") or {} for event in reversed(events)
                        if event.get("event_type")=="revision_todo_verification"
                        and (event.get("payload") or {}).get("todo_id")==todo_id
                        and (event.get("payload") or {}).get("response_digest")==verification_digest
                    ),None) if verification_digest else None
                    if verification_applied:
                        st.success("이 해결 여부 검증 응답은 이미 반영되었습니다.")
                    if st.button("외부 해결 여부 검증 반영",disabled=not can_verify or not verification_response.strip() or bool(verification_applied),key=f"m2-external-verify-{current_id}-{todo_id}"):
                        try:
                            verification=parse_todo_verification(verification_response,todo)
                            _record_paper_todo_verification(current_id,verification_todo,verification,source="external_llm",response_text=verification_response)
                            st.rerun()
                        except ValueError as error:st.error(str(error))
def m2_screen(model: str, use_ollama: bool, semantic: bool, embedding_model: str) -> None:
    render_paper_project_hub(model, use_ollama, semantic, embedding_model)
    return
    st.header(ui_text("M2 · 연구 방향·논증 자문", "M2 · Research Direction & Argument Advisory"))
    st.caption(ui_text("새 지식·연구자 논점·외부 요청을 하나의 연구 논점으로 관리합니다. 각 논점은 승인 지식에 근거한 연구 판단안, 필요한 M1 근거보강, 연구자의 결정으로 이어집니다.", "Manage new knowledge, researcher issues, and external requests as one research issue. Each issue leads to an evidence-grounded decision brief, M1 supplementation when needed, and a researcher decision."))
    pages = [
        "전체 연구 논점",
        "M1 지식 변화",
        "연구 논점 등록",
        "외부 자문 요청",
        "연구 판단안 이력",
    ]
    page = st.radio(
        ui_text("M2 작업", "M2 Work"),
        pages, horizontal=True, key="m2-work-page",
        format_func=lambda v: {
            "전체 연구 논점": ui_text("전체 연구 논점", "All Research Issues"),
            "M1 지식 변화": ui_text("M1 지식 변화", "M1 Knowledge Changes"),
            "연구 논점 등록": ui_text("연구 논점 등록", "Create Research Issue"),
            "외부 자문 요청": ui_text("외부 자문 요청", "External Advisory"),
            "연구 판단안 이력": ui_text("연구 판단안 이력", "Decision Brief History"),
        }.get(v, v),
    )
    st.divider()
    if page == "전체 연구 논점":
        _render_all_question_threads(model, use_ollama, semantic, embedding_model)
    elif page == "M1 지식 변화":
        _render_m1_new_information(model, use_ollama, semantic, embedding_model)
    elif page == "연구 논점 등록":
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
    """Synchronize every active local workspace with one shared server root."""
    with st.sidebar.expander(ui_text("작업공간 동기화", "Workspace Sync"), expanded=False):
        st.caption("한 번의 동기화로 활성 워크스페이스의 DB와 다운로드한 논문 원문을 모두 통합합니다.")
        st.caption(f"활성 워크스페이스 · {len(WORKSPACE_PROFILES)}개")
        default_server = os.environ.get("RESEARCH_FELLOW_SERVER_DIR", "")
        if not default_server:
            default_server = os.environ.get(f"RESEARCH_FELLOW_SERVER_DIR_{WORKSPACE_KEY.upper()}", "")
        if not default_server:
            legacy = os.environ.get("RESEARCH_FELLOW_SERVER_DB", "")
            if legacy:
                legacy_path = Path(legacy).expanduser()
                default_server = str(legacy_path.parent if legacy_path.suffix.lower() == ".db" else legacy_path)
        server_value = st.text_input(
            "서버 루트 디렉터리",
            value=default_server,
            placeholder="예: ~/Library/Mobile Documents/com~apple~CloudDocs/ResearchFellow",
            key="workspace-sync-server-dir",
            help="Google Drive, iCloud Drive, NAS 등에 위치한 공유 폴더입니다. 워크스페이스별 DB와 원문은 workspaces/<key>/ 아래에 보관됩니다.",
        ).strip()
        if not server_value:
            st.info("서버 루트 디렉터리를 지정하면 전체 워크스페이스 동기화가 활성화됩니다.")
            return
        try:
            server_root = Path(server_value).expanduser()
            sync = AllWorkspacesSync(WORKSPACE_PROFILES, DATA, server_root)
            st.caption(f"Server · {server_root / 'workspaces'}")
            completed = st.session_state.pop("workspace-sync-batch-result", None)
            if completed:
                st.success(completed["message"])
                st.dataframe(completed["rows"], use_container_width=True, hide_index=True)

            statuses = sync.preview()
            rows = [
                {
                    "Workspace": item.label,
                    "상태": "동기화 가능" if item.server_exists else "초기화 필요",
                    "서버로": item.upload_changes,
                    "로컬로": item.download_changes,
                    "충돌": item.conflicts,
                    "초기 데이터": item.initialization_rows,
                }
                for item in statuses
            ]
            st.dataframe(rows, use_container_width=True, hide_index=True, height=min(250, 38 + 35 * len(rows)))
            total_conflicts = sum(item.conflicts for item in statuses)
            if total_conflicts:
                st.warning(f"양쪽에서 같은 레코드를 변경한 충돌 {total_conflicts}건은 자동으로 덮어쓰지 않고 보류합니다.")
            if st.button("모든 워크스페이스 동기화", type="primary", use_container_width=True, key="workspace-sync-apply-all"):
                results = sync.apply()
                result_rows = []
                for item in results:
                    label = WORKSPACE_PROFILES[item.key].label
                    result_rows.append({
                        "Workspace": label,
                        "DB": "서버 초기화" if item.initialized else f"변경 {item.applied}건 적용",
                        "충돌": item.conflicts,
                        "원문 업로드": item.assets.uploaded,
                        "원문 다운로드": item.assets.downloaded,
                        "원문 유지": item.assets.unchanged,
                        "원문 없음": item.assets.missing,
                    })
                unresolved = sum(item.conflicts for item in results)
                st.session_state["workspace-sync-batch-result"] = {
                    "message": f"전체 동기화 완료 · {len(results)}개 Workspace · 보류 충돌 {unresolved}건",
                    "rows": result_rows,
                }
                st.rerun()
        except Exception as exc:
            st.error(f"동기화 준비 실패: {exc}")

def main() -> None:
    st.set_page_config(page_title=WORKSPACE_PROFILE.browser_title, layout="wide")
    st.markdown("""<style>
    [data-testid="stStatusWidget"], div[data-testid="stStatusWidget"], button[data-testid="stStatusWidget"], [data-testid="stToolbar"] [aria-label*="Running"] { background:#f79009 !important; color:#1f1300 !important; border-color:#f79009 !important; font-weight:700 !important; }
    [data-testid="stStatusWidget"] *, [data-testid="stToolbar"] [aria-label*="Running"] * { color:#1f1300 !important; }
    .rf-running { position: fixed; top: 0.55rem; right: 5.9rem; z-index: 999999; max-width: 30rem; padding: 0.42rem 0.75rem; border: 1px solid #b54708; border-radius: 0.45rem; background: #f79009; color: #1f1300; font-weight: 700; box-shadow: 0 2px 7px rgba(0,0,0,.22); }
    hr { border: 0 !important; border-top: 2px solid #2F80ED !important; margin: 1.45rem 0 1.15rem 0 !important; opacity: .72 !important; }
    </style>""", unsafe_allow_html=True)
    st.sidebar.title("Research Fellow")
    if workspace_restore_message := st.session_state.pop("workspace-restore-message", ""):
        st.sidebar.success(str(workspace_restore_message))
    workspace_keys = list(WORKSPACE_PROFILES)
    selected_workspace = st.sidebar.selectbox(
        ui_text("연구 작업공간", "Research workspace"),
        workspace_keys,
        index=workspace_keys.index(WORKSPACE_KEY) if WORKSPACE_KEY in workspace_keys else 0,
        format_func=lambda key: f"{WORKSPACE_PROFILES[key].short_label} · {WORKSPACE_PROFILES[key].label}",
        key="research-workspace-selector",
        help=ui_text("기능과 코드는 공유하고, DB·검색 인덱스·전문성 컨텍스트만 분리합니다. 두 브라우저 탭에서 서로 다른 ?workspace= 값을 사용하면 동시에 작업할 수 있습니다.", "The code and workflows are shared; only the DB, retrieval index, and expertise context are separated. Open different ?workspace= values in separate browser tabs to work with both at once."),
    )
    if selected_workspace != WORKSPACE_KEY:
        st.query_params["workspace"] = selected_workspace
        st.rerun()
    with st.sidebar.expander(ui_text("워크스페이스 추가·아카이브", "Add or archive workspaces"), expanded=False):
        st.caption(ui_text(
            "사용 중인 워크스페이스를 확인하고 새 공간을 만들거나 ZIP으로 아카이브할 수 있습니다.",
            "Review active workspaces, create a new one, or archive a custom workspace as ZIP.",
        ))
        st.markdown(ui_text("**현재 워크스페이스**", "**Active workspaces**"))
        for key, profile in WORKSPACE_PROFILES.items():
            kind = ui_text("기본·보호됨", "built-in · protected") if key in BUILTIN_WORKSPACE_KEYS else ui_text("사용자 생성", "custom")
            st.caption(f"• {profile.label} · `{key}` · {kind}")
        archived_bytes = st.session_state.get("workspace-archive-download-bytes")
        archived_name = str(st.session_state.get("workspace-archive-download-name") or "workspace.zip")
        if archived_bytes:
            st.success(ui_text("아카이브가 준비되었습니다. 아래 ZIP을 내려받아 보관하세요.", "The archive is ready. Download and keep the ZIP below."))
            st.download_button(
                ui_text("워크스페이스 ZIP 내려받기", "Download workspace ZIP"),
                data=archived_bytes, file_name=archived_name, mime="application/zip",
                key="download-archived-workspace", use_container_width=True,
            )
            if st.button(ui_text("다운로드 안내 닫기", "Dismiss download notice"), key="dismiss-workspace-archive-download"):
                st.session_state.pop("workspace-archive-download-bytes", None)
                st.session_state.pop("workspace-archive-download-name", None)
                st.rerun()
        st.divider()
        st.markdown(ui_text("**새 워크스페이스**", "**New workspace**"))
        with st.form("create-research-workspace"):
            new_workspace_label = st.text_input(ui_text("표시 이름", "Display name"), placeholder="예: Manufacturing AI")
            new_workspace_key = st.text_input(ui_text("영문 키", "Key"), placeholder="manufacturing_ai")
            new_workspace_purpose = st.text_area(ui_text("연구 목적·범위", "Research purpose and scope"), height=90)
            new_workspace_expertise = st.text_area(ui_text("LLM 전문성 지침 (선택)", "LLM expertise instruction (optional)"), height=110)
            create_workspace = st.form_submit_button(ui_text("새 워크스페이스 만들기", "Create workspace"))
        if create_workspace:
            try:
                if not new_workspace_label.strip() or not new_workspace_key.strip():
                    raise ValueError("표시 이름과 영문 키를 모두 입력하세요.")
                created_workspace = save_custom_workspace(
                    WORKSPACE_CONFIG, key=new_workspace_key, label=new_workspace_label,
                    purpose=new_workspace_purpose, expertise_instruction=new_workspace_expertise,
                )
                st.query_params["workspace"] = created_workspace.key
                st.rerun()
            except ValueError as error:
                st.error(str(error))
        st.divider()
        st.markdown(ui_text("**ZIP에서 다시 불러오기**", "**Reload from ZIP**"))
        restore_upload = st.file_uploader(
            ui_text("워크스페이스 아카이브 ZIP", "Workspace archive ZIP"),
            type=["zip"], key="restore-workspace-archive-upload",
        )
        if st.button(
            ui_text("ZIP 워크스페이스 불러오기", "Reload workspace from ZIP"),
            disabled=restore_upload is None, key="restore-workspace-archive",
            use_container_width=True,
        ):
            try:
                restored_profile, database_restored = restore_workspace_archive(
                    restore_upload.getvalue(), config_path=WORKSPACE_CONFIG, data_dir=DATA,
                )
                st.session_state["workspace-restore-message"] = ui_text(
                    f"{restored_profile.label}을 불러왔습니다. " + ("DB도 복원했습니다." if database_restored else "기존 로컬 DB에 다시 연결했습니다."),
                    f"Reloaded {restored_profile.label}. " + ("The database was restored." if database_restored else "Reconnected the existing local database."),
                )
                st.query_params["workspace"] = restored_profile.key
                st.rerun()
            except ValueError as error:
                st.error(str(error))
        custom_workspace_keys = [key for key in workspace_keys if key not in BUILTIN_WORKSPACE_KEYS]
        if custom_workspace_keys:
            st.divider()
            st.markdown(ui_text("**사용자 워크스페이스 아카이브**", "**Archive a custom workspace**"))
            remove_workspace_key = st.selectbox(
                ui_text("아카이브할 워크스페이스", "Workspace to archive"), custom_workspace_keys,
                format_func=lambda key: WORKSPACE_PROFILES[key].label, key="remove-research-workspace-key",
            )
            remove_confirmed = st.checkbox(ui_text("ZIP 생성 후 활성 목록에서 제거하며 로컬 DB는 보존됨을 확인", "Create a ZIP, remove it from the active list, and preserve its local database"), key="remove-research-workspace-confirm")
            if st.button(ui_text("ZIP 생성·워크스페이스 아카이브", "Create ZIP and archive workspace"), disabled=not remove_confirmed, key="remove-research-workspace"):
                try:
                    archive_profile = WORKSPACE_PROFILES[remove_workspace_key]
                    archive_bytes = build_workspace_archive(archive_profile, DATA)
                    st.session_state["workspace-archive-download-bytes"] = archive_bytes
                    st.session_state["workspace-archive-download-name"] = f"research-fellow-{remove_workspace_key}.zip"
                    delete_custom_workspace(WORKSPACE_CONFIG, remove_workspace_key)
                    if remove_workspace_key == WORKSPACE_KEY:
                        st.query_params["workspace"] = "general"
                    st.rerun()
                except ValueError as error:
                    st.error(str(error))
    st.sidebar.caption(f"{WORKSPACE_PROFILE.label} · DB · {LOCAL_DB}")
    st.sidebar.caption(ui_text(WORKSPACE_PROFILE.purpose, {
        "general": "Broad, long-term research memory across the researcher’s interests",
        "agent_development": "Specialized workspace for AI agent development, specification, workflows, memory, and evaluation",
        "vision_ai": "Specialized workspace for computer vision, multimodal AI, industrial inspection, representation learning, and deployment",
    }.get(WORKSPACE_KEY, WORKSPACE_PROFILE.purpose)))
    st.sidebar.markdown(
        ui_text("새 탭으로 열기 · ", "Open in new tab · ")
        + " · ".join(f"[{profile.short_label}](?workspace={key})" for key, profile in WORKSPACE_PROFILES.items())
    )
    response_language = st.sidebar.radio(
        ui_text("응답 언어", "Response language"), ["English", "한국어"], horizontal=True,
        key="response-language", help=ui_text("영어가 기본입니다. 한국어 선택 시 같은 DB·프롬프트·워크플로우를 유지하면서 표현 언어만 한국어로 바꿉니다.", "English is the default. Korean changes only the presentation language while keeping the same DB, prompts, and workflows.")
    )
    st.sidebar.caption(ui_text("하나의 코드베이스 · 작업공간별 연구 메모리/전문성 분리 · 언어는 출력 Injection", "One codebase · research memory/expertise separated by workspace · language via output injection"))
    render_workspace_sync()
    paper_provider = st.sidebar.radio(
        ui_text("본문 읽기·비교", "Paper reading & comparison"), ["gemini", "ollama"], horizontal=True,
        format_func={"ollama": "Ollama 로컬", "gemini": "Gemini 외부 API"}.get,
        key="llm-provider-paper-choice",
    )
    internal_provider = st.sidebar.radio(
        ui_text("내부 지식·M2 해석", "Internal knowledge & M2 interpretation"), ["ollama", "gemini"], horizontal=True,
        format_func={"ollama": "Ollama 로컬", "gemini": "Gemini 외부 API"}.get,
        key="llm-provider-internal-choice",
    )
    if "gemini" in {paper_provider, internal_provider} and not gemini_api_available():
        st.sidebar.warning("GEMINI_API_KEY가 없어 Gemini 외부 API를 선택할 수 없습니다.")
        paper_provider = "ollama" if paper_provider == "gemini" else paper_provider
        internal_provider = "ollama" if internal_provider == "gemini" else internal_provider
    st.session_state["llm-provider-paper"] = paper_provider
    st.session_state["llm-provider-internal"] = internal_provider
    model = st.sidebar.text_input(ui_text("Ollama 모델", "Ollama model"), value="gpt-oss:20b", disabled="ollama" not in {paper_provider, internal_provider})
    use_ollama = "ollama" in {paper_provider, internal_provider}
    semantic = st.sidebar.checkbox(ui_text("시드카드 임베딩 검색", "Semantic seed-card search"), value=True, help=ui_text("질문과 표현이 다른 카드도 시드 후보로 찾습니다. 사용할 수 없으면 lexical 검색으로 자동 전환됩니다.", "Finds seed cards even when their wording differs from the question. Falls back to lexical search if unavailable."))
    embedding_model = st.sidebar.text_input(ui_text("임베딩 모델", "Embedding model"), value="nomic-embed-text", disabled=not semantic)
    if use_ollama:
        connected, status = ollama_status(model)
        st.sidebar.caption(f"Ollama · {status}")
        st.sidebar.caption("첫 모델 호출 시 Ollama 모델 기본 설정을 읽습니다. 논문 읽기는 Mac 메모리 사용을 위해 8K 컨텍스트를 명시합니다.")
        if not connected:
            st.sidebar.warning("초안 없이도 P1·P2 흐름은 동작합니다.")
    if paper_provider == "gemini":
        st.sidebar.caption("본문 원문은 Gemini 외부 API로 전송됩니다.")
    pending_workspace = st.session_state.pop("_navigate_workspace", None)
    if pending_workspace:
        st.session_state["main-workspace"] = pending_workspace
    workspace_items = [
        ("연구위원 데스크", ui_text("연구위원 데스크", "Research Fellow Desk")),
        ("M1 · 문헌조사·지식화", ui_text("M1 · 문헌조사·지식화", "M1 · Literature & Knowledge")),
        ("M2 · 지식 기반 자문", ui_text("M2 · 논문 프로젝트", "M2 · Paper Projects")),
        ("논문 작업실", ui_text("논문 작업실", "Paper Workspace")),
        ("지식 베이스·운영", ui_text("지식 베이스·운영", "Knowledge Base & Operations")),
        ("개발·프롬프트", ui_text("개발·프롬프트", "Development & Prompts")),
    ]
    workspace_labels = {key: label for key, label in workspace_items}
    screen = st.sidebar.radio(
        ui_text("작업공간", "Workspace"),
        [key for key, _ in workspace_items],
        format_func=lambda value: workspace_labels[value],
        key="main-workspace",
    )
    if screen == "연구위원 데스크":
        home(model, use_ollama, semantic, embedding_model)
    elif screen == "M1 · 문헌조사·지식화":
        m1_screen(model, use_ollama, semantic, embedding_model)
    elif screen == "M2 · 지식 기반 자문":
        m2_screen(model, use_ollama, semantic, embedding_model)
    elif screen == "논문 작업실":
        render_m2_coauthor_workspace(model, use_ollama, semantic, embedding_model)
    elif screen == "지식 베이스·운영":
        overview_tab, delta_tab, manage_tab = st.tabs([ui_text("승인 지식·관계", "Approved Knowledge & Relations"), ui_text("연구 활동 Delta", "Research Activity Delta"), ui_text("지식 관리", "Knowledge Management")])
        with overview_tab:
            st.header(ui_text("지식 베이스", "Knowledge Base"))
            st.caption(ui_text("이 화면은 M1·M2가 함께 참조하는 승인 지식과 승인 관계의 읽기·관리 투영입니다.", "Read and manage the approved knowledge and relations shared by M1 and M2."))
            approved_card_count = memory.count()
            active_relations = ledger.active_knowledge_relations()
            st.metric(ui_text("승인 지식카드", "Approved knowledge cards"), approved_card_count)
            st.metric(ui_text("승인 관계", "Approved relations"), len(active_relations))
            query = st.text_input(ui_text("승인 지식 검색", "Search approved knowledge"), placeholder=ui_text("예: multi LLM design feasibility", "e.g., multi LLM design feasibility"), key="knowledge-base-query")
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
