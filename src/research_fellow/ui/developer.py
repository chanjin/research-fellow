"""Developer-only Streamlit playground for editable Jinja prompt assets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import streamlit as st

from research_fellow.application.prompt_tasks import evidence_views
from research_fellow.infrastructure.document_reader import extract_document
from research_fellow.infrastructure.prompt_templates import (
    PROMPT_CATALOG, read_prompt_template, render_prompt_source, save_prompt_template,
)
from research_fellow.infrastructure.retrieval import RetrievalResult
from research_fellow.llm import ollama_draft_result


def _retrieval_views(cards: list[dict[str, Any]]) -> list[RetrievalResult]:
    return [RetrievalResult(card, 1.0, "developer", "개발 화면에서 선택한 승인 지식") for card in cards]


def _selected_cards(cards: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    options = {f"{card['title']} [{card['card_id']}]": card for card in cards}
    selected = st.multiselect("프롬프트에 제공할 승인 지식", list(options), key=key)
    return [options[label] for label in selected]


def _source_candidates(value: str) -> list[dict[str, Any]]:
    if not value.strip():
        return []
    parsed = json.loads(value)
    if not isinstance(parsed, list) or not all(isinstance(item, dict) for item in parsed):
        raise ValueError("문헌 후보 JSON은 객체 목록이어야 합니다.")
    return parsed


def _build_context(template_name: str, cards: list[dict[str, Any]], prefix: str, extraction_cache_dir: Path | None) -> dict[str, Any]:
    """Collect template-specific inputs; no result is written from this screen."""
    if template_name in {"m1_curation.j2", "m1_page_curation.j2"}:
        uploaded = st.file_uploader("개발용 PDF·TXT·MD", type=["pdf", "txt", "md"], key=f"{prefix}-document")
        source_kind = st.selectbox("자료 성격", ["외부 논문", "연구자의 확정 문서", "연구자의 아이디어 노트"], key=f"{prefix}-kind")
        if uploaded is None:
            raise ValueError("M1 카드 프롬프트에는 개발용 문서를 업로드하세요.")
        document = extract_document(uploaded, cache_dir=extraction_cache_dir)
        if template_name == "m1_page_curation.j2":
            return {"document_title": uploaded.name, "source_kind": source_kind, "page": document.pages[0], "max_cards": 2}
        return {"document_title": uploaded.name, "source_kind": source_kind, "pages": document.pages[:3], "max_cards": 3}

    if template_name == "m1_candidate_consolidation.j2":
        if len(cards) < 2:
            raise ValueError("통합 판단 프롬프트에는 승인 지식 카드가 두 장 이상 필요합니다.")
        selected = _selected_cards(cards, f"{prefix}-consolidation-cards")
        selected = selected[:6]
        pairs = []
        for index in range(0, len(selected) - 1, 2):
            first, second = selected[index], selected[index + 1]
            def view(card: dict[str, Any]) -> dict[str, Any]:
                return {"candidate_id": card["card_id"], "title": card["title"], "claim": card["claim"],
                        "evidence_excerpt": card["evidence_excerpt"], "evidence_pages": card["evidence_pages"],
                        "labels": card.get("labels", []), "conditions": card["conditions"], "limits": card["limits"]}
            pairs.append({"pair_id": f"pair-{index // 2 + 1:03d}", "selection_reason": "개발 화면에서 선택", "first": view(first), "second": view(second)})
        if not pairs:
            raise ValueError("두 장씩 선택한 뒤 통합 판단을 실행하세요.")
        return {"candidate_pairs": pairs}

    if template_name == "m1_relation_proposal.j2":
        if len(cards) < 2:
            raise ValueError("관계 프롬프트에는 승인 지식 카드가 두 장 이상 필요합니다.")
        card_map = {f"{card['title']} [{card['card_id']}]": card for card in cards}
        source_name = st.selectbox("출발 카드", list(card_map), key=f"{prefix}-source")
        target_options = [name for name in card_map if name != source_name]
        target_name = st.selectbox("도착 카드", target_options, key=f"{prefix}-target")
        return {"source": card_map[source_name], "target": card_map[target_name]}

    chosen = _selected_cards(cards, f"{prefix}-cards")
    question = st.text_area("연구 질문·주장·검토 주제", key=f"{prefix}-question", placeholder="예: 명세 우선 설계는 구현 품질을 높이는가?")

    if template_name == "m1_claim_verification.j2":
        return {"assertion": question, "evidence": evidence_views(chosen)}
    if template_name == "m1_gap_search_plan.j2":
        focus = st.text_input("탐색 초점", key=f"{prefix}-focus", placeholder="예: 평가 기준과 비교 연구")
        return {"knowledge_gap": question, "focus": focus, "existing_evidence": evidence_views(chosen)}
    if template_name == "m1_source_triage.j2":
        raw = st.text_area(
            "도구가 수집한 문헌 후보 JSON", key=f"{prefix}-sources",
            placeholder='[{"source_id":"arxiv:1234.5678", "title":"...", "authors":["..."], "published":"2026-01-01", "summary":"...", "url":"..."}]',
        )
        candidates = [
            {"reference": f"S{index}", "source_id": str(item.get("source_id", "")), "title": str(item.get("title", "")),
             "authors": list(item.get("authors", [])), "published": str(item.get("published", "")),
             "summary": str(item.get("summary", "")), "url": str(item.get("url", ""))}
            for index, item in enumerate(_source_candidates(raw)[:15], start=1)
        ]
        return {"knowledge_gap": question, "source_candidates": candidates}
    if template_name == "m1_lineage_review.j2":
        return {"topic": question, "knowledge": evidence_views(chosen, prefix="K", limit=10)}
    if template_name == "m1_revalidation_review.j2":
        return {"reason": question, "evidence": evidence_views(chosen)}
    if template_name == "m2_research_review.j2":
        return {"question": question, "evidence": _retrieval_views(chosen)}
    if template_name == "m2_research_direction.j2":
        from research_fellow.domain.research import ResearchState

        return {
            "state": ResearchState(question=question or "개발용 연구 질문"), "evidence": _retrieval_views(chosen),
            "recent_updates": [], "max_intents": 3,
        }
    if template_name == "m2_knowledge_update_report.j2":
        return {"research_question": question, "evidence": evidence_views(chosen)}
    if template_name == "m2_external_interpretation.j2":
        requester = st.text_input("요청자", key=f"{prefix}-requester", value="외부 요청자")
        expertise = st.text_input("M2 전문성", key=f"{prefix}-expertise", value="에이전트 공학과 연구위원 설계")
        context = st.text_area("요청 맥락·제약", key=f"{prefix}-context")
        return {"requester": requester, "expertise": expertise, "question": question, "context": context}
    if template_name == "m2_external_response.j2":
        expertise = st.text_input("M2 전문성", key=f"{prefix}-expertise", value="에이전트 공학과 연구위원 설계")
        return {"expertise": expertise, "question": question, "evidence": _retrieval_views(chosen)}
    raise ValueError(f"지원하지 않는 프롬프트입니다: {template_name}")


def render_developer_screen(
    cards: list[dict[str, Any]], model: str, use_ollama: bool, extraction_cache_dir: Path | None = None,
) -> None:
    st.header("개발 · Jinja 프롬프트 작업실")
    st.caption("여기서 저장한 템플릿은 즉시 운영 화면에도 적용됩니다. 실행 결과는 초안이며, 지식·관계·승인 상태를 변경하지 않습니다.")

    catalog = {f"{item.label} — {item.purpose}": item for item in PROMPT_CATALOG}
    selected_label = st.selectbox("프롬프트", list(catalog), key="developer-template")
    template = catalog[selected_label]
    editor_key = f"developer-editor-{template.name}"
    source = read_prompt_template(template.name)

    editor_column, run_column = st.columns([1, 1], gap="large")
    with editor_column:
        st.subheader("프롬프트 편집")
        st.caption(f"파일: `src/research_fellow/prompts/{template.name}`")
        st.text_area("전체 Jinja 템플릿", value=source, height=760, key=editor_key)
        if st.button("Jinja 문법 검증 후 저장", type="primary", key=f"save-{template.name}"):
            try:
                save_prompt_template(template.name, st.session_state[editor_key])
                st.success("저장했습니다. 다음 운영 화면 실행부터 같은 템플릿이 적용됩니다.")
            except ValueError as error:
                st.error(str(error))

    with run_column:
        st.subheader("문맥 렌더링·실행")
        st.caption("왼쪽의 현재 편집본(저장 전 변경 포함)으로 렌더링합니다. 실행은 지식·관계·승인 상태를 변경하지 않습니다.")
        try:
            context = _build_context(template.name, cards, f"developer-{template.name}", extraction_cache_dir)
            prompt = render_prompt_source(st.session_state[editor_key], **context)
            with st.expander("현재 편집본의 렌더링 결과", expanded=True):
                st.code(prompt, language="markdown")
            if st.button("Ollama로 초안 실행", type="primary", key=f"run-{template.name}"):
                result = ollama_draft_result(prompt, model, use_ollama)
                if result.ok:
                    st.text_area("LLM 초안 결과", value=result.text, height=360, key=f"output-{template.name}")
                else:
                    st.error(result.error or "Ollama 초안을 만들지 못했습니다.")
        except (ValueError, json.JSONDecodeError) as error:
            st.info(f"실행 문맥을 준비하세요: {error}")
        except Exception as error:
            st.error(f"프롬프트 렌더링에 실패했습니다: {error}")
