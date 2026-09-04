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
from research_fellow.llm import LLM_PROFILES, gemini_draft_result, llm_profile, ollama_draft_result
from research_fellow.storage import Ledger


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
    if template_name in {"m1_curation.j2", "m1_page_curation.j2", "m1_claim_discovery.j2", "m1_claim_qualification.j2"}:
        uploaded = st.file_uploader("개발용 PDF·TXT·MD", type=["pdf", "txt", "md"], key=f"{prefix}-document")
        source_kind = st.selectbox("자료 성격", ["외부 논문", "연구자의 확정 문서", "연구자의 아이디어 노트"], key=f"{prefix}-kind")
        if uploaded is None:
            raise ValueError("M1 카드 프롬프트에는 개발용 문서를 업로드하세요.")
        document = extract_document(uploaded, cache_dir=extraction_cache_dir)
        if template_name == "m1_page_curation.j2":
            return {"document_title": uploaded.name, "source_kind": source_kind, "page": document.pages[0], "max_cards": 2}
        source_text = "\n\n".join(page.text for page in document.pages)[:14_000]
        if template_name == "m1_claim_discovery.j2":
            return {"source_text": source_text, "max_claims": 10}
        if template_name == "m1_claim_qualification.j2":
            claims = [{"claim_id": "C1", "claim": "Example claim to qualify from the supplied source text."}]
            return {"document_title": uploaded.name, "source_kind": source_kind, "source_text": source_text, "claims": claims}
        return {"document_title": uploaded.name, "source_kind": source_kind, "source_text": source_text, "max_cards": 3}

    if template_name in {"m1_claim_consolidation.j2", "m1_claim_labels.j2"}:
        raw_claims = st.text_area("Candidate claims (one per line)", key=f"{prefix}-claims", placeholder="First claim\nSecond claim")
        claims = [{"claim_id": f"C{index}", "claim": value.strip()} for index, value in enumerate(raw_claims.splitlines(), start=1) if value.strip()]
        if not claims:
            raise ValueError("Enter at least one candidate claim.")
        return {"claims": claims[:10]}

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
                        "evidence_excerpt": card["evidence_excerpt"],
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

    if template_name == "m1_relation_batch_proposal.j2":
        if len(cards) < 2:
            raise ValueError("다중 관계 프롬프트에는 승인 지식 카드가 두 장 이상 필요합니다.")
        card_map = {f"{card['title']} [{card['card_id']}]": card for card in cards}
        source_name = st.selectbox("소스 카드", list(card_map), key=f"{prefix}-batch-source")
        target_options = [name for name in card_map if name != source_name]
        selected_target_names = st.multiselect(
            "타겟 카드 (최대 5개)",
            target_options,
            default=target_options[: min(5, len(target_options))],
            key=f"{prefix}-batch-targets",
        )
        if not selected_target_names:
            raise ValueError("타겟 카드를 한 장 이상 선택하세요.")
        return {"source": card_map[source_name], "targets": [card_map[name] for name in selected_target_names[:5]]}

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
    if template_name == "m1_paper_shelf_analysis.j2":
        uploaded = st.file_uploader("개발용 논문 PDF·TXT·MD", type=["pdf", "txt", "md"], key=f"{prefix}-paper")
        if uploaded is None:
            raise ValueError("서재 분석 프롬프트에는 개발용 논문을 업로드하세요.")
        document = extract_document(uploaded, cache_dir=extraction_cache_dir)
        source_text = "\n\n".join(f"[p.{page.page_number}]\n{page.text[:3000]}" for page in document.pages[:8])[:18_000]
        return {"title": document.title, "authors": document.author, "research_question": question, "source_text": source_text}
    if template_name == "m1_lineage_review.j2":
        return {"topic": question, "knowledge": evidence_views(chosen, prefix="K", limit=10)}
    if template_name == "m1_lineage_overview.j2":
        if not chosen:
            raise ValueError("계보 종합 의견 프롬프트에는 승인 지식 카드를 한 장 이상 선택하세요.")
        lineage_cards = [
            {
                "card_id": card["card_id"], "title": card["title"], "claim": card["claim"],
                "explanation": card.get("explanation", ""), "labels": card.get("labels", []),
                "concepts": card.get("concepts", []), "applies_to": card.get("applies_to", []),
                "conditions": card.get("conditions", ""), "limits": card.get("limits", ""),
            }
            for card in chosen[:20]
        ]
        return {"cards": lineage_cards, "relations": []}
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
    if template_name == "m2_meaning_summary.j2":
        if not chosen:
            raise ValueError("의미 요약 프롬프트에는 승인 지식 카드를 한 장 이상 선택하세요.")
        cards_for_summary = [
            {
                "reference": f"K1-{index}", "title": card["title"], "claim": card["claim"],
                "source": card.get("provenance", {}).get("source_name", "미상"),
                "labels": card.get("labels", []), "conditions": card.get("conditions", ""), "limits": card.get("limits", ""),
            }
            for index, card in enumerate(chosen[:6], start=1)
        ]
        return {"groups": [{"number": 1, "cards": cards_for_summary, "relations": [], "reports": []}], "unmatched_reports": []}
    if template_name == "m2_delta_meaning_summary.j2":
        if not chosen:
            raise ValueError("Delta 요약 프롬프트에는 승인 지식 카드를 한 장 이상 선택하세요.")
        return {
            "is_initial_baseline": False, "new_card_ids": {str(chosen[0]["card_id"])}, "new_relation_ids": set(),
            "cards": [
                {"card_id": str(card["card_id"]), "title": card["title"], "claim": card["claim"],
                 "source": card.get("provenance", {}).get("source_name", "미상"), "labels": card.get("labels", []),
                 "conditions": card.get("conditions", ""), "limits": card.get("limits", "")}
                for card in chosen[:6]
            ],
            "relations": [], "reports": [],
        }
    if template_name == "m2_search_keywords.j2":
        if not chosen:
            raise ValueError("탐색 키워드 프롬프트에는 승인 지식 카드를 한 장 이상 선택하세요.")
        return {"profile": {"title": "개발용 탐색 프로필", "question": question or "개발용 연구 질문", "context": "방법과 평가 기준을 함께 탐색", "keywords": chosen[0].get("labels", []) or ["research method"]}}
    if template_name == "m2_abstract_relevance.j2":
        return {
            "profile": {"title": "개발용 탐색 프로필", "question": question or "개발용 연구 질문", "context": "방법과 평가 기준을 함께 탐색", "keywords": ["research method"]},
            "candidates": [{"source_id": "arxiv:demo", "title": "Demo paper", "summary": "This abstract describes a method and an evaluation study."}],
        }
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
    cards: list[dict[str, Any]], model: str, use_ollama: bool, extraction_cache_dir: Path | None = None, ledger: Ledger | None = None,
    llm_audit_path: Path | None = None, provider: str = "ollama",
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
        profile_name = st.selectbox("LLM 호출 프로파일", list(LLM_PROFILES), index=list(LLM_PROFILES).index("m2_report" if template.name.startswith("m2_") else "p1_card_draft"))
        _, base = llm_profile(profile_name)
        overrides: dict[str, object] | None = None
        if provider == "ollama":
            with st.expander("이번 실행의 Ollama 설정", expanded=True):
                temperature = st.slider("temperature", 0.0, 1.0, float(base["temperature"]), 0.05)
                think = st.selectbox("think", ["low", "medium", "high"], index=["low", "medium", "high"].index(str(base["think"])))
                num_ctx = st.select_slider("num_ctx", options=[2048, 4096, 6144, 8192], value=int(base["num_ctx"]))
                num_predict = st.select_slider("num_predict", options=[400, 700, 900, 1400, 1600, 1800, 2400], value=int(base["num_predict"]))
                timeout_seconds = st.number_input("timeout_seconds", min_value=30, max_value=900, value=int(base["timeout_seconds"]), step=10)
                overrides = {"temperature": temperature, "think": think, "num_ctx": num_ctx, "num_predict": num_predict, "timeout_seconds": timeout_seconds}
        else:
            st.caption("Gemini 외부 API를 사용합니다. Ollama 전용 실행 설정은 적용되지 않습니다.")
        try:
            context = _build_context(template.name, cards, f"developer-{template.name}", extraction_cache_dir)
            prompt = render_prompt_source(st.session_state[editor_key], **context)
            with st.expander("현재 편집본의 렌더링 결과", expanded=True):
                prompt_size = len(prompt.encode("utf-8"))
                st.caption(f"입력 크기: {len(prompt):,}자 · {prompt_size:,} UTF-8 bytes · 추정 {((prompt_size + 3) // 4):,} tokens (정확한 토큰 수는 Ollama 응답의 prompt_eval_count 참조)")
                st.code(prompt, language="markdown")
            run_label = "Gemini로 초안 실행" if provider == "gemini" else "Ollama로 초안 실행"
            if st.button(run_label, type="primary", key=f"run-{template.name}"):
                result = gemini_draft_result(prompt, profile=profile_name) if provider == "gemini" else ollama_draft_result(prompt, model, use_ollama, profile=profile_name, overrides=overrides)
                if result.ok:
                    st.text_area("LLM 초안 결과", value=result.text, height=360, key=f"output-{template.name}")
                    st.caption(f"종료: {result.diagnostics}")
                else:
                    st.error(result.error or "LLM 초안을 만들지 못했습니다.")
        except (ValueError, json.JSONDecodeError) as error:
            st.info(f"실행 문맥을 준비하세요: {error}")
        except Exception as error:
            st.error(f"프롬프트 렌더링에 실패했습니다: {error}")
    if ledger:
        st.subheader("LLM 호출 로그 · 입력·설정·출력 재현")
        if llm_audit_path:
            st.caption(f"파일 감사 로그: `{llm_audit_path}` · 한 줄이 한 호출인 JSONL 형식입니다.")
            if llm_audit_path.exists():
                st.download_button("LLM 호출 JSONL 로그 내려받기", data=llm_audit_path.read_bytes(), file_name=llm_audit_path.name, mime="application/x-ndjson", key="download-llm-audit-jsonl")
        if st.button("SQLite·JSONL LLM 호출 로그 전체 삭제", key="clear-llm-audit-logs"):
            deleted_count = ledger.clear_llm_calls()
            file_removed = False
            if llm_audit_path and llm_audit_path.exists():
                llm_audit_path.unlink()
                file_removed = True
            file_status = "JSONL 파일을 삭제했습니다" if file_removed else "삭제할 JSONL 파일은 없었습니다"
            st.success(f"SQLite 호출 로그 {deleted_count}건을 삭제했고, {file_status}. 승인 지식·관계·연구 기록은 변경되지 않습니다.")
        for call in ledger.llm_calls(limit=30):
            summary = f"{call['created_at'][:19].replace('T', ' ')} · {call['profile_name']} · {call['model']} · {call['diagnostics'].get('done_reason', 'error')}"
            with st.expander(summary):
                st.json({"settings": call["settings"], "diagnostics": call["diagnostics"], "error": call["error"]})
                diagnostics = call["diagnostics"]
                request_bytes = diagnostics.get("request_prompt_utf8_bytes")
                response_bytes = diagnostics.get("response_utf8_bytes")
                if request_bytes is not None:
                    st.caption(f"입력: {diagnostics.get('request_prompt_chars', 0):,}자 · {request_bytes:,} bytes · 추정 {diagnostics.get('request_prompt_estimated_tokens', 0):,} tokens · Ollama 실제 {diagnostics.get('prompt_eval_count', '미상')} tokens")
                if response_bytes is not None:
                    st.caption(f"출력: {diagnostics.get('response_chars', 0):,}자 · {response_bytes:,} bytes · 추정 {diagnostics.get('response_estimated_tokens', 0):,} tokens · Ollama 실제 {diagnostics.get('eval_count', '미상')} tokens")
                st.markdown("**입력 프롬프트**")
                st.code(call["prompt"], language="markdown")
                st.markdown("**출력**")
                st.code(call["response"] or "(응답 없음)", language="markdown")
