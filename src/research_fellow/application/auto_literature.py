"""End-to-end execution of M1 auto literature review through the AJD workflow DSL."""
from __future__ import annotations

import json
import re

from pathlib import Path
import hashlib
import json
from typing import Any, Callable

from research_fellow.application.dsl import prepare_workflow_run
from research_fellow.application.llm_retry import LLMRetryExhausted
from research_fellow.application.llm_execution import execute_llm_stage
from research_fellow.application.paper_batch import process_top_papers
from research_fellow.application.run_tracking import ExecutionRunTracker
from research_fellow.application.search_profile_strategy import auto_search_strategy_prompt
from research_fellow.application.literature_discovery_parsers import parse_external_literature_results
from research_fellow.application.literature_discovery_formats import _json_payload
from research_fellow.services import complete_intent
from research_fellow.domain.knowledge import KnowledgeCard
import uuid
from research_fellow.storage import Ledger
from research_fellow.infrastructure.prompt_renderer import apply_review_language_policy

Draft = Callable[[str], str | None]
WORKFLOW_PATH = "m1/auto_literature_review.yaml"


def _run_tracker(context: dict[str, Any]) -> ExecutionRunTracker:
    tracker = context.get("run_tracker")
    if isinstance(tracker, ExecutionRunTracker):
        return tracker
    run_id = str(context.get("auto_run_id") or "").strip()
    if not run_id:
        raise ValueError("M1 workflow is missing auto_run_id for execution tracking")
    tracker = ExecutionRunTracker(context["ledger"], run_id)
    context["run_tracker"] = tracker
    return tracker


def selected_paper_review_prompt(profile: dict[str, Any], papers: list[dict[str, Any]]) -> str:
    payload = []
    full_text_sections: list[str] = []
    for paper in papers:
        payload.append({
            "source_id": paper.get("source_id", ""),
            "title": paper.get("title", ""),
            "authors": paper.get("authors", []),
            "year": paper.get("published") or paper.get("publication_year") or "",
            "local_pdf_filename": paper.get("local_pdf_filename") or "",
            "discovery_summary": paper.get("summary") or "",
            "why_relevant": paper.get("why_relevant") or "",
        })
        full_text = str(paper.get("full_text_content") or "").strip()
        if full_text:
            source_id = str(paper.get("source_id") or "")
            title = str(paper.get("title") or "Untitled paper")
            note = str(paper.get("full_text_extraction_note") or "").strip()
            header = f"SOURCE_ID: {source_id}\nTITLE: {title}"
            if note:
                header += f"\nEXTRACTION NOTE: {note}"
            full_text_sections.append(
                f"--- PAPER FULL TEXT START ---\n{header}\n\n{full_text}\n--- PAPER FULL TEXT END ---"
            )
    full_text_context = "\n\n".join(full_text_sections).strip()
    if not full_text_context:
        raise ValueError(
            "Paper Review에는 추출된 원문 텍스트가 필요합니다. "
            "원문 HTML/PDF를 먼저 추출하거나 로컬 PDF를 연결해 주세요."
        )
    return apply_review_language_policy(
        "You are a senior researcher in computer science/AI with rigorous analytical ability. "
        "You are assisting a researcher with the first review of papers selected for one research question.\n\n"
        f"RESEARCH QUESTION\n{profile.get('question', '')}\n\n"
        f"RESEARCH CONTEXT\n{profile.get('context', '')}\n\n"
        f"SELECTED PAPERS\n{json.dumps(payload, ensure_ascii=False, indent=2)}\n\n"
        + ((f"PAPER FULL TEXT\n{full_text_context}\n\n") if full_text_context else "")
        + "TASK\n"
        "For each selected paper, prepare a source-grounded first review. "
        "Use the supplied PAPER FULL TEXT block as the paper-content source of truth. "
        "External paper URLs are intentionally not supplied to this review step. Do not use external web access or model prior knowledge to fill gaps in the supplied text. "
        "Use the supplied metadata only to identify the paper, never as a substitute for full text.\n"
        "Do not invent claims, mechanisms, limitations, numbers, or missing sections that are not supported by the supplied PAPER FULL TEXT. "
        "If a detail cannot be verified from the supplied text, write '논문에서 확인되지 않음'.\n"
        "Create an Executive 1-Page Summary that can be scanned within one A4 page. Exclude generic background, rhetorical modifiers, and filler. "
        "Compress around technical facts, causal relationships, concrete mechanisms, and quantitative results. "
        "The summary is NOT a generic paper summary. It is a research-question-conditioned interpretation. Every section must prioritize what this paper contributes to answering the RESEARCH QUESTION above, while faithfully distinguishing the paper's own claims from implications for the researcher's question.\n"
        "If the paper only provides indirect evidence for the research question, say so explicitly. Do not make the paper appear to answer the research question more directly than it does.\n"
        "Write all researcher-facing prose in Korean. Keep source_id, short taxonomy/type values, paper/model/method names, DOI/arXiv IDs, and URLs in English/original form.\n"
        "For unavailable or unsupported information, explicitly write '논문에서 확인되지 않음' rather than guessing.\n"
        "For every candidate knowledge claim, propose a concise title in Korean. The title should be a short, review-friendly noun phrase or proposition label that captures the knowledge, not a truncated copy of the claim.\n"
        "Write each candidate knowledge claim as a self-contained 2-3 sentence knowledge unit that can be understood without reopening the source paper. The claim must name the relevant subject/context, state the substantive relationship, mechanism, or finding, and include the most important boundary or interpretation needed to avoid ambiguity. Do not use dangling references such as this paper, this model, the above result, or it unless the referent is explicitly named in the same claim. Keep evidence details and verbatim quotations in evidence/source_quotes rather than bloating the claim.\n"
        "For every candidate knowledge claim, also extract up to three verbatim sentences from the accessible paper full text that directly support or qualify that claim. Put them in source_quotes. Preserve the original language and wording, but remove double-quotation mark characters (straight or curly) from source_quotes before returning JSON so pasted output remains parse-safe. Do not paraphrase inside source_quotes, and never invent a quote. If full text is unavailable or no sentence can be verified, return an empty source_quotes array.\n\n"
        "EXECUTIVE SUMMARY CONTENT\n"
        "1. Bottom Line: one or two sentences answering: 'What does this paper tell us about the RESEARCH QUESTION?' State whether the evidence is direct or indirect.\n"
        "2. Problem & RQ: first restate the researcher's RESEARCH QUESTION lens, then summarize 1-2 critical prior limitations and the paper's own research question.\n"
        "3. Key Mechanism: explain only the mechanisms needed to understand why the paper matters for the RESEARCH QUESTION; include a 2-3 step input -> processing/reasoning loop -> output/verification flow.\n"
        "4. Contributions & Results: select up to three contributions and quantitative results that are most informative for answering or constraining the RESEARCH QUESTION, not merely the paper's most prominent headline results.\n"
        "5. Limitations & Boundaries: emphasize conditions under which the paper's findings should NOT be transferred to the RESEARCH QUESTION, plus engineering trade-offs relevant to applying the result.\n\n"
        "Return ONLY this JSON shape:\n"
        "{\n"
        "  \"papers\": [\n"
        "    {\n"
        "      \"source_id\": \"exact source_id from input\",\n"
        "      \"executive_summary\": {\n"
        "        \"research_question\": \"입력된 연구질문을 그대로 재진술\",\n"
        "        \"bottom_line\": \"이 논문이 해당 연구질문에 주는 핵심 답변 또는 근거 1~2문장; 직접/간접 근거 여부 명시\",\n"
        "        \"existing_limitations\": [\"기존 한계 1\", \"기존 한계 2\"],\n"
        "        \"paper_rq\": \"논문이 해결하고자 하는 핵심 질문\",\n"
        "        \"key_idea\": \"기존 방식과의 본질적 차이\",\n"
        "        \"mechanism_flow\": [\"입력/1단계\", \"중간 처리·추론/2단계\", \"출력·검증/3단계\"],\n"
        "        \"contributions\": [\n"
        "          {\"type\": \"Concept/Theory\", \"content\": \"기여 내용\"},\n"
        "          {\"type\": \"Architecture/Algorithm\", \"content\": \"기여 내용\"},\n"
        "          {\"type\": \"Validation/Benchmark\", \"content\": \"기여 내용\"}\n"
        "        ],\n"
        "        \"quantitative_results\": [\"베이스라인 대비 핵심 정량 결과\"],\n"
        "        \"limitations\": [\"전제 조건 또는 실패 경계\"],\n"
        "        \"engineering_tradeoffs\": [\"컴퓨팅·지연·비용·구현 복잡도 등의 트레이드오프\"]\n"
        "      },\n"
        "      \"claims\": [\n"
        "        {\"title\": \"concise knowledge-card title\", \"claim\": \"candidate knowledge claim\", \"evidence\": \"supporting evidence or clearly attributed result\", \"source_quotes\": [\"verbatim supporting sentence 1\", \"verbatim supporting sentence 2\", \"verbatim supporting sentence 3\"], \"limits\": \"conditions/limitations\"}\n"
        "      ],\n"
        "      \"review_note\": \"원문 접근성·근거 품질 등 리뷰 주의사항\"\n"
        "    }\n"
        "  ]\n"
        "}\n"
        "Do not add a separate generic summary outside executive_summary."
    )


def _summary_text(value: Any, fallback: str = "논문에서 확인되지 않음") -> str:
    text = str(value or "").strip()
    return text or fallback


def _summary_items(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value or "").strip()
    return [text] if text else []


def _render_executive_one_page_summary(title: str, executive: dict[str, Any], research_question: str = "") -> str:
    limitations = _summary_items(executive.get("existing_limitations")) or ["논문에서 확인되지 않음"]
    flow = _summary_items(executive.get("mechanism_flow")) or ["논문에서 확인되지 않음"]
    quantitative = _summary_items(executive.get("quantitative_results")) or ["논문에서 확인되지 않음"]
    boundaries = _summary_items(executive.get("limitations")) or ["논문에서 확인되지 않음"]
    tradeoffs = _summary_items(executive.get("engineering_tradeoffs")) or ["논문에서 확인되지 않음"]

    contributions: list[str] = []
    for item in executive.get("contributions") or []:
        if isinstance(item, dict):
            kind = str(item.get("type") or "Contribution").strip()
            content = str(item.get("content") or "").strip()
            if content:
                contributions.append(f"**{kind}:** {content}")
        elif str(item).strip():
            contributions.append(str(item).strip())
    if not contributions:
        contributions = ["논문에서 확인되지 않음"]

    rq_text = _summary_text(research_question or executive.get("research_question"), "연구질문 정보 없음")
    lines = [
        f"**{_summary_text(title, 'Untitled paper')} — 연구질문 관점 1-Page Summary**",
        "",
        f"**연구질문:** {rq_text}",
        "",
        "**1. 핵심 결론 (Bottom Line)**",
        f"* **핵심 명제:** {_summary_text(executive.get('bottom_line'))}",
        "",
        "**2. 배경과 문제 정의 (Problem & RQ)**",
        f"* **기존 한계:** {'; '.join(limitations)}",
        f"* **연구 질문(RQ):** {_summary_text(executive.get('paper_rq'))}",
        "",
        "**3. 제안 방법론 및 아키텍처 (Key Mechanism)**",
        f"* **핵심 아이디어:** {_summary_text(executive.get('key_idea'))}",
        f"* **핵심 구조/흐름:** {' → '.join(flow)}",
        "",
        "**4. 핵심 기여 & 실증 성과 (Contributions & Results)**",
        "* **주요 기여:**",
    ]
    lines.extend(f"  {idx}. {item}" for idx, item in enumerate(contributions[:3], start=1))
    lines.extend([
        f"* **정량 성과:** {'; '.join(quantitative)}",
        "",
        "**5. 한계점 및 적용 조건 (Limitations & Boundaries)**",
        f"* **전제 조건 및 한계:** {'; '.join(boundaries)}",
        f"* **공학적 트레이드오프:** {'; '.join(tradeoffs)}",
    ])
    return "\n".join(lines).strip()


def _normalize_review_source_id(value: Any) -> str:
    """Normalize DOI/arXiv identifiers for tolerant review-response matching."""
    raw = str(value or "").strip().casefold()
    if not raw:
        return ""
    # Strip common DOI wrappers while keeping the DOI payload itself.
    for prefix in (
        "https://doi.org/", "http://doi.org/",
        "https://dx.doi.org/", "http://dx.doi.org/",
        "doi:",
    ):
        if raw.startswith(prefix):
            raw = raw[len(prefix):].strip()
            break
    # Strip common arXiv wrappers and PDF suffixes.
    for prefix in (
        "https://arxiv.org/abs/", "http://arxiv.org/abs/",
        "https://arxiv.org/pdf/", "http://arxiv.org/pdf/",
        "arxiv:",
    ):
        if raw.startswith(prefix):
            raw = raw[len(prefix):].strip()
            break
    raw = raw.split("?", 1)[0].split("#", 1)[0].strip().rstrip("/")
    if raw.endswith(".pdf"):
        raw = raw[:-4]
    # arXiv version suffixes (v1, v2, ...) do not identify a different paper.
    if re.fullmatch(r"(?:\d{4}\.\d{4,5}|[a-z-]+/\d{7})(?:v\d+)?", raw):
        raw = re.sub(r"v\d+$", "", raw)
    return raw


def _parse_selected_paper_review(
    text: str,
    selected: list[dict[str, Any]],
    research_question: str = "",
    *,
    allow_legacy_summary: bool = False,
) -> list[dict[str, Any]]:
    raw = text.strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        raw = "\n".join(lines).strip()
    payload = _json_payload(raw)
    rows = payload.get("papers") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise ValueError("선택 논문 리뷰 응답에 papers 배열이 없습니다.")
    by_id: dict[str, dict[str, Any]] = {}
    for item in selected:
        normalized_id = _normalize_review_source_id(item.get("source_id"))
        if normalized_id:
            by_id[normalized_id] = dict(item)
    reviewed: list[dict[str, Any]] = []
    received_source_ids: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        source_id = str(row.get("source_id") or "")
        if source_id:
            received_source_ids.append(source_id)
        base = by_id.get(_normalize_review_source_id(source_id))
        if base is None:
            continue
        claims = []
        for claim in row.get("claims") or []:
            if not isinstance(claim, dict):
                continue
            claim_text = str(claim.get("claim") or "").strip()
            evidence = str(claim.get("evidence") or "").strip()
            if claim_text and evidence:
                source_quotes = []
                for quote in claim.get("source_quotes") or []:
                    quote_text = str(quote or "").translate(str.maketrans("", "", '"“”')).strip()
                    if quote_text and quote_text not in source_quotes:
                        source_quotes.append(quote_text)
                    if len(source_quotes) >= 3:
                        break
                claims.append({
                    "title": str(claim.get("title") or "").strip(),
                    "claim": claim_text,
                    "evidence": evidence,
                    "source_quotes": source_quotes,
                    "limits": str(claim.get("limits") or "").strip(),
                })
        executive = row.get("executive_summary")
        if isinstance(executive, dict):
            summary = _render_executive_one_page_summary(
                str(base.get("title") or "Untitled paper"),
                executive,
                research_question=research_question,
            )
        elif allow_legacy_summary:
            # Transitional compatibility for historical durable paper_first_review
            # tasks. New inline literature rounds must use executive_summary.
            summary = str(row.get("paper_summary") or "").strip()
        else:
            raise ValueError(
                "현재 논문 리뷰 응답에는 executive_summary가 필요합니다. "
                "과거 paper_summary 형식은 새 문헌조사 라운드에서 사용할 수 없습니다."
            )
        if not summary:
            continue
        review_note = str(row.get("review_note") or "").strip()
        # Claims are an intermediate representation used to create Knowledge Card
        # candidates.  Do not duplicate them in the durable Paper Review body;
        # their source quotes travel with the KC evidence instead.
        review_raw = summary
        if review_note:
            review_raw += "\n\n**Review note:** " + review_note
        reviewed.append({
            **base,
            "paper_summary": summary,
            "executive_summary": dict(executive) if isinstance(executive, dict) else {},
            "review_note": review_note,
            "full_text_review": review_raw,
            "full_text_status": "completed",
            "full_text_similarity": int(base.get("relevance_score") or 0),
            "knowledge_candidates": claims,
        })
    if not reviewed:
        expected_ids = [str(item.get("source_id") or "") for item in selected if str(item.get("source_id") or "")]
        if received_source_ids and expected_ids:
            raise ValueError(
                "선택 논문 리뷰의 source_id를 현재 논문과 매칭하지 못했습니다. "
                f"응답: {received_source_ids}; 선택 논문: {expected_ids}"
            )
        raise ValueError("선택 논문 리뷰에서 유효한 결과를 찾지 못했습니다.")
    return reviewed


def literature_synthesis_prompt(profile: dict[str, Any], run: dict[str, Any], papers: list[dict[str, Any]]) -> str:
    from research_fellow.infrastructure.prompt_renderer import render_prompt
    return render_prompt(
        "m1_literature_review_synthesis.j2",
        profile=profile,
        run=run,
        papers=papers,
    )


def deterministic_literature_report(profile: dict[str, Any], run: dict[str, Any], papers: list[dict[str, Any]]) -> str:
    completed = [paper for paper in papers if paper.get("full_text_status") == "completed"]
    lines = [
        "## 자동 문헌탐색 결과",
        f"- 연구질문: {profile.get('question', '')}",
        f"- 초록 검토: {len(run.get('candidates', []))}편",
        f"- 본문 비교 완료: {len(completed)}편",
        "",
        "## 문헌 발견 방식",
        "- 외부 LLM이 승인된 연구 맥락을 바탕으로 실제 논문 후보를 직접 탐색했습니다.",
        "",
        "## 상위 논문",
    ]
    for index, paper in enumerate(sorted(completed, key=lambda item: item.get("full_text_similarity", 0), reverse=True), 1):
        citations = paper.get("citation_count")
        citation_text = "확인 불가" if citations is None else f"{citations:,}회"
        lines.extend([
            f"{index}. **{paper.get('title', '')}**",
            f"   - 연도: {str(paper.get('published', ''))[:4]} · 인용: {citation_text} · 본문 적합성: {paper.get('full_text_similarity', 0)}/100",
            f"   - {paper.get('full_text_review', '')[:700]}",
        ])
    if not completed:
        lines.append("본문 비교를 완료한 논문이 없습니다. 검색 로그에서 PDF 확보 실패 여부를 확인하세요.")
    lines.extend([
        "",
        "## 연구자 확인 필요",
        "이 보고서는 자동 탐색·초록 선별·본문 비교 결과입니다. 지식카드로 승인되기 전에는 검증된 연구 지식으로 취급하지 않습니다.",
    ])
    return "\n".join(lines)


def execute_auto_literature_review(
    ledger: Ledger,
    intent_event: dict[str, Any],
    data_dir: Path,
    *,
    keyword_drafter: Draft,
    abstract_reviewer: Draft,
    fulltext_drafter: Draft,
    synthesis_drafter: Draft,
    parent_run_id: str = "",
    resume_run_id: str = "",
    checkpoint_dir: Path | None = None,
) -> dict[str, Any]:
    """Execute one M1 Curation Intent using the YAML workflow definition.

    The YAML owns the M1 orchestration. Python handlers retain stateful ledger
    operations, parsing/validation, search execution, and deterministic fallback.
    """
    workflow = prepare_workflow_run(WORKFLOW_PATH, {
        "ledger": ledger,
        "curation_intent": intent_event,
        "data_dir": data_dir,
        "keyword_drafter": keyword_drafter,
        "abstract_reviewer": abstract_reviewer,
        "fulltext_drafter": fulltext_drafter,
        "synthesis_drafter": synthesis_drafter,
        "parent_run_id": parent_run_id,
        "resume_run_id": resume_run_id,
    }, globals())

    try:
        result = workflow.execute()
        if result.get("status") == "waiting_for_interaction" and checkpoint_dir is not None:
            from research_fellow.infrastructure.workflow_checkpoint import JsonFileCheckpointStore
            intent = dict(intent_event.get("payload") or {})
            intent_id = str(intent.get("intent_id") or intent_event.get("subject_id") or "")
            checkpoint_id = f"m1-{intent_id}"
            payload = workflow.checkpoint_payload(checkpoint_id=checkpoint_id)
            profile = dict(workflow.context.get("search_profile") or {})
            payload["resume_metadata"] = {
                "kind": "m1_literature_review",
                "intent_id": intent_id,
                "research_question": str(profile.get("question") or intent.get("question") or ""),
                "profile_id": str(profile.get("profile_id") or ""),
            }
            JsonFileCheckpointStore(checkpoint_dir).save(checkpoint_id, payload)
            # The candidate-review checkpoint is now durable, so the discovery
            # response no longer needs to act as crash/retry protection.
            tracker = workflow.context.get("run_tracker")
            if tracker is not None:
                tracker.clear_manual_override(stage="search_strategy")
            return {**result, "status": "needs_attention", "checkpoint_id": checkpoint_id}
        if result.get("status") == "completed":
            tracker = workflow.context.get("run_tracker")
            if tracker is not None:
                tracker.clear_manual_override(stage="search_strategy")
        return result
    except LLMRetryExhausted as error:
        return _handle_retry_exhausted(workflow.context, error, workflow.definition.workflow_id)
    except Exception as error:
        return _handle_unexpected_error(workflow.context, error, workflow.definition.workflow_id)


def _plan_literature_search(context: dict[str, Any]) -> None:
    ledger: Ledger = context["ledger"]
    intent_event = context["curation_intent"]
    intent = dict(intent_event.get("payload") or {})
    intent_id = str(intent.get("intent_id") or intent_event.get("subject_id") or "")
    profiles = [item for item in ledger.search_profiles(include_deleted=True) if item.get("intent_id") == intent_id]
    profile = profiles[0] if profiles else ledger.create_search_profile(intent)
    tracker = ExecutionRunTracker.start(ledger, run_id=str(context.get("resume_run_id") or ""), intent_id=intent_id, stage="search_strategy")
    context.update({
        "intent_id": intent_id, "search_profile": profile,
        "auto_run_id": tracker.run_id, "run_tracker": tracker,
    })

    profile = context["search_profile"]
    search_prompt = auto_search_strategy_prompt(profile)
    manual_search = tracker.manual_override(stage="search_strategy")
    discovery_result = execute_llm_stage(
        context["keyword_drafter"],
        search_prompt,
        stage="search_strategy",
        parser=lambda text: parse_external_literature_results(text, max_results=20),
        accept=lambda value: bool(value.get("papers")),
        manual_response=manual_search,
        invalid_manual_message="외부 LLM 문헌 탐색 응답에서 유효한 논문 후보를 찾지 못했습니다.",
    )
    # Keep a validated manual response until the workflow reaches a durable
    # next boundary.  Clearing it here can lose the pasted response if a later
    # step or checkpoint write fails, causing the same External-LLM Input to
    # reappear with an empty text area on the next rerun.
    tracker.enter("search_strategy", retry_count=discovery_result.attempts - 1)
    discovery = discovery_result.value
    candidates = list(discovery.get("papers") or [])
    if not candidates:
        raise ValueError("외부 LLM 문헌 탐색에서 유효한 논문 후보를 찾지 못했습니다.")

    # SearchProfile remains the durable Intent-scoped work item, but its keywords
    # are no longer the execution mechanism for automatic M1 discovery.
    ledger.update_search_profile_policy(
        profile["profile_id"],
        context=profile.get("context", ""),
        cadence=profile.get("cadence", "manual"),
        is_active=True,
    )
    context["search_profile"] = profile
    context["discovered_candidates"] = candidates
    context["search_strategy"] = {
        "mode": "external_llm_paper_discovery",
        "search_summary": str(discovery.get("search_summary") or "").strip(),
        "candidate_count": len(candidates),
        "sources": sorted({str(item.get("source") or "external_llm") for item in candidates}),
    }

def _preserve_selected_literature_candidates(context: dict[str, Any]) -> None:
    """Confirm the papers already preserved during inline researcher Review.

    The Review UI writes immediately so the paper never disappears between UI
    phases.  This workflow action is intentionally idempotent and only reconciles
    the selected PaperRefs with the durable shelf before report finalization.
    """
    ledger: Ledger = context["ledger"]
    selected = [dict(item) for item in (context.get("selected_papers") or [])]
    profile = dict(context.get("search_profile") or {})
    preserved: list[dict[str, Any]] = []
    selected_source_ids: set[str] = set()
    for item in selected:
        source_id = str(item.get("source_id") or "").strip()
        if source_id:
            selected_source_ids.add(source_id)
        paper_id = str(item.get("paper_id") or "").strip()
        current = ledger.shelf_paper(paper_id) if paper_id else None
        authors = item.get("authors") or []
        if isinstance(authors, str):
            authors = [x.strip() for x in authors.split(",") if x.strip()]
        year = str(item.get("publication_year") or item.get("year") or item.get("published") or "")[:4]
        abstract_url = str(item.get("abstract_url") or item.get("url") or item.get("source_url") or "").strip()
        full_text_url = str(item.get("full_text_url") or "").strip()
        pdf_url = str(item.get("pdf_url") or "").strip()
        origin_links = list(item.get("origin_links") or profile.get("origin_links") or [])
        paper = ledger.upsert_shelf_paper({
            "paper_id": paper_id,
            "title": str(item.get("title") or (current or {}).get("title") or "Untitled paper"),
            "authors": list(authors) or list((current or {}).get("authors") or []),
            "publication_year": year or str((current or {}).get("publication_year") or ""),
            "source_url": abstract_url,
            "abstract_url": abstract_url,
            "full_text_url": full_text_url,
            "pdf_url": pdf_url,
            "source_id": source_id or str((current or {}).get("source_id") or ""),
            "pdf_path": str((current or {}).get("pdf_path") or ""),
            "labels": list((current or {}).get("labels") or []) or ["M1 discovery"],
            "shelf_status": str((current or {}).get("shelf_status") or "reference"),
            "reading_status": str((current or {}).get("reading_status") or "unread"),
            "asset_type": "paper",
            "intake_source": str((current or {}).get("intake_source") or "m1_external_discovery_researcher_selected"),
            "origin_links": origin_links,
            "abstract": str(item.get("summary") or item.get("abstract") or (current or {}).get("abstract") or ""),
        })
        preserved.append(paper)
    context["preserved_papers"] = preserved
    outcome = dict(context.get("search_outcome") or {})
    candidates = list(outcome.get("candidates") or [])
    outcome["candidates"] = [
        {**item, "abstract_shortlist": True}
        for item in candidates
        if str(item.get("source_id") or "") in selected_source_ids
    ]
    context["search_outcome"] = outcome
    context["paper_selection_completed"] = True

def _record_failure(context: dict[str, Any], error: LLMRetryExhausted, stage: str, extra: dict[str, Any] | None = None) -> None:
    tracker = _run_tracker(context)
    tracker.needs_attention(
        stage=stage, error_type=error.error_type,
        error_message=error.message, retry_count=error.attempts,
    )
    tracker.record_llm_failure(
        error,
        context={"kind": "auto_literature", "parent_run_id": context.get("parent_run_id", ""), **(extra or {})},
        intent_id=context["intent_id"],
        item_key=error.item_key,
    )




def _discover_and_screen_literature(context: dict[str, Any]) -> None:
    """Persist and shortlist papers already discovered by the external LLM.

    The previous implementation generated arXiv Boolean queries and then ran an
    arXiv search + a second abstract-screening LLM pass.  In the current
    operating model the external LLM itself performs broad web-enabled paper
    discovery, so this step only normalizes that candidate set, marks the top
    papers for full-text work, and records the durable search run.
    """
    ledger: Ledger = context["ledger"]
    _run_tracker(context).enter("paper_candidate_intake")
    candidates = list(context.get("discovered_candidates") or [])
    if not candidates:
        outcome = {"run_id": "", "query": "external_llm_paper_discovery", "candidates": [], "status": "failed", "error": "외부 LLM 논문 후보가 없습니다."}
        context["search_outcome"] = outcome
        context["search_succeeded"] = False
        context["search_failed"] = True
        return

    ranked = sorted(candidates, key=lambda item: int(item.get("relevance_score") or 0), reverse=True)
    shortlist_ids = {str(item.get("source_id") or "") for item in ranked[:5]}
    enriched = []
    for item in ranked:
        score = max(0, min(int(item.get("relevance_score") or 0), 100))
        level = "high" if score >= 75 else "medium" if score >= 50 else "low"
        enriched.append({
            **item,
            "citation_count": item.get("citation_count"),
            "influential_citation_count": item.get("influential_citation_count"),
            "relevance": {
                "level": level,
                "rationale": str(item.get("why_relevant") or item.get("quick_take") or "외부 LLM 문헌탐색 후보"),
            },
            "abstract_shortlist": str(item.get("source_id") or "") in shortlist_ids,
            "evidence_status": "abstract_only_pending",
        })
    query = str((context.get("search_strategy") or {}).get("search_summary") or "external_llm_paper_discovery")
    run_id = ledger.record_search_run(context["search_profile"]["profile_id"], "auto_external_llm", query, enriched, "completed")
    outcome = {"run_id": run_id, "query": query, "candidates": enriched, "status": "completed", "error": ""}
    context["search_outcome"] = outcome
    context["search_succeeded"] = True
    context["search_failed"] = False


def _record_search_failure(context: dict[str, Any]) -> None:
    ledger: Ledger = context["ledger"]
    outcome = context["search_outcome"]
    profile = context["search_profile"]
    intent_event = context["curation_intent"]
    ledger.transition(intent_event["phenomenon_id"], "ready", "failed")
    report = f"자동 문헌탐색을 완료하지 못했습니다. {outcome.get('error') or '검색 후보가 없습니다.'}"
    ledger.record(
        intent_event["case_id"], "advice_report", "m1", ["researcher", "m2"], "auto_literature_report",
        {"title": f"M1 자동 문헌탐색 실패 · {profile.get('title', '')}", "report": report,
         "search_run_id": outcome.get("run_id", ""), "intent_id": context["intent_id"], "auto_mode": True},
        subject_id=context["intent_id"], status="failed",
    )
    context.update({"status": "failed", "report": report, "run": outcome, "papers": []})


def _review_full_texts(context: dict[str, Any]) -> None:
    ledger: Ledger = context["ledger"]
    outcome = context["search_outcome"]
    _run_tracker(context).enter("fulltext_review")

    def retrying_fulltext_drafter(prompt: str) -> str:
        try:
            item_key = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]
            tracker = _run_tracker(context)
            manual = tracker.manual_override(stage="fulltext_review", item_key=item_key)
            result = execute_llm_stage(
                context["fulltext_drafter"],
                prompt,
                stage="fulltext_review",
                parser=lambda value: value.strip(),
                accept=lambda value: len(value) >= 20,
                manual_response=manual,
                item_key=item_key,
            )
            if result.source == "manual":
                tracker.clear_manual_override(stage="fulltext_review", item_key=item_key)
            return result.value
        except LLMRetryExhausted:
            raise

    processed = process_top_papers(
        context["search_profile"], outcome["candidates"], context["data_dir"], retrying_fulltext_drafter,
        make_cards=False, review_full_text=True,
    )
    merged = {item["source_id"]: item for item in outcome["candidates"]}
    merged.update({item["source_id"]: item for item in processed})
    merged_candidates = list(merged.values())
    ledger.update_search_run_candidates(outcome["run_id"], merged_candidates)
    context["reviewed_papers"] = processed
    context["report_run"] = {**outcome, "candidates": merged_candidates}



def _prepare_selected_paper_review(context: dict[str, Any]) -> None:
    """Verify that researcher-selected papers completed inline first review."""
    selected = [dict(item) for item in (context.get("selected_papers") or [])]
    incomplete = [
        item for item in selected
        if not str(item.get("paper_id") or "").strip() or not str(item.get("paper_summary") or "").strip()
    ]
    if incomplete:
        titles = ", ".join(str(item.get("title") or "Untitled")[:60] for item in incomplete[:5])
        raise ValueError(f"Review 단계에서 논문별 요약이 완료되지 않았습니다: {titles}")
    context["inline_review_ready"] = True

def _apply_selected_paper_review(context: dict[str, Any]) -> None:
    """Collect paper-by-paper Review results already created in the UI boundary."""
    selected = [dict(item) for item in (context.get("selected_papers") or [])]
    reviewed: list[dict[str, Any]] = []
    analysis_ids: list[str] = []
    request_ids: list[str] = []
    for item in selected:
        row = dict(item)
        row.setdefault("paper_summary", str(item.get("paper_summary") or item.get("summary") or ""))
        row.setdefault("knowledge_candidates", list(item.get("knowledge_candidates") or []))
        row["full_text_status"] = "completed" if str(row.get("paper_summary") or "").strip() else "not_reviewed"
        reviewed.append(row)
        paper_id = str(row.get("paper_id") or "")
        if paper_id and str(row.get("paper_summary") or "").strip():
            analysis_ids.append(paper_id)
        request_ids.extend(str(x) for x in (row.get("knowledge_request_ids") or []) if str(x))
    context["reviewed_papers"] = reviewed
    context["paper_analysis_ids"] = analysis_ids
    context["knowledge_request_ids"] = request_ids
    outcome = dict(context.get("search_outcome") or {})
    merged = {str(item.get("source_id") or ""): dict(item) for item in (outcome.get("candidates") or [])}
    for item in reviewed:
        merged[str(item.get("source_id") or "")] = item
    merged_candidates = list(merged.values())
    context["ledger"].update_search_run_candidates(outcome["run_id"], merged_candidates)
    context["report_run"] = {**outcome, "candidates": merged_candidates}

def _synthesize_literature_report_deterministic(context: dict[str, Any]) -> None:
    _run_tracker(context).enter("synthesis")
    context["synthesis"] = deterministic_literature_report(
        context["search_profile"], context["report_run"], context["reviewed_papers"]
    )


def _knowledge_candidate_title(claim: dict[str, Any], claim_text: str) -> str:
    """Prefer the LLM-proposed review title; retain the old claim-prefix fallback."""
    proposed = str(claim.get("title") or "").strip()
    if proposed:
        return proposed[:120]
    return claim_text[:72]


def _normalized_candidate_quotes(value: object) -> list[str]:
    quotes: list[str] = []
    for item in value or []:
        text = str(item or "").strip()
        if text and text not in quotes:
            quotes.append(text)
        if len(quotes) >= 3:
            break
    return quotes


def _knowledge_claim_signature(claim: dict[str, Any]) -> tuple[str, str, str, str, tuple[str, ...], str, str, str]:
    """Content identity for a parsed Paper Review claim.

    A pending Knowledge Card decision may be reused only when it represents the
    exact current review asset.  In particular, a historical candidate without
    source_quotes must not mask a newly parsed, grounded candidate.
    """
    claim_text = str(claim.get("claim") or "").strip()
    return (
        _knowledge_candidate_title(claim, claim_text),
        claim_text,
        str(claim.get("evidence") or "").strip()[:3200],
        str(claim.get("limits") or "").strip(),
        tuple(_normalized_candidate_quotes(claim.get("source_quotes"))),
        "",  # Paper Review claims do not author KC context.
        "",  # Paper Review claims do not author KC implication.
        "",  # Paper Review claims do not author KC conditions.
    )


def _knowledge_card_signature(card: dict[str, Any]) -> tuple[str, str, str, str, tuple[str, ...], str, str, str]:
    return (
        str(card.get("title") or "").strip()[:120],
        str(card.get("claim") or "").strip(),
        str(card.get("source_excerpt") or card.get("evidence_excerpt") or "").strip()[:3200],
        str(card.get("limits") or "").strip(),
        tuple(_normalized_candidate_quotes(card.get("source_quotes"))),
        str(card.get("context") or "").strip(),
        str(card.get("implication") or "").strip(),
        str(card.get("conditions") or "").strip(),
    )


def _reconcile_pending_knowledge_requests(
    ledger: Ledger, *, intent_id: str, paper_id: str, claims: list[dict[str, Any]]
) -> tuple[dict[int, dict[str, Any]], list[dict[str, Any]]]:
    """Match only current proposed KC requests to the latest parsed claims.

    Historical approved/rejected/deferred requests never block a fresh review.
    Proposed requests whose content no longer matches the latest Paper Review are
    retained in the ledger for audit but marked ``superseded`` so Attention will
    not surface stale cards.
    """
    prior: list[dict[str, Any]] = []
    for row in ledger.phenomena(type_="decision_request", status="proposed"):
        if str(row.get("subject_type") or "") != "knowledge_card":
            continue
        payload = dict(row.get("payload") or {})
        if str(payload.get("intent_id") or "") == intent_id and str(payload.get("paper_id") or "") == paper_id:
            prior.append(row)

    matched: dict[int, dict[str, Any]] = {}
    used_request_ids: set[str] = set()
    for index, claim in enumerate(claims):
        signature = _knowledge_claim_signature(claim)
        for row in prior:
            request_id = str(row.get("phenomenon_id") or "")
            if not request_id or request_id in used_request_ids:
                continue
            payload = dict(row.get("payload") or {})
            card = payload.get("card") if isinstance(payload.get("card"), dict) else {}
            if _knowledge_card_signature(dict(card)) == signature:
                matched[index] = row
                used_request_ids.add(request_id)
                break

    stale = [row for row in prior if str(row.get("phenomenon_id") or "") not in used_request_ids]
    for row in stale:
        request_id = str(row.get("phenomenon_id") or "")
        if request_id:
            ledger.set_status(request_id, "superseded")
    return matched, stale


def _stage_literature_knowledge_candidates(context: dict[str, Any]) -> None:
    """Aggregate inline Review assets, with a legacy-safe write fallback.

    New reviews persist Summary/Knowledge candidates inside the Review boundary.
    Older checkpoints may reach this capability with reviewed_papers but without
    those writes, so we complete the missing durable state exactly once.
    """
    ledger: Ledger = context["ledger"]
    profile = dict(context.get("search_profile") or {})
    intent_id = str(context.get("intent_id") or "")
    preserved_by_source = {str(x.get("source_id") or ""): dict(x) for x in (context.get("preserved_papers") or [])}
    linked_rqs = ledger.research_questions_for_intent(intent_id)
    rq = dict(linked_rqs[0]) if linked_rqs else {}
    rq_id = str(rq.get("rq_id") or "")
    rq_text = str(rq.get("question") or profile.get("question") or "")
    analysis_ids: list[str] = []
    request_ids: list[str] = []

    for reviewed in context.get("reviewed_papers") or []:
        source_id = str(reviewed.get("source_id") or "")
        shelf = preserved_by_source.get(source_id)
        if not shelf:
            continue
        paper_id = str(shelf.get("paper_id") or "")
        summary = str(reviewed.get("paper_summary") or reviewed.get("summary") or "").strip()
        review_note = str(reviewed.get("review_note") or "").strip()
        if paper_id and rq_id and summary and not ledger.paper_question_analysis(paper_id, rq_id):
            ledger.save_paper_question_analysis(
                paper_id,
                research_question_id=rq_id,
                intent_id=intent_id,
                research_question=rq_text,
                summary=summary,
                reading_raw_output=str(reviewed.get("full_text_review") or (summary + ("\n\n**Review note:** " + review_note if review_note else ""))),
                generated=True,
            )
        elif paper_id and summary and not rq_id and not ledger.paper_analysis(paper_id):
            # Legacy/unscoped fallback only. RQ-bound reviews must not replace a
            # paper-level summary with the interpretation from the latest question.
            ledger.save_paper_analysis(
                paper_id,
                research_question=rq_text,
                summary=summary,
                reading_raw_output=str(reviewed.get("full_text_review") or (summary + ("\n\n**Review note:** " + review_note if review_note else ""))),
                generated=True,
            )
        if paper_id and summary:
            ledger.update_shelf_paper(
                paper_id,
                shelf_status=str(shelf.get("shelf_status") or "reference"),
                reading_status="read",
            )
        if paper_id and (ledger.paper_question_analysis(paper_id, rq_id) if rq_id else ledger.paper_analysis(paper_id)):
            analysis_ids.append(paper_id)

        claims = [dict(x) for x in (reviewed.get("knowledge_candidates") or []) if isinstance(x, dict)]
        matched, _stale = _reconcile_pending_knowledge_requests(
            ledger, intent_id=intent_id, paper_id=paper_id, claims=claims
        )
        for claim_index, claim in enumerate(claims):
            claim_text = str(claim.get("claim") or "").strip()
            evidence = str(claim.get("evidence") or "").strip()
            quotes = _normalized_candidate_quotes(claim.get("source_quotes"))
            if len(claim_text) < 8 or len(evidence) < 8:
                continue
            prior = matched.get(claim_index)
            if prior is not None:
                request_ids.append(str(prior.get("phenomenon_id") or ""))
                continue
            card = KnowledgeCard(
                card_id=f"kc-candidate-{uuid.uuid4().hex[:12]}",
                title=_knowledge_candidate_title(claim, claim_text), source_kind="external_paper", claim=claim_text,
                context="", implication="", source_excerpt=evidence[:3200], source_quotes=quotes,
                labels=list(profile.get("labels") or []), evidence_level="provisional", status="verified",
                evidence_excerpt=evidence[:1600], conditions="", limits=str(claim.get("limits") or ""),
                provenance={"source_name": str(shelf.get("title") or "paper"), "publication_year": str(shelf.get("publication_year") or ""), "paper_id": paper_id, "research_question_id": rq_id, "intent_id": intent_id, "grounding": "m1_first_literature_round"},
                origin_links=list(shelf.get("origin_links") or []),
            ).model_dump(mode="json")
            case_id = ledger.create_case("research", f"Literature knowledge candidate: {str(shelf.get('title') or '')[:72]}")
            rid = ledger.record(
                case_id, "decision_request", "m1", ["researcher"], "knowledge_card",
                {
                    "title": f"문헌조사 지식카드 후보 승인: {card['title']}",
                    "card": card, "paper_id": paper_id, "intent_id": intent_id, "rq_id": rq_id,
                    "publication_year": str(shelf.get("publication_year") or ""),
                    "research_question": rq_text,
                    "review_note": review_note,
                    "literature_round": "first",
                    "next_action": "승인 시 지식카드로 등록합니다.",
                },
                subject_id=card["card_id"],
            )
            request_ids.append(rid)

    context["paper_analysis_ids"] = list(dict.fromkeys(analysis_ids))
    context["knowledge_request_ids"] = list(dict.fromkeys(x for x in request_ids if x))

def _synthesize_literature_report(context: dict[str, Any]) -> None:
    tracker = _run_tracker(context)
    tracker.enter("synthesis")
    prompt = literature_synthesis_prompt(context["search_profile"], context["report_run"], context["reviewed_papers"])
    manual = tracker.manual_override(stage="synthesis")
    try:
        synthesis_result = execute_llm_stage(
            context["synthesis_drafter"], prompt, stage="synthesis",
            parser=lambda value: value.strip(), accept=lambda value: len(value) >= 20,
            manual_response=manual,
        )
        if synthesis_result.source == "manual":
            tracker.clear_manual_override(stage="synthesis")
        context["synthesis"] = synthesis_result.value
        tracker.note_attempts(synthesis_result.attempts)
    except LLMRetryExhausted as error:
        if error.error_type == "external_llm_required":
            raise
        _record_failure(context, error, "synthesis", {"search_run_id": context["search_outcome"].get("run_id", "")})
        context["synthesis"] = deterministic_literature_report(context["search_profile"], context["report_run"], context["reviewed_papers"])


def _retire_stale_search_strategy_attention(ledger: Ledger, intent_id: str, *, keep_run_id: str = "") -> None:
    """Close historical search-strategy leftovers after a literature round finishes.

    They remain as audit history but must not reappear as fresh Inputs cards.
    """
    if not intent_id:
        return
    for run in ledger.auto_research_runs(statuses=("needs_attention",), limit=200):
        run_id = str(run.get("run_id") or "").strip()
        if not run_id or run_id == keep_run_id:
            continue
        if str(run.get("intent_id") or "").strip() != intent_id:
            continue
        if str(run.get("current_stage") or "").strip() != "search_strategy":
            continue
        ledger.update_auto_research_run(
            run_id, status="completed", stage="superseded_after_literature_completion"
        )
    for failure in ledger.auto_research_failures(status="needs_attention", limit=200):
        if str(failure.get("intent_id") or "").strip() != intent_id:
            continue
        if str(failure.get("stage") or "").strip() != "search_strategy":
            continue
        failure_id = str(failure.get("failure_id") or failure.get("id") or "").strip()
        if failure_id:
            ledger.resolve_auto_research_failure(failure_id, status="superseded")

def _finalize_literature_review(context: dict[str, Any]) -> None:
    ledger: Ledger = context["ledger"]
    intent_event = context["curation_intent"]
    profile = context["search_profile"]
    processed = context["reviewed_papers"]
    completed = [item for item in processed if item.get("full_text_status") == "completed"]
    report_id = ledger.record(
        intent_event["case_id"], "advice_report", "m1", ["researcher", "m2"], "auto_literature_report",
        {
            "title": f"M1 자동 문헌탐색 보고 · {profile.get('title', '')}",
            "report": context["synthesis"],
            "intent_id": context["intent_id"],
            "search_run_id": context["search_outcome"]["run_id"],
            "auto_mode": True,
            "search_strategy": context["search_strategy"],
            "abstract_review_count": len(context["search_outcome"]["candidates"]),
            "fulltext_review_count": len(completed),
            "paper_analysis_count": len(context.get("paper_analysis_ids") or []),
            "knowledge_request_ids": list(context.get("knowledge_request_ids") or []),
            "top_papers": [
                {
                    "source_id": item.get("source_id"), "title": item.get("title"),
                    "published": item.get("published"), "citation_count": item.get("citation_count"),
                    "influential_citation_count": item.get("influential_citation_count"),
                    "full_text_similarity": item.get("full_text_similarity"), "url": item.get("url"),
                    "pdf_path": item.get("pdf_path", ""), "abstract": item.get("summary", ""),
                }
                for item in sorted(completed, key=lambda p: p.get("full_text_similarity", 0), reverse=True)[:5]
            ],
        },
        subject_id=context["intent_id"], status="completed",
    )
    context["report_id"] = report_id
    context["completed_papers"] = completed

    # Complete the business intent after publishing the report.
    tracker = _run_tracker(context)
    run_id = tracker.run_id
    unresolved = [f for f in ledger.auto_research_failures() if f.get("run_id") == run_id]
    if unresolved:
        tracker.needs_attention(stage="completed_with_attention")
        ledger.transition(context["curation_intent"]["phenomenon_id"], "ready", "failed")
        context.update({
            "status": "needs_attention",
            "report": context["synthesis"],
            "run": context["report_run"],
            "papers": context["reviewed_papers"],
            "retry_run_id": run_id,
        })
        return

    completed = context.get("completed_papers", [])
    tracker.complete()
    _retire_stale_search_strategy_attention(
        ledger, str(context.get("intent_id") or ""), keep_run_id=run_id
    )
    complete_intent(
        ledger, context["curation_intent"],
        f"초록 {len(context['search_outcome']['candidates'])}편을 검토하고 상위 {len(completed)}편의 본문을 비교하여 자동 탐색 보고서 {context['report_id']}를 생성했습니다.",
    )
    context.update({
        "status": "completed",
        "report": context["synthesis"],
        "run": context["report_run"],
        "papers": context["reviewed_papers"],
    })


def _handle_retry_exhausted(context: dict[str, Any], error: LLMRetryExhausted, workflow_id: str) -> dict[str, Any]:
    ledger: Ledger = context["ledger"]
    if context.get("auto_run_id"):
        _record_failure(context, error, error.stage)
    intent_event = context["curation_intent"]
    external_manual = error.error_type == "external_llm_required"
    if not external_manual and (intent_event.get("status") == "ready" or ledger.phenomenon(intent_event["phenomenon_id"]).get("status") == "ready"):
        ledger.transition(intent_event["phenomenon_id"], "ready", "failed")
    profile = context.get("search_profile") or {}
    if external_manual:
        return {
            "status": "needs_attention",
            "attention_kind": "external_llm_input",
            "report": f"외부 LLM 실행이 필요한 정상 입력 단계에서 대기 중입니다: {error.stage}",
            "run": {}, "papers": [], "retry_run_id": context.get("auto_run_id", ""),
            "workflow_id": workflow_id, "workflow_trace": list(context.get("workflow_trace", [])),
        }
    report = f"자동 문헌탐색 중 LLM Retry가 모두 실패했습니다: {error.message}"
    ledger.record(
        intent_event["case_id"], "advice_report", "m1", ["researcher", "m2"], "auto_literature_report",
        {"title": f"M1 자동 문헌탐색 주의 필요 · {profile.get('title', '')}", "report": report,
         "intent_id": context.get("intent_id", ""), "auto_mode": True,
         "retry_run_id": context.get("auto_run_id", ""), "needs_attention": True},
        subject_id=context.get("intent_id", ""), status="failed",
    )
    return {
        "status": "needs_attention", "report": report, "run": {}, "papers": [],
        "retry_run_id": context.get("auto_run_id", ""), "workflow_id": workflow_id,
        "workflow_trace": list(context.get("workflow_trace", [])),
    }


def _handle_unexpected_error(context: dict[str, Any], error: Exception, workflow_id: str) -> dict[str, Any]:
    ledger: Ledger = context["ledger"]
    tracker = context.get("run_tracker")
    if tracker is not None:
        tracker.needs_attention(
            stage="unexpected_error", error_type="unexpected_error", error_message=str(error),
        )
    intent_event = context["curation_intent"]
    current = ledger.phenomenon(intent_event["phenomenon_id"])
    if current and current.get("status") == "ready":
        ledger.transition(intent_event["phenomenon_id"], "ready", "failed")
    profile = context.get("search_profile") or {}
    report = f"자동 문헌탐색 중 오류가 발생했습니다: {error}"
    ledger.record(
        intent_event["case_id"], "advice_report", "m1", ["researcher", "m2"], "auto_literature_report",
        {"title": f"M1 자동 문헌탐색 실패 · {profile.get('title', '')}", "report": report,
         "intent_id": context.get("intent_id", ""), "auto_mode": True},
        subject_id=context.get("intent_id", ""), status="failed",
    )
    return {
        "status": "failed", "report": report, "run": {}, "papers": [],
        "workflow_id": workflow_id, "workflow_trace": list(context.get("workflow_trace", [])),
    }
