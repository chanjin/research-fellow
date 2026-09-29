"""End-to-end execution of M1 auto literature review through the AJD workflow DSL."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from research_fellow.application.dsl import WorkflowDefinition, prepare_workflow_run, workflow_result
from research_fellow.application.llm_retry import LLMRetryExhausted
from research_fellow.application.llm_execution import execute_llm_stage
from research_fellow.application.paper_batch import process_top_papers
from research_fellow.application.run_tracking import ExecutionRunTracker
from research_fellow.application.search_profiles import (
    auto_search_strategy_prompt,
    keyword_prompt,
    parse_auto_search_strategy,
    parse_keyword_plan,
    run_profile,
)
from research_fellow.services import complete_intent
from research_fellow.storage import Ledger

Draft = Callable[[str], str | None]
WORKFLOW_PATH = "m1/auto_literature_review.yaml"


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
        "## 자동 영문 검색전략",
    ]
    for group in profile.get("concept_groups", []):
        lines.append(f"- {group.get('concept', '')}: " + " OR ".join(group.get("terms", [])))
    for index, query in enumerate(profile.get("boolean_queries", []), 1):
        lines.append(f"- Q{index}: `{query}`")
    lines.extend(["", "## 상위 논문"])
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
    }, globals())
    context = workflow.context

    return workflow.execute(recover=[
        (LLMRetryExhausted, _recover_retry_exhausted),
        (Exception, _recover_unexpected_error),
    ])


def _plan_literature_search(context: dict[str, Any]) -> None:
    ledger: Ledger = context["ledger"]
    intent_event = context["curation_intent"]
    intent = dict(intent_event.get("payload") or {})
    intent_id = str(intent.get("intent_id") or intent_event.get("subject_id") or "")
    profiles = [item for item in ledger.search_profiles(include_deleted=True) if item.get("intent_id") == intent_id]
    profile = profiles[0] if profiles else ledger.create_search_profile(intent)
    tracker = ExecutionRunTracker.start(ledger, intent_id=intent_id, stage="search_strategy")
    context.update({
        "intent_id": intent_id, "search_profile": profile,
        "auto_run_id": tracker.run_id, "run_tracker": tracker,
    })

    profile = context["search_profile"]
    strategy_result = execute_llm_stage(
        context["keyword_drafter"],
        auto_search_strategy_prompt(profile),
        stage="search_strategy",
        parser=parse_auto_search_strategy,
        accept=lambda value: bool(value.get("phrases") or value.get("queries")),
    )
    tracker.enter("search_strategy", retry_count=strategy_result.attempts - 1)
    strategy = strategy_result.value
    phrases = strategy["phrases"]
    core_terms = strategy["core_terms"]
    if not phrases and not strategy["queries"]:
        legacy_result = execute_llm_stage(
            context["keyword_drafter"],
            keyword_prompt(profile),
            stage="search_strategy_legacy",
            parser=parse_keyword_plan,
            accept=lambda value: bool(value[0]),
        )
        phrases, core_terms = legacy_result.value
        strategy = {**strategy, "phrases": phrases, "core_terms": core_terms, "queries": []}
    if not phrases and not strategy["queries"]:
        raise ValueError("자동 탐색용 영문 검색전략을 생성하지 못했습니다.")
    if phrases:
        ledger.update_search_profile(
            profile["profile_id"], context=profile.get("context", ""), keywords=phrases,
            core_terms=core_terms, cadence=profile.get("cadence", "manual"), is_active=True,
        )
        profile = next(item for item in ledger.search_profiles(include_deleted=True) if item["profile_id"] == profile["profile_id"])
    profile = {**profile, "concept_groups": strategy["concept_groups"], "boolean_queries": strategy["queries"]}
    context["search_profile"] = profile
    context["search_strategy"] = {
        "concept_groups": profile.get("concept_groups", []),
        "boolean_queries": profile.get("boolean_queries", []),
        "keywords": profile.get("keywords", []),
    }

def _record_failure(context: dict[str, Any], error: LLMRetryExhausted, stage: str, extra: dict[str, Any] | None = None) -> None:
    tracker: ExecutionRunTracker = context["run_tracker"]
    tracker.fail_from_llm(
        error, stage=stage,
        context={"kind": "auto_literature", "parent_run_id": context.get("parent_run_id", ""), **(extra or {})},
        intent_id=context["intent_id"],
    )




def _discover_and_screen_literature(context: dict[str, Any]) -> None:
    ledger: Ledger = context["ledger"]
    context["run_tracker"].enter("abstract_screening")

    def retrying_abstract_reviewer(prompt: str) -> str:
        return execute_llm_stage(
            context["abstract_reviewer"],
            prompt,
            stage="abstract_screening",
            parser=lambda value: value.strip(),
            accept=lambda value: len(value) >= 20,
        ).value

    outcome = run_profile(ledger, context["search_profile"], "auto", reviewer=retrying_abstract_reviewer)
    context["search_outcome"] = outcome
    context["search_succeeded"] = outcome.get("status") == "completed"
    context["search_failed"] = not context["search_succeeded"]


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
    context["run_tracker"].enter("fulltext_review")

    def retrying_fulltext_drafter(prompt: str) -> str:
        try:
            return execute_llm_stage(
                context["fulltext_drafter"],
                prompt,
                stage="fulltext_review",
                parser=lambda value: value.strip(),
                accept=lambda value: len(value) >= 20,
            ).value
        except LLMRetryExhausted as error:
            _record_failure(context, error, "fulltext_review", {"retry_hint": "실패한 논문만 다시 본문 검토"})
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


def _synthesize_literature_report(context: dict[str, Any]) -> None:
    ledger: Ledger = context["ledger"]
    tracker: ExecutionRunTracker = context["run_tracker"]
    tracker.enter("synthesis")
    try:
        synthesis_result = execute_llm_stage(
            context["synthesis_drafter"],
            literature_synthesis_prompt(context["search_profile"], context["report_run"], context["reviewed_papers"]),
            stage="synthesis",
            parser=lambda value: value.strip(),
            accept=lambda value: len(value) >= 20,
        )
        synthesis = synthesis_result.value
        tracker.note_attempts(synthesis_result.attempts)
    except LLMRetryExhausted as error:
        _record_failure(context, error, "synthesis", {"search_run_id": context["search_outcome"].get("run_id", "")})
        synthesis = deterministic_literature_report(context["search_profile"], context["report_run"], context["reviewed_papers"])
    context["synthesis"] = synthesis


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
    tracker: ExecutionRunTracker = context["run_tracker"]
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


def _recover_retry_exhausted(
    context: dict[str, Any], error: Exception, definition: WorkflowDefinition,
) -> dict[str, Any]:
    assert isinstance(error, LLMRetryExhausted)
    ledger: Ledger = context["ledger"]
    if context.get("auto_run_id"):
        _record_failure(context, error, error.stage)
    intent_event = context["curation_intent"]
    if intent_event.get("status") == "ready" or ledger.phenomenon(intent_event["phenomenon_id"]).get("status") == "ready":
        ledger.transition(intent_event["phenomenon_id"], "ready", "failed")
    profile = context.get("search_profile") or {}
    report = f"자동 문헌탐색 중 LLM Retry가 모두 실패했습니다: {error.message}"
    ledger.record(
        intent_event["case_id"], "advice_report", "m1", ["researcher", "m2"], "auto_literature_report",
        {"title": f"M1 자동 문헌탐색 주의 필요 · {profile.get('title', '')}", "report": report,
         "intent_id": context.get("intent_id", ""), "auto_mode": True,
         "retry_run_id": context.get("auto_run_id", ""), "needs_attention": True},
        subject_id=context.get("intent_id", ""), status="failed",
    )
    return workflow_result(
        definition, context, status="needs_attention",
        values={
            "report": report, "run": {}, "papers": [],
            "retry_run_id": context.get("auto_run_id", ""),
        },
    )


def _recover_unexpected_error(
    context: dict[str, Any], error: Exception, definition: WorkflowDefinition,
) -> dict[str, Any]:
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
    return workflow_result(
        definition, context, status="failed",
        values={"report": report, "run": {}, "papers": []},
    )
