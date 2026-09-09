"""End-to-end execution of M2 auto-dispatched literature exploration intents."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from research_fellow.application.paper_batch import process_top_papers
from research_fellow.application.search_profiles import auto_search_strategy_prompt, keyword_prompt, parse_auto_search_strategy, parse_keyword_plan, search_profile_candidates, shortlist_candidates
from research_fellow.services import complete_intent
from research_fellow.application.llm_retry import LLMRetryExhausted, call_with_retry, failure_payload
from research_fellow.storage import Ledger

Draft = Callable[[str], str | None]


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
    resume_run_id: str = "",
    abstract_batch_size: int = 20,
) -> dict[str, Any]:
    """Run one auto Intent from planning through a researcher-facing report.

    This function is synchronous by design. It is called by an explicit auto-mode
    action and leaves every durable result in SQLite; it is not a background job.
    """
    intent = dict(intent_event.get("payload") or {})
    intent_id = str(intent.get("intent_id") or intent_event.get("subject_id") or "")
    profiles = [item for item in ledger.search_profiles(include_deleted=True) if item.get("intent_id") == intent_id]
    profile = profiles[0] if profiles else ledger.create_search_profile(intent)
    run_id = resume_run_id or ledger.create_auto_research_run(intent_id=intent_id, stage="search_strategy")
    prior_run = ledger.auto_research_run(run_id) if resume_run_id else None
    checkpoint = dict((prior_run or {}).get("checkpoint") or {})

    def record_failure(error: LLMRetryExhausted, stage: str, context: dict[str, Any] | None = None, item_key: str = "") -> None:
        ledger.update_auto_research_run(
            run_id, status="needs_attention", stage=stage,
            error_type=error.error_type, error_message=error.message, retry_count=error.attempts,
        )
        ledger.record_auto_research_failure(
            failure_payload(error, context={"kind": "auto_literature", "parent_run_id": parent_run_id, **(context or {})}),
            run_id=run_id, intent_id=intent_id, item_key=item_key,
        )

    try:
        # Resume from the last durable checkpoint whenever possible.
        strategy = checkpoint.get("search_strategy")
        if strategy:
            phrases = list(strategy.get("phrases") or [])
            core_terms = list(strategy.get("core_terms") or [])
        else:
            strategy_prompt = auto_search_strategy_prompt(profile)
            manual_strategy = ledger.manual_recovery_override(run_id, stage="search_strategy")
            if manual_strategy:
                raw_strategy, attempts = manual_strategy, 1
                parsed_manual = parse_auto_search_strategy(raw_strategy)
                if not (parsed_manual.get("phrases") or parsed_manual.get("queries")):
                    raise ValueError("수동 복구 검색전략을 파싱하지 못했습니다.")
                ledger.clear_manual_recovery_override(run_id, stage="search_strategy")
            else:
                raw_strategy, attempts = call_with_retry(
                    keyword_drafter, strategy_prompt, stage="search_strategy",
                    validator=lambda text: bool(parse_auto_search_strategy(text).get("phrases") or parse_auto_search_strategy(text).get("queries")),
                )
            ledger.update_auto_research_run(run_id, stage="search_strategy", retry_count=attempts - 1)
            strategy = parse_auto_search_strategy(raw_strategy)
            phrases = strategy["phrases"]
            core_terms = strategy["core_terms"]
            if not phrases and not strategy["queries"]:
                legacy_raw, attempts = call_with_retry(
                    keyword_drafter, keyword_prompt(profile), stage="search_strategy_legacy",
                    validator=lambda text: bool(parse_keyword_plan(text)[0]),
                )
                phrases, core_terms = parse_keyword_plan(legacy_raw)
                strategy = {**strategy, "phrases": phrases, "core_terms": core_terms, "queries": []}
            checkpoint["search_strategy"] = strategy
            ledger.update_auto_research_run(run_id, checkpoint=checkpoint, stage="search_strategy_completed")
        if not phrases and not strategy.get("queries"):
            raise ValueError("자동 탐색용 영문 검색전략을 생성하지 못했습니다.")
        if phrases:
            ledger.update_search_profile(
                profile["profile_id"], context=profile.get("context", ""), keywords=phrases,
                core_terms=core_terms, cadence=profile.get("cadence", "manual"), is_active=True,
            )
            profile = next(item for item in ledger.search_profiles(include_deleted=True) if item["profile_id"] == profile["profile_id"])
        profile = {**profile, "concept_groups": strategy.get("concept_groups", []), "boolean_queries": strategy.get("queries", [])}

        # Search is separated from abstract triage so a failed LLM batch can resume
        # without querying arXiv/Semantic Scholar again.
        raw_candidates = checkpoint.get("raw_candidates")
        query = str(checkpoint.get("search_query") or "")
        if raw_candidates is None:
            query, raw_candidates = search_profile_candidates(profile)
            checkpoint["search_query"] = query
            checkpoint["raw_candidates"] = raw_candidates
            checkpoint["abstract_reviews"] = {}
            checkpoint["abstract_last_completed_batch"] = 0
            ledger.update_auto_research_run(run_id, checkpoint=checkpoint, stage="search_completed")

        ledger.update_auto_research_run(run_id, stage="abstract_screening", checkpoint=checkpoint)
        existing_reviews = dict(checkpoint.get("abstract_reviews") or {})

        abstract_batch_cursor = int(checkpoint.get("abstract_last_completed_batch", 0)) + 1

        def retrying_abstract_reviewer(prompt: str) -> str:
            nonlocal abstract_batch_cursor
            item_key = f"batch-{abstract_batch_cursor}"
            manual_response = ledger.manual_recovery_override(run_id, stage="abstract_screening", item_key=item_key)
            if manual_response:
                text = manual_response
                ledger.clear_manual_recovery_override(run_id, stage="abstract_screening", item_key=item_key)
            else:
                text, _attempts = call_with_retry(
                    abstract_reviewer, prompt, stage="abstract_screening",
                    validator=lambda value: bool(value.strip()) and len(value.strip()) >= 20,
                )
            abstract_batch_cursor += 1
            return text

        def save_abstract_batch(batch_no: int, reviewed: list[dict[str, Any]]) -> None:
            reviews = dict(checkpoint.get("abstract_reviews") or {})
            for item in reviewed:
                reviews[str(item.get("source_id", ""))] = item
            checkpoint["abstract_reviews"] = reviews
            checkpoint["abstract_last_completed_batch"] = batch_no
            checkpoint["abstract_reviewed_count"] = len(reviews)
            ledger.update_auto_research_run(run_id, stage="abstract_screening", checkpoint=checkpoint)

        try:
            candidates = shortlist_candidates(
                profile, list(raw_candidates or []), retrying_abstract_reviewer, batch_size=abstract_batch_size,
                existing_reviews=existing_reviews, on_batch_completed=save_abstract_batch,
            )
        except LLMRetryExhausted as error:
            next_batch = int(checkpoint.get("abstract_last_completed_batch", 0)) + 1
            batch_start = (next_batch - 1) * max(5, min(int(abstract_batch_size), 25))
            batch_items = list(raw_candidates or [])[batch_start : batch_start + max(5, min(int(abstract_batch_size), 25))]
            record_failure(error, "abstract_screening", {
                "retry_hint": f"완료된 초록 batch는 유지합니다. Batch {next_batch}부터 재시도하며 반복 실패 시 batch 크기를 10편으로 줄이세요.",
                "resume_run_id": run_id, "batch_no": next_batch, "search_query": query, "batch_size": abstract_batch_size,
                "source_ids": [str(item.get("source_id", "")) for item in batch_items if item.get("source_id")],
            }, item_key=f"batch-{next_batch}")
            ledger.update_auto_research_run(run_id, checkpoint=checkpoint, status="needs_attention", stage="abstract_screening")
            ledger.transition(intent_event["phenomenon_id"], "ready", "failed")
            for rq in ledger.research_questions_for_intent(intent_id):
                ledger.add_research_question_change(str(rq.get("rq_id", "")), "exploration_failed", f"후속 문헌탐색이 초록 스크리닝 Batch {next_batch}에서 중단되어 재시도가 필요합니다.")
            report = f"초록 스크리닝 Batch {next_batch}에서 자동 Retry가 모두 실패했습니다. 앞 단계 검색 결과와 완료된 batch는 보존되었습니다."
            ledger.record(
                intent_event["case_id"], "advice_report", "m1", ["researcher", "m2"], "auto_literature_report",
                {"title": f"M1 자동 문헌탐색 주의 필요 · {profile.get('title', '')}", "report": report,
                 "intent_id": intent_id, "auto_mode": True, "retry_run_id": run_id, "needs_attention": True,
                 "abstract_review_count": len(existing_reviews), "fulltext_review_count": 0},
                subject_id=intent_id, status="failed",
            )
            return {"status": "needs_attention", "report": report, "run": {"query": query, "candidates": list(raw_candidates or [])}, "papers": [], "retry_run_id": run_id}

        search_run_id = str(checkpoint.get("search_run_id") or "")
        if search_run_id:
            ledger.update_search_run_candidates(search_run_id, candidates)
        else:
            search_run_id = ledger.record_search_run(profile["profile_id"], "auto", query, candidates, "completed" if candidates else "completed_no_candidates")
            checkpoint["search_run_id"] = search_run_id
            ledger.complete_search_profile(profile["profile_id"])
        checkpoint["screened_candidates"] = candidates
        ledger.update_auto_research_run(run_id, checkpoint=checkpoint, stage="abstract_screening_completed")
        outcome = {"run_id": search_run_id, "query": query, "candidates": candidates, "status": "completed" if candidates else "completed_no_candidates", "error": ""}
        if not candidates:
            ledger.transition(intent_event["phenomenon_id"], "ready", "failed")
            report = "자동 문헌탐색 검색 후보가 없습니다."
            return {"status": "failed", "report": report, "run": outcome, "papers": []}

        ledger.update_auto_research_run(run_id, stage="fulltext_review")
        def retrying_fulltext_drafter(prompt: str) -> str:
            try:
                manual_response = ledger.manual_recovery_override(run_id, stage="fulltext_review")
                if manual_response:
                    ledger.clear_manual_recovery_override(run_id, stage="fulltext_review")
                    return manual_response
                text, _attempts = call_with_retry(
                    fulltext_drafter, prompt, stage="fulltext_review",
                    validator=lambda value: bool(value.strip()) and len(value.strip()) >= 20,
                )
                return text
            except LLMRetryExhausted as error:
                record_failure(error, "fulltext_review", {"retry_hint": "실패한 논문만 다시 본문 검토"})
                raise
        processed = process_top_papers(
            profile, outcome["candidates"], data_dir, retrying_fulltext_drafter,
            make_cards=False, review_full_text=True,
        )
        merged = {item["source_id"]: item for item in outcome["candidates"]}
        merged.update({item["source_id"]: item for item in processed})
        merged_candidates = list(merged.values())
        ledger.update_search_run_candidates(outcome["run_id"], merged_candidates)
        checkpoint["fulltext_results"] = processed
        checkpoint["screened_candidates"] = merged_candidates
        ledger.update_auto_research_run(run_id, checkpoint=checkpoint, stage="fulltext_review_completed")
        run_for_report = {**outcome, "candidates": merged_candidates}
        ledger.update_auto_research_run(run_id, stage="synthesis")
        try:
            synthesis_prompt = literature_synthesis_prompt(profile, run_for_report, processed)
            manual_synthesis = ledger.manual_recovery_override(run_id, stage="synthesis")
            if manual_synthesis:
                synthesis, attempts = manual_synthesis, 1
                ledger.clear_manual_recovery_override(run_id, stage="synthesis")
            else:
                synthesis, attempts = call_with_retry(
                    synthesis_drafter, synthesis_prompt,
                    stage="synthesis", validator=lambda value: bool(value.strip()) and len(value.strip()) >= 20,
                )
            ledger.update_auto_research_run(run_id, retry_count=attempts - 1)
        except LLMRetryExhausted as error:
            record_failure(error, "synthesis", {"search_run_id": outcome.get("run_id", "")})
            # A deterministic report preserves completed search/full-text work while
            # making the synthesis failure visible on the desk for optional retry.
            synthesis = deterministic_literature_report(profile, run_for_report, processed)

        completed = [item for item in processed if item.get("full_text_status") == "completed"]
        report_id = ledger.record(
            intent_event["case_id"], "advice_report", "m1", ["researcher", "m2"], "auto_literature_report",
            {
                "title": f"M1 자동 문헌탐색 보고 · {profile.get('title', '')}",
                "report": synthesis,
                "intent_id": intent_id,
                "search_run_id": outcome["run_id"],
                "auto_mode": True,
                "search_strategy": {
                    "concept_groups": profile.get("concept_groups", []),
                    "boolean_queries": profile.get("boolean_queries", []),
                    "keywords": profile.get("keywords", []),
                },
                "abstract_review_count": len(outcome["candidates"]),
                "fulltext_review_count": len(completed),
                "top_papers": [
                    {
                        "source_id": item.get("source_id"), "title": item.get("title"),
                        "published": item.get("published"), "citation_count": item.get("citation_count"),
                        "influential_citation_count": item.get("influential_citation_count"),
                        "full_text_similarity": item.get("full_text_similarity"), "url": item.get("url"),
                        "pdf_path": item.get("pdf_path", ""),
                    }
                    for item in sorted(completed, key=lambda p: p.get("full_text_similarity", 0), reverse=True)[:5]
                ],
            },
            subject_id=intent_id, status="completed",
        )
        unresolved = [f for f in ledger.auto_research_failures() if f.get("run_id") == run_id]
        if unresolved:
            ledger.update_auto_research_run(run_id, status="needs_attention", stage="completed_with_attention")
            ledger.transition(intent_event["phenomenon_id"], "ready", "failed")
            return {"status": "needs_attention", "report": synthesis, "report_id": report_id, "run": run_for_report, "papers": processed, "retry_run_id": run_id, "search_strategy": {"concept_groups": profile.get("concept_groups", []), "boolean_queries": profile.get("boolean_queries", []), "keywords": profile.get("keywords", [])}}
        ledger.update_auto_research_run(run_id, status="completed", stage="completed", checkpoint=checkpoint)
        for rq in ledger.research_questions_for_intent(intent_id):
            ledger.add_research_question_change(str(rq.get("rq_id", "")), "exploration_completed", f"M1 후속 문헌탐색이 완료되어 초록 {len(outcome['candidates'])}편과 상위 {len(completed)}편의 본문 비교 결과가 보고되었습니다.")
        complete_intent(
            ledger, intent_event,
            f"초록 {len(outcome['candidates'])}편을 검토하고 상위 {len(completed)}편의 본문을 비교하여 자동 탐색 보고서 {report_id}를 생성했습니다.",
        )
        return {"status": "completed", "report": synthesis, "report_id": report_id, "run": run_for_report, "papers": processed, "search_strategy": {"concept_groups": profile.get("concept_groups", []), "boolean_queries": profile.get("boolean_queries", []), "keywords": profile.get("keywords", [])}}
    except LLMRetryExhausted as error:
        record_failure(error, error.stage)
        ledger.transition(intent_event["phenomenon_id"], "ready", "failed")
        report = f"자동 문헌탐색 중 LLM Retry가 모두 실패했습니다: {error.message}"
        ledger.record(
            intent_event["case_id"], "advice_report", "m1", ["researcher", "m2"], "auto_literature_report",
            {"title": f"M1 자동 문헌탐색 주의 필요 · {profile.get('title', '')}", "report": report,
             "intent_id": intent_id, "auto_mode": True, "retry_run_id": run_id, "needs_attention": True},
            subject_id=intent_id, status="failed",
        )
        return {"status": "needs_attention", "report": report, "run": {}, "papers": [], "retry_run_id": run_id}
    except Exception as error:
        ledger.update_auto_research_run(run_id, status="needs_attention", stage="unexpected_error", error_type="unexpected_error", error_message=str(error))
        ledger.transition(intent_event["phenomenon_id"], "ready", "failed")
        report = f"자동 문헌탐색 중 오류가 발생했습니다: {error}"
        ledger.record(
            intent_event["case_id"], "advice_report", "m1", ["researcher", "m2"], "auto_literature_report",
            {"title": f"M1 자동 문헌탐색 실패 · {profile.get('title', '')}", "report": report,
             "intent_id": intent_id, "auto_mode": True},
            subject_id=intent_id, status="failed",
        )
        return {"status": "failed", "report": report, "run": {}, "papers": []}
