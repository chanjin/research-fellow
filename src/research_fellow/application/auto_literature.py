"""End-to-end execution of M2 auto-dispatched literature exploration intents."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from research_fellow.application.paper_batch import process_top_papers
from research_fellow.application.search_profiles import auto_search_strategy_prompt, keyword_prompt, parse_auto_search_strategy, parse_keyword_plan, run_profile
from research_fellow.services import complete_intent
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
) -> dict[str, Any]:
    """Run one auto Intent from planning through a researcher-facing report.

    This function is synchronous by design. It is called by an explicit auto-mode
    action and leaves every durable result in SQLite; it is not a background job.
    """
    intent = dict(intent_event.get("payload") or {})
    intent_id = str(intent.get("intent_id") or intent_event.get("subject_id") or "")
    profiles = [item for item in ledger.search_profiles(include_deleted=True) if item.get("intent_id") == intent_id]
    profile = profiles[0] if profiles else ledger.create_search_profile(intent)

    try:
        # Auto mode always plans its own English Boolean search strategy. Existing
        # profile keywords remain useful as hints, but never require a researcher click.
        raw_strategy = keyword_drafter(auto_search_strategy_prompt(profile)) or ""
        strategy = parse_auto_search_strategy(raw_strategy)
        phrases = strategy["phrases"]
        core_terms = strategy["core_terms"]
        if not phrases and not strategy["queries"]:
            # Backward-compatible fallback for a local model returning the old format.
            legacy_raw = keyword_drafter(keyword_prompt(profile)) or ""
            phrases, core_terms = parse_keyword_plan(legacy_raw)
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

        outcome = run_profile(ledger, profile, "auto", reviewer=abstract_reviewer)
        if outcome["status"] != "completed":
            ledger.transition(intent_event["phenomenon_id"], "ready", "failed")
            report = f"자동 문헌탐색을 완료하지 못했습니다. {outcome.get('error') or '검색 후보가 없습니다.'}"
            ledger.record(
                intent_event["case_id"], "advice_report", "m1", ["researcher", "m2"], "auto_literature_report",
                {"title": f"M1 자동 문헌탐색 실패 · {profile.get('title', '')}", "report": report,
                 "search_run_id": outcome.get("run_id", ""), "intent_id": intent_id, "auto_mode": True},
                subject_id=intent_id, status="failed",
            )
            return {"status": "failed", "report": report, "run": outcome, "papers": []}

        processed = process_top_papers(
            profile, outcome["candidates"], data_dir, fulltext_drafter,
            make_cards=False, review_full_text=True,
        )
        merged = {item["source_id"]: item for item in outcome["candidates"]}
        merged.update({item["source_id"]: item for item in processed})
        merged_candidates = list(merged.values())
        ledger.update_search_run_candidates(outcome["run_id"], merged_candidates)
        run_for_report = {**outcome, "candidates": merged_candidates}
        synthesis = synthesis_drafter(literature_synthesis_prompt(profile, run_for_report, processed)) or deterministic_literature_report(profile, run_for_report, processed)

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
                    }
                    for item in sorted(completed, key=lambda p: p.get("full_text_similarity", 0), reverse=True)[:5]
                ],
            },
            subject_id=intent_id, status="completed",
        )
        complete_intent(
            ledger, intent_event,
            f"초록 {len(outcome['candidates'])}편을 검토하고 상위 {len(completed)}편의 본문을 비교하여 자동 탐색 보고서 {report_id}를 생성했습니다.",
        )
        return {"status": "completed", "report": synthesis, "report_id": report_id, "run": run_for_report, "papers": processed, "search_strategy": {"concept_groups": profile.get("concept_groups", []), "boolean_queries": profile.get("boolean_queries", []), "keywords": profile.get("keywords", [])}}
    except Exception as error:
        ledger.transition(intent_event["phenomenon_id"], "ready", "failed")
        report = f"자동 문헌탐색 중 오류가 발생했습니다: {error}"
        ledger.record(
            intent_event["case_id"], "advice_report", "m1", ["researcher", "m2"], "auto_literature_report",
            {"title": f"M1 자동 문헌탐색 실패 · {profile.get('title', '')}", "report": report,
             "intent_id": intent_id, "auto_mode": True},
            subject_id=intent_id, status="failed",
        )
        return {"status": "failed", "report": report, "run": {}, "papers": []}
