"""Structured-output parsers and validators for paper coauthor capabilities.

These functions normalize model output; they do not own workflow control.
"""
from __future__ import annotations
import copy, json, re
from typing import Any
from research_fellow.application.structured_output import extract_json_object
from research_fellow.application.paper_coauthor_manuscript import (
    ANNOTATION_TYPES, ANNOTATION_GUIDES, _annotation, _literature_candidate_title, sentences,
)

TODO_RECONCILIATION_STATUSES = {
    "resolved", "retained", "modified", "obsolete", "merged", "split", "researcher_review",
}

REVISION_ACTION_TYPES = {
    "M1_LITERATURE_SEARCH", "KNOWLEDGE_CARD_REVIEW", "RESEARCHER_DECISION",
    "EXPERIMENT", "SURVEY", "CASE_ANALYSIS", "DATA_ANALYSIS", "TRACE_REVIEW", "WRITING_ONLY",
}

def parse_paper_proposal(text: str) -> dict[str,Any]:
    """Normalize an LLM proposal review without treating model memory as evidence."""
    raw=extract_json_object(text, message="Short Paper 응답은 JSON object여야 합니다.")

    def assessment_rows(name: str) -> list[dict[str,Any]]:
        rows=[]
        for index,item in enumerate(raw.get(name) or [],1):
            if isinstance(item,str):item={"subject":item,"assessment":item}
            if not isinstance(item,dict):continue
            status=str(item.get("evidence_status") or "verification_required")
            if status not in {"grounded_internal","model_prior","verification_required","researcher_decision"}:
                status="verification_required"
            rows.append({
                "item_id":str(item.get("item_id") or f"{name}-{index:02d}"),
                "subject":str(item.get("subject") or item.get("title") or "").strip(),
                "assessment":str(item.get("assessment") or item.get("comment") or "").strip(),
                "evidence_status":status,
                "source_ids":list(dict.fromkeys(str(value) for value in item.get("source_ids") or [] if str(value))),
                "verification_need":str(item.get("verification_need") or "").strip(),
            })
        return [item for item in rows if item["subject"] or item["assessment"]][:12]

    candidates=[]
    for index,item in enumerate(raw.get("candidate_pairs") or [],1):
        if not isinstance(item,dict):continue
        candidate_title=str(item.get("title") or "").strip()
        candidate_rq=str(item.get("research_question") or "").strip()
        if not candidate_title or not candidate_rq:continue
        candidates.append({
            "candidate_id":str(item.get("candidate_id") or f"candidate-{index}"),
            "title":candidate_title,"research_question":candidate_rq,
            "contribution":str(item.get("contribution") or "").strip(),
            "scope":str(item.get("scope") or "").strip(),
            "rationale":str(item.get("rationale") or "").strip(),
        })
    if not candidates:raise ValueError("제목·연구질문 후보가 포함되어야 합니다.")

    searches=[]
    for index,item in enumerate(raw.get("m1_verification_candidates") or [],1):
        if not isinstance(item,dict):continue
        title=str(item.get("title") or "").strip();target=str(item.get("target") or "").strip()
        if not title or not target:continue
        searches.append({
            "candidate_id":str(item.get("candidate_id") or f"proposal-search-{index}"),
            "title":title,"target":target,
            "research_context":str(item.get("research_context") or "").strip(),
            "expected_evidence":str(item.get("expected_evidence") or "").strip(),
            "completion_condition":str(item.get("completion_condition") or "").strip(),
        })
    recommended=str(raw.get("recommended_candidate_id") or "")
    if recommended not in {item["candidate_id"] for item in candidates}:recommended=candidates[0]["candidate_id"]
    decisions=[]
    for item in raw.get("researcher_decisions") or []:
        if isinstance(item,dict):
            decisions.append({
                "question":str(item.get("question") or item.get("decision") or "").strip(),
                "options":[str(value) for value in item.get("options") or [] if str(value).strip()],
                "reason":str(item.get("reason") or "").strip(),
            })
        elif str(item).strip():decisions.append({"question":str(item).strip(),"options":[],"reason":""})
    return {
        "planning_summary":str(raw.get("planning_summary") or "").strip(),
        "internal_similarity_assessment":assessment_rows("internal_similarity_assessment"),
        "external_landscape_assessment":assessment_rows("external_landscape_assessment"),
        "novelty_risks":assessment_rows("novelty_risks"),
        "candidate_pairs":candidates[:5],"recommended_candidate_id":recommended,
        "m1_verification_candidates":searches[:5],"researcher_decisions":decisions[:8],
    }

def parse_writing_spec_guidance(text: str, *, valid_card_ids: set[str]) -> dict[str,Any]:
    raw=extract_json_object(text, message="Short Paper 응답은 JSON object여야 합니다.");claims=[]
    for index,item in enumerate(raw.get("central_claim_candidates") or [],1):
        if isinstance(item,str):item={"claim":item}
        if not isinstance(item,dict):continue
        claim=str(item.get("claim") or item.get("text") or "").strip()
        if not claim:continue
        claims.append({
            "claim_id":str(item.get("claim_id") or f"claim-{index}"),"claim":claim,
            "rationale":str(item.get("rationale") or "").strip(),
            "evidence_card_ids":list(dict.fromkeys(
                str(value) for value in item.get("evidence_card_ids") or [] if str(value) in valid_card_ids
            )),
            "risk_or_condition":str(item.get("risk_or_condition") or "").strip(),
        })
    if not claims:raise ValueError("중심 주장 후보가 하나 이상 필요합니다.")
    recommended=str(raw.get("recommended_claim_id") or "")
    if recommended not in {item["claim_id"] for item in claims}:recommended=claims[0]["claim_id"]

    def strings(name: str, limit: int) -> list[str]:
        values=raw.get(name) or []
        if isinstance(values,str):values=values.splitlines()
        return list(dict.fromkeys(str(value).strip() for value in values if str(value).strip()))[:limit]

    def strings_from(values: Any, limit: int) -> list[str]:
        values=values or []
        if isinstance(values,str):values=values.splitlines()
        return list(dict.fromkeys(str(value).strip() for value in values if str(value).strip()))[:limit]

    try:minimum=max(100,int(raw.get("target_min_chars") or 4000))
    except (TypeError,ValueError):minimum=4000
    try:maximum=max(minimum,int(raw.get("target_max_chars") or 9000))
    except (TypeError,ValueError):maximum=max(minimum,9000)

    design_candidates=[]
    for index,item in enumerate(raw.get("research_design_candidates") or [],1):
        if not isinstance(item,dict):continue
        method=str(item.get("method") or item.get("label") or "").strip()
        if not method:continue
        plans=[]
        for plan_index,plan in enumerate(item.get("verification_plan") or [],1):
            if not isinstance(plan,dict):continue
            metrics=[]
            for metric in plan.get("metrics") or []:
                if isinstance(metric,str):metric={"name":metric}
                if isinstance(metric,dict) and str(metric.get("name") or "").strip():
                    example=str(metric.get("example_record") or "[조건] | [관측값] | [해석 대기]").strip()
                    if not example.startswith("예시·미수행"):example="예시·미수행: "+example
                    metrics.append({
                        "name":str(metric.get("name") or "").strip(),
                        "definition":str(metric.get("definition") or "").strip(),
                        "example_record":example,
                    })
            plans.append({
                "plan_id":str(plan.get("plan_id") or f"plan-{index}-{plan_index}"),
                "research_question":str(plan.get("research_question") or "").strip(),
                "claim_to_verify":str(plan.get("claim_to_verify") or "").strip(),
                "method":str(plan.get("method") or method).strip(),
                "baseline":str(plan.get("baseline") or "").strip(),
                "unit_of_analysis":str(plan.get("unit_of_analysis") or "").strip(),
                "required_data":list(dict.fromkeys(str(value).strip() for value in plan.get("required_data") or [] if str(value).strip()))[:10],
                "metrics":metrics[:8],
                "analysis_method":str(plan.get("analysis_method") or "").strip(),
                "success_criteria":str(plan.get("success_criteria") or "").strip(),
                "falsification_condition":str(plan.get("falsification_condition") or "").strip(),
                "validity_threats":list(dict.fromkeys(str(value).strip() for value in plan.get("validity_threats") or [] if str(value).strip()))[:8],
            })
        recording=[]
        for artifact in item.get("result_recording_plan") or []:
            if not isinstance(artifact,dict):continue
            example_row=str(artifact.get("example_row") or "[조건] | [관측값] | [해석 대기]").strip()
            if not example_row.startswith("예시·미수행"):example_row="예시·미수행: "+example_row
            recording.append({
                "artifact":str(artifact.get("artifact") or "").strip(),
                "fields":[str(value).strip() for value in artifact.get("fields") or [] if str(value).strip()][:12],
                "example_row":example_row,
                "status":"planned",
            })
        design_candidates.append({
            "design_id":str(item.get("design_id") or f"design-{index}"),"method":method,
            "compatible_claim_ids":[str(value) for value in item.get("compatible_claim_ids") or [] if str(value) in {claim["claim_id"] for claim in claims}],
            "rationale":str(item.get("rationale") or "").strip(),
            "verification_plan":plans[:6],
            "execution_guide":strings_from(item.get("execution_guide"),10),
            "result_recording_plan":recording[:6],
        })
    recommended_design=str(raw.get("recommended_design_id") or "")
    if design_candidates and recommended_design not in {item["design_id"] for item in design_candidates}:
        recommended_design=design_candidates[0]["design_id"]

    searches=[]
    for index,item in enumerate(raw.get("literature_search_candidates") or [],1):
        if not isinstance(item,dict):continue
        target=str(item.get("target") or "").strip()
        if not target:continue
        searches.append({
            "candidate_id":str(item.get("candidate_id") or f"method-search-{index}"),
            "title":_literature_candidate_title(item.get("title"),target),"target":target,
            "research_context":str(item.get("research_context") or "").strip(),
            "expected_evidence":str(item.get("expected_evidence") or "").strip(),
            "completion_condition":str(item.get("completion_condition") or "").strip(),
        })
    return {
        "guidance_summary":str(raw.get("guidance_summary") or "").strip(),
        "audience":str(raw.get("audience") or "").strip(),
        "central_claim_candidates":claims[:5],"recommended_claim_id":recommended,
        "target_min_chars":minimum,"target_max_chars":maximum,
        "required_section_terms":strings("required_section_terms",8),
        "writing_rules":strings("writing_rules",10),
        "open_decisions":strings("open_decisions",8),
        "research_design_candidates":design_candidates[:4],
        "recommended_design_id":recommended_design,
        "literature_search_candidates":searches[:5],
    }

def parse_todo_group_plan(text: str, todos: list[dict[str,Any]]) -> dict[str,Any]:
    """Validate an LLM grouping plan; each active To-do may appear in at most one group."""
    raw=extract_json_object(text, message="Short Paper 응답은 JSON object여야 합니다.");valid_order=[str(item.get("todo_id") or "") for item in todos];valid_ids=set(valid_order);used:set[str]=set();group_ids:set[str]=set();groups=[]
    for index,item in enumerate(raw.get("groups") or [],1):
        if not isinstance(item,dict):continue
        todo_ids=[]
        for todo_id in item.get("todo_ids") or []:
            value=str(todo_id)
            if value in valid_ids and value not in used and value not in todo_ids:todo_ids.append(value)
        todo_ids=todo_ids[:4]
        if len(todo_ids)<2:continue
        used.update(todo_ids)
        search=item.get("shared_search") if isinstance(item.get("shared_search"),dict) else {}
        group_id=str(item.get("group_id") or f"tg-{index:02d}").strip()
        if group_id in group_ids:group_id=f"{group_id}-{index}"
        group_ids.add(group_id)
        groups.append({
            "group_id":group_id,
            "title":str(item.get("title") or f"유사 To-do 그룹 {index}").strip(),
            "reason":str(item.get("reason") or "").strip(),"todo_ids":todo_ids,
            "shared_evidence_need":str(item.get("shared_evidence_need") or "").strip(),
            "recommended_action":str(item.get("recommended_action") or "revision").strip(),
            "shared_search":{
                "title":str(search.get("title") or "").strip(),"target":str(search.get("target") or "").strip(),
                "research_context":str(search.get("research_context") or "").strip(),
                "expected_evidence":str(search.get("expected_evidence") or "").strip(),
                "completion_condition":str(search.get("completion_condition") or "").strip(),
            },
        })
    return {"groups":groups,"ungrouped_todo_ids":[todo_id for todo_id in valid_order if todo_id not in used]}

def parse_group_resolution(
    text: str, group: dict[str,Any], todos: list[dict[str,Any]], *,
    valid_card_ids:set[str], valid_paper_ids:set[str],
) -> dict[str,Any]:
    """Validate a group response while retaining an independent verdict for every To-do."""
    raw=extract_json_object(text, message="Short Paper 응답은 JSON object여야 합니다.")
    if raw.get("group_id") and str(raw.get("group_id"))!=str(group.get("group_id") or ""):
        raise ValueError("그룹 해결 응답의 group_id가 현재 확정 그룹과 일치하지 않습니다.")
    todo_by_id={str(item.get("todo_id") or ""):item for item in todos};results=[];seen=set()
    for item in raw.get("todo_results") or []:
        if not isinstance(item,dict):continue
        todo_id=str(item.get("todo_id") or "")
        if todo_id not in todo_by_id or todo_id in seen:continue
        seen.add(todo_id)
        results.append(parse_resolution_proposal(
            json.dumps(item,ensure_ascii=False),todo_by_id[todo_id],
            valid_card_ids=valid_card_ids,valid_paper_ids=valid_paper_ids,
        ))
    missing=[todo_id for todo_id in todo_by_id if todo_id not in seen]
    if missing:raise ValueError("그룹 해결 응답에 모든 선택 To-do의 개별 판정이 필요합니다: "+", ".join(missing))
    resolved_count=sum(1 for item in results if item["verdict"]=="resolved")
    verdict="resolved" if resolved_count==len(results) else "partial" if resolved_count else "needs_more_work"
    consistency=raw.get("cross_todo_consistency") if isinstance(raw.get("cross_todo_consistency"),dict) else {}
    return {
        "group_id":str(group.get("group_id") or ""),"group_verdict":verdict,"todo_results":results,
        "cross_todo_consistency":{
            "summary":str(consistency.get("summary") or "").strip(),
            "conflicts":[str(value) for value in consistency.get("conflicts") or [] if str(value).strip()][:10],
        },
    }

def parse_resolution_proposal(
    text: str, todo: dict[str,Any], *, valid_card_ids:set[str], valid_paper_ids:set[str],
) -> dict[str,Any]:
    raw=extract_json_object(text, message="Short Paper 응답은 JSON object여야 합니다.");verdict=str(raw.get("verdict") or "").strip()
    if verdict not in {"resolved","needs_more_work"}:
        raise ValueError("해결 제안 verdict는 resolved 또는 needs_more_work여야 합니다.")
    if str(raw.get("todo_id") or "")!=str(todo.get("todo_id") or ""):
        raise ValueError("해결 제안의 todo_id가 현재 To-do와 일치하지 않습니다.")
    revisions=[]
    for item in raw.get("revisions") or []:
        if not isinstance(item,dict):continue
        sentence_id=str(item.get("sentence_id") or "").strip();after=str(item.get("after") or "").strip()
        if not sentence_id or not after:continue
        revisions.append({
            "sentence_id":sentence_id,"after":after,"reason":str(item.get("reason") or "").strip(),
            "resolved_annotation_ids":[str(todo.get("todo_id") or "")] if sentence_id==str(todo.get("sentence_id") or "") else [],
            "new_evidence_card_ids":list(dict.fromkeys(
                str(card_id) for card_id in item.get("new_evidence_card_ids") or []
                if str(card_id) in valid_card_ids
            )),
            "new_reference_paper_ids":list(dict.fromkeys(
                str(paper_id) for paper_id in item.get("new_reference_paper_ids") or []
                if str(paper_id) in valid_paper_ids
            )),
        })
    if verdict=="resolved" and not any(item["sentence_id"]==str(todo.get("sentence_id") or "") for item in revisions):
        raise ValueError("resolved 제안에는 To-do 대상 문장의 수정안이 필요합니다.")
    target_revision=next((item for item in revisions if item["sentence_id"]==str(todo.get("sentence_id") or "")),None)
    if verdict=="resolved" and target_revision and not (
        target_revision["new_evidence_card_ids"] or target_revision["new_reference_paper_ids"]
    ):
        raise ValueError("resolved 제안의 대상 문장에는 선택한 지식카드 또는 원문 논문 근거가 하나 이상 연결되어야 합니다.")
    next_search=raw.get("next_search") if isinstance(raw.get("next_search"),dict) else {}
    return {
        "todo_id":str(todo.get("todo_id") or ""),"sentence_id":str(todo.get("sentence_id") or ""),
        "verdict":verdict,"reason":str(raw.get("reason") or "").strip(),"revisions":revisions,
        "appendix_updates":[item for item in raw.get("appendix_updates") or [] if isinstance(item,dict)],
        "evidence_connections":[item for item in raw.get("evidence_connections") or [] if isinstance(item,dict)][:20],
        "suggested_reference_paper_ids":list(dict.fromkeys(
            str(paper_id) for paper_id in raw.get("suggested_reference_paper_ids") or []
            if str(paper_id) in valid_paper_ids
        )),
        "next_search":{
            "title":str(next_search.get("title") or "").strip(),
            "target":str(next_search.get("target") or "").strip(),
            "research_context":str(next_search.get("research_context") or "").strip(),
            "expected_evidence":str(next_search.get("expected_evidence") or "").strip(),
            "completion_condition":str(next_search.get("completion_condition") or "").strip(),
        },
    }

def parse_todo_verification(text: str, todo: dict[str,Any]) -> dict[str,Any]:
    raw = extract_json_object(text, message="Short Paper 응답은 JSON object여야 합니다.")
    verdict = str(raw.get("verdict") or "").strip()
    if verdict not in {"resolved", "needs_more_work"}:
        raise ValueError("verdict는 resolved 또는 needs_more_work여야 합니다.")
    if str(raw.get("todo_id") or "") != str(todo.get("todo_id") or ""):
        raise ValueError("검증 응답의 todo_id가 현재 To-do와 일치하지 않습니다.")
    if str(raw.get("sentence_id") or "") != str(todo.get("sentence_id") or ""):
        raise ValueError("검증 응답의 sentence_id가 현재 문장과 일치하지 않습니다.")
    next_action = str(raw.get("next_action") or ("close" if verdict == "resolved" else "revise_again"))
    allowed_actions = {"close", "researcher_input", "knowledge_search", "literature_search", "revise_again"}
    if next_action not in allowed_actions:
        raise ValueError("지원하지 않는 next_action입니다.")
    if verdict == "resolved" and next_action != "close":
        raise ValueError("resolved 판정의 next_action은 close여야 합니다.")
    if verdict == "needs_more_work" and next_action == "close":
        raise ValueError("needs_more_work 판정은 보완 next_action이 필요합니다.")
    checks=[]
    for item in raw.get("criteria_checks") or []:
        if not isinstance(item,dict):
            continue
        checks.append({
            "criterion":str(item.get("criterion") or "").strip(),
            "satisfied":item.get("satisfied") is True,
            "evidence":str(item.get("evidence") or "").strip(),
        })
    return {
        "todo_id":str(todo.get("todo_id") or ""),
        "sentence_id":str(todo.get("sentence_id") or ""),
        "verdict":verdict,
        "reason":str(raw.get("reason") or "").strip(),
        "criteria_checks":checks,
        "remaining_gap":str(raw.get("remaining_gap") or "").strip(),
        "next_action":next_action,
    }

def parse_manuscript(text: str, *, valid_card_ids: set[str], version: int) -> dict[str,Any]:
    raw=extract_json_object(text, message="Short Paper 응답은 JSON object여야 합니다."); sections=[]; sentence_index=0
    for sidx,section in enumerate(raw.get("sections") or [],1):
        paragraphs=[]
        for pidx,paragraph in enumerate(section.get("paragraphs") or [],1):
            sentences=[]
            for sentence in paragraph.get("sentences") or []:
                body=str(sentence.get("text") or "").strip()
                if not body: continue
                sentence_index+=1; sid=str(sentence.get("sentence_id") or f"s-{sentence_index:03d}")
                evidence=[str(x) for x in sentence.get("evidence_card_ids") or [] if str(x) in valid_card_ids]
                annotations=[]
                for aidx,item in enumerate(sentence.get("annotations") or [],1):
                    kind=str(item.get("type") or "");
                    if kind in ANNOTATION_TYPES:
                        annotations.append(_annotation(item, annotation_id=str(item.get("annotation_id") or f"{sid}-a{aidx}"), kind=kind))
                sentences.append({"sentence_id":sid,"text":body,"role":str(sentence.get("role") or "supporting"),"evidence_card_ids":evidence,"annotations":annotations})
            if sentences: paragraphs.append({"paragraph_id":str(paragraph.get("paragraph_id") or f"p-{sidx:02d}-{pidx:02d}"),"sentences":sentences})
        if paragraphs: sections.append({"section_id":str(section.get("section_id") or f"section-{sidx}"),"title":str(section.get("title") or f"Section {sidx}"),"paragraphs":paragraphs})
    if not sections: raise ValueError("유효한 section/paragraph/sentence가 없습니다.")
    appendix_claims=[]
    seen_claims:set[str]=set()
    body_claims={
        re.sub(r"\s+"," ",str(sentence.get("text") or "")).casefold()
        for section in sections for paragraph in section.get("paragraphs",[]) for sentence in paragraph.get("sentences",[])
    }
    for index,item in enumerate(raw.get("appendix_claims") or [],1):
        if not isinstance(item,dict):
            continue
        claim=str(item.get("claim") or "").strip()
        evidence=list(dict.fromkeys(
            str(card_id) for card_id in item.get("evidence_card_ids") or []
            if str(card_id) in valid_card_ids
        ))
        claim_key=re.sub(r"\s+"," ",claim).casefold()
        if len(claim)<8 or not evidence or claim_key in seen_claims or claim_key in body_claims:
            continue
        seen_claims.add(claim_key)
        appendix_claims.append({
            "appendix_id":str(item.get("appendix_id") or f"a-{index:02d}"),
            "title":str(item.get("title") or claim[:72]).strip(),
            "claim":claim,
            "why_excluded":str(item.get("why_excluded") or "2페이지 본문의 우선순위에서 제외").strip(),
            "full_paper_value":str(item.get("full_paper_value") or "Full Paper 확장 시 검토").strip(),
            "evidence_card_ids":evidence,
            "status":"full_paper_candidate",
        })
    return {
        "version":version,"title":str(raw.get("title") or "Short Paper"),"sections":sections,
        "appendix_claims":appendix_claims[:8],
        "overall_assessment":str(raw.get("overall_assessment") or ""),
    }

def parse_review(text: str, manuscript: dict[str,Any]) -> list[dict[str,Any]]:
    raw=extract_json_object(text, message="Short Paper 응답은 JSON object여야 합니다."); valid={s["sentence_id"] for s in sentences(manuscript)}; result=[]
    for idx,item in enumerate(raw.get("annotations") or [],1):
        sid=str(item.get("sentence_id") or ""); kind=str(item.get("type") or "")
        if sid in valid and kind in ANNOTATION_TYPES:
            annotation = _annotation(item, annotation_id=str(item.get("annotation_id") or f"review-a{idx}"), kind=kind)
            annotation["sentence_id"] = sid
            annotation["status"] = "open"
            result.append(annotation)
    return result

def parse_revision_resolution_plan(text: str, todo: dict[str,Any]) -> dict[str,Any]:
    raw=extract_json_object(text, message="Short Paper 응답은 JSON object여야 합니다.");actions=[]
    for index,item in enumerate(raw.get("actions") or [],1):
        if not isinstance(item,dict):continue
        kind=str(item.get("type") or "WRITING_ONLY").strip().upper()
        if kind not in REVISION_ACTION_TYPES:kind="WRITING_ONLY"
        title=str(item.get("title") or "").strip()
        if not title:continue
        procedure=item.get("procedure") or []
        if isinstance(procedure,str):procedure=procedure.splitlines()
        outputs=item.get("expected_artifacts") or []
        if isinstance(outputs,str):outputs=outputs.splitlines()
        actions.append({
            "action_id":str(item.get("action_id") or f"action-{index}"),"type":kind,"title":title,
            "purpose":str(item.get("purpose") or "").strip(),
            "procedure":[str(value).strip() for value in procedure if str(value).strip()][:12],
            "expected_artifacts":[str(value).strip() for value in outputs if str(value).strip()][:10],
            "completion_condition":str(item.get("completion_condition") or "").strip(),
            "status":"planned",
        })
    if not actions:raise ValueError("To-do 해결 계획에는 하나 이상의 실행 작업이 필요합니다.")
    return {
        "todo_id":str(todo.get("todo_id") or ""),
        "strategy_summary":str(raw.get("strategy_summary") or "").strip(),
        "revision_target":str(raw.get("revision_target") or todo.get("sentence_id") or "").strip(),
        "actions":actions[:10],
        "risks":[str(value).strip() for value in raw.get("risks") or [] if str(value).strip()][:8],
        "expected_resolution":str(raw.get("expected_resolution") or todo.get("completion_criteria") or "").strip(),
    }

def parse_todo_reconciliation(
    text: str, *, existing_todos: list[dict[str,Any]], manuscript: dict[str,Any],
) -> dict[str,Any]:
    """Normalize LLM backlog changes without silently deleting historical To-dos."""
    raw=extract_json_object(text, message="Short Paper 응답은 JSON object여야 합니다.")
    existing={str(item.get("todo_id") or ""):item for item in existing_todos if item.get("todo_id")}
    valid_sentences={str(item.get("sentence_id") or ""):item for item in sentences(manuscript)}
    assessments=[];seen=set()
    for item in raw.get("existing_todo_assessments") or []:
        if not isinstance(item,dict):continue
        todo_id=str(item.get("todo_id") or "")
        status=str(item.get("status") or "retained").strip().lower()
        if todo_id not in existing or todo_id in seen or status not in TODO_RECONCILIATION_STATUSES:continue
        seen.add(todo_id);prior=existing[todo_id]
        assessments.append({
            "todo_id":todo_id,"status":status,"reason":str(item.get("reason") or "").strip(),
            "evidence":[str(value).strip() for value in item.get("evidence") or [] if str(value).strip()][:8],
            "sentence_id":str(item.get("sentence_id") or prior.get("sentence_id") or ""),
            "updated_problem":str(item.get("updated_problem") or prior.get("problem") or "").strip(),
            "updated_completion_criteria":str(item.get("updated_completion_criteria") or prior.get("completion_criteria") or "").strip(),
            "updated_recommended_action":str(item.get("updated_recommended_action") or prior.get("recommended_action") or "researcher_input").strip(),
            "merged_into":str(item.get("merged_into") or "").strip(),
        })
    for todo_id,prior in existing.items():
        if todo_id not in seen:
            assessments.append({
                "todo_id":todo_id,"status":"researcher_review",
                "reason":"LLM 응답에서 기존 To-do 평가가 누락되어 연구자 확인이 필요합니다.","evidence":[],
                "sentence_id":str(prior.get("sentence_id") or ""),
                "updated_problem":str(prior.get("problem") or ""),
                "updated_completion_criteria":str(prior.get("completion_criteria") or ""),
                "updated_recommended_action":str(prior.get("recommended_action") or "researcher_input"),"merged_into":"",
            })

    new_todos=[];known_ids=set(existing)
    for index,item in enumerate(raw.get("new_todos") or [],1):
        if not isinstance(item,dict):continue
        sentence_id=str(item.get("sentence_id") or "")
        kind=str(item.get("type") or "logic_gap")
        if sentence_id not in valid_sentences or kind not in ANNOTATION_TYPES:continue
        todo_id=str(item.get("todo_id") or f"reconcile-{sentence_id}-{index}")
        if todo_id in known_ids:todo_id=f"{todo_id}-new-{index}"
        known_ids.add(todo_id);guide=ANNOTATION_GUIDES[kind]
        new_todos.append({
            "todo_id":todo_id,"parent_todo_id":str(item.get("parent_todo_id") or ""),
            "sentence_id":sentence_id,"sentence_text":str(valid_sentences[sentence_id].get("text") or ""),
            "type":kind,"label":guide["label"],
            "priority":str(item.get("priority") or "P1") if str(item.get("priority") or "P1") in {"P0","P1","P2"} else "P1",
            "problem":str(item.get("problem") or guide["description"]).strip(),
            "completion_criteria":str(item.get("completion_criteria") or guide["completion_criteria"]).strip(),
            "recommended_action":str(item.get("recommended_action") or guide["action"]).strip(),
            "search_guide":str(item.get("search_guide") or guide["search_guide"]).strip(),
            "question_for_researcher":str(item.get("question_for_researcher") or "").strip(),
            "literature_search_candidates":[],
            "reason":str(item.get("reason") or "리비전 후 새로 발견된 보완점").strip(),
        })
    return {
        "summary":str(raw.get("summary") or "").strip(),
        "revision_achievements":[str(value).strip() for value in raw.get("revision_achievements") or [] if str(value).strip()][:12],
        "existing_todo_assessments":assessments,"new_todos":new_todos[:20],
        "next_revision_recommendations":[str(value).strip() for value in raw.get("next_revision_recommendations") or [] if str(value).strip()][:12],
    }
