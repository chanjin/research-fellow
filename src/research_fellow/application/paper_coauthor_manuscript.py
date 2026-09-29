"""Deterministic manuscript and revision-domain transformations for paper coauthoring.

No workflow sequencing belongs here.
"""
from __future__ import annotations
import copy, re
from typing import Any
from research_fellow.application.structured_output import extract_json_object

ANNOTATION_TYPES = {"insufficient_evidence","citation_needed","researcher_input","decision","scope_unclear","overclaim","counterargument_needed","logic_gap","contribution_unclear"}

ANNOTATION_GUIDES: dict[str, dict[str, str]] = {
    "insufficient_evidence": {
        "label": "근거 보강 필요", "description": "현재 연결된 지식카드만으로 이 주장을 지지하기 어렵습니다.",
        "action": "knowledge_search", "search_guide": "주장의 핵심 개념, 적용 조건, 관찰·실험 결과를 뒷받침하는 근거를 탐색합니다.",
        "completion_criteria": "근거카드가 연결되고 문장의 주장 강도가 근거 수준과 일치합니다.",
    },
    "citation_needed": {
        "label": "인용 확인 필요", "description": "외부 사실·수치·선행연구에 대한 출처가 필요합니다.",
        "action": "literature_search", "search_guide": "고유명사, 정량 수치, 비교 결과의 원 논문과 측정 조건을 탐색합니다.",
        "completion_criteria": "검증 가능한 원 출처가 연결되거나 확인되지 않은 세부 주장이 제거됩니다.",
    },
    "researcher_input": {
        "label": "연구자 입력 필요", "description": "지식카드만으로 결정할 수 없는 연구자의 경험·관찰·입장이 필요합니다.",
        "action": "researcher_input", "search_guide": "문헌보다 사례 맥락, 관찰 사실, 저자의 해석을 먼저 정리합니다.",
        "completion_criteria": "연구자의 답변이 기록되고 해당 답변이 문장에 구체적으로 반영됩니다.",
    },
    "decision": {
        "label": "연구자 결정 필요", "description": "주장 범위나 대안 중 연구자가 선택해야 하는 사항입니다.",
        "action": "decision", "search_guide": "선택지별 근거, 반례, 적용 조건을 비교한 뒤 저자의 입장을 결정합니다.",
        "completion_criteria": "선택한 입장과 선택 이유가 기록되고 문장에 일관되게 반영됩니다.",
    },
    "scope_unclear": {
        "label": "범위 명확화 필요", "description": "대상, 조건, 제외 범위 또는 핵심 용어가 충분히 한정되지 않았습니다.",
        "action": "researcher_input", "search_guide": "핵심 용어의 정의와 논문이 포함·제외할 환경 및 사례를 확인합니다.",
        "completion_criteria": "대상·조건·제외 범위가 문장 또는 인접 문장에서 명시됩니다.",
    },
    "overclaim": {
        "label": "과잉 주장 조정", "description": "근거보다 강한 일반화나 인과 주장이 포함되어 있습니다.",
        "action": "decision", "search_guide": "반례와 적용 한계를 확인하고 주장 강도를 비교합니다.",
        "completion_criteria": "주장이 근거 수준에 맞게 한정되거나 추가 근거로 정당화됩니다.",
    },
    "counterargument_needed": {
        "label": "반론 보완 필요", "description": "대안 설명, 반례 또는 상반된 연구 결과가 다뤄지지 않았습니다.",
        "action": "literature_search", "search_guide": "반대 결과, 대안 접근, 성공 조건과 실패 조건을 중심으로 탐색합니다.",
        "completion_criteria": "주요 반론을 공정하게 제시하고 현재 주장의 경계가 설명됩니다.",
    },
    "logic_gap": {
        "label": "논리 연결 보완", "description": "전제와 결론 사이의 추론 단계가 누락되었거나 연결이 약합니다.",
        "action": "researcher_input", "search_guide": "누락된 전제, 메커니즘, 인과 경로를 확인합니다.",
        "completion_criteria": "전제에서 결론으로 이어지는 중간 논리가 명시됩니다.",
    },
    "contribution_unclear": {
        "label": "기여 명확화 필요", "description": "기존 연구와 비교한 새로운 기여가 분명하지 않습니다.",
        "action": "literature_search", "search_guide": "가장 가까운 선행연구의 문제, 방법, 결과와 본 연구의 차이를 비교합니다.",
        "completion_criteria": "기존 연구 대비 차이와 독자가 얻는 새로운 가치가 한 문장으로 표현됩니다.",
    },
}

def _literature_candidate_title(title: Any, target: str) -> str:
    value = str(title or "").strip()
    generic = (
        "관련 문헌 탐색", "문헌 탐색 후보", "bounded m1 search task",
        "bounded literature-search task", "literature search candidate",
    )
    if not value or any(marker in value.lower() for marker in generic):
        value = target.rstrip("?. ")[:100].strip()
    return value

def _annotation(item: dict[str, Any], *, annotation_id: str, kind: str) -> dict[str, Any]:
    guide = ANNOTATION_GUIDES[kind]
    literature_candidates = []
    for index, candidate in enumerate(item.get("literature_search_candidates") or [], 1):
        if not isinstance(candidate, dict):
            continue
        target = str(candidate.get("target") or candidate.get("question") or "").strip()
        if not target:
            continue
        title = _literature_candidate_title(candidate.get("title"), target)
        literature_candidates.append({
            "candidate_id": str(candidate.get("candidate_id") or f"{annotation_id}-lit-{index}"),
            "title": title,
            "target": target,
            "research_context": str(candidate.get("research_context") or "").strip(),
            "expected_evidence": str(candidate.get("expected_evidence") or guide["search_guide"]).strip(),
            "completion_condition": str(candidate.get("completion_condition") or guide["completion_criteria"]).strip(),
        })
        if len(literature_candidates) >= 3:
            break
    return {
        "annotation_id": annotation_id,
        "type": kind,
        "severity": str(item.get("severity") or "medium"),
        "comment": str(item.get("reason") or item.get("comment") or guide["description"]),
        "recommended_action": str(item.get("recommended_action") or guide["action"]),
        "question_for_researcher": str(item.get("question_for_researcher") or ""),
        "search_guide": str(item.get("search_guide") or guide["search_guide"]),
        "completion_criteria": str(item.get("completion_criteria") or guide["completion_criteria"]),
        "literature_search_candidates": literature_candidates,
        "status": str(item.get("status") or "open"),
    }

def annotation_legend() -> list[dict[str, str]]:
    return [
        {"마크업": kind, "표시": guide["label"], "의미": guide["description"], "기본 조치": guide["action"]}
        for kind, guide in ANNOTATION_GUIDES.items()
    ]

def apply_appendix_refresh(
    text: str, manuscript: dict[str,Any], *, valid_card_ids:set[str], version:int,
) -> tuple[dict[str,Any],list[dict[str,Any]]]:
    """Replace Full Paper candidates without rewriting the two-page body."""
    raw=extract_json_object(text, message="Short Paper 응답은 JSON object여야 합니다.");result=copy.deepcopy(manuscript);before=list(result.get("appendix_claims") or [])
    body_claims={re.sub(r"\s+"," ",str(sentence.get("text") or "")).casefold() for sentence in sentences(result)}
    refreshed=[];seen:set[str]=set()
    for index,item in enumerate(raw.get("appendix_claims") or [],1):
        if not isinstance(item,dict):continue
        claim=str(item.get("claim") or "").strip();claim_key=re.sub(r"\s+"," ",claim).casefold()
        evidence=list(dict.fromkeys(
            str(card_id) for card_id in item.get("evidence_card_ids") or []
            if str(card_id) in valid_card_ids
        ))
        if len(claim)<8 or not evidence or claim_key in seen or claim_key in body_claims:continue
        seen.add(claim_key)
        refreshed.append({
            "appendix_id":str(item.get("appendix_id") or f"a-{index:02d}"),
            "title":str(item.get("title") or claim[:72]).strip(),"claim":claim,
            "why_excluded":str(item.get("why_excluded") or "2페이지 본문의 우선순위에서 제외").strip(),
            "full_paper_value":str(item.get("full_paper_value") or "Full Paper 확장 시 검토").strip(),
            "evidence_card_ids":evidence,"status":"full_paper_candidate",
        })
        if len(refreshed)>=8:break
    before_by_id={str(item.get("appendix_id") or ""):item for item in before}
    after_by_id={str(item.get("appendix_id") or ""):item for item in refreshed}
    diffs=[]
    for appendix_id in sorted(set(before_by_id)|set(after_by_id)):
        old=before_by_id.get(appendix_id);new=after_by_id.get(appendix_id)
        old_claim=str((old or {}).get("claim") or "");new_claim=str((new or {}).get("claim") or "")
        if old==new:continue
        diffs.append({
            "sentence_id":appendix_id,"before":old_claim or "(신규)","after":new_claim or "(Appendix에서 제거)",
            "reason":"Full Paper 확장 후보 재평가","change_scope":"appendix",
        })
    result["appendix_claims"]=refreshed;result["version"]=version
    return result,diffs

def targeted_revision_context(
    manuscript: dict[str,Any], comments: list[dict[str,Any]],
) -> list[dict[str,Any]]:
    requested_ids = {str(item.get("sentence_id", "")) for item in comments if item.get("sentence_id")}
    ordered = sentences(manuscript)
    targets = []
    for index, sentence in enumerate(ordered):
        if sentence.get("sentence_id") not in requested_ids:
            continue
        targets.append({
            "sentence": sentence,
            "previous_sentence": ordered[index-1] if index > 0 else None,
            "next_sentence": ordered[index+1] if index + 1 < len(ordered) else None,
        })
    return targets

def selected_group_revisions(
    group_result: dict[str,Any], selected_todo_ids: set[str],
) -> tuple[list[dict[str,Any]],list[dict[str,Any]]]:
    """Merge only researcher-selected resolved results and reject conflicting sentence rewrites."""
    merged:dict[str,dict[str,Any]]={};appendix=[]
    for result in group_result.get("todo_results") or []:
        if result.get("verdict")!="resolved" or str(result.get("todo_id") or "") not in selected_todo_ids:continue
        for item in result.get("revisions") or []:
            sentence_id=str(item.get("sentence_id") or "")
            if not sentence_id:continue
            existing=merged.get(sentence_id)
            if existing and existing.get("after")!=item.get("after"):
                raise ValueError(f"문장 {sentence_id}에 서로 다른 그룹 수정안이 있어 함께 적용할 수 없습니다.")
            if existing:
                for field in ("new_evidence_card_ids","new_reference_paper_ids","resolved_annotation_ids"):
                    existing[field]=list(dict.fromkeys((existing.get(field) or [])+(item.get(field) or [])))
                if item.get("reason") and item.get("reason") not in existing.get("reason",""):
                    existing["reason"]="; ".join(filter(None,[existing.get("reason",""),item.get("reason","")]))
            else:merged[sentence_id]=copy.deepcopy(item)
        appendix.extend(copy.deepcopy(result.get("appendix_updates") or []))
    return list(merged.values()),appendix

def revision_focus_context(manuscript: dict[str,Any], todo: dict[str,Any]) -> dict[str,Any]:
    """Return only the section and paragraph that contain the active To-do sentence."""
    target_id=str(todo.get("sentence_id") or "")
    for section in manuscript.get("sections",[]):
        for paragraph in section.get("paragraphs",[]):
            paragraph_sentences=list(paragraph.get("sentences") or [])
            if any(str(item.get("sentence_id") or "")==target_id for item in paragraph_sentences):
                return {
                    "section_id":str(section.get("section_id") or ""),
                    "section_title":str(section.get("title") or ""),
                    "paragraph_id":str(paragraph.get("paragraph_id") or ""),
                    "sentences":paragraph_sentences,
                }
    return {"section_id":"","section_title":"","paragraph_id":"","sentences":[]}

def apply_review(manuscript: dict[str,Any], annotations: list[dict[str,Any]]) -> dict[str,Any]:
    result=copy.deepcopy(manuscript); by={}
    for item in annotations: by.setdefault(item["sentence_id"],[]).append({k:v for k,v in item.items() if k!="sentence_id"})
    for sentence in sentences(result): sentence["annotations"]=by.get(sentence["sentence_id"],[])
    return result

def apply_revisions(
    text: str, manuscript: dict[str,Any], *, valid_card_ids:set[str], version:int,
    allowed_sentence_ids: set[str] | None = None,
) -> tuple[dict[str,Any],list[dict[str,Any]]]:
    raw=extract_json_object(text, message="Short Paper 응답은 JSON object여야 합니다."); result=copy.deepcopy(manuscript); by={s["sentence_id"]:s for s in sentences(result)}; diffs=[]
    for item in raw.get("revisions") or []:
        sid=str(item.get("sentence_id") or ""); after=str(item.get("after") or "").strip()
        if sid not in by or not after or (allowed_sentence_ids is not None and sid not in allowed_sentence_ids): continue
        before=by[sid]["text"]; by[sid]["text"]=after
        by[sid]["evidence_card_ids"]=sorted(set(by[sid].get("evidence_card_ids",[])+[str(x) for x in item.get("new_evidence_card_ids") or [] if str(x) in valid_card_ids]))
        by[sid]["reference_paper_ids"]=sorted(set(
            by[sid].get("reference_paper_ids",[])+[str(x) for x in item.get("new_reference_paper_ids") or []]
        ))
        resolved=set(str(x) for x in item.get("resolved_annotation_ids") or [])
        for annotation in by[sid].get("annotations",[]):
            if annotation.get("annotation_id") in resolved:
                annotation["status"]="revised_pending_review"
        diffs.append({"sentence_id":sid,"before":before,"after":after,"reason":str(item.get("reason") or ""),"change_scope":"sentence"})
    if allowed_sentence_ids is None:
        appendix=list(result.get("appendix_claims") or [])
        appendix_by_id={str(item.get("appendix_id") or ""):item for item in appendix}
        body_claims={re.sub(r"\s+"," ",str(sentence.get("text") or "")).casefold() for sentence in sentences(result)}
        for existing in list(appendix):
            existing_key=re.sub(r"\s+"," ",str(existing.get("claim") or "")).casefold()
            if existing_key and existing_key in body_claims:
                appendix.remove(existing);appendix_by_id.pop(str(existing.get("appendix_id") or ""),None)
                diffs.append({
                    "sentence_id":str(existing.get("appendix_id") or ""),
                    "before":str(existing.get("claim") or ""),"after":"(본문으로 승격되어 Appendix에서 제거)",
                    "reason":"동일 주장의 본문·Appendix 중복 방지","change_scope":"appendix",
                })
        for index,item in enumerate(raw.get("appendix_updates") or [],1):
            if not isinstance(item,dict):
                continue
            action=str(item.get("action") or "add").strip().lower()
            appendix_id=str(item.get("appendix_id") or f"a-{len(appendix)+index:02d}").strip()
            existing=appendix_by_id.get(appendix_id)
            if action=="remove":
                if existing:
                    appendix.remove(existing);appendix_by_id.pop(appendix_id,None)
                    diffs.append({
                        "sentence_id":appendix_id,"before":str(existing.get("claim") or ""),"after":"(Appendix에서 제거)",
                        "reason":str(item.get("reason") or "Full Paper 후보에서 제외"),"change_scope":"appendix",
                    })
                continue
            claim=str(item.get("claim") or (existing or {}).get("claim") or "").strip()
            supplied_evidence=[
                str(card_id) for card_id in item.get("evidence_card_ids") or []
                if str(card_id) in valid_card_ids
            ]
            evidence=list(dict.fromkeys(supplied_evidence or list((existing or {}).get("evidence_card_ids") or [])))
            claim_key=re.sub(r"\s+"," ",claim).casefold()
            duplicate_appendix=any(
                other is not existing and re.sub(r"\s+"," ",str(other.get("claim") or "")).casefold()==claim_key
                for other in appendix
            )
            if len(claim)<8 or not evidence or claim_key in body_claims or duplicate_appendix:
                continue
            if existing is None and len(appendix)>=8:
                continue
            candidate={
                "appendix_id":appendix_id,
                "title":str(item.get("title") or (existing or {}).get("title") or claim[:72]).strip(),
                "claim":claim,
                "why_excluded":str(item.get("why_excluded") or (existing or {}).get("why_excluded") or "2페이지 본문의 우선순위에서 제외").strip(),
                "full_paper_value":str(item.get("full_paper_value") or (existing or {}).get("full_paper_value") or "Full Paper 확장 시 검토").strip(),
                "evidence_card_ids":evidence,
                "status":"full_paper_candidate",
            }
            old_snapshot=dict(existing) if existing else {}
            before=str(old_snapshot.get("claim") or "")
            if existing:
                existing.clear();existing.update(candidate)
            else:
                appendix.append(candidate);appendix_by_id[appendix_id]=candidate
            if old_snapshot!=candidate:
                diffs.append({
                    "sentence_id":appendix_id,"before":before or "(신규)","after":claim,
                    "reason":str(item.get("reason") or "Full Paper 확장 후보 주장 보존"),"change_scope":"appendix",
                })
        result["appendix_claims"]=appendix[:8]
    result["version"]=version
    return result,diffs

def sentences(manuscript: dict[str,Any]) -> list[dict[str,Any]]:
    return [s for section in manuscript.get("sections",[]) for paragraph in section.get("paragraphs",[]) for s in paragraph.get("sentences",[])]

def manuscript_markdown(manuscript: dict[str,Any], *, markup: bool=True) -> str:
    lines=[f"# {manuscript.get('title','Short Paper')}"]
    for section in manuscript.get("sections",[]):
        lines.append(f"\n## {section['title']}")
        for paragraph in section.get("paragraphs",[]):
            parts=[]
            for s in paragraph.get("sentences",[]):
                badges=" ".join(
                    f"`{ANNOTATION_GUIDES.get(a['type'],{}).get('label',a['type'])}`"
                    for a in s.get("annotations",[]) if a.get("status")=="open"
                ) if markup else ""
                parts.append(f"{s['text']} {badges}".strip())
            lines.append(" ".join(parts))
    appendix=list(manuscript.get("appendix_claims") or [])
    if appendix:
        lines.append("\n## Appendix · Full Paper 확장 후보 주장")
        lines.append("2페이지 본문의 핵심 논지에서는 제외했지만, 이후 Full Paper를 구성할 때 검토할 중요한 주장입니다.")
        for index,item in enumerate(appendix,1):
            lines.append(f"\n### A{index}. {item.get('title') or item.get('appendix_id','')}")
            lines.append(str(item.get("claim") or ""))
            lines.append(f"- **본문 제외 이유:** {item.get('why_excluded') or '-'}")
            lines.append(f"- **Full Paper 활용:** {item.get('full_paper_value') or '-'}")
            evidence=", ".join(str(card_id) for card_id in item.get("evidence_card_ids") or []) or "없음"
            lines.append(f"- **근거 지식카드:** {evidence}")
    return "\n".join(lines)

def diagnostics(manuscript: dict[str,Any], *, card_titles: dict[str,str] | None=None) -> list[dict[str,Any]]:
    titles = card_titles or {}
    result=[]
    for sentence in sentences(manuscript):
        evidence_ids=[str(item) for item in sentence.get("evidence_card_ids",[])]
        open_annotations=[item for item in sentence.get("annotations",[]) if item.get("status")=="open"]
        result.append({
            "문장 ID":sentence["sentence_id"],
            "문장 내용":sentence.get("text", ""),
            "역할":sentence.get("role", ""),
            "연결 근거":", ".join(titles.get(card_id,card_id) for card_id in evidence_ids) or "없음",
            "보완점":", ".join(ANNOTATION_GUIDES.get(a["type"],{}).get("label",a["type"]) for a in open_annotations) or "완료",
            "진단 내용":" | ".join(str(a.get("comment", "")) for a in open_annotations if a.get("comment")) or "-",
        })
    return result

def revision_todos(manuscript: dict[str, Any]) -> list[dict[str, Any]]:
    severity_priority = {"high": "P0", "medium": "P1", "low": "P2"}
    result = []
    for sentence in sentences(manuscript):
        for annotation in sentence.get("annotations", []):
            status = str(annotation.get("status") or "open")
            if status not in {"open", "ready", "revised_pending_review"}:
                continue
            kind = str(annotation.get("type") or "")
            if kind not in ANNOTATION_GUIDES:
                continue
            guide = ANNOTATION_GUIDES[kind]
            literature_candidates = list(annotation.get("literature_search_candidates") or [])
            if not literature_candidates:
                target = str(annotation.get("question_for_researcher") or f"이 문장의 주장을 검증하거나 한정하는 선행연구는 무엇인가: {sentence['text']}")
                literature_candidates = [{
                    "candidate_id": f"{annotation.get('annotation_id','')}-lit-1",
                    "title": f"{sentence['text'][:72].rstrip(' .')}의 근거와 적용 조건",
                    "target": target,
                    "research_context": f"검토 문장: {sentence['text']}\n보완 필요: {annotation.get('comment') or guide['description']}",
                    "expected_evidence": str(annotation.get("search_guide") or guide["search_guide"]),
                    "completion_condition": str(annotation.get("completion_criteria") or guide["completion_criteria"]),
                }]
            result.append({
                "todo_id": str(annotation.get("annotation_id") or ""),
                "sentence_id": sentence["sentence_id"],
                "sentence_text": sentence["text"],
                "type": kind,
                "label": guide["label"],
                "severity": str(annotation.get("severity") or "medium"),
                "priority": severity_priority.get(str(annotation.get("severity") or "medium"), "P1"),
                "status": status,
                "problem": str(annotation.get("comment") or guide["description"]),
                "question_for_researcher": str(annotation.get("question_for_researcher") or ""),
                "recommended_action": str(annotation.get("recommended_action") or guide["action"]),
                "search_guide": str(annotation.get("search_guide") or guide["search_guide"]),
                "completion_criteria": str(annotation.get("completion_criteria") or guide["completion_criteria"]),
                "literature_search_candidates": literature_candidates,
            })
    order = {"P0": 0, "P1": 1, "P2": 2}
    return sorted(result, key=lambda item: (order[item["priority"]], item["sentence_id"], item["todo_id"]))

def revision_todo_timeline(
    active_todos: list[dict[str, Any]], updates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge active To-dos with durable resolved updates for progress display."""
    latest_updates = {
        str(item.get("todo_id", "")): item
        for item in updates if item.get("todo_id")
    }
    terminal={"resolved","obsolete","merged","split"}
    history = {
        todo_id: {**item, "todo_id": todo_id}
        for todo_id, item in latest_updates.items() if item.get("status") in terminal
    }
    for item in active_todos:
        history[str(item.get("todo_id", ""))] = item
    order = {"P0": 0, "P1": 1, "P2": 2}
    return sorted(history.values(), key=lambda item: (
        1 if item.get("status") in terminal else 0,
        order.get(str(item.get("priority", "P1")), 1),
        str(item.get("sentence_id", "")), str(item.get("todo_id", "")),
    ))

def review_change_set(
    before_todos: list[dict[str, Any]], annotations: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Describe exactly which sentence-linked To-dos a review kept, added, or resolved."""
    before_by_id = {
        str(item.get("todo_id", "")): item for item in before_todos if item.get("todo_id")
    }
    after_by_id = {
        str(item.get("annotation_id", "")): item for item in annotations if item.get("annotation_id")
    }
    labels = {kind: guide["label"] for kind, guide in ANNOTATION_GUIDES.items()}

    def row(todo_id: str, item: dict[str, Any], change: str) -> dict[str, Any]:
        kind = str(item.get("type", ""))
        return {
            "change": change, "todo_id": todo_id,
            "sentence_id": str(item.get("sentence_id", "")), "type": kind,
            "label": str(item.get("label") or labels.get(kind, kind)),
            "problem": str(item.get("problem") or item.get("comment") or item.get("reason") or ""),
        }

    before_ids, after_ids = set(before_by_id), set(after_by_id)
    retained = sorted(before_ids & after_ids)
    new = sorted(after_ids - before_ids)
    resolved = sorted(before_ids - after_ids)
    return {
        "retained_todos": [row(todo_id, after_by_id[todo_id], "유지") for todo_id in retained],
        "new_todos": [row(todo_id, after_by_id[todo_id], "신규") for todo_id in new],
        "resolved_todos": [row(todo_id, before_by_id[todo_id], "해결") for todo_id in resolved],
    }
