"""Apply researcher review of a paper-reading claim to durable knowledge state."""
from __future__ import annotations

from typing import Any, Mapping

from research_fellow.application.episodic_memory import store_researcher_curation_episode


def _split_csv(value: Any) -> list[str]:
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def apply_paper_reading_review(
    ledger: Any,
    memory: Any,
    *,
    paper: Mapping[str, Any],
    reading_question: Mapping[str, Any],
    review: Mapping[str, Any],
) -> dict[str, Any]:
    decision = str(review.get("decision") or "").strip()
    if decision not in {"register", "defer", "irrelevant"}:
        raise ValueError(f"unsupported paper-reading decision: {decision!r}")

    question_id = str(reading_question.get("question_id") or "").strip()
    paper_id = str(paper.get("paper_id") or "").strip()
    if not question_id or not paper_id:
        raise ValueError("paper and reading question identifiers are required")

    comment = str(review.get("comment") or "").strip()
    evidence = [str(x).strip(" -•") for x in (review.get("evidence") or []) if str(x).strip(" -•")]
    duplicate_mode = str(review.get("duplicate_mode") or "separate")
    duplicate_target_id = str(review.get("duplicate_target_id") or "").strip()

    if decision == "register":
        title = str(review.get("card_title") or "").strip()
        claim = str(review.get("card_claim") or "").strip()
        if len(title) < 4 or len(claim) < 8:
            raise ValueError("지식카드 등록에는 4자 이상의 카드 제목과 8자 이상의 주장(Claim)이 필요합니다.")
        if len(evidence) > 5:
            raise ValueError("원문 근거는 핵심 위치 최대 5개까지만 유지해 주세요.")
        if duplicate_mode == "defer":
            ledger.update_paper_reading_question(question_id, researcher_comment=comment, status="deferred")
            return {"action": "deferred", "question_id": question_id}
        if duplicate_mode == "enrich":
            if not duplicate_target_id:
                raise ValueError("근거를 보강할 기존 지식카드를 선택하세요.")
            enriched = memory.add_supporting_evidence(duplicate_target_id, {
                "source_name": str(paper.get("title") or ""),
                "paper_id": paper_id,
                "reading_question": str(reading_question.get("question") or ""),
                "evidence_excerpt": "\n".join(evidence)[:1600],
                "citation_markers": evidence,
                "conditions": str(review.get("card_conditions") or "").strip(),
                "limits": str(review.get("card_limits") or "").strip(),
                "researcher_comment": comment,
                "research_context": str(review.get("card_context") or "").strip(),
                "origin_links": list(paper.get("origin_links") or []),
            })
            existing_cards = ledger.paper_card_ids(paper_id)
            ledger.set_paper_card_links(paper_id, [*existing_cards, enriched["card_id"]])
            case_id = ledger.create_case("research", f"Knowledge evidence enrichment: {str(paper.get('title') or '')[:72]}")
            ledger.record(case_id, "knowledge_update", "m1", ["m2", "researcher"], "knowledge_card", {
                "title": f"기존 지식카드 근거 보강: {enriched['title']}",
                "card_id": enriched["card_id"], "paper_id": paper_id,
            }, enriched["card_id"], status="completed")
            ledger.update_paper_reading_question(question_id, researcher_comment=comment, status="registered")
            return {"action": "enriched", "question_id": question_id, "card_id": enriched["card_id"]}

    case_id = ledger.create_case("research", f"Researcher paper curation: {str(paper.get('title') or '')[:72]}")
    if decision == "register":
        card = memory.add({
            "title": str(review.get("card_title") or "").strip(),
            "source_kind": "external_paper" if paper.get("asset_type") in {"paper", "web_page"} else "researcher_idea_note",
            "claim": str(review.get("card_claim") or "").strip(),
            "explanation": "\n".join(part for part in [
                f"읽기 질문: {reading_question.get('question', '')}",
                f"연구자 해석·첨삭: {comment}" if comment else "",
            ] if part),
            "context": str(review.get("card_context") or "").strip(),
            "labels": _split_csv(review.get("card_labels")),
            "concepts": _split_csv(review.get("card_concepts")),
            "applies_to": _split_csv(review.get("card_applies_to")),
            "evidence_level": str(review.get("card_evidence_level") or "empirical"),
            "status": "verified",
            "evidence_excerpt": "\n".join(evidence)[:1600],
            "evidence_pages": [],
            "citation_markers": evidence,
            "conditions": str(review.get("card_conditions") or "").strip(),
            "limits": str(review.get("card_limits") or "").strip(),
            "provenance": {
                "source_name": str(paper.get("title") or ""), "paper_id": paper_id,
                "grounding": "paper_reading_researcher_registration",
                "reading_question": str(reading_question.get("question") or ""),
            },
            "origin_links": list(paper.get("origin_links") or []),
        })
        existing_cards = ledger.paper_card_ids(paper_id)
        ledger.set_paper_card_links(paper_id, [*existing_cards, card["card_id"]])
        ledger.record(case_id, "knowledge_update", "m1", ["m2", "researcher"], "knowledge_card", {
            "title": f"논문 읽기에서 등록한 지식카드: {card['title']}", "card_id": card["card_id"], "paper_id": paper_id,
        }, card["card_id"], status="completed")
        ledger.update_paper_reading_question(question_id, researcher_comment=comment, status="registered")
        store_researcher_curation_episode(
            ledger, case_id=case_id, episode_type="paper_card_registration", paper_title=str(paper.get("title") or ""),
            situation=f"읽기 질문: {reading_question.get('question','')}\n원문 근거: {'; '.join(evidence)}",
            decision="이 해석을 지식카드로 등록한다.",
            action_summary=f"등록 카드: {card['title']}\n주장: {card['claim']}",
            action_steps=["원문 근거 확인", "잠정 해석 첨삭", "Claim·개념·조건·한계 입력", "지식카드 등록"],
            evidence_card_ids=[card["card_id"]], unresolved_items=[card["limits"]] if card.get("limits") else [],
        )
        return {"action": "registered", "question_id": question_id, "card_id": card["card_id"]}

    if decision == "irrelevant":
        ledger.update_paper_reading_question(question_id, researcher_comment=comment, status="irrelevant")
        store_researcher_curation_episode(
            ledger, case_id=case_id, episode_type="paper_card_registration", paper_title=str(paper.get("title") or ""),
            situation=f"읽기 질문: {reading_question.get('question','')}\n원문 근거: {'; '.join(evidence)}",
            decision="이 해석은 현재 연구 주제의 지식카드로 등록하지 않는다.",
            action_summary=f"무관 판단 사유: {comment or '연구자 판단에 따라 현재 지식화 범위에서 제외'}",
            action_steps=["원문 근거 확인", "현재 연구 주제와의 관련성 판단", "무관 처리"],
            evidence_card_ids=[], unresolved_items=[],
        )
        return {"action": "irrelevant", "question_id": question_id}

    ledger.update_paper_reading_question(question_id, researcher_comment=comment, status="deferred")
    store_researcher_curation_episode(
        ledger, case_id=case_id, episode_type="paper_card_registration", paper_title=str(paper.get("title") or ""),
        situation=f"읽기 질문: {reading_question.get('question','')}\n원문 근거: {'; '.join(evidence)}",
        decision="이 해석은 근거 또는 연구 맥락 확인이 더 필요해 보류한다.",
        action_summary=f"보류 사유: {comment or '근거·조건을 추가 확인한 뒤 등록 또는 무관 판단'}",
        action_steps=["원문 근거 확인", "Claim·조건 보완 필요성 판단", "보류 처리"],
        evidence_card_ids=[], unresolved_items=[str(reading_question.get("question") or "")],
    )
    return {"action": "deferred", "question_id": question_id}
