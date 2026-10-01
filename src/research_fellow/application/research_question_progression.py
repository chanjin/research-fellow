"""Initial progression for a newly submitted Research Question.

The workflow does not introduce a second research-task store. It inspects the
approved knowledge memory, records the inspection through the existing research
state review trail, and reuses the existing M2 -> M1 CurationIntent/SearchProfile
path for follow-up evidence acquisition.
"""
from __future__ import annotations

from typing import Any, Mapping

from research_fellow.application.advising_rq_intents import create_auto_exploration_intent_for_rq
from research_fellow.application.dsl import prepare_workflow_run

WORKFLOW_PATH = "m2/research_question_intake.yaml"


def start_research_question_progression(
    ledger: Any,
    memory: Any,
    research_question: Mapping[str, Any],
) -> dict[str, Any]:
    """Run the deterministic initial M2 progression for one newly created RQ."""
    run = prepare_workflow_run(
        WORKFLOW_PATH,
        {
            "ledger": ledger,
            "memory": memory,
            "research_question": dict(research_question),
        },
        globals(),
    )
    return run.execute()


def _inspect_research_question_context(context: dict[str, Any]) -> None:
    ledger = context["ledger"]
    memory = context["memory"]
    rq = dict(context["research_question"])
    rq_id = str(rq.get("rq_id") or "")
    question = str(rq.get("question") or "").strip()
    research_context = str(rq.get("research_context") or "").strip()

    related = list(memory.search(" ".join(part for part in (question, research_context) if part), limit=6))
    review_id = ledger.create_research_state_review("auto", [])
    ledger.link_research_question_review(review_id, rq_id, "new")
    for card in related:
        card_id = str(card.get("card_id") or "")
        if card_id:
            ledger.add_research_question_source(
                rq_id,
                review_id,
                card_id,
                relation_reason="신규 연구질문 초기 검토에서 관련 승인 지식으로 검색되었습니다.",
            )

    if related:
        exploration_need = (
            f"관련 승인 지식 {len(related)}건을 출발점으로 사용하되, 질문의 전제·반대 근거·적용 조건과 "
            "아직 확인되지 않은 공백을 외부 문헌에서 검증한다."
        )
        inspection_summary = f"관련 승인 지식 {len(related)}건을 확인했습니다. 추가 외부 근거 검증을 시작합니다."
    else:
        exploration_need = "현재 승인 지식에서 직접 관련 근거가 확인되지 않아 외부 문헌에서 기초 근거와 적용 조건을 탐색한다."
        inspection_summary = "직접 관련된 승인 지식이 없어 새로운 근거 탐색을 시작합니다."

    ledger.update_research_question_exploration_need(rq_id, exploration_need)
    refreshed = ledger.research_question(rq_id) or rq
    context.update(
        {
            "research_question": refreshed,
            "review_id": review_id,
            "matched_knowledge": related,
            "matched_knowledge_count": len(related),
            "exploration_need": exploration_need,
            "inspection_summary": inspection_summary,
        }
    )


def _start_research_question_exploration(context: dict[str, Any]) -> None:
    ledger = context["ledger"]
    rq = dict(context["research_question"])
    review_id = str(context["review_id"])
    summary = str(context.get("inspection_summary") or "")

    dispatched = create_auto_exploration_intent_for_rq(
        ledger,
        rq,
        score=4,
        selection_reason="연구자가 새 연구질문을 직접 제시했으며 초기 근거 검토 후 후속 evidence acquisition이 필요합니다.",
    )
    ledger.update_review_question_selection(
        review_id,
        str(rq.get("rq_id") or ""),
        score=4,
        reason="신규 연구질문 초기 progression에서 M1 근거 탐색 대상으로 연결했습니다.",
        selected=True,
    )
    ledger.complete_research_state_review(
        review_id,
        generated_rq_count=1,
        selected_rq_count=1,
        summary=summary,
    )

    intent = dispatched["intent"]
    case_id = ledger.create_case("research", f"RQ intake · {str(rq.get('question') or '')[:64]}")
    ledger.record(
        case_id,
        "advice_report",
        "m2",
        ["researcher"],
        "research_question_intake",
        {
            "title": "신규 연구질문 접수 및 다음 단계",
            "report": summary,
            "rq_id": str(rq.get("rq_id") or ""),
            "matched_knowledge_count": int(context.get("matched_knowledge_count") or 0),
            "intent_id": intent.intent_id,
            "next_action": "M1이 연결된 탐색 Intent를 바탕으로 근거를 확보합니다.",
            "human_attention_required": False,
        },
        subject_id=str(rq.get("rq_id") or ""),
        status="completed",
    )

    context.update(
        {
            "status": "exploring",
            "dispatched_intents": [
                {
                    "intent_id": intent.intent_id,
                    "title": intent.title,
                    "question": intent.question,
                    "profile_created": not bool(dispatched.get("reused")),
                    "reused": bool(dispatched.get("reused")),
                }
            ],
            "next_action": "M1 evidence acquisition queued",
            "human_attention_required": False,
        }
    )
