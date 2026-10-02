"""Initial progression for a newly submitted Research Question.

The workflow does not introduce a second research-task store. A newly submitted
question starts from the researcher-provided question/context only; prior knowledge
or literature is deliberately not injected into the first literature round. The
existing M2 -> M1 CurationIntent/SearchProfile path is reused for evidence acquisition.
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
    """Prepare a fresh first-round context for a newly submitted RQ.

    Existing approved knowledge or previously collected literature is intentionally
    not searched here. Those assets belong to researcher-approved follow-up rounds
    for the same RQ, after the question has accumulated its own evidence lineage.
    """
    ledger = context["ledger"]
    rq = dict(context["research_question"])
    rq_id = str(rq.get("rq_id") or "")

    review_id = ledger.create_research_state_review("auto", [])
    ledger.link_research_question_review(review_id, rq_id, "new")

    exploration_need = (
        "새 연구질문 자체와 연구자가 제공한 맥락만을 기준으로 외부 문헌에서 기초 근거, "
        "반대 근거, 적용 조건, 관련 방법과 사례를 탐색한다."
    )
    inspection_summary = (
        "신규 연구질문을 기존 지식카드나 문헌에 의존하지 않고 독립적인 첫 문헌 탐색으로 시작합니다."
    )

    ledger.update_research_question_exploration_need(rq_id, exploration_need)
    refreshed = ledger.research_question(rq_id) or rq
    # Keep the existing workflow contract stable while making the first round
    # explicitly fresh. Follow-up literature uses research_actions.py instead.
    context.update(
        {
            "research_question": refreshed,
            "review_id": review_id,
            "matched_knowledge": [],
            "matched_knowledge_count": 0,
            "exploration_need": exploration_need,
            "inspection_summary": inspection_summary,
        }
    )


def _start_research_question_exploration(context: dict[str, Any]) -> None:
    ledger = context["ledger"]
    rq = dict(context["research_question"])
    review_id = str(context["review_id"])
    summary = str(context.get("inspection_summary") or "")

    fresh_rq = {**rq, "_fresh_intake": True}
    dispatched = create_auto_exploration_intent_for_rq(
        ledger,
        fresh_rq,
        score=4,
        selection_reason="연구자가 새 연구질문을 직접 제시했으며, 기존 지식·문헌을 주입하지 않은 독립적인 첫 evidence acquisition이 필요합니다.",
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
            "matched_knowledge_count": 0,
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
