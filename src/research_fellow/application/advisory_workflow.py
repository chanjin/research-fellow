"""Plan-first, evidence-clustered processing for M2 questions and advisories."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from research_fellow.application.dsl import prepare_workflow_run
from research_fellow.infrastructure.prompt_renderer import render_prompt
from research_fellow.infrastructure.retrieval import KnowledgeCluster, KnowledgeRetriever


@dataclass(frozen=True)
class SubQuestion:
    question: str
    purpose: str


@dataclass(frozen=True)
class AdvisoryPlan:
    request_type: str
    decision_question: str
    subquestions: tuple[SubQuestion, ...]


def advisory_plan_prompt(request_type: str, question: str, context: str, recipient: str, recalled_context: str = "") -> str:
    return render_prompt("m2_advisory_plan.j2", request_type=request_type, question=question, context=context, recipient=recipient, recalled_context=recalled_context)


def parse_advisory_plan(text: str, request_type: str, question: str) -> AdvisoryPlan:
    """Accept a readable LLM draft, with a deterministic safe plan as fallback."""
    decision_question = question.strip()
    subquestions: list[SubQuestion] = []
    section = ""
    for raw in text.splitlines():
        line = raw.strip()
        if line.lower().startswith("##"):
            section = line.lstrip("# ").lower()
            continue
        if not line:
            continue
        if "decision question" in section or "핵심 판단" in section:
            decision_question = line.lstrip("-• ").strip() or decision_question
        elif "subquestion" in section or "하위 질문" in section:
            item = line.lstrip("-•0123456789. ").strip()
            if item:
                subquestions.append(SubQuestion(item, "최종 판단에 필요한 세부 근거 확인"))
    if not subquestions:
        subquestions = [
            SubQuestion(question.strip(), "요청의 핵심 주장·권고를 판단"),
            SubQuestion(f"{question.strip()}의 적용 조건과 대상 범위는 무엇인가?", "적용 가능 범위를 판정"),
            SubQuestion(f"{question.strip()}에 대한 반론, 한계, 추가 확인 사항은 무엇인가?", "단정 가능한 범위를 제한"),
        ]
    return AdvisoryPlan(request_type, decision_question, tuple(subquestions[:3]))


def collect_evidence_clusters(
    plan: AdvisoryPlan, retriever: KnowledgeRetriever, cards: list[dict[str, object]], relations: list[dict[str, object]], *,
    context: str = "", semantic: bool = False, embedding_model: str = "nomic-embed-text",
) -> list[tuple[SubQuestion, KnowledgeCluster]]:
    """The retrieval boundary: deterministic search and relation expansion only."""
    return [
        (subquestion, retriever.cluster(
            list(cards), list(relations), f"{subquestion.question}\n{context}", semantic=semantic, embedding_model=embedding_model,
        ))
        for subquestion in plan.subquestions
    ]


def subquestion_judgment_prompt(subquestion: SubQuestion, cluster: KnowledgeCluster, context: str) -> str:
    return render_prompt("m2_subquestion_judgment.j2", subquestion=subquestion, cluster=cluster, context=context)


def advisory_synthesis_prompt(plan: AdvisoryPlan, judgments: list[str], clusters: list[KnowledgeCluster], recipient: str) -> str:
    return render_prompt("m2_advisory_synthesis.j2", plan=plan, judgments=judgments, clusters=clusters, recipient=recipient)


def deterministic_subquestion_judgment(subquestion: SubQuestion, cluster: KnowledgeCluster) -> str:
    if not cluster.members:
        return f"### {subquestion.question}\n승인 지식에서 직접 근거를 찾지 못했습니다. 추가 탐색 또는 사실 확인이 필요합니다."
    lines = [f"### {subquestion.question}", "**현재 근거**"]
    for member in cluster.members:
        card = member.result.card
        route = "직접 일치" if member.distance == 0 else " → ".join(member.relation_types)
        lines.append(f"- [{card['card_id']}] {card['claim']} ({route})")
    lines.append("**유보 사항**: 카드의 적용 조건과 한계를 비교한 뒤에만 최종 결론으로 통합합니다.")
    return "\n".join(lines)


def deterministic_advisory(plan: AdvisoryPlan, judgments: list[str]) -> str:
    return "\n\n".join([
        f"## 판단 대상\n{plan.decision_question}",
        "## 하위 판단\n" + "\n\n".join(judgments),
        "## 결론 범위\n현재 승인 지식의 적용 조건과 한계를 벗어나는 사항은 추가 탐색 또는 확인이 필요합니다.",
    ])


DraftFunction = Callable[[str], str | None]


def _plan_advisory(context: dict[str, Any]) -> None:
    draft_fn: DraftFunction = context["draft_fn"]
    prompt = advisory_plan_prompt(
        str(context["request_type"]),
        str(context["question"]),
        str(context.get("context") or ""),
        str(context["recipient"]),
        str(context.get("recalled_context") or ""),
    )
    draft = draft_fn(prompt)
    context["plan_draft"] = draft
    context["plan"] = parse_advisory_plan(
        draft or "",
        str(context["request_type"]),
        str(context["question"]),
    )


def execute_advisory_planning(
    *,
    request_type: str,
    question: str,
    context: str,
    recipient: str,
    recalled_context: str,
    draft_fn: DraftFunction,
) -> dict[str, Any]:
    """Run the plan-first M2 advisory planning workflow."""
    workflow = prepare_workflow_run(
        "m2/advisory_planning.yaml",
        {
            "request_type": request_type,
            "question": question,
            "context": context,
            "recipient": recipient,
            "recalled_context": recalled_context,
            "draft_fn": draft_fn,
        },
        globals(),
    )
    return workflow.execute()


def _collect_advisory_evidence(context: dict[str, Any]) -> None:
    retriever: KnowledgeRetriever = context["retriever"]
    context["evidence_clusters"] = collect_evidence_clusters(
        context["plan"],
        retriever,
        list(context.get("cards") or []),
        list(context.get("relations") or []),
        context=str(context.get("context") or ""),
        semantic=bool(context.get("semantic", False)),
        embedding_model=str(context.get("embedding_model") or "nomic-embed-text"),
    )
    context["judgments"] = []


def _judge_advisory_subquestion(context: dict[str, Any]) -> None:
    draft_fn: DraftFunction = context["draft_fn"]
    subquestion, cluster = context["current_item"]
    draft = draft_fn(subquestion_judgment_prompt(subquestion, cluster, str(context.get("context") or "")))
    judgments = context.get("judgments")
    if not isinstance(judgments, list):
        judgments = []
        context["judgments"] = judgments
    judgments.append(draft or deterministic_subquestion_judgment(subquestion, cluster))


def _synthesize_advisory_response(context: dict[str, Any]) -> None:
    draft_fn: DraftFunction = context["draft_fn"]
    evidence_clusters = list(context.get("evidence_clusters") or [])
    judgments = list(context.get("judgments") or [])
    plan: AdvisoryPlan = context["plan"]
    draft = draft_fn(
        advisory_synthesis_prompt(
            plan,
            judgments,
            [cluster for _, cluster in evidence_clusters],
            str(context["recipient"]),
        )
    )
    context["answer"] = draft or deterministic_advisory(plan, judgments)
    context["evidence_card_ids"] = list(dict.fromkeys(
        card_id for _, cluster in evidence_clusters for card_id in cluster.card_ids
    ))
    context["evidence_relation_ids"] = list(dict.fromkeys(
        relation_id for _, cluster in evidence_clusters for relation_id in cluster.relation_ids
    ))
    context["unresolved_items"] = [
        subquestion.question for subquestion, cluster in evidence_clusters if not cluster.members
    ]


def execute_advisory_response(
    *,
    plan: AdvisoryPlan,
    context: str,
    recipient: str,
    cards: list[dict[str, object]],
    relations: list[dict[str, object]],
    retriever: KnowledgeRetriever,
    draft_fn: DraftFunction,
    semantic: bool = False,
    embedding_model: str = "nomic-embed-text",
) -> dict[str, Any]:
    """Run evidence collection, per-question judgment, and advisory synthesis."""
    workflow = prepare_workflow_run(
        "m2/advisory_response.yaml",
        {
            "plan": plan,
            "context": context,
            "recipient": recipient,
            "cards": cards,
            "relations": relations,
            "retriever": retriever,
            "draft_fn": draft_fn,
            "semantic": semantic,
            "embedding_model": embedding_model,
        },
        globals(),
    )
    return workflow.execute()
