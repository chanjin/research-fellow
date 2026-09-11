from __future__ import annotations

import re
import uuid
from typing import Any

from research_fellow.domain.knowledge import KnowledgeCard, parse_text_draft
from research_fellow.infrastructure.arxiv import ArxivError, search as arxiv_search
from research_fellow.infrastructure.semantic_scholar import enrich_citation_counts
from research_fellow.application.search_profiles import auto_search_strategy_prompt, parse_auto_search_strategy


def sensemaking_answer_prompt(*, thread_title: str, conversation: list[dict[str, Any]], question: str, cards: list[dict[str, Any]], papers: list[dict[str, Any]] | None = None) -> str:
    history = "\n\n".join(f"{t.get('role','user').upper()}: {t.get('content','')}" for t in conversation[-8:]) or "(new thread)"
    evidence = []
    for card in cards:
        evidence.append(
            f"[{card.get('card_id','')}] {card.get('title','')}\n"
            f"Claim: {card.get('claim','')}\nContext: {card.get('context','')}\n"
            f"Implication: {card.get('implication','')}\nLimits: {card.get('limits','')}"
        )
    paper_text = []
    for paper in (papers or [])[:5]:
        cites = paper.get('citation_count')
        paper_text.append(
            f"- {paper.get('title','')} ({str(paper.get('published',''))[:4]}) · citations: {cites if cites is not None else 'unavailable'}\n"
            f"  Abstract: {paper.get('summary','')[:1200]}"
        )
    return f"""# Research Sensemaking

You are a domain research fellow helping a researcher think quickly, not performing a full systematic literature review.
Interpret the new question in light of the accumulated approved knowledge. Use quick-literature abstracts only as provisional external context, never as verified knowledge cards.
Distinguish clearly between supported knowledge, interpretation, and open questions. Keep the response concise enough to support another conversational turn.

## Thread
{thread_title}

## Conversation so far
{history}

## New researcher question / claim
{question.strip()}

## Approved knowledge cards
{chr(10).join(evidence) if evidence else '(none found)'}

## Quick literature context (abstract-only, provisional)
{chr(10).join(paper_text) if paper_text else '(not used)'}

## Respond in Korean using this compact structure
### 빠른 해석
### 기존 지식과의 연결
### 의미 / 시사점
### 아직 불확실한 점
### 다음에 물어볼 만한 질문
Cite approved card IDs like [kc-...] where relevant. If quick literature is used, label it explicitly as provisional literature context.
"""


def quick_search_plan_prompt(question: str, conversation: list[dict[str, Any]]) -> str:
    recent = "\n".join(str(t.get('content','')) for t in conversation[-4:])
    profile = {
        "title": "Quick Sensemaking Literature",
        "question": question,
        "context": recent,
        "keywords": [],
    }
    return auto_search_strategy_prompt(profile)


def quick_literature_search(strategy_text: str, *, max_papers: int = 20) -> dict[str, Any]:
    plan = parse_auto_search_strategy(strategy_text)
    queries = list(plan.get("queries") or [])[:4]
    if not queries:
        phrases = list(plan.get("phrases") or [])[:4]
        queries = [f'all:"{re.sub(r"[^A-Za-z0-9 -]", "", p).strip()}"' for p in phrases if p.strip()]
    collected: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    per_query = max(5, min(10, max_papers))
    for query in queries:
        try:
            for paper in arxiv_search(query, max_results=per_query):
                sid = str(paper.get("source_id", ""))
                if sid and sid not in collected:
                    collected[sid] = paper
                if len(collected) >= max_papers:
                    break
        except ArxivError as exc:
            errors.append(str(exc))
        if len(collected) >= max_papers:
            break
    papers = enrich_citation_counts(list(collected.values())[:max_papers])
    return {"plan": plan, "papers": papers, "errors": errors}


def knowledge_card_candidate_prompt(*, thread_title: str, latest_question: str, latest_answer: str) -> str:
    return f"""# Knowledge Card Candidate from Sensemaking
Create ONE conservative candidate knowledge card from the exchange below. This is not automatically approved knowledge.
Use English for machine fields. Only state what is supported by the exchange; if the exchange is mostly interpretation, mark limits explicitly.

Thread: {thread_title}
Researcher input: {latest_question}
Research fellow interpretation: {latest_answer}

Return exactly these one-line fields:
Title: ...
Claim: ...
Context: ...
Implication: ...
Labels: label1, label2
Evidence: ...
Conditions: ...
Limits: ...
"""


def parse_sensemaking_card_candidate(text: str, *, thread_title: str) -> dict[str, Any] | None:
    fields = parse_text_draft(text or "")
    if not fields.get("claim"):
        return None
    card = KnowledgeCard(
        card_id=f"kc-candidate-{uuid.uuid4().hex[:12]}",
        title=fields.get("title") or fields["claim"][:80],
        source_kind="researcher_idea_note",
        claim=fields["claim"],
        context=fields.get("context", ""),
        implication=fields.get("implication", ""),
        labels=[x.strip() for x in fields.get("labels", "").split(",") if x.strip()],
        evidence_excerpt=fields.get("evidence_excerpt", ""),
        conditions=fields.get("conditions", ""),
        limits=fields.get("limits", ""),
        provenance={"source_name": f"Sensemaking: {thread_title}"},
    )
    return card.model_dump(mode="json")
