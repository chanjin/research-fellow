"""Parsing of M2 research-question suggestions and priority assessments."""
from __future__ import annotations
import re
from typing import Any
from research_fellow.domain.research import ResearchQuestionCandidate

def parse_research_question_suggestions(
    text: str, *, valid_card_ids: set[str] | None = None, limit: int = 10,
) -> list[ResearchQuestionCandidate]:
    """Parse evidence-grounded RQ blocks while tolerating the legacy numbered-list format."""
    valid_card_ids = valid_card_ids or set()
    candidates: list[ResearchQuestionCandidate] = []
    blocks = [part.strip() for part in re.split(r"(?im)^##\s*RQ\s*\d+\s*$", text) if part.strip()]
    for block in blocks:
        fields: dict[str, str] = {}
        aliases = {
            "question": "question", "질문": "question",
            "why now": "rationale", "왜 지금": "rationale", "도출 이유": "rationale",
            "gap/tension": "gap_or_tension", "gap or tension": "gap_or_tension", "공백/긴장": "gap_or_tension",
            "research context": "research_context", "연구 맥락": "research_context",
            "source card ids": "source_card_ids", "근거 카드 ids": "source_card_ids", "근거 카드": "source_card_ids",
            "exploration need": "exploration_need", "추가 탐색 필요": "exploration_need",
        }
        current: str | None = None
        for line in block.splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                current = aliases.get(key.strip().lower())
                if current:
                    fields[current] = value.strip()
            elif current and line.strip():
                fields[current] = (fields.get(current, "") + " " + line.strip()).strip()
        question = fields.get("question", "").strip()
        rationale = fields.get("rationale", "").strip()
        if not question or not rationale:
            continue
        raw_ids = [item.strip() for item in re.split(r"[,;\s]+", fields.get("source_card_ids", "")) if item.strip()]
        source_ids = [item for item in raw_ids if not valid_card_ids or item in valid_card_ids]
        try:
            candidates.append(ResearchQuestionCandidate(
                question=question, rationale=rationale, gap_or_tension=fields.get("gap_or_tension", ""),
                research_context=fields.get("research_context", ""), exploration_need=fields.get("exploration_need", ""),
                source_card_ids=source_ids,
            ))
        except ValueError:
            continue
        if len(candidates) >= limit:
            return candidates

    # Backward compatibility for pasted legacy output: keep it as a candidate,
    # but make the missing rationale explicit rather than silently inventing one.
    if not candidates:
        for line in text.splitlines():
            matched = re.match(r"^\s*\d{1,2}[.)]\s+(.+?)\s*$", line)
            if not matched:
                continue
            question = matched.group(1).strip()
            if question:
                candidates.append(ResearchQuestionCandidate(
                    question=question,
                    rationale="외부 응답이 질문만 제공하여 도출 이유가 기록되지 않았습니다. 저장 후 연구자가 맥락을 보완해야 합니다.",
                ))
            if len(candidates) >= limit:
                break
    return candidates


def parse_rq_priority_assessment(
    text: str, backlog: list[dict[str, Any]], *, limit: int = 3,
) -> list[dict[str, Any]]:
    """Parse RQ_ID/SCORE/REASON blocks and fall back deterministically when needed."""
    by_id = {str(item.get("rq_id", "")): item for item in backlog}
    ranked: list[dict[str, Any]] = []
    seen: set[str] = set()
    for block in re.split(r"(?im)^##\s*(?:Priority|RQ)\s*\d+\s*$", text or ""):
        fields: dict[str, str] = {}
        for line in block.splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip().lower().replace(" ", "_")
            if key in {"rq_id", "score", "reason"}:
                fields[key] = value.strip()
        rq_id = fields.get("rq_id", "")
        if rq_id not in by_id or rq_id in seen:
            continue
        try:
            score = max(1, min(5, int(float(fields.get("score", "0")))))
        except ValueError:
            score = 0
        if score <= 0:
            continue
        ranked.append({"rq": by_id[rq_id], "score": score, "reason": fields.get("reason", "")})
        seen.add(rq_id)

    if ranked:
        ranked.sort(key=lambda item: item["score"], reverse=True)
        return ranked[:limit]

    # Deterministic fallback: researcher interest first, then stronger provenance/context.
    def fallback_key(item: dict[str, Any]) -> tuple[int, int, int, str]:
        status_weight = 2 if item.get("status") == "interested" else 1
        evidence = len(item.get("source_card_ids", [])) + len(item.get("source_update_ids", []))
        context = sum(bool(str(item.get(key, "")).strip()) for key in ("rationale", "gap_or_tension", "research_context", "exploration_need"))
        return status_weight, evidence, context, str(item.get("updated_at", ""))

    result = []
    for rq in sorted(backlog, key=fallback_key, reverse=True)[:limit]:
        score = 5 if rq.get("status") == "interested" else 4
        result.append({
            "rq": rq, "score": score,
            "reason": "LLM 우선순위 평가를 읽지 못해 관심 상태, 근거 수, 연구 맥락 충실도를 기준으로 선택했습니다.",
        })
    return result
