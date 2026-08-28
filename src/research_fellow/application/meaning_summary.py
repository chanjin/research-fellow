"""Read-only, fact-first projection of approved knowledge and M2 reports."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from research_fellow.storage import Ledger

def build_fact_groups(cards: list[dict[str, Any]], relations: list[dict[str, Any]], limit: int = 20) -> list[dict[str, Any]]:
    """Group cards only by explicit approved relations, never by an LLM guess."""
    visible = cards[:limit]
    by_id = {str(card["card_id"]): card for card in visible}
    neighbors: dict[str, set[str]] = defaultdict(set)
    active_relations = []
    for relation in relations:
        source, target = str(relation["source_card_id"]), str(relation["target_card_id"])
        if source in by_id and target in by_id:
            neighbors[source].add(target)
            neighbors[target].add(source)
            active_relations.append(relation)

    groups, visited = [], set()
    for card in visible:
        root = str(card["card_id"])
        if root in visited:
            continue
        stack, component = [root], set()
        while stack:
            current = stack.pop()
            if current in component:
                continue
            component.add(current)
            stack.extend(neighbors[current] - component)
        visited.update(component)
        group_cards = [item for item in visible if str(item["card_id"]) in component]
        group_relations = [
            item for item in active_relations
            if str(item["source_card_id"]) in component and str(item["target_card_id"]) in component
        ]
        groups.append({"card_ids": component, "cards": group_cards, "relations": group_relations})
    return groups


def attach_reports(groups: list[dict[str, Any]], reports: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Attach an M2 report only when its ledger payload explicitly cites a card."""
    unmatched = []
    for group in groups:
        group["reports"] = []
    for report in reports:
        evidence_ids = set(map(str, report["payload"].get("evidence_card_ids", [])))
        matches = [group for group in groups if evidence_ids & group["card_ids"]]
        if matches:
            matches[0]["reports"].append(report)
        else:
            unmatched.append(report)
    return groups, unmatched


def meaning_summary_prompt(groups: list[dict[str, Any]], unmatched_reports: list[dict[str, Any]]) -> str:
    """Make a bounded prompt in which facts and M2 interpretations are separate."""
    from research_fellow.infrastructure.prompt_renderer import render_prompt

    prompt_groups = []
    for index, group in enumerate(groups[:6], start=1):
        cards = [
            {
                "reference": f"K{index}-{card_index}", "title": card["title"], "claim": card["claim"],
                "source": card.get("provenance", {}).get("source_name", "미상"), "labels": card.get("labels", []),
                "conditions": card.get("conditions", ""), "limits": card.get("limits", ""),
            }
            for card_index, card in enumerate(group["cards"], start=1)
        ]
        card_titles = {str(card["card_id"]): card["title"] for card in group["cards"]}
        relations = [
            {
                "source": card_titles.get(str(relation["source_card_id"]), ""),
                "target": card_titles.get(str(relation["target_card_id"]), ""),
                "type": relation["relation_type"], "evidence": relation["evidence"],
                "conditions": relation["conditions"], "confidence": relation["confidence"],
            }
            for relation in group["relations"]
        ]
        reports = [
            {
                "reference": f"R{index}-{report_index}", "title": report["payload"].get("title", "M2 보고서"),
                "question": (report["payload"].get("state") or {}).get("question", "연결된 연구질문 없음"),
                "report": str(report["payload"].get("report", ""))[:1800],
            }
            for report_index, report in enumerate(group.get("reports", [])[:3], start=1)
        ]
        prompt_groups.append({"number": index, "cards": cards, "relations": relations, "reports": reports})
    standalone_reports = [
        {
            "title": report["payload"].get("title", "M2 보고서"),
            "question": (report["payload"].get("state") or {}).get("question", "연결된 연구질문 없음"),
        }
        for report in unmatched_reports[:5]
    ]
    return render_prompt("m2_meaning_summary.j2", groups=prompt_groups, unmatched_reports=standalone_reports)


def latest_summary(ledger: Ledger) -> dict[str, Any] | None:
    """The latest saved projection is the delta cursor; no knowledge store is duplicated."""
    summaries = ledger.phenomena(recipient="researcher", type_="activity_summary")
    return summaries[0] if summaries else None


def delta_inputs(
    cards: list[dict[str, Any]], relations: list[dict[str, Any]], reports: list[dict[str, Any]], previous: dict[str, Any] | None,
) -> dict[str, Any]:
    """Return new items plus the smallest explicit graph context needed to read them."""
    baseline = (previous or {}).get("payload", {}).get("baseline", {})
    known_cards = set(map(str, baseline.get("card_ids", [])))
    known_relations = set(map(str, baseline.get("relation_ids", [])))
    known_reports = set(map(str, baseline.get("report_ids", [])))
    new_cards = [card for card in cards if str(card["card_id"]) not in known_cards]
    new_relations = [relation for relation in relations if str(relation["relation_id"]) not in known_relations]
    new_reports = [report for report in reports if str(report["phenomenon_id"]) not in known_reports]
    changed_card_ids = {str(card["card_id"]) for card in new_cards}
    for relation in new_relations:
        changed_card_ids.update((str(relation["source_card_id"]), str(relation["target_card_id"])))
    for report in new_reports:
        changed_card_ids.update(map(str, report["payload"].get("evidence_card_ids", [])))
    context_cards = [card for card in cards if str(card["card_id"]) in changed_card_ids]
    relevant_relations = [
        relation for relation in relations
        if str(relation["source_card_id"]) in changed_card_ids or str(relation["target_card_id"]) in changed_card_ids
    ]
    return {
        "cards": context_cards, "relations": relevant_relations, "reports": new_reports,
        "delta": {
            "card_ids": [str(card["card_id"]) for card in new_cards],
            "relation_ids": [str(relation["relation_id"]) for relation in new_relations],
            "report_ids": [str(report["phenomenon_id"]) for report in new_reports],
        },
        "baseline": {
            "card_ids": [str(card["card_id"]) for card in cards],
            "relation_ids": [str(relation["relation_id"]) for relation in relations],
            "report_ids": [str(report["phenomenon_id"]) for report in reports],
        },
        "is_initial_baseline": previous is None,
    }


def delta_meaning_summary_prompt(delta: dict[str, Any]) -> str:
    """Give M2 the delta and only the graph context explicitly connected to it."""
    from research_fellow.infrastructure.prompt_renderer import render_prompt

    cards = delta["cards"]
    card_titles = {str(card["card_id"]): card["title"] for card in cards}
    return render_prompt(
        "m2_delta_meaning_summary.j2",
        is_initial_baseline=delta["is_initial_baseline"],
        new_card_ids=set(delta["delta"]["card_ids"]), new_relation_ids=set(delta["delta"]["relation_ids"]),
        cards=[{
            "card_id": str(card["card_id"]), "title": card["title"], "claim": card["claim"],
            "source": card.get("provenance", {}).get("source_name", "미상"), "labels": card.get("labels", []),
            "conditions": card.get("conditions", ""), "limits": card.get("limits", ""),
        } for card in cards[:20]],
        relations=[{
            "relation_id": str(relation["relation_id"]),
            "source": card_titles.get(str(relation["source_card_id"]), str(relation["source_card_id"])),
            "target": card_titles.get(str(relation["target_card_id"]), str(relation["target_card_id"])),
            "type": relation["relation_type"], "evidence": relation["evidence"],
            "conditions": relation["conditions"], "confidence": relation["confidence"],
        } for relation in delta["relations"][:20]],
        reports=[{
            "reference": str(report["phenomenon_id"]), "title": report["payload"].get("title", "M2 보고서"),
            "question": (report["payload"].get("state") or {}).get("question", "연결된 연구질문 없음"),
            "report": str(report["payload"].get("report", ""))[:1800],
        } for report in delta["reports"][:6]],
    )


def deterministic_delta_summary(delta: dict[str, Any]) -> str:
    """A saveable, provenance-preserving fallback when the local LLM is unavailable."""
    change = delta["delta"]
    if not any(change.values()):
        return "## 이번 Delta\n마지막 의미 요약 이후 승인 지식, 승인 관계, 새 M2 보고서가 없습니다."
    lines = ["## 이번 Delta", f"- 새 승인 지식카드: {len(change['card_ids'])}건", f"- 새 승인 관계: {len(change['relation_ids'])}건", f"- 새 M2 보고서: {len(change['report_ids'])}건", "", "## 새 승인 사실"]
    new_ids = set(change["card_ids"])
    for card in delta["cards"]:
        if str(card["card_id"]) in new_ids:
            lines.append(f"- {card['title']}: {card['claim']} (출처: {card.get('provenance', {}).get('source_name', '미상')})")
    if not new_ids:
        lines.append("- 이번에는 새 지식카드가 없습니다.")
    lines.extend(["", "## 승인 관계 변화"])
    new_relation_ids = set(change["relation_ids"])
    for relation in delta["relations"]:
        if str(relation["relation_id"]) in new_relation_ids:
            lines.append(f"- {relation['relation_type']} · 근거: {relation['evidence']} · 조건: {relation['conditions']}")
    if not new_relation_ids:
        lines.append("- 이번에는 새 승인 관계가 없습니다.")
    lines.extend(["", "## M2의 기존 해석 변화"])
    for report in delta["reports"]:
        lines.append(f"- {report['payload'].get('title', 'M2 보고서')} (해석·판단이며 승인 사실과 구분)")
    if not delta["reports"]:
        lines.append("- 이번에는 새 M2 보고서가 없습니다.")
    return "\n".join(lines)


def record_delta_summary(ledger: Ledger, delta: dict[str, Any], summary: str, trigger: str) -> str:
    """Persist a derived briefing with its cursor; source knowledge remains untouched."""
    case_id = ledger.create_case("research", "연구 활동 Delta 요약")
    return ledger.record(
        case_id, "activity_summary", "m2", ["researcher"], "activity_delta_summary",
        {
            "title": "연구 활동 Delta 요약", "summary": summary, "baseline": delta["baseline"],
            "delta": delta["delta"], "is_initial_baseline": delta["is_initial_baseline"], "trigger": trigger,
        }, status="completed",
    )
