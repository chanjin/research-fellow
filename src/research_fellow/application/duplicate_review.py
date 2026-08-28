"""Conservative duplicate detection for candidate knowledge cards.

This is a reviewer aid, not a semantic verdict. It blocks no researcher choice;
the UI defaults likely duplicates out of the new-card approval flow.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher


def similar_approved_cards(candidates: list[dict[str, object]], approved_cards: list[dict[str, object]], threshold: float = 0.52) -> dict[str, list[dict[str, object]]]:
    matches: dict[str, list[dict[str, object]]] = {}
    for candidate in candidates:
        candidate_id = str(candidate.get("card_id", ""))
        candidate_claim = str(candidate.get("claim", ""))
        ranked = []
        for card in approved_cards:
            score = _similarity(candidate_claim, str(card.get("claim", "")))
            if score >= threshold:
                ranked.append({"card_id": card["card_id"], "title": card["title"], "claim": card["claim"], "score": round(score, 2)})
        if ranked:
            matches[candidate_id] = sorted(ranked, key=lambda item: float(item["score"]), reverse=True)[:3]
    return matches


def _similarity(first: str, second: str) -> float:
    first_tokens, second_tokens = _tokens(first), _tokens(second)
    if not first_tokens or not second_tokens:
        return 0.0
    overlap = len(first_tokens & second_tokens) / len(first_tokens | second_tokens)
    sequence = SequenceMatcher(None, " ".join(sorted(first_tokens)), " ".join(sorted(second_tokens))).ratio()
    return 0.65 * overlap + 0.35 * sequence


def _tokens(value: str) -> set[str]:
    return {token.casefold() for token in re.findall(r"[A-Za-z0-9가-힣][A-Za-z0-9가-힣_-]+", value) if len(token) > 1}
