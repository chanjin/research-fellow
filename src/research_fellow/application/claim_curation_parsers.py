"""Structured-output parsers and validation helpers for claim-first curation."""
from __future__ import annotations
import re
from dataclasses import dataclass

@dataclass(frozen=True)
class ClaimReview:
    claim_id: str
    claim: str
    decision: str = "retain"
    merge_target: str | None = None
    reason: str = ""


@dataclass(frozen=True)
class CandidateClaim:
    claim: str
    explanation: str = ""
    labels: tuple[str, ...] = ()


def parse_candidate_claims(text: str, limit: int = 10) -> list[CandidateClaim]:
    """Parse the small text format without depending on LLM-controlled JSON."""
    # Local models often add Markdown and omit the requested item number.  A
    # Claim field is the reliable boundary in both cases.
    starts = list(re.finditer(r"(?im)^\s*(?:\d{1,2}[.)]\s*)?(?:\*{1,2}\s*)?claim\s*(?:\*{1,2})?\s*:", text))
    items = [text[match.start() : starts[index + 1].start() if index + 1 < len(starts) else len(text)] for index, match in enumerate(starts)]
    if not items:
        items = re.split(r"(?m)^\s*(?:\d{1,2}[.)]|[-*])\s+", text.strip())
    cleaned: list[CandidateClaim] = []
    seen = set()
    for item in items:
        if not item.strip():
            continue
        fields = _fields(item)
        raw_claim = fields.get("claim") or item.splitlines()[0]
        claim = re.sub(r"\s+", " ", raw_claim).strip(" -•\t")
        explanation = re.sub(r"\s+", " ", fields.get("explanation", "")).strip()
        labels = tuple(label.strip() for label in fields.get("labels", "").split(",") if 1 < len(label.strip()) <= 48)[:3]
        key = re.sub(r"[^a-z0-9]+", " ", claim.lower()).strip()
        if len(claim) < 20 or key in seen or _looks_like_metadata(claim):
            continue
        seen.add(key)
        cleaned.append(CandidateClaim(claim, explanation, labels))
        if len(cleaned) >= limit:
            break
    return cleaned


def parse_discovered_claims(text: str, limit: int = 10) -> list[str]:
    """Compatibility helper for earlier callers."""
    return [candidate.claim for candidate in parse_candidate_claims(text, limit)]


def parse_label_suggestions(text: str, claims: list[str]) -> dict[str, list[str]]:
    """Parse only claim IDs and short labels; malformed output is harmless."""
    valid = {f"C{index}" for index in range(1, len(claims) + 1)}
    result = {claim_id: [] for claim_id in valid}
    for line in text.splitlines():
        match = re.match(r"^\s*(C\d+)\s*:\s*(.+?)\s*$", line, flags=re.IGNORECASE)
        if not match or match.group(1).upper() not in valid:
            continue
        labels = [item.strip() for item in match.group(2).split(",") if 1 < len(item.strip()) <= 48]
        result[match.group(1).upper()] = labels[:3]
    return result


def parse_claim_review(text: str, claims: list[str]) -> list[ClaimReview]:
    valid = {f"C{index}": claim for index, claim in enumerate(claims, start=1)}
    reviews: dict[str, ClaimReview] = {claim_id: ClaimReview(claim_id, claim) for claim_id, claim in valid.items()}
    for block in re.split(r"(?m)^---+\s*$", text):
        fields = _fields(block)
        claim_id = fields.get("claim")
        decision = fields.get("decision", "retain").lower()
        target = fields.get("merge with") or None
        if claim_id not in valid or decision not in {"retain", "merge", "discard"}:
            continue
        if decision == "merge" and target not in valid:
            continue
        reviews[claim_id] = ClaimReview(claim_id, valid[claim_id], decision, target, fields.get("reason", ""))
    return list(reviews.values())


def retained_claims(reviews: list[ClaimReview]) -> list[str]:
    merged = {review.claim_id for review in reviews if review.decision == "merge"}
    return [review.claim for review in reviews if review.decision == "retain" and review.claim_id not in merged]


def _fields(block: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in block.splitlines():
        match = re.match(r"^\s*(?:[-#]\s*)?(?:\*{1,2}\s*)?([A-Za-z][A-Za-z ]*?)(?:\s*\*{1,2})?\s*:\s*(.*)$", line)
        if match:
            value = re.sub(r"^\*+\s*|\s*\*+$", "", match.group(2).strip())
            fields[match.group(1).strip().lower()] = value
    return fields


def _looks_like_metadata(value: str) -> bool:
    normalized = value.lower()
    return any(token in normalized for token in ("creative commons", "all rights reserved", "ieee software", "http://", "https://"))
