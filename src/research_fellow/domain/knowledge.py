from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator


SourceKind = Literal["external_paper", "researcher_published_work", "researcher_idea_note"]
RelationType = Literal["supports", "extends", "contradicts", "qualifies", "uses_method", "addresses_gap"]


class Evidence(BaseModel):
    """A page-addressable extract; the original document remains the authority."""

    excerpt: str = Field(min_length=1, max_length=1600)
    pages: list[int] = Field(min_length=1)
    citation_markers: list[str] = Field(min_length=1)
    section: str | None = None

    @field_validator("pages")
    @classmethod
    def pages_are_positive(cls, value: list[int]) -> list[int]:
        if any(page < 1 for page in value):
            raise ValueError("근거 쪽수는 1 이상이어야 합니다.")
        return sorted(set(value))


class KnowledgeCard(BaseModel):
    """The only card shape permitted in approved JSONL semantic memory."""

    card_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    source_kind: SourceKind
    claim: str = Field(min_length=8)
    labels: list[str] = Field(default_factory=list)
    evidence_excerpt: str = Field(min_length=1, max_length=1600)
    evidence_pages: list[int] = Field(min_length=1)
    citation_markers: list[str] = Field(min_length=1)
    conditions: str = Field(min_length=1)
    limits: str = Field(min_length=1)
    provenance: dict[str, str]

    @field_validator("evidence_pages")
    @classmethod
    def evidence_pages_are_positive(cls, value: list[int]) -> list[int]:
        if any(page < 1 for page in value):
            raise ValueError("근거 쪽수는 1 이상이어야 합니다.")
        return sorted(set(value))


class KnowledgeRelation(BaseModel):
    """An approved, evidence-bearing assertion about two approved knowledge cards."""

    relation_id: str = Field(min_length=1)
    source_card_id: str = Field(min_length=1)
    target_card_id: str = Field(min_length=1)
    relation_type: RelationType
    evidence: str = Field(min_length=8)
    conditions: str = Field(min_length=1)
    confidence: Literal["low", "medium", "high"]

    @field_validator("target_card_id")
    @classmethod
    def cards_must_differ(cls, target: str, info: object) -> str:
        source = getattr(info, "data", {}).get("source_card_id")
        if source == target:
            raise ValueError("관계의 출발·도착 카드는 달라야 합니다.")
        return target


FIELD_ALIASES = {
    "title": "title", "제목": "title", "claim": "claim", "주장": "claim",
    "evidence": "evidence_excerpt", "evidence_excerpt": "evidence_excerpt", "근거": "evidence_excerpt",
    "pages": "evidence_pages", "쪽수": "evidence_pages", "citation": "citation_markers", "인용": "citation_markers",
    "labels": "labels", "레이블": "labels", "conditions": "conditions", "조건": "conditions",
    "limits": "limits", "한계": "limits",
}


def parse_text_draft(text: str) -> dict[str, str]:
    """Parse a deliberately simple text draft, never an LLM-controlled JSON object."""
    fields: dict[str, str] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        normalized = FIELD_ALIASES.get(key.strip().lower())
        if normalized and value.strip():
            fields[normalized] = value.strip()
    return fields


def page_markers(page: int, text: str) -> list[str]:
    doi = re.findall(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", text, flags=re.IGNORECASE)
    return [f"p.{page}", *sorted(set(doi))]
