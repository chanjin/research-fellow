from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class OntologyFacet(BaseModel):
    """Meta-category used to organize ontology types along a semantic axis."""

    facet_id: str = Field(min_length=1)
    name: str = Field(min_length=1, max_length=120)
    description: str = Field(default="", max_length=1200)


class OntologyType(BaseModel):
    """Researcher-defined schema type assigned to one or more approved knowledge cards."""

    type_id: str = Field(min_length=1)
    name: str = Field(min_length=1, max_length=120)
    description: str = Field(default="", max_length=1200)
    facet_id: str | None = None


class OntologyTypeRelation(BaseModel):
    """Directed, researcher-defined relation between ontology types."""

    relation_id: str = Field(min_length=1)
    source_type_id: str = Field(min_length=1)
    target_type_id: str = Field(min_length=1)
    relation_name: str = Field(min_length=1, max_length=120)
    description: str = Field(default="", max_length=1200)

    @field_validator("target_type_id")
    @classmethod
    def types_must_differ(cls, target: str, info: object) -> str:
        source = getattr(info, "data", {}).get("source_type_id")
        if source == target:
            raise ValueError("타입 관계의 출발·도착 타입은 달라야 합니다.")
        return target
