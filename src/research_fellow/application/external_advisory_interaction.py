"""Application service for accepting an interpreted external advisory request."""
from __future__ import annotations

from typing import Any, Mapping

from research_fellow.application.input_interaction import create_external_advisory_thread


def apply_external_advisory_interpretation(
    ledger: Any,
    request: Mapping[str, Any],
    review: Mapping[str, Any],
) -> dict[str, Any]:
    interpreted_question = str(review.get("interpreted_question") or "").strip()
    interpretation = str(review.get("interpretation") or "").strip()
    if not interpreted_question:
        raise ValueError("interpreted advisory question is required")
    return create_external_advisory_thread(
        ledger,
        request,
        interpreted_question=interpreted_question,
        interpretation=interpretation,
    )
