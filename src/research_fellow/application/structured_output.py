"""Small shared helpers for defensive extraction of structured LLM output."""
from __future__ import annotations

import json
import re
from typing import Any, TypeVar

T = TypeVar("T")


def strip_json_fence(text: str) -> str:
    value = (text or "").strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", value, flags=re.I | re.S)
    if fenced:
        return fenced.group(1).strip()
    return re.sub(r"^```(?:json)?\s*|\s*```$", "", value, flags=re.I | re.S).strip()


def extract_json_value(text: str) -> Any:
    """Extract a JSON object/array from an LLM response without repairing content.

    This is intentionally conservative. Modules that need domain-specific JSON
    repair (for example literature discovery) should keep their stronger parser.
    """
    raw = strip_json_fence(text)
    if not raw:
        raise ValueError("LLM 응답이 비어 있습니다.")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        positions = [(raw.find("{"), "}"), (raw.find("["), "]")]
        positions = [(pos, closer) for pos, closer in positions if pos >= 0]
        if not positions:
            raise ValueError("JSON 응답을 찾을 수 없습니다.")
        start, closer = min(positions, key=lambda item: item[0])
        end = raw.rfind(closer)
        if end <= start:
            raise ValueError("JSON 응답이 중간에서 잘린 것으로 보입니다.")
        try:
            return json.loads(raw[start : end + 1])
        except json.JSONDecodeError as error:
            raise ValueError(f"JSON 형식을 해석할 수 없습니다: {error}") from error


def extract_json_object(text: str, *, message: str = "응답은 JSON object여야 합니다.") -> dict[str, Any]:
    value = extract_json_value(text)
    if not isinstance(value, dict):
        raise ValueError(message)
    return value
