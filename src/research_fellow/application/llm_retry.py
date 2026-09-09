"""Retry policy and durable failure records for long-running LLM workflows."""
from __future__ import annotations

import json
import random
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable

Draft = Callable[[str], str | None]
Validator = Callable[[str], bool]


@dataclass
class LLMRetryExhausted(RuntimeError):
    stage: str
    error_type: str
    message: str
    attempts: int
    recommended_action: str
    prompt: str = ""

    def __str__(self) -> str:
        return f"{self.stage}: {self.error_type} after {self.attempts} attempts - {self.message}"


def classify_output(text: str | None, validator: Validator | None = None) -> tuple[bool, str, str]:
    if text is None or not str(text).strip():
        return False, "empty_response", "LLM이 빈 응답을 반환했습니다."
    value = str(text).strip()
    # Common truncation signals for structured output. Validation below is the
    # stronger signal; these heuristics catch obviously unfinished responses.
    if value.count("{") > value.count("}") or value.count("[") > value.count("]"):
        return False, "truncated_response", "구조화 응답이 중간에서 잘린 것으로 보입니다."
    if validator is not None:
        try:
            if not validator(value):
                return False, "schema_validation_error", "필수 구조 또는 필드를 만족하지 못했습니다."
        except (ValueError, TypeError, json.JSONDecodeError) as error:
            return False, "json_parse_error", str(error)
    return True, "", ""


def recommendation_for(error_type: str, stage: str) -> str:
    if error_type == "empty_response":
        return "동일 입력으로 다시 시도하세요. 반복되면 모델 연결 상태와 API 설정을 확인하세요."
    if error_type == "truncated_response":
        return "출력 길이를 줄이고 strict 형식으로 다시 시도하세요. 초록 검토라면 batch 크기를 줄이는 것을 권장합니다."
    if error_type in {"json_parse_error", "schema_validation_error"}:
        return "설명문 없이 필수 필드만 반환하도록 recovery prompt로 다시 시도하세요."
    if error_type in {"timeout", "provider_error"}:
        return "일시적 호출 실패일 수 있으므로 동일 단계부터 다시 시도하세요."
    return f"{stage} 단계의 입력과 프롬프트를 확인한 뒤 실패 단계부터 다시 시도하세요."


def recovery_prompt(prompt: str, error_type: str, attempt: int) -> str:
    suffix = [
        "",
        "--- RETRY INSTRUCTION ---",
        f"Previous attempt failed with: {error_type}.",
        "Return a complete response. Do not stop mid-item.",
    ]
    if error_type in {"truncated_response", "json_parse_error", "schema_validation_error"}:
        suffix.extend([
            "Use a shorter response than before.",
            "Return only the required structure and fields; omit commentary and repetition.",
            "Make every requested item concise (1-3 short sentences per field).",
        ])
    if attempt >= 3:
        suffix.append("This is the final automatic retry. Prefer completeness and valid structure over detail.")
    return prompt + "\n" + "\n".join(suffix)


def call_with_retry(
    draft: Draft,
    prompt: str,
    *,
    stage: str,
    validator: Validator | None = None,
    max_attempts: int = 3,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[str, int]:
    """Call an LLM with bounded automatic retry and output validation."""
    current_prompt = prompt
    last_type, last_message = "unexpected_error", "알 수 없는 오류"
    for attempt in range(1, max(1, max_attempts) + 1):
        try:
            response = draft(current_prompt)
            ok, error_type, message = classify_output(response, validator)
            if ok:
                return str(response), attempt
            last_type, last_message = error_type, message
        except TimeoutError as error:
            last_type, last_message = "timeout", str(error) or "LLM 호출 시간이 초과되었습니다."
        except (ConnectionError, OSError) as error:
            last_type, last_message = "provider_error", str(error)
        if attempt < max_attempts:
            current_prompt = recovery_prompt(prompt, last_type, attempt + 1)
            # Keep retries responsive in the Streamlit request while still avoiding
            # immediate hammering of a transient provider failure.
            sleep(min(0.35 * (2 ** (attempt - 1)) + random.random() * 0.1, 1.5))
    raise LLMRetryExhausted(
        stage=stage,
        error_type=last_type,
        message=last_message,
        attempts=max_attempts,
        recommended_action=recommendation_for(last_type, stage),
        prompt=recovery_prompt(prompt, last_type, max_attempts + 1),
    )


def failure_payload(error: LLMRetryExhausted, *, context: dict[str, Any] | None = None) -> dict[str, Any]:
    merged_context = dict(context or {})
    if error.prompt:
        merged_context.setdefault("recovery_prompt", error.prompt)
    return {
        "failure_id": f"arf-{uuid.uuid4().hex[:12]}",
        "stage": error.stage,
        "attempt_count": error.attempts,
        "error_type": error.error_type,
        "error_message": error.message,
        "recommended_action": error.recommended_action,
        "context": merged_context,
    }
