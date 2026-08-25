"""Non-authoritative Ollama draft generation with actionable diagnostics."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

import requests


LOGGER = logging.getLogger(__name__)
DEFAULT_OLLAMA_URL = "http://localhost:11434"


@dataclass(frozen=True)
class OllamaDraftResult:
    """A draft or a user-readable failure; it never controls application state."""

    text: str | None
    error: str | None = None
    status_code: int | None = None

    @property
    def ok(self) -> bool:
        return bool(self.text)


def ollama_base_url() -> str:
    """Allow a non-default local endpoint without hard-coding environment details."""
    return os.getenv("OLLAMA_HOST", DEFAULT_OLLAMA_URL).rstrip("/")


def ollama_draft_result(prompt: str, model: str, enabled: bool) -> OllamaDraftResult:
    """Generate prose and retain enough diagnostic information to fix failures."""
    if not enabled:
        return OllamaDraftResult(None, "Ollama 초안 사용 옵션이 꺼져 있습니다.")

    endpoint = f"{ollama_base_url()}/api/generate"
    try:
        response = requests.post(
            endpoint,
            json={"model": model, "prompt": prompt, "stream": False, "options": {"temperature": 0.2}},
            timeout=90,
        )
    except requests.Timeout as error:
        LOGGER.warning("Ollama draft timed out after 90 seconds: model=%s endpoint=%s", model, endpoint)
        return OllamaDraftResult(None, "Ollama 응답이 90초 안에 완료되지 않았습니다. 모델을 확인하거나 더 짧은 문서로 시도하세요.")
    except requests.ConnectionError as error:
        LOGGER.warning("Ollama connection failed: model=%s endpoint=%s error=%s", model, endpoint, error)
        return OllamaDraftResult(
            None,
            f"Ollama 서버에 연결할 수 없습니다 ({ollama_base_url()}). `ollama serve` 실행 여부를 확인하세요.",
        )
    except requests.RequestException as error:
        LOGGER.warning("Ollama request failed: model=%s endpoint=%s error=%s", model, endpoint, error)
        return OllamaDraftResult(None, f"Ollama 요청을 완료하지 못했습니다: {error}")

    if not response.ok:
        detail = _response_error_detail(response)
        LOGGER.warning("Ollama returned HTTP %s: model=%s detail=%s", response.status_code, model, detail)
        if response.status_code == 404:
            detail = f"모델 `{model}`을 찾지 못했습니다. `ollama list`로 설치 모델을 확인하세요. {detail}".strip()
        return OllamaDraftResult(None, detail, response.status_code)

    try:
        payload: dict[str, Any] = response.json()
    except ValueError:
        LOGGER.warning("Ollama returned a non-JSON response: model=%s", model)
        return OllamaDraftResult(None, "Ollama 응답이 JSON 형식이 아닙니다. Streamlit 실행 로그를 확인하세요.")

    if payload.get("error"):
        detail = str(payload["error"])
        LOGGER.warning("Ollama returned an API error: model=%s detail=%s", model, detail)
        return OllamaDraftResult(None, detail, response.status_code)
    text = str(payload.get("response") or "").strip()
    if not text:
        LOGGER.warning("Ollama returned an empty draft: model=%s", model)
        return OllamaDraftResult(None, "Ollama가 빈 응답을 반환했습니다. 모델 로그와 입력 문서 크기를 확인하세요.")
    return OllamaDraftResult(text)


def ollama_draft(prompt: str, model: str, enabled: bool) -> str | None:
    """Compatibility wrapper for call sites that only need best-effort prose."""
    return ollama_draft_result(prompt, model, enabled).text


def ollama_status(model: str) -> tuple[bool, str]:
    """Check connection and model availability without changing workflow state."""
    endpoint = f"{ollama_base_url()}/api/tags"
    try:
        response = requests.get(endpoint, timeout=3)
        response.raise_for_status()
        models = {str(item.get("name")) for item in response.json().get("models", [])}
        if model in models:
            return True, f"연결됨 · {model} 사용 가능"
        available = ", ".join(sorted(models)) or "없음"
        return False, f"연결됨 · `{model}` 모델 없음 (설치됨: {available})"
    except requests.ConnectionError:
        return False, f"Ollama에 연결할 수 없음 ({ollama_base_url()})"
    except requests.Timeout:
        return False, "Ollama 상태 확인 시간이 초과되었습니다. 서버 상태를 확인하세요."
    except (requests.RequestException, ValueError) as error:
        LOGGER.warning("Ollama status check failed: endpoint=%s error=%s", endpoint, error)
        return False, f"Ollama 상태 확인 실패: {error}"


def _response_error_detail(response: requests.Response) -> str:
    try:
        payload = response.json()
        if isinstance(payload, dict) and payload.get("error"):
            return str(payload["error"])
    except ValueError:
        pass
    body = response.text.strip().replace("\n", " ")
    return body[:400] or f"Ollama가 HTTP {response.status_code} 오류를 반환했습니다."
