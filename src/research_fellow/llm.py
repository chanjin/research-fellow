"""Non-authoritative Ollama draft generation with actionable diagnostics."""

from __future__ import annotations

import logging
import os
import time
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import requests


LOGGER = logging.getLogger(__name__)
DEFAULT_OLLAMA_URL = "http://localhost:11434"
# GEMINI_MODEL이 설정되어 있으면 그 값을 우선 사용한다. 기본값은 현재
# 안정적으로 사용할 수 있는 Flash 계열 모델로 둔다.
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"

LLM_PROFILES: dict[str, dict[str, object]] = {
    "default": {"temperature": 0.2, "think": "medium", "num_ctx": 4096, "num_predict": 1600, "timeout_seconds": 200, "min_response_chars": 240, "max_short_retries": 1},
    "abstract_triage": {"temperature": 0.1, "think": "low", "num_ctx": 4096, "num_predict": 900, "timeout_seconds": 150, "min_response_chars": 80, "max_short_retries": 1},
    "full_text_similarity": {"temperature": 0.15, "think": "low", "num_ctx": 6144, "num_predict": 800, "timeout_seconds": 220, "min_response_chars": 100, "max_short_retries": 1},
    "p1_card_draft": {"temperature": 0.2, "think": "medium", "num_ctx": 6144, "num_predict": 1400, "timeout_seconds": 260, "min_response_chars": 180, "max_short_retries": 1},
    # gpt-oss:20b on a 16GB Mac can swap heavily above an 8K context. The
    # paper-reading prompt is capped below that, so match Ollama Chat's 8K
    # setting explicitly for this one long-context workload.
    # 논문 요약과 근거 기반 읽기 질문은 한 요청에 함께 생성한다. 질문 수는
    # 최대 10개이나, 완결된 응답을 우선하기 위해 충분한 출력 여유를 둔다.
    "paper_reading": {"temperature": 0.15, "think": "low", "num_ctx": 8192, "num_predict": 6000, "timeout_seconds": 360, "min_response_chars": 180, "max_short_retries": 1},
    "m2_report": {"temperature": 0.25, "think": "medium", "num_ctx": 6144, "num_predict": 1800, "timeout_seconds": 260, "min_response_chars": 300, "max_short_retries": 1},
}
_AUDIT_LOGGER: Any | None = None
_AUDIT_LOG_PATH: Path | None = None
_OLLAMA_MODEL_DEFAULTS: dict[str, dict[str, object]] = {}


def set_llm_audit_logger(logger: Any | None) -> None:
    """Accept a Ledger.record_llm_call-compatible function without importing storage."""
    global _AUDIT_LOGGER
    _AUDIT_LOGGER = logger


def set_llm_audit_log_path(path: str | Path | None) -> None:
    """Set an append-only JSONL mirror for calls without coupling LLM code to SQLite."""
    global _AUDIT_LOG_PATH
    _AUDIT_LOG_PATH = Path(path) if path else None


def llm_audit_log_path() -> Path | None:
    """Return the configured JSONL audit location for UI download and diagnostics."""
    return _AUDIT_LOG_PATH


def llm_profile(name: str | None, overrides: dict[str, object] | None = None) -> tuple[str, dict[str, object]]:
    profile_name = name or "default"
    settings = dict(LLM_PROFILES.get(profile_name, LLM_PROFILES["default"]))
    if overrides:
        settings.update({key: value for key, value in overrides.items() if value is not None})
    return profile_name, settings


def _positive_int_env(name: str, default: int) -> int:
    try:
        return max(1, int(os.getenv(name, str(default))))
    except ValueError:
        return default


@dataclass(frozen=True)
class OllamaDraftResult:
    """A draft or a user-readable failure; it never controls application state."""

    text: str | None
    error: str | None = None
    status_code: int | None = None
    diagnostics: dict[str, object] | None = None

    @property
    def ok(self) -> bool:
        return bool(self.text) and not self.error


def ollama_base_url() -> str:
    """Allow a non-default local endpoint without hard-coding environment details."""
    return os.getenv("OLLAMA_HOST", DEFAULT_OLLAMA_URL).rstrip("/")


def _think_level() -> str:
    value = os.getenv("OLLAMA_THINK_LEVEL", "medium").lower()
    return value if value in {"low", "medium", "high"} else "medium"


def _parse_ollama_parameters(parameters: str) -> dict[str, object]:
    """Parse the Modelfile-style `parameters` text returned by `/api/show`."""
    parsed: dict[str, object] = {}
    for line in parameters.splitlines():
        key, separator, value = line.strip().partition(" ")
        if not key or not separator or not value.strip():
            continue
        value = value.strip()
        try:
            parsed[key] = int(value)
        except ValueError:
            try:
                parsed[key] = float(value)
            except ValueError:
                parsed[key] = value
    return parsed


def ollama_model_defaults(model: str) -> dict[str, object]:
    """Read and cache the model's Ollama/Modelfile defaults without failing a draft."""
    if model in _OLLAMA_MODEL_DEFAULTS:
        return _OLLAMA_MODEL_DEFAULTS[model]
    try:
        response = requests.post(f"{ollama_base_url()}/api/show", json={"model": model}, timeout=5)
        payload = response.json() if response.ok else {}
        raw_parameters = str(payload.get("parameters") or "")
        defaults: dict[str, object] = {
            "parameters": _parse_ollama_parameters(raw_parameters),
            "raw_parameters": raw_parameters,
        }
    except (requests.RequestException, ValueError):
        # Older Ollama versions or a temporary connection error must not block
        # a model request; Ollama will still apply its own server defaults.
        defaults = {"parameters": {}, "raw_parameters": ""}
    _OLLAMA_MODEL_DEFAULTS[model] = defaults
    return defaults


def use_ollama_model_defaults() -> bool:
    """Use the model's own settings unless an operator explicitly opts out."""
    return os.getenv("OLLAMA_USE_MODEL_DEFAULTS", "1").strip().lower() not in {"0", "false", "no"}


def text_size_metrics(text: str, prefix: str) -> dict[str, int]:
    """Cheap, model-agnostic prompt-size diagnostics.

    Ollama reports exact token counts only after a successful response. The
    estimate is deliberately labelled as an estimate: UTF-8 bytes divided by
    four is useful for spotting unexpectedly large prompts across languages,
    but is not a tokenizer replacement.
    """
    encoded = text.encode("utf-8")
    return {
        f"{prefix}_chars": len(text),
        f"{prefix}_utf8_bytes": len(encoded),
        f"{prefix}_estimated_tokens": (len(encoded) + 3) // 4,
    }


def ollama_draft_result(
    prompt: str,
    model: str,
    enabled: bool,
    profile: str | None = None,
    overrides: dict[str, object] | None = None,
    on_chunk: Callable[[str], None] | None = None,
) -> OllamaDraftResult:
    """Generate an Ollama draft, optionally streaming visible text to a caller."""
    if not enabled:
        return OllamaDraftResult(None, "Ollama 초안 사용 옵션이 꺼져 있습니다.")

    profile_name, settings = llm_profile(profile, overrides)
    endpoint = f"{ollama_base_url()}/api/generate"
    timeout_seconds = int(settings["timeout_seconds"])
    min_response_chars = int(settings["min_response_chars"])
    max_retries = int(settings["max_short_retries"])
    model_defaults = ollama_model_defaults(model)
    use_model_defaults = use_ollama_model_defaults()
    request_prompt = prompt
    longest_short_text = ""
    for attempt in range(max_retries + 1):
        started = time.monotonic()
        request_sizes = {
            **text_size_metrics(prompt, "original_prompt"),
            **text_size_metrics(request_prompt, "request_prompt"),
        }
        try:
            request_payload: dict[str, object] = {
                "model": model,
                "prompt": request_prompt,
                "stream": bool(on_chunk),
                "keep_alive": os.getenv("OLLAMA_KEEP_ALIVE", "10m"),
            }
            # Chat normally leaves Modelfile defaults in control. Profiles still
            # set retry/timeout policy, but do not override the model by default.
            effective_options: dict[str, object] = {}
            # `/api/show` cannot read the Ollama Chat application's per-chat
            # 8K setting. Keep model defaults for ordinary calls, but make the
            # long paper-reading call explicitly match that safe context size.
            if profile_name == "paper_reading":
                effective_options["num_ctx"] = settings["num_ctx"]
            if not use_model_defaults:
                request_payload.update({
                    "think": str(settings["think"]),
                })
                effective_options.update({
                    "temperature": settings["temperature"],
                    "num_ctx": settings["num_ctx"],
                    "num_predict": settings["num_predict"],
                })
            if effective_options:
                request_payload["options"] = effective_options
            response = requests.post(
                endpoint,
                json=request_payload,
                timeout=timeout_seconds,
                stream=bool(on_chunk),
            )
        except requests.Timeout:
            LOGGER.warning("Ollama draft timed out after %s seconds: model=%s endpoint=%s", timeout_seconds, model, endpoint)
            return _audit(OllamaDraftResult(None, f"Ollama 응답이 {timeout_seconds}초 안에 완료되지 않았습니다. 모델을 확인하거나 더 짧은 문서로 시도하세요."), profile_name, model, request_prompt, settings, {**request_sizes, "attempt": attempt + 1, "elapsed_seconds": round(time.monotonic() - started, 2)})
        except requests.ConnectionError as error:
            LOGGER.warning("Ollama connection failed: model=%s endpoint=%s error=%s", model, endpoint, error)
            return _audit(OllamaDraftResult(None, f"Ollama 서버에 연결할 수 없습니다 ({ollama_base_url()}). `ollama serve` 실행 여부를 확인하세요."), profile_name, model, request_prompt, settings, {**request_sizes, "attempt": attempt + 1, "elapsed_seconds": round(time.monotonic() - started, 2)})
        except requests.RequestException as error:
            LOGGER.warning("Ollama request failed: model=%s endpoint=%s error=%s", model, endpoint, error)
            return _audit(OllamaDraftResult(None, f"Ollama 요청을 완료하지 못했습니다: {error}"), profile_name, model, request_prompt, settings, {**request_sizes, "attempt": attempt + 1, "elapsed_seconds": round(time.monotonic() - started, 2)})

        if not response.ok:
            detail = _response_error_detail(response)
            LOGGER.warning("Ollama returned HTTP %s: model=%s detail=%s", response.status_code, model, detail)
            if response.status_code == 404:
                detail = f"모델 `{model}`을 찾지 못했습니다. `ollama list`로 설치 모델을 확인하세요. {detail}".strip()
            return _audit(OllamaDraftResult(None, detail, response.status_code), profile_name, model, request_prompt, settings, {**request_sizes, "attempt": attempt + 1, "elapsed_seconds": round(time.monotonic() - started, 2)})
        if on_chunk:
            payload: dict[str, Any] = {}
            chunks: list[str] = []
            try:
                for raw_line in response.iter_lines(decode_unicode=True):
                    if not raw_line:
                        continue
                    payload = json.loads(raw_line)
                    if payload.get("error"):
                        break
                    fragment = str(payload.get("response") or "")
                    if fragment:
                        chunks.append(fragment)
                        try:
                            on_chunk(fragment)
                        except Exception:  # UI feedback must never cancel the stored result
                            LOGGER.debug("Ollama stream callback failed", exc_info=True)
                text = "".join(chunks).strip()
            except (ValueError, requests.RequestException) as error:
                LOGGER.warning("Ollama stream was not valid JSON: model=%s error=%s", model, error)
                return _audit(OllamaDraftResult(None, "Ollama 스트리밍 응답을 해석하지 못했습니다."), profile_name, model, request_prompt, settings, {**request_sizes, "attempt": attempt + 1, "elapsed_seconds": round(time.monotonic() - started, 2)})
        else:
            try:
                payload = response.json()
            except ValueError:
                LOGGER.warning("Ollama returned a non-JSON response: model=%s", model)
                return _audit(OllamaDraftResult(None, "Ollama 응답이 JSON 형식이 아닙니다. Streamlit 실행 로그를 확인하세요."), profile_name, model, request_prompt, settings, {**request_sizes, "attempt": attempt + 1, "elapsed_seconds": round(time.monotonic() - started, 2)})
            text = str(payload.get("response") or "").strip()
        if payload.get("error"):
            detail = str(payload["error"])
            LOGGER.warning("Ollama returned an API error: model=%s detail=%s", model, detail)
            return _audit(OllamaDraftResult(None, detail, response.status_code), profile_name, model, request_prompt, settings, {**request_sizes, "attempt": attempt + 1, "elapsed_seconds": round(time.monotonic() - started, 2)})
        diagnostics = {
            **request_sizes,
            "use_model_defaults": use_model_defaults,
            "model_default_parameters": model_defaults.get("parameters", {}),
            "effective_request_options": effective_options,
            "attempt": attempt + 1,
            "done_reason": payload.get("done_reason"),
            **text_size_metrics(text, "response"),
            "thinking_chars": len(str(payload.get("thinking") or "")),
            "prompt_eval_count": payload.get("prompt_eval_count"),
            "eval_count": payload.get("eval_count"),
            "load_seconds": round(float(payload.get("load_duration") or 0) / 1_000_000_000, 3),
            "prompt_eval_seconds": round(float(payload.get("prompt_eval_duration") or 0) / 1_000_000_000, 3),
            "eval_seconds": round(float(payload.get("eval_duration") or 0) / 1_000_000_000, 3),
            "elapsed_seconds": round(time.monotonic() - started, 2),
        }
        if text.upper() == "NO_CANDIDATE" or len(text) >= min_response_chars:
            return _audit(OllamaDraftResult(text, diagnostics=diagnostics), profile_name, model, request_prompt, settings, diagnostics)
        longest_short_text = max(longest_short_text, text, key=len)
        LOGGER.warning("Ollama returned a short draft (%s/%s chars), attempt %s/%s: model=%s", len(text), min_response_chars, attempt + 1, max_retries + 1, model)
        request_prompt = (
            f"{prompt}\n\nYour previous response was incomplete or too short. "
            "Return a complete answer in the same language and format requested in the prompt."
        )
    if longest_short_text:
        return _audit(OllamaDraftResult(longest_short_text, f"Ollama returned a short response after {max_retries} retries.", diagnostics=diagnostics), profile_name, model, request_prompt, settings, diagnostics)
    return _audit(OllamaDraftResult(None, f"Ollama returned empty responses after {max_retries} retries. Check the model log and input size.", diagnostics=diagnostics), profile_name, model, request_prompt, settings, diagnostics)


def ollama_draft(prompt: str, model: str, enabled: bool, profile: str | None = None, overrides: dict[str, object] | None = None) -> str | None:
    """Compatibility wrapper for call sites that only need best-effort prose."""
    return ollama_draft_result(prompt, model, enabled, profile, overrides).text


def gemini_api_available() -> bool:
    """External API is opt-in at the UI; the key is never stored in the ledger."""
    return bool(os.getenv("GEMINI_API_KEY", "").strip())


def _gemini_thinking_config(model: str, think: object) -> dict[str, object]:
    """Translate shared profiles into the provider's version-specific controls."""
    level = str(think or "medium").lower()
    if model.startswith("gemini-2.5-"):
        budgets = {"low": 1024, "medium": 4096, "high": 8192}
        return {"thinkingBudget": budgets.get(level, 4096)}
    # Gemini 3 models use a reasoning level instead of the legacy budget.
    return {"thinkingLevel": {"low": "low", "medium": "medium", "high": "high"}.get(level, "medium")}


def gemini_draft_result(prompt: str, profile: str | None = None, model: str | None = None) -> OllamaDraftResult:
    """Use the Gemini Developer API only after the caller obtained explicit UI consent."""
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    selected_model = model or os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
    profile_name, settings = llm_profile(profile)
    if not api_key:
        return OllamaDraftResult(None, "GEMINI_API_KEY가 설정되지 않아 외부 모델을 사용할 수 없습니다.")
    started = time.monotonic()
    request_sizes = text_size_metrics(prompt, "request_prompt")
    min_response_chars = int(settings["min_response_chars"])
    max_retries = int(settings["max_short_retries"])
    request_prompt = prompt
    longest_short_text = ""
    thinking_config = _gemini_thinking_config(selected_model, settings.get("think"))
    try:
        for attempt in range(max_retries + 1):
            response = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{selected_model}:generateContent",
                params={"key": api_key},
                json={"contents": [{"role": "user", "parts": [{"text": request_prompt}]}], "generationConfig": {
                    "temperature": settings["temperature"], "maxOutputTokens": settings["num_predict"],
                    "thinkingConfig": thinking_config,
                }},
                timeout=int(settings["timeout_seconds"]),
            )
            if not response.ok:
                return _audit(OllamaDraftResult(None, _response_error_detail(response), response.status_code), profile_name, selected_model, request_prompt, settings, {**request_sizes, "provider": "gemini", "attempt": attempt + 1, "elapsed_seconds": round(time.monotonic() - started, 2)})
            payload = response.json()
            candidate = payload.get("candidates", [{}])[0]
            parts = candidate.get("content", {}).get("parts", [])
            text = "".join(str(part.get("text", "")) for part in parts).strip()
            finish_reason = str(candidate.get("finishReason") or "UNKNOWN")
            diagnostics = {
                **request_sizes,
                "provider": "gemini",
                "requested_model": selected_model,
                "response_model": payload.get("modelVersion", selected_model),
                "finish_reason": finish_reason,
                "attempt": attempt + 1,
                "max_output_tokens": settings["num_predict"],
                "thinking_config": thinking_config,
                "prompt_token_count": payload.get("usageMetadata", {}).get("promptTokenCount"),
                "candidate_token_count": payload.get("usageMetadata", {}).get("candidatesTokenCount"),
                "thoughts_token_count": payload.get("usageMetadata", {}).get("thoughtsTokenCount"),
                "total_token_count": payload.get("usageMetadata", {}).get("totalTokenCount"),
                **text_size_metrics(text, "response"),
                "elapsed_seconds": round(time.monotonic() - started, 2),
            }
            if finish_reason == "MAX_TOKENS" and attempt < max_retries:
                longest_short_text = max(longest_short_text, text, key=len)
                request_prompt = (
                    f"{prompt}\n\nThe previous response reached its output limit before it was complete. "
                    "Return a compact but complete response: a 500–800 Korean-character summary and only one to three "
                    "highest-value question blocks. Keep each field to one sentence, except Evidence which has exactly two concise p.N hints."
                )
                continue
            if finish_reason not in {"STOP", "UNKNOWN"}:
                return _audit(OllamaDraftResult(None, f"Gemini가 '{finish_reason}' 사유로 출력을 중단했습니다. 이 결과는 저장하지 않았습니다.", diagnostics=diagnostics), profile_name, selected_model, request_prompt, settings, diagnostics)
            if text.upper() == "NO_CANDIDATE" or len(text) >= min_response_chars:
                return _audit(OllamaDraftResult(text, diagnostics=diagnostics), profile_name, selected_model, request_prompt, settings, diagnostics)
            longest_short_text = max(longest_short_text, text, key=len)
            request_prompt = (
                f"{prompt}\n\nYour previous response was only {len(text)} characters and incomplete. "
                "Return a complete response in the exact requested format. Prefer fewer complete items over partial content."
            )
        return _audit(OllamaDraftResult(None, f"Gemini가 {max_retries + 1}회 연속 너무 짧은 응답을 반환했습니다 ({len(longest_short_text)}자). 이 결과는 저장하지 않았습니다.", diagnostics=diagnostics), profile_name, selected_model, request_prompt, settings, diagnostics)
    except requests.Timeout:
        return _audit(OllamaDraftResult(None, "Gemini API 응답 시간이 초과되었습니다."), profile_name, selected_model, prompt, settings, {**request_sizes, "provider": "gemini", "elapsed_seconds": round(time.monotonic() - started, 2)})
    except (requests.RequestException, ValueError, KeyError, IndexError) as error:
        return _audit(OllamaDraftResult(None, f"Gemini API 요청을 완료하지 못했습니다: {error}"), profile_name, selected_model, prompt, settings, {**request_sizes, "provider": "gemini", "elapsed_seconds": round(time.monotonic() - started, 2)})


def _audit(result: OllamaDraftResult, profile_name: str, model: str, prompt: str, settings: dict[str, object], diagnostics: dict[str, object]) -> OllamaDraftResult:
    if _AUDIT_LOGGER:
        try:
            _AUDIT_LOGGER(profile_name=profile_name, model=model, prompt=prompt, settings=settings, response=result.text, error=result.error, diagnostics=diagnostics)
        except Exception as error:  # diagnostics must never break research work
            LOGGER.warning("Could not persist LLM audit record: %s", error)
    if _AUDIT_LOG_PATH:
        try:
            _AUDIT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            entry = {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "profile_name": profile_name,
                "model": model,
                "prompt": prompt,
                "settings": settings,
                "response": result.text or "",
                "error": result.error or "",
                "diagnostics": diagnostics,
            }
            with _AUDIT_LOG_PATH.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError as error:  # file diagnostics must never break research work
            LOGGER.warning("Could not append LLM JSONL audit record: %s", error)
    return result


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
