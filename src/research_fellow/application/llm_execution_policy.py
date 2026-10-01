"""Stage-level LLM execution policy.

Research-quality stages default to external manual execution until a capable local
server is available.  The policy is configuration, not workflow state: changing a
stage from external_manual to local does not require DSL or application changes.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any
import yaml

POLICY_PATH = Path(__file__).resolve().parents[1] / "llm_execution_policy.yaml"
LOCAL = "local"
EXTERNAL_MANUAL = "external_manual"
_ALLOWED = {LOCAL, EXTERNAL_MANUAL}

@lru_cache(maxsize=1)
def load_llm_execution_policy() -> dict[str, Any]:
    data = yaml.safe_load(POLICY_PATH.read_text(encoding="utf-8")) or {}
    if str(data.get("policy") or "") != "research-fellow-llm-execution/v0.1":
        raise ValueError("Unsupported LLM execution policy schema")
    default = str(data.get("default") or EXTERNAL_MANUAL)
    if default not in _ALLOWED:
        raise ValueError(f"Unsupported default LLM execution mode: {default}")
    stages = {str(k): str(v) for k, v in dict(data.get("stages") or {}).items()}
    bad = {k: v for k, v in stages.items() if v not in _ALLOWED}
    if bad:
        raise ValueError(f"Unsupported LLM execution modes: {bad}")
    return {"default": default, "stages": stages}


def llm_execution_mode(stage: str) -> str:
    policy = load_llm_execution_policy()
    return str(policy["stages"].get(str(stage), policy["default"]))
