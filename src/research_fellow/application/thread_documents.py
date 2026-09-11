from __future__ import annotations

from typing import Any


def current_state_prompt(*, thread_kind: str, title: str, current_question: str, prior_state: str, conversation: list[dict[str, Any]] | None = None, reports: list[dict[str, Any]] | None = None) -> str:
    turns = conversation or []
    report_items = reports or []
    history = "\n\n".join(f"{str(t.get('role','')).upper()}: {str(t.get('content',''))}" for t in turns[-10:]) or "(none)"
    report_text = "\n\n".join(str(r.get("report_text") or r.get("body_text") or "") for r in report_items[:3]) or "(none)"
    return f"""# Thread Current State Update

You maintain one living current-state document for a long-running research thread.
Do not write a chronological transcript. Produce the best compact representation of the CURRENT state after incorporating the latest evidence, conversation and reports.
When the question has evolved, preserve the earlier direction briefly under Question Evolution, but make the latest question explicit.
Separate supported findings from interpretation. Preserve unresolved issues instead of forcing closure.

Thread kind: {thread_kind}
Thread title: {title}
Current question: {current_question}

## Prior Current State
{prior_state or '(first update)'}

## Recent Conversation
{history}

## Recent M2 Reports
{report_text}

## Required document structure
### 현재 질문
### 질문의 변화
### 현재까지 확인된 핵심 사실
### 현재 해석 / 잠정 결론
### 핵심 근거
### 남은 불확실성 / 논쟁점
### 연구자의 현재 관심 방향
### 다음 탐색 또는 질문 후보

Write in Korean. Keep it concise but complete enough that a researcher can reopen the thread later and immediately understand where the work stands.
"""


def report_snapshot_prompt(*, thread_kind: str, title: str, current_state: str, current_question: str) -> str:
    return f"""# Current Research Conclusion Report

Create a report snapshot representing the conclusion that can be responsibly stated NOW for this research thread.
This is a point-in-time report, not a living document. Do not overstate unresolved issues.

Thread kind: {thread_kind}
Title: {title}
Current question: {current_question}

## Current State Document
{current_state}

## Required report structure
# {title}
## 질문 / 목적
## Executive Summary
## 핵심 결론
## 근거와 해석
## 반대 근거 / 한계
## 아직 검증이 필요한 사항
## 향후 연구 제안

Write in Korean. Make the report suitable for saving and revisiting later.
"""
