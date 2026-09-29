"""Execution-run tracking for long-running agent workflows.

This module owns *execution observation state* only: current runtime stage,
retry metadata, manual recovery overrides and technical failure records.

It deliberately does not own business/domain state transitions such as
completing a research-state review or a curation intent. Those remain explicit
application/domain operations so the future workflow DSL can distinguish
business completion from runtime execution status.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from research_fellow.application.llm_retry import LLMRetryExhausted, failure_payload
from research_fellow.storage import Ledger


@dataclass
class ExecutionRunTracker:
    ledger: Ledger
    run_id: str

    @classmethod
    def start(
        cls,
        ledger: Ledger,
        *,
        run_id: str = "",
        review_id: str = "",
        intent_id: str = "",
        stage: str = "",
    ) -> "ExecutionRunTracker":
        actual_run_id = run_id or ledger.create_auto_research_run(
            review_id=review_id,
            intent_id=intent_id,
            stage=stage,
        )
        return cls(ledger=ledger, run_id=actual_run_id)

    def bind_review(self, review_id: str) -> None:
        self.ledger.bind_auto_research_run_review(self.run_id, review_id)

    def enter(self, stage: str, *, retry_count: int | None = None) -> None:
        self.ledger.update_auto_research_run(
            self.run_id,
            stage=stage,
            retry_count=retry_count,
        )

    def note_attempts(self, attempts: int) -> None:
        self.ledger.update_auto_research_run(
            self.run_id,
            retry_count=max(0, int(attempts) - 1),
        )

    def complete(self, *, stage: str = "completed") -> None:
        self.ledger.update_auto_research_run(
            self.run_id,
            status="completed",
            stage=stage,
        )

    def needs_attention(
        self,
        *,
        stage: str,
        error_type: str = "",
        error_message: str = "",
        retry_count: int | None = None,
    ) -> None:
        self.ledger.update_auto_research_run(
            self.run_id,
            status="needs_attention",
            stage=stage,
            error_type=error_type or None,
            error_message=error_message or None,
            retry_count=retry_count,
        )

    def manual_override(self, *, stage: str, item_key: str = "") -> str:
        return self.ledger.manual_recovery_override(
            self.run_id,
            stage=stage,
            item_key=item_key,
        )

    def clear_manual_override(self, *, stage: str, item_key: str = "") -> None:
        self.ledger.clear_manual_recovery_override(
            self.run_id,
            stage=stage,
            item_key=item_key,
        )

    def fail_from_llm(
        self,
        error: LLMRetryExhausted,
        *,
        stage: str | None = None,
        context: dict[str, Any] | None = None,
        review_id: str = "",
        intent_id: str = "",
        item_key: str = "",
    ) -> str:
        """Mark execution attention and persist one LLM failure in one operation."""
        actual_stage = stage or error.stage
        self.needs_attention(
            stage=actual_stage,
            error_type=error.error_type,
            error_message=error.message,
            retry_count=error.attempts,
        )
        return self.record_llm_failure(
            error,
            context=context or {},
            review_id=review_id,
            intent_id=intent_id,
            item_key=item_key,
        )

    def record_llm_failure(
        self,
        error: LLMRetryExhausted,
        *,
        context: dict[str, Any] | None = None,
        review_id: str = "",
        intent_id: str = "",
        item_key: str = "",
    ) -> str:
        return self.ledger.record_auto_research_failure(
            failure_payload(error, context=context or {}),
            run_id=self.run_id,
            review_id=review_id,
            intent_id=intent_id,
            item_key=item_key,
        )
