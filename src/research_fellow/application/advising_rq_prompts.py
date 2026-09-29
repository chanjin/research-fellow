"""Prompt construction for M2 research-question prioritization."""
from __future__ import annotations
from typing import Any

def auto_rq_priority_prompt(backlog: list[dict[str, Any]], max_select: int = 3) -> str:
    """Ask M2 to rank only currently actionable backlog questions for autonomous exploration."""
    from research_fellow.infrastructure.prompt_renderer import render_prompt

    return render_prompt("m2_auto_rq_priority.j2", backlog=backlog, max_select=max_select)
