"""Navigation policy for the job-centred Research Fellow UI.

R48 keeps the normal researcher surface intentionally small: the Operating Desk
is the only job workspace.  Compatibility and diagnostic pages remain available
only when Developer mode is explicitly enabled.  This module contains no domain
state and performs no durable writes.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

OPERATING_DESK = "운영 데스크"
LEGACY_DESK = "기존 연구위원 데스크"
LEGACY_M1 = "M1 · 문헌조사·지식화"
LEGACY_M2 = "M2 · 지식 기반 자문"
LEGACY_PAPER = "논문 작업실"
LEGACY_KNOWLEDGE = "지식 베이스·운영"
LEGACY_DEVELOPER = "개발·프롬프트"

LEGACY_WORKSPACES = (
    LEGACY_DESK,
    LEGACY_M1,
    LEGACY_M2,
    LEGACY_PAPER,
    LEGACY_KNOWLEDGE,
    LEGACY_DEVELOPER,
)


@dataclass(frozen=True)
class NavigationItem:
    key: str
    ko: str
    en: str
    developer_only: bool = False


NAVIGATION_ITEMS = (
    NavigationItem(OPERATING_DESK, "운영 데스크", "Operating Desk"),
    NavigationItem(LEGACY_DESK, "기존 연구위원 데스크", "Legacy Research Fellow Desk", True),
    NavigationItem(LEGACY_M1, "M1 · 문헌조사·지식화", "Legacy M1 · Literature & Knowledge", True),
    NavigationItem(LEGACY_M2, "M2 · 논문 프로젝트", "Legacy M2 · Paper Projects", True),
    NavigationItem(LEGACY_PAPER, "논문 작업실", "Legacy Paper Workspace", True),
    NavigationItem(LEGACY_KNOWLEDGE, "지식 베이스·운영", "Legacy Knowledge Base & Operations", True),
    NavigationItem(LEGACY_DEVELOPER, "개발·프롬프트", "Legacy Prompt / Developer Tools", True),
)


def developer_mode_default() -> bool:
    """Return an opt-in default controlled only by process configuration."""
    value = os.environ.get("RESEARCH_FELLOW_DEVELOPER_MODE", "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def visible_navigation_items(*, developer_mode: bool) -> tuple[NavigationItem, ...]:
    if developer_mode:
        return NAVIGATION_ITEMS
    return tuple(item for item in NAVIGATION_ITEMS if not item.developer_only)


def normalize_workspace(requested: str | None, *, developer_mode: bool) -> str:
    """Keep stale/deep-linked Legacy destinations out of the researcher surface."""
    if requested == "연구위원 데스크":  # pre-R42 compatibility key
        requested = OPERATING_DESK
    valid = {item.key for item in visible_navigation_items(developer_mode=developer_mode)}
    return requested if requested in valid else OPERATING_DESK
