"""Compatibility facade for paper coauthor capabilities.

Workflow orchestration has moved to AJD Workflow DSL. Capability implementations
are split by concern so this module remains only a stable import surface.
"""
from research_fellow.application.paper_coauthor_search import *  # noqa: F401,F403
from research_fellow.application.paper_coauthor_prompts import *  # noqa: F401,F403
from research_fellow.application.paper_coauthor_parsers import *  # noqa: F401,F403
from research_fellow.application.paper_coauthor_manuscript import *  # noqa: F401,F403
