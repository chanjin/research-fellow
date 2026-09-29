"""Compatibility facade for progressive curation capabilities.

New code should import from progressive_curation_* modules by concern.
"""
from research_fellow.application.progressive_curation_models import *  # noqa: F401,F403
from research_fellow.application.progressive_curation_generation import *  # noqa: F401,F403
from research_fellow.application.progressive_curation_consolidation import *  # noqa: F401,F403
from research_fellow.application.progressive_curation_submit import *  # noqa: F401,F403
