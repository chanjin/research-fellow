"""Compatibility facade for search-profile capabilities.

New code should import the concern-specific modules directly.
"""
from research_fellow.application.search_profile_intent import *  # noqa: F401,F403
from research_fellow.application.search_profile_strategy import *  # noqa: F401,F403
from research_fellow.application.search_profile_execution import *  # noqa: F401,F403
