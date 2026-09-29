"""Compatibility facade for claim-first curation capabilities.

New code should import from claim_curation_prompts, claim_curation_parsers, or claim_curation_cards.
"""
from research_fellow.application.claim_curation_prompts import *  # noqa: F401,F403
from research_fellow.application.claim_curation_parsers import *  # noqa: F401,F403
from research_fellow.application.claim_curation_cards import *  # noqa: F401,F403
