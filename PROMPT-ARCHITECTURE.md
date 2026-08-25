# Python–Jinja Prompt Architecture

## Boundary

Python owns document extraction, retrieval, bounded candidate selection, IDs,
payload validation, SQLite/JSONL writes, approval transitions, and audit
records. Jinja templates contain the revisable instructions that ask an LLM to
write a **draft only**. An LLM output is never allowed to create a case, approve
knowledge, choose an unavailable source/page/reference, or update memory.

`src/research_fellow/infrastructure/prompt_renderer.py` is the single renderer.
It uses `StrictUndefined`, so a missing context variable fails early rather than
silently changing a prompt.

## Existing implementation analysis and adaptation

| Attached implementation | Durable Python responsibility retained/adapted | Jinja prompt asset prepared here |
| --- | --- | --- |
| `pdf_ingestion.py`, `pdf_scenario.py` | bounded page extraction, page markers, source metadata, evidence-page validation | `m1_curation.j2` |
| `agents.py`, `contracts.py`, `schemas.py` | task inputs, allowed values, reference validation, approval/state handling | existing M1/M2 templates and `application/prompt_tasks.py` |
| `claim_verification.py` | assign stable `E#` references and verify output only against supplied cards | `m1_claim_verification.j2` |
| `gap_search.py`, `arxiv.py` | run search tools; create stable `S#` references; validate selected sources | `m1_gap_search_plan.j2`, `m1_source_triage.j2` |
| `concept_lineage.py`, `relation_review.py` | restrict node/pair candidates and store a relation only after approval | `m1_lineage_review.j2`, existing `m1_relation_proposal.j2` |
| `revalidation_review.py` | surface conflict/freshness candidates; never automatically retract knowledge | `m1_revalidation_review.j2` |
| `knowledge_overview.py` | produce read-model projections from approved records | `m2_knowledge_update_report.j2` |
| `memory.py`, `io.py` | append-only canonical knowledge and ledger/adapters | no prompt: storage is deterministic |
| `llm.py`, `prompting.py` | transport, retry, diagnostic logging, and strict Jinja rendering | no business policy in Python strings |

## Prompt entry points

`application/prompt_tasks.py` creates bounded contexts for the additional
templates. Its functions return prompt text only:

- `claim_verification_prompt(assertion, cards)`
- `gap_search_plan_prompt(gap, focus, cards)`
- `source_triage_prompt(gap, sources)`
- `lineage_review_prompt(topic, cards)`
- `revalidation_review_prompt(reason, cards)`
- `knowledge_update_report_prompt(cards, research_question)`

Before an associated UI/workflow is added, call the relevant function, ask the
LLM for a draft, then implement a deterministic parser/validator that accepts
only the displayed `E#`, `S#`, or `K#` references. No template should be wired
directly to SQLite or JSONL.
