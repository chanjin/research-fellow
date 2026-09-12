# Research Fellow — Single Codebase

This version intentionally uses **one application, one SQLite database, one cache, and one set of task prompts**.

## Runtime model

- App: `app.py`
- DB: `data/research_fellow.db` (override with `RESEARCH_FELLOW_DATA_DIR` / `RESEARCH_FELLOW_DB_FILENAME`)
- Cache: `.cache/research-fellow`
- Workspace sync server directory: `RESEARCH_FELLOW_SERVER_DIR`
- Default response language: **English**
- Optional presentation language: **Korean**, selected in the sidebar

The Korean option does not launch a second agent or use a second DB. It injects only a presentation-language rule into the same task prompt.

## Prompt architecture

Existing task prompts remain the source of task-specific instructions. Every LLM call passes through:

`src/research_fellow/prompt_profiles.py`

This central layer adds:

1. English-first academic research conventions
2. evidence / inference / uncertainty rules
3. task-family-specific rules (Sensemaking, M2 report, ontology, paper reading, etc.)
4. a runtime output-language instruction (`English` or `한국어`)

Machine-readable JSON keys, enum values, IDs, DB fields, and required schemas must not be translated.

## Run

```bash
./run_research_fellow.sh
```

or

```bash
streamlit run app.py
```

Use another port if needed:

```bash
./run_research_fellow.sh 8502
```

## Data migration from the former dual-profile experiment

The canonical DB is now the original single DB:

`data/research_fellow.db`

The former `data_paper_en/research_fellow_paper_en.db` is **not automatically merged**. Keep it as a backup if you created English-only data there. Merge only after reviewing duplicates and conflicts.

## Removed dual-profile components

The single-codebase package does not use:

- `app_en.py`
- `runtime_profile.py`
- `RESEARCH_FELLOW_PROFILE`
- separate `paper_en` DB/cache
- dual-process launch scripts

## Validation

`python -m compileall -q app.py src` passes.

With `PYTHONPATH=src python -m pytest -q`, the reconstructed current project reports 81 passing tests and 2 previously known failures unrelated to this single-profile change:

- an outdated auto-literature test monkeypatching removed `search_profile_candidates`
- an existing `create_document_candidates()` expectation mismatch
