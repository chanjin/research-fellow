# Shared-Phenomena Research Fellow

M1 Curator, M2 Advisor, researcher, and external requester collaborate through
**shared phenomena**: observable requests, reports, intents, updates, and decisions.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
streamlit run app.py
```

The app stores local runtime data below `data/`:

- `research_fellow.db`: cases, shared phenomena, and researcher decisions.
- `knowledge_cards.jsonl`: approved semantic knowledge cards only.

To use a local Ollama model, start Ollama and set the model in the sidebar.
The application still works without Ollama: it produces concise template-based
drafts rather than failing a workflow.

## Simplified model

| Store | Meaning |
| --- | --- |
| Knowledge cards (JSONL) | Approved facts and evidence used as semantic memory. |
| Shared-phenomena ledger (SQLite) | What one domain exposed to another: request, report, intent, update, or decision. |
| UI | Role-specific projections of the same ledger, not separate queues and notification databases. |

The seven phenomenon types are `research_update`, `advice_report`,
`decision_request`, `decision`, `curation_intent`, `knowledge_update`, and
`advisory_exchange`.

## P1: document curation

M1 uses PyMuPDF as its primary PDF reader, retaining page-addressable text
blocks and coordinates before proposing at most three independently approvable
cards per document. `pypdf` is used only as a compatibility fallback. Each card
carries a claim, excerpt, page number, citation marker, labels, conditions,
limits, and provenance.

Derived extraction results are content-addressed and kept under
`data/extracted_documents/`. The cache contains only page/block text and source
metadata—not the uploaded PDF bytes or images—so uploading the identical file
again reuses the structured text rather than parsing the PDF again. Delete a
specific cache JSON only when intentionally forcing re-extraction.

An optional Ollama draft is deliberately plain text rather than structured JSON.
Deterministic code normalizes it, fills only document-derived evidence fields,
and displays warnings for fields that need researcher correction. The researcher
can batch approve, request correction, or reject cards; only approved cards enter
`knowledge_cards.jsonl`.

## P2: knowledge relations and lineage

P2 works only with approved cards. M1 proposes one of `supports`, `extends`,
`contradicts`, `qualifies`, `uses_method`, or `addresses_gap`, with relation
evidence, conditions, and confidence. A researcher decision request records the
proposal; only approval appends it to `knowledge_relations.jsonl` and emits a
`knowledge_update` for M2 and the researcher.

The P2 Streamlit tab renders at most ten approved-card nodes as a Graphviz
lineage projection. It is a view of canonical card and relation memory, not a
separate graph database. The sidebar checks the local Ollama endpoint and the
default `gemma4:e4b` model. Gemma may produce P1/P2 text drafts only; all IDs,
relation types, approvals, and stores are validated by deterministic code.

## Knowledge maintenance

The **지식 관리** screen can logically delete active cards and relations. Deletion
creates a tombstone plus a ledger update instead of overwriting JSONL history.
Deleted cards disappear from search and P1/P2 projections; relations connected
to a deleted card are also omitted from the lineage map. P2 now generates up to
three relation proposals from approved cards for researcher confirmation, rather
than requiring manual relation authoring.

## P3: retrieval and semantic memory

`knowledge_cards.jsonl` remains the canonical approved-memory record. P3 always
uses lexical search over title, claim, labels, evidence, conditions, and limits.
Its result view and M2 prompts show the card ID, original source/page, search
method, score, and selection reason.

When the sidebar's **P3 Ollama 임베딩 검색** option is enabled, P3 additionally
uses the local `nomic-embed-text` model. It keeps a regenerable local index only:
the index is rebuilt automatically whenever the approved-card text, embedding
model, or index schema version changes. `LlamaIndexAdapter` is an optional
ingestion adapter; install it with `pip install -e '.[llamaindex]'` when moving
to a LlamaIndex-backed semantic engine.

## P4: M2 research state and direction

The **M2 · 연구 상태·방향** screen records a validated `ResearchState` as a
`research_update`: question, current hypothesis, constraints, unresolved issues,
recent evidence changes, and confidence. M2 reads the resulting state together
with approved-card retrieval and observable M1 `knowledge_update` phenomena.
It creates one concise advice report and no more than three structured M1
curation-intent decision requests. Each intent contains purpose, question,
labels, priority, expected evidence, and completion condition. The researcher
approves, defers, or rejects those requests from the home inbox; only approval
creates a `ready` `curation_intent` in M1's execution queue.

## Prompt assets

M1/M2 prompts are versioned `.j2` assets under `src/research_fellow/prompts/`.
Python uses Jinja2 with `StrictUndefined`: it supplies only validated context and
keeps IDs, state transitions, approval, storage, and retrieval deterministic.
Install the regular project dependencies before running the app so these prompt
assets are available.

## Developer prompt workspace

The sidebar's **개발·프롬프트** screen is a development-only playground. Select
one registered `.j2` asset, edit it, and save only after Jinja syntax validation.
The saved asset is immediately reused by the corresponding operational screen.
The second tab supplies bounded development context (approved cards, a local
document, or deterministic search metadata), shows the fully rendered prompt,
and can request an Ollama draft. It never writes a card, relation, decision, or
shared phenomenon. See `PROMPT-ARCHITECTURE.md` for the Python/Jinja boundary
and the additional review-task templates.
