# M2 Research Co-author

M2 is reorganized around a durable Paper Project rather than an open-ended advice report.

1. A project starts through one of three paths: derive an RQ from newly approved
   Knowledge Cards, select a previously defined RQ, or enter an RQ directly.
2. Selecting an existing RQ inherits its linked Knowledge Cards, rationale,
   research context, gap/tension, exploration need, and source provenance into a
   durable `project_origin` event. The same RQ reopens its existing Paper Project
   instead of silently creating a duplicate.
3. M2 drafts a two-page Short Paper using only linked evidence and marks limitations.
4. Researcher positions, decisions, research gaps, and M1 search intents are stored as distinct events.
5. A static Gate requires a Short Draft, evidence, and a researcher Decision before expansion.
6. An approved project advances to Full Paper and receives a ten-page outline/section plan.

## Research-question entry

The restored **New Knowledge → Research Question** screen remains the intake for
recently reviewed papers. It can use either the internal LLM or the editable
external-LLM prompt/response flow. Saved candidates immediately become selectable
in **Existing Research Questions**, where the researcher can inspect the derivation
reason and source cards before starting the paper project.

## Related Knowledge Cards

The researcher no longer writes a separate card-search query. The project title
and confirmed research question form the retrieval context. The static retrieval
pipeline first takes up to 12 lexical/embedding matches as seed cards, then adds
cards sharing their ontology types and cards assigned to directly connected
types (one hop). Existing RQ cards are always retained; newly discovered cards
fill the remaining capacity of a 24-card working set. Search hits remain primary,
while same-type and related-type expansions are interleaved for diversity. The
screen displays each card's search or ontology entry path for researcher review.
The query, all candidates, their entry paths, and the final selection are stored
as an `evidence_selection` Paper Project event.

Static tasks own IDs, state transitions, event history, and Gates. LLM tasks own
interpretation, drafting, outlining, and review. Researcher decisions remain explicit,
auditable events. Existing M2 advisory data is retained but the new UI enters the
Research Co-author workspace by default.

## Short Paper Iteration

The two-page manuscript is the shared researcher-facing artifact. Each sentence
has a stable ID, evidence-card links, and review annotations. The diagnostic table
is derived from the manuscript instead of maintained as a separate artifact.

Annotations cover insufficient evidence, citation need, researcher input,
decisions, unclear scope, overclaim, missing counterarguments, logic gaps, and
unclear contribution. A researcher selects a sentence, adds a comment, optionally
searches for more Knowledge Cards, and runs a targeted internal or external LLM
revision. Each revision creates a manuscript version and sentence-level Diff.

## Revision To-do workflow

Sentence markup and Revision To-dos are two views of the same issue, but they
serve different purposes. Markup is a diagnostic attached to the manuscript
sentence: it explains what is weak and why. A Revision To-do is the executable
work item derived from that diagnostic: it carries priority, researcher input,
evidence-search work, revision state, and an observable completion criterion.

Sentence annotations are converted into prioritized Revision To-dos. Each To-do
contains a readable markup label, problem explanation, researcher question,
information-search guide, recommended action, and observable completion
criterion. The workflow is:

1. `Open`: M2 review creates a sentence-linked To-do.
2. `Ready`: the researcher records a decision or answer and optionally selects
   cards found through the same hybrid/ontology evidence retrieval.
3. `Revised pending review`: an internal or external LLM revises only the linked
   sentence while preserving its stable ID and records a sentence Diff.
4. `Resolved` or reopened: a full sentence review removes a satisfied annotation
   or produces a new To-do when the completion criterion is still unmet.

The Revision screen retains resolved To-dos as a checked, struck-through project
checklist and shows completed/total progress. Active To-dos remain selectable;
resolved items remain visible as history. If a later review returns the same
stable annotation ID, that item becomes active again.

Every response, evidence selection, targeted revision, review result, and
manuscript version is retained in the Paper Project event history.

After an internal or external review is applied, the UI now shows a durable
review receipt: source, previous/result manuscript versions, retained/new/
resolved To-do counts, annotation-type counts, and the exact sentence-linked
items in each category. The same pasted external response cannot be applied
twice; its apply button remains disabled even if the manuscript advances. A new
external response has a new digest and enables the button. The most recent
receipt remains available in the review screen and the complete receipt is also
visible in project history.

The current manuscript is followed immediately by a sentence-level revision
history. Revision versions are shown newest first with the originating To-do,
revision source, reason, stable sentence ID, and side-by-side before/after text.
Older revision events without explicit To-do metadata are linked to the nearest
recorded `revision_todo_update` when possible.

## Literature search candidates

Every Revision To-do carries one to three bounded literature-search candidates.
Each candidate specifies a target question, paper/revision context, expected
evidence, completion condition, and priority rather than a bare keyword list.
The researcher can edit a candidate and submit it as a `CurationIntent` approval
request. After approval, the existing decision workflow creates an M1 search
profile and places it in the unified literature-discovery queue. The Paper
Project retains the To-do, candidate, Intent, and approval-request identifiers
in a `revision_literature_intent` event. Legacy To-dos receive one deterministic
candidate derived from their sentence, problem, search guide, and completion
criterion.
