# Versioned Ontology Change Review

The ontology is the approved graph of **Types** and **Type-to-Type Relations**.
Facets are stereotypes used to group and colour Types. Knowledge Cards are not
permanent graph nodes: a card can have multiple Type assignments.

## Workflow

1. The reviewer selects newly approved cards and, when needed, existing typed
   cards whose assignments should be revalidated.
2. The LLM compares them with the current Facet–Type graph and creates a Diff:
   existing-Type assignments, new Types, and new Type relations.
   The same fully rendered prompt is available for copying to an external LLM;
   its JSON response can be pasted, validated, and stored as the proposal.
3. The researcher reviews the evidence and rationale, records corrections, and
   creates another Diff if the proposal needs revision.
4. Each reviewed card returns its complete desired multi-Type set. Approval can
   therefore add Types or remove an incorrect previous assignment.
5. Approval applies the Diff and emits the next `OntologyVersion`.

No proposal changes the current ontology before approval. Each version records
its base version, source cards, full Facet–Type–Relation–Assignment snapshot,
assignment Diff, summary, approver, and time. The newly published graph is shown
immediately and every historical version can be reopened. Type selection in the
map reveals the cards assigned to that Type; card-to-card relations remain
separate from the ontology schema.

In a version graph, Facet colours remain the normal classification colours.
Version-specific overlays make the Diff visible: green means a newly created
Type and yellow means an existing Type affected by a definition or card-assignment
change.

## Graph review and revision

- Proposal and published Type graphs support wheel zoom and drag pan.
- A researcher can request a Type name/description update, a relation
  name/description update, or a relation add/delete operation while viewing the graph.
- Structured instructions and free-form comments can generate another internal-LLM
  Diff. The editable prompt and JSON response fields support the same workflow with
  an external LLM.
- Approved Type and relation changes are included in the version snapshot and Diff.

## Card batch size

All newly untyped cards are selected by default without a fixed count limit. A
researcher may deselect cards only when deliberately splitting a review into batches.
