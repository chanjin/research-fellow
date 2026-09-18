# M1 Workspace Integration

M1 now presents four researcher-facing workspaces instead of seven implementation-oriented tabs.

## 1. Literature Discovery

- Direct topic/context entry and Knowledge-Card-driven discovery share one workflow.
- Selected approved cards prefill an editable topic and research context using their
  claims, conditions, limits, and concepts.
- Discovery history, reference candidates, and the Research Intent search queue stay
  in this workspace. The queue is an execution state, not a separate job.
- The history is a unified read model over ad-hoc Knowledge-Card searches and
  periodic M2 Research-Intent runs. Both expose the same topic, context, status,
  results, source, and follow-up fields.
- A `Search Run` completes independently. Daily and weekly `Search Profiles` remain
  active for their next cadence; only a manual one-shot profile closes after running.
- Research-Intent profiles no longer depend on a researcher-maintained fixed keyword
  list. Every run is an LLM Discovery Task that receives the full Intent context and
  recent run strategies/results, returns a fresh JSON search plan, retrieves from the
  selected scholarly sources, and applies contextual abstract triage.
- The same editable task prompt accepts an external LLM JSON plan. Internal and
  external plans execute through the same retrieval, triage, and Run-history path.

## 2. Paper Shelf

- Papers selected from discovery and externally supplied documents remain here for
  preservation, reading, and later conversion into reviewed Knowledge Cards.

## 3. Approved Knowledge

- Search and the full card list share the `Search & List` subview.
- Card-to-card evidence, contradiction, extension, and provenance links are handled
  in the `Relations & Lineage Maintenance` subview.
- Relations and lineage are corrective/detail work over approved knowledge; they are
  not presented as a separate top-level M1 job.

## 4. Ontology

- Versioned Change Review governs Type assignment and Type-to-Type structure.
- Ontology Map presents the currently approved Facet–Type graph.

Type-to-Type ontology relations and Card-to-Card knowledge relations remain distinct:
the former describe the domain schema, while the latter record evidence-level links
between concrete approved claims.
