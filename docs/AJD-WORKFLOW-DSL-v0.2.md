# AJD Workflow DSL v0.2

## Purpose

AJD Workflow DSL is an executable job-level specification between AJD and Python implementation.
It describes **what the agent does, why it does it, what job data it consumes/produces, and what shared phenomena cross the agent boundary**. Implementation details such as LLM retry, parsing, SQL, persistence, provider exceptions, and logging remain outside the DSL.

## Frozen core model

```text
AJD responsibility
    ↓ realizes
Workflow
    ├─ trigger                shared phenomenon entering the workflow
    ├─ inputs                 invocation data
    ├─ memory                 persistent job memory access
    ├─ steps                  job capabilities / decisions / interactions / subworkflows
    │    ├─ uses              required / optional local data
    │    ├─ produces          local data products
    │    ├─ when / foreach    orchestration
    │    └─ emits             shared phenomena leaving the workflow
    ├─ outcomes               exhaustive branch alternatives when needed
    ├─ outputs                required / optional public results
    ├─ completion.criteria    job completion semantics
    └─ invariants             truths that must remain valid
```

## Canonical workflow form

```yaml
dsl: ajd-workflow/v0.2
id: example_workflow
revision: 1
agent: m1
realizes: [responsibility_id]
purpose: >
  Explain the job-level purpose.

trigger:
  phenomenon: request
  from: researcher
  condition: ready

inputs:
  - request

memory:
  reads: [knowledge_cards]
  writes: [paper_analysis]

steps:
  - id: analyze_request
    kind: action
    uses:
      required: [request]
    produces: [analysis, needs_review]

  - id: decide_next_action
    kind: decision
    authority: m1
    when: needs_review
    uses:
      required: [analysis]
    produces: [decision]

  - id: publish_result
    kind: action
    uses:
      required: [analysis]
      optional: [decision]
    produces: [result]
    emits:
      - phenomenon: advice_report
        to: [researcher]

outputs:
  required: [result]
  optional: [decision]

completion:
  criteria:
    - researcher_facing_result_is_recorded

invariants:
  - result_is_traceable_to_input
```

## Step kinds

- `action`: performs a job capability.
- `decision`: performs a job judgment. `authority` is mandatory.
- `interaction`: represents an explicit external/human action. `actor` is mandatory.
- `workflow`: invokes a child workflow.

For `action` and `decision`, `capability` defaults to the step `id`; therefore repeating the same name is unnecessary. `capability` is used only when the implementation binding intentionally differs from the semantic step id.

For ordinary steps, `actor` defaults to the owning workflow `agent`. A child workflow's agent is inferred from the child definition. Explicit `actor` remains available when semantically necessary.

## Data hierarchy

`inputs → step uses → step produces → outputs`

`uses` distinguishes required and path-dependent data:

```yaml
uses:
  required: [a, b]
  optional: [c]
```

Workflow outputs use the same hierarchy:

```yaml
outputs:
  required: [status, report]
  optional: [retry_run_id]
```

This replaces the v0.1 sibling keys `optional_uses` and `optional_outputs`.

## Completion vs runtime status

`completion.criteria` expresses when the **job responsibility is considered fulfilled**. It is not the technical run status. Runtime states such as retry count, current stage, `running`, or `needs_attention` remain runtime concerns.

## Shared phenomena

`trigger` and `emits` are the DSL representation of APF shared phenomena.

```yaml
trigger:
  phenomenon: curation_intent
  from: m2
  condition: ready
```

```yaml
emits:
  - phenomenon: knowledge_update
    to: [m2]
```

The workflow catalog validates compatible producer/consumer routing while allowing system-boundary phenomena such as researcher requests or reports.

## Memory

`memory` refers only to persistent job memory assets declared in the owning agent's machine-readable AJD contract.

```yaml
memory:
  reads: [knowledge_cards]
  writes: [research_questions]
```

Local workflow data belongs to `inputs / uses / produces / outputs`, not `memory`.

## Outcomes

`outcomes` declares exhaustive mutually exclusive alternatives needed for branch-sensitive validation.

```yaml
outcomes:
  search_result:
    one_of: [search_succeeded, search_failed]
```

Required outputs must be available across every normal outcome path; otherwise they belong under `outputs.optional`.

## Deliberately excluded from the DSL

The following remain implementation/runtime concerns:

- prompt templates and rendering
- LLM/provider selection
- retry/backoff
- JSON extraction and parser functions
- validator function names
- SQL/table/repository details
- logging and technical execution traces
- Python handler paths

## v0.2 freeze decisions

v0.2 freezes the following vocabulary for the next implementation phase:

`dsl, id, revision, agent, realizes, purpose, trigger, inputs, memory, outcomes, steps, kind, capability, actor, authority, when, foreach, uses, produces, emits, outputs, completion, invariants`.

New language elements should not be added until additional job workflows expose a repeated semantic need that cannot be represented with this vocabulary.
