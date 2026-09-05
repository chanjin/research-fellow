# Ontology workspace implementation

## Definition
Ontology is represented as:
1. researcher-defined types assigned to approved knowledge cards, and
2. researcher-defined directed relations between those types.

The prior paper-reading ontology-candidate UI is no longer part of the active workflow. Legacy database fields/tables remain only for backward compatibility with existing local databases.

## New schema
- `ontology_types`
- `ontology_card_assignments`
- `ontology_type_relations`

## New code
- `src/research_fellow/domain/ontology.py`
- `src/research_fellow/application/ontology.py`
- `tests/test_ontology.py`

## UI workflow
`지식 관리 > 온톨로지`

### 타입 구성
1. Search approved cards with keyword / local embedding / approved card-relation expansion.
2. Select a group of cards.
3. Create a new type or assign the cards to an existing type.
4. Review the cards already assigned to each type.

### 타입 관계
1. Inspect the ontology type graph.
2. Choose source and target types.
3. Enter a free-form relation name and optional explanation.
4. Save or delete relations.

## Legacy behavior removed from active UI
- paper-reading `온톨로지 일반화 메모로 보내기`
- ontology candidate review workspace
- ontology-candidate generation requirement in the reading prompt

## Future retrieval integration
The new schema is intentionally separate from `knowledge_relations` so advisory retrieval can later use:
`question -> seed cards -> assigned ontology types -> related types -> additional cards -> knowledge-relation expansion`.
