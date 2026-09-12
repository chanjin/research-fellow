# Ontology Facet + Multi-Type Builder

- A knowledge card may have zero or more ontology Types.
- A Type may optionally belong to one Facet.
- Facet is a meta-classification axis for Types, not a graph relation node.
- Type-to-Type relations remain the ontology graph edges.
- Builder uses a per-card multi-select; changes replace that card's complete assignment set.
- The contextual graph displays all Types assigned to the focused card and their 1-hop neighbours, grouped by Facet.

Example:

- Requirement -> Non-functional Requirement
- Agent Case -> Architect Agent
- Target Domain -> Software Architecture

The same knowledge card may carry all three Types.
