from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from research_fellow.infrastructure.retrieval import KnowledgeRetriever, RetrievalResult


@dataclass(frozen=True)
class OntologyCardHit:
    result: RetrievalResult
    relation_distance: int = 0


def search_cards_for_ontology(
    retriever: KnowledgeRetriever,
    cards: list[dict[str, Any]],
    relations: list[dict[str, Any]],
    query: str,
    *,
    use_keyword: bool = True,
    use_embedding: bool = True,
    use_relations: bool = True,
    embedding_model: str = "nomic-embed-text",
    limit: int = 20,
) -> list[OntologyCardHit]:
    """Find cards for researcher-led ontology construction.

    Keyword and embedding similarity identify seed cards. Optional relation expansion
    adds approved neighbouring cards, but never assigns a type automatically.
    """
    query = query.strip()
    if not query:
        return []

    if use_relations:
        cluster = retriever.cluster(
            cards,
            relations,
            query,
            seed_limit=min(6, max(1, limit // 3)),
            max_hops=1,
            limit=limit,
            semantic=use_embedding,
            embedding_model=embedding_model,
        )
        hits = [OntologyCardHit(member.result, member.distance) for member in cluster.members]
    else:
        hits = [
            OntologyCardHit(result, 0)
            for result in retriever.search(
                cards,
                query,
                limit=limit,
                semantic=use_embedding,
                embedding_model=embedding_model,
            )
        ]

    if use_keyword:
        return hits[:limit]

    # If keyword matching is disabled, keep semantic or relation-derived results only.
    return [
        hit for hit in hits
        if hit.result.method != "lexical" or hit.relation_distance > 0
    ][:limit]


def _escape(value: Any) -> str:
    return str(value).replace('"', '\\"')


def ontology_dot(
    types: list[dict[str, Any]],
    relations: list[dict[str, Any]],
    facets: list[dict[str, Any]] | None = None,
) -> str:
    """Graphviz projection of the ontology schema, grouped by facet when available."""
    facets = facets or []
    lines = [
        "digraph ontology {",
        '  graph [rankdir="LR", bgcolor="transparent", pad="0.2", compound="true"];',
        '  node [shape="box", style="rounded", fontname="Arial", fontsize="11"];',
        '  edge [fontname="Arial", fontsize="10"];',
    ]
    type_ids = {str(item["type_id"]) for item in types}
    facet_by_id = {str(item["facet_id"]): item for item in facets}
    grouped: dict[str | None, list[dict[str, Any]]] = {}
    for item in types:
        grouped.setdefault(item.get("facet_id"), []).append(item)
    for facet_id, members in grouped.items():
        if facet_id and str(facet_id) in facet_by_id:
            facet = facet_by_id[str(facet_id)]
            cluster_id = "cluster_" + str(facet_id).replace("-", "_")
            lines.append(f'  subgraph "{cluster_id}" {{')
            lines.append(f'    label="{_escape(facet["name"])}";')
            lines.append('    style="rounded,dashed";')
            for item in members:
                lines.append(f'    "{item["type_id"]}" [label="{_escape(item["name"])}"];')
            lines.append("  }")
        else:
            for item in members:
                lines.append(f'  "{item["type_id"]}" [label="{_escape(item["name"])}"];')
    for relation in relations:
        source = str(relation["source_type_id"]); target = str(relation["target_type_id"])
        if source in type_ids and target in type_ids:
            lines.append(f'  "{source}" -> "{target}" [label="{_escape(relation["relation_name"])}"];')
    lines.append("}")
    return "\n".join(lines)


def ontology_context_dot(
    types: list[dict[str, Any]],
    relations: list[dict[str, Any]],
    focus_type_ids: list[str] | set[str] | tuple[str, ...] | str | None,
    facets: list[dict[str, Any]] | None = None,
) -> str:
    """Show all types assigned to one card plus their one-hop type neighbourhood."""
    if isinstance(focus_type_ids, str):
        focus_ids = {focus_type_ids}
    else:
        focus_ids = {str(value) for value in (focus_type_ids or []) if value}
    if not focus_ids:
        return ontology_dot(types, relations, facets)
    neighbour_ids = set(focus_ids); local_relations = []
    for relation in relations:
        source = str(relation["source_type_id"]); target = str(relation["target_type_id"])
        if source in focus_ids or target in focus_ids:
            local_relations.append(relation); neighbour_ids.update((source, target))
    local_types = [item for item in types if str(item["type_id"]) in neighbour_ids]
    lines = [
        "digraph ontology_context {",
        '  graph [rankdir="LR", bgcolor="transparent", pad="0.2", compound="true"];',
        '  node [shape="box", style="rounded", fontname="Arial", fontsize="11"];',
        '  edge [fontname="Arial", fontsize="10"];',
    ]
    facet_by_id = {str(item["facet_id"]): item for item in (facets or [])}
    grouped: dict[str | None, list[dict[str, Any]]] = {}
    for item in local_types:
        grouped.setdefault(item.get("facet_id"), []).append(item)
    for facet_id, members in grouped.items():
        if facet_id and str(facet_id) in facet_by_id:
            facet = facet_by_id[str(facet_id)]
            cluster_id = "cluster_" + str(facet_id).replace("-", "_")
            lines.append(f'  subgraph "{cluster_id}" {{')
            lines.append(f'    label="{_escape(facet["name"])}";')
            lines.append('    style="rounded,dashed";')
            for item in members:
                pen = "2.8" if str(item["type_id"]) in focus_ids else "1"
                lines.append(f'    "{item["type_id"]}" [label="{_escape(item["name"])}", penwidth="{pen}"];')
            lines.append("  }")
        else:
            for item in members:
                pen = "2.8" if str(item["type_id"]) in focus_ids else "1"
                lines.append(f'  "{item["type_id"]}" [label="{_escape(item["name"])}", penwidth="{pen}"];')
    for relation in local_relations:
        lines.append(f'  "{relation["source_type_id"]}" -> "{relation["target_type_id"]}" [label="{_escape(relation["relation_name"])}"];')
    lines.append("}")
    return "\n".join(lines)
