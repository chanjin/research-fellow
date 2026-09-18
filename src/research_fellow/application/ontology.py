from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import math

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
    type_highlights: dict[str, str] | None = None,
) -> str:
    """Graphviz projection, with optional version-change status colours."""
    facets = facets or []
    type_highlights = type_highlights or {}
    status_colours = {"new": "#C6F6D5", "changed": "#FDE68A"}

    def node_line(item: dict[str, Any], base_colour: str = "#F5F7FA", indent: str = "  ") -> str:
        type_id = str(item["type_id"])
        status = type_highlights.get(type_id, "")
        fill = status_colours.get(status, base_colour)
        border = "#15803D" if status == "new" else "#B45309" if status == "changed" else "#667085"
        penwidth = "2.8" if status else "1"
        return f'{indent}"{type_id}" [label="{_escape(item["name"])}", fillcolor="{fill}", color="{border}", penwidth="{penwidth}"];'
    lines = [
        "digraph ontology {",
        '  graph [rankdir="LR", bgcolor="transparent", pad="0.2", compound="true"];',
        '  node [shape="box", style="rounded,filled", fillcolor="#F5F7FA", fontname="Arial", fontsize="11"];',
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
            lines.append(f'    label="FACET · {_escape(facet["name"])}"; labelloc="t"; labeljust="l"; fontsize="13"; fontname="Arial Bold";')
            lines.append(f'    style="rounded,dashed"; color="{_escape(facet.get("color") or "#9AA4B2")}"; penwidth="2"; margin="18";')
            for item in members:
                lines.append(node_line(item, str(facet.get("color") or "#F5F7FA"), "    "))
            lines.append("  }")
        else:
            for item in members:
                lines.append(node_line(item))
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


def ontology_plotly_figure(
    types: list[dict[str, Any]], relations: list[dict[str, Any]],
    facets: list[dict[str, Any]] | None = None,
    type_highlights: dict[str, str] | None = None,
) -> Any | None:
    """Zoomable/pannable Type graph. Returns None when Plotly is unavailable."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        return None
    facets = facets or []; type_highlights = type_highlights or {}
    facet_by_id = {str(item["facet_id"]): item for item in facets}
    ordered = sorted(types, key=lambda item: (str(item.get("facet_name") or ""), str(item.get("name") or "")))
    count = max(1, len(ordered)); positions: dict[str, tuple[float, float]] = {}
    for index, item in enumerate(ordered):
        angle = (2 * math.pi * index / count) - math.pi / 2
        radius = 1.0 + 0.16 * (index % 3)
        positions[str(item["type_id"])] = (radius * math.cos(angle), radius * math.sin(angle))
    traces: list[Any] = []
    for relation in relations:
        source = positions.get(str(relation.get("source_type_id"))); target = positions.get(str(relation.get("target_type_id")))
        if not source or not target: continue
        traces.append(go.Scatter(x=[source[0], target[0]], y=[source[1], target[1]], mode="lines", line={"color":"#98A2B3","width":1.5}, hoverinfo="text", text=[str(relation.get("description") or relation.get("relation_name") or ""), ""], showlegend=False))
        mx, my = (source[0]+target[0])/2, (source[1]+target[1])/2
        traces.append(go.Scatter(x=[mx], y=[my], mode="text", text=[str(relation.get("relation_name") or "")], textfont={"size":10,"color":"#475467"}, hoverinfo="text", hovertext=[str(relation.get("description") or "")], showlegend=False))
    x=[]; y=[]; labels=[]; colours=[]; borders=[]; hover=[]
    for item in ordered:
        type_id=str(item["type_id"]); px,py=positions[type_id]; status=type_highlights.get(type_id, "")
        facet=facet_by_id.get(str(item.get("facet_id") or ""), {})
        colours.append("#C6F6D5" if status=="new" else "#FDE68A" if status=="changed" else str(facet.get("color") or "#E4E7EC"))
        borders.append("#15803D" if status=="new" else "#B45309" if status=="changed" else "#667085")
        x.append(px); y.append(py); labels.append(str(item.get("name") or type_id))
        hover.append(f"<b>{item.get('name')}</b><br>Facet: {item.get('facet_name') or '미지정'}<br>{item.get('description') or '설명 없음'}<br>Cards: {item.get('card_count', 0)}")
    traces.append(go.Scatter(x=x,y=y,mode="markers+text",text=labels,textposition="top center",customdata=[str(item["type_id"]) for item in ordered],hovertext=hover,hoverinfo="text",marker={"size":34,"color":colours,"line":{"color":borders,"width":3}},showlegend=False))
    figure=go.Figure(data=traces)
    figure.update_layout(height=650, margin={"l":20,"r":20,"t":20,"b":20}, dragmode="pan", hovermode="closest", xaxis={"visible":False}, yaxis={"visible":False,"scaleanchor":"x","scaleratio":1}, plot_bgcolor="#FFFFFF")
    return figure
