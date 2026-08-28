"""Small, local NetworkX projection of approved knowledge relations.

The graph is deliberately reconstructed from the approved card and SQLite
relation ledgers.  It is an execution aid for path retrieval, never a second
source of truth and never a place for unapproved LLM inferences.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import networkx as nx


@dataclass(frozen=True)
class EvidencePath:
    card_ids: tuple[str, ...]
    relation_ids: tuple[str, ...]
    relation_types: tuple[str, ...]


def build_graph(cards: Iterable[dict[str, Any]], relations: Iterable[dict[str, Any]]) -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()
    for card in cards:
        graph.add_node(
            str(card["card_id"]), title=str(card.get("title", "")), claim=str(card.get("claim", "")),
            labels=tuple(card.get("labels", [])), provenance=dict(card.get("provenance", {})),
        )
    for relation in relations:
        source, target = str(relation["source_card_id"]), str(relation["target_card_id"])
        if source not in graph or target not in graph:
            continue
        graph.add_edge(
            source, target, key=str(relation["relation_id"]), relation_id=str(relation["relation_id"]),
            relation_type=str(relation["relation_type"]), evidence=str(relation["evidence"]),
            conditions=str(relation["conditions"]), confidence=str(relation["confidence"]),
        )
    return graph


def evidence_paths(
    graph: nx.MultiDiGraph, source_card_id: str, target_card_id: str, *, max_hops: int = 3,
    allowed_relation_types: set[str] | None = None, limit: int = 5,
) -> list[EvidencePath]:
    """Return short, approved directed paths with their exact relation IDs."""
    if max_hops < 1 or source_card_id not in graph or target_card_id not in graph:
        return []
    result: list[EvidencePath] = []
    for nodes in nx.all_simple_paths(graph, source_card_id, target_card_id, cutoff=max_hops):
        relation_ids: list[str] = []
        relation_types: list[str] = []
        valid = True
        for source, target in zip(nodes, nodes[1:]):
            edge_options = graph.get_edge_data(source, target, default={})
            options = [item for item in edge_options.values() if allowed_relation_types is None or item["relation_type"] in allowed_relation_types]
            if not options:
                valid = False
                break
            # Multiple approved relations may connect a pair.  Keep the most
            # confident edge deterministicly, while retaining its exact ID.
            confidence_order = {"high": 0, "medium": 1, "low": 2}
            chosen = sorted(options, key=lambda item: (confidence_order.get(item["confidence"], 3), item["relation_id"]))[0]
            relation_ids.append(chosen["relation_id"])
            relation_types.append(chosen["relation_type"])
        if valid:
            result.append(EvidencePath(tuple(nodes), tuple(relation_ids), tuple(relation_types)))
            if len(result) >= limit:
                break
    return result
