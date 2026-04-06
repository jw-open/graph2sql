"""
Personalized PageRank (PPR) algorithm for graph2sql.

Given a natural language query and a schema graph, this module computes
PPR scores to identify the most relevant nodes (tables/fields), then
returns an extended subgraph (top-k nodes + their 1-hop neighbours)
as structured context for LLM-based SQL generation.

No database or LLM dependency — nodes carry their own content.
"""

import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .types import GraphDict


def personalized_page_rank(
    query: str,
    graph: GraphDict,
    alpha: float = 0.85,
    tol: float = 1e-6,
    max_iter: int = 50,
    k: int = 3,
) -> Dict[str, Any]:
    """
    Compute Personalized PageRank over a schema graph and return the
    most relevant subgraph for the given natural language query.

    Parameters
    ----------
    query : str
        Natural language question (e.g. "total revenue by customer last month").
    graph : GraphDict
        Schema graph: ``{"nodes": [...], "edges": [...]}``.
        Each node must have ``id`` and ``label``; ``content`` and ``attributes``
        are optional but recommended.
        Each edge must have ``"from"``, ``"to"``, and ``"label"``.
    alpha : float
        Damping factor (probability of following an edge). Default 0.85.
    tol : float
        Convergence tolerance for the power iteration. Default 1e-6.
    max_iter : int
        Maximum power-iteration steps. Default 50.
    k : int
        Number of top-ranked seed nodes to use. Default 3.

    Returns
    -------
    dict
        ``{"nodes": [...], "edges": [...]}`` — the extended subgraph containing
        the top-k PPR nodes plus their immediate 1-hop neighbours, with a
        ``"score"`` field on the top-k nodes.
        Returns ``{"nodes": [], "edges": []}`` if no query tokens match any label.
    """
    nodes: List[Dict[str, Any]] = graph.get("nodes", [])
    edges: List[Dict[str, Any]] = graph.get("edges", [])
    n = len(nodes)

    if n == 0:
        return {"nodes": [], "edges": []}

    # Map node id → matrix index; store lower-cased labels for matching.
    node_id_to_index: Dict[str, int] = {}
    labels: List[str] = []
    for i, node in enumerate(nodes):
        node_id_to_index[node["id"]] = i
        labels.append(node["label"].lower())

    # Build column-stochastic transition matrix M.
    M = np.zeros((n, n))
    outlinks: Dict[int, List[int]] = {i: [] for i in range(n)}
    for edge in edges:
        src = edge.get("from")
        tgt = edge.get("to")
        if src in node_id_to_index and tgt in node_id_to_index:
            j = node_id_to_index[src]
            i = node_id_to_index[tgt]
            outlinks[j].append(i)

    for j in range(n):
        if outlinks[j]:
            prob = 1.0 / len(outlinks[j])
            for i in outlinks[j]:
                M[i, j] = prob
        else:
            # Dangling node: distribute probability uniformly.
            M[:, j] = 1.0 / n

    # Build personalization vector using token overlap between query and labels.
    # Also matches against attribute values — use attributes like {"alias": "customers"}
    # to make a node labeled "users" match queries containing "customers".
    query_tokens = set(re.findall(r"\w+", query.lower()))
    p = np.zeros(n)
    for i, node in enumerate(nodes):
        label_tokens = re.findall(r"\w+", node["label"].lower())
        match_count = sum(1 for token in label_tokens if token in query_tokens)

        # Also check attribute values for additional token matches.
        attrs = node.get("attributes") or {}
        for attr_value in attrs.values():
            if isinstance(attr_value, str):
                attr_tokens = re.findall(r"\w+", attr_value.lower())
                match_count += sum(1 for token in attr_tokens if token in query_tokens)

        p[i] = match_count

    if p.sum() == 0:
        return {"nodes": [], "edges": []}

    p = p / p.sum()

    # Power iteration.
    r = np.ones(n) / n
    for _ in range(max_iter):
        r_new = (1 - alpha) * p + alpha * M.dot(r)
        if np.linalg.norm(r_new - r, 1) < tol:
            r = r_new
            break
        r = r_new

    # Select top-k nodes by PPR score.
    scored = sorted(
        ((r[node_id_to_index[node["id"]]], node["id"], node["label"]) for node in nodes),
        key=lambda t: t[0],
        reverse=True,
    )
    top_tuples = scored[:k]
    top_ids = {node_id for _, node_id, _ in top_tuples}
    top_scores = {node_id: float(score) for score, node_id, _ in top_tuples}

    top_node_objs = [node for node in nodes if node["id"] in top_ids]

    # Extend to 1-hop neighbours.
    extended_nodes, extended_edges = _retrieve_extended_subgraph(top_node_objs, graph)

    id_to_label = {node["id"]: node["label"] for node in extended_nodes}

    # Format nodes — include label, content, attributes, and score for top-k.
    formatted_nodes = []
    for node in extended_nodes:
        node_dict: Dict[str, Any] = {
            "label": node["label"],
            "content": node.get("content"),
            "attributes": node.get("attributes", {}),
        }
        if node["id"] in top_scores:
            node_dict["score"] = top_scores[node["id"]]
        formatted_nodes.append(node_dict)

    # Format edges — replace ids with labels.
    formatted_edges = []
    for edge in extended_edges:
        from_label = id_to_label.get(edge.get("from"))
        to_label = id_to_label.get(edge.get("to"))
        if from_label and to_label:
            formatted_edges.append({
                "label": edge.get("label", ""),
                "from": from_label,
                "to": to_label,
            })

    return {"nodes": formatted_nodes, "edges": formatted_edges}


def _retrieve_extended_subgraph(
    top_nodes: List[Dict[str, Any]],
    graph: GraphDict,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Extend top-k nodes with their immediate (1-hop) neighbours and all
    edges that connect any two nodes in the combined set.
    """
    top_ids = {node["id"] for node in top_nodes}
    extended_ids = set(top_ids)

    for edge in graph.get("edges", []):
        if edge.get("from") in top_ids or edge.get("to") in top_ids:
            extended_ids.add(edge.get("from"))
            extended_ids.add(edge.get("to"))

    extended_nodes = [n for n in graph.get("nodes", []) if n["id"] in extended_ids]
    extended_edges = [
        e for e in graph.get("edges", [])
        if e.get("from") in extended_ids and e.get("to") in extended_ids
    ]
    return extended_nodes, extended_edges
