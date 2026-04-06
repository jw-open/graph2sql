from typing import Any, Dict, List, Optional

# Node dict shape:
#   id         (str)  — unique identifier
#   label      (str)  — name of the node (table name, field name, etc.); also used in query matching
#   content    (str, optional) — rich description of the node (column types, constraints, notes)
#   attributes (dict, optional) — any extra metadata
Node = Dict[str, Any]

# Edge dict shape uses the string key "from" (not a Python keyword in dict context):
#   "from"  (str) — source node id
#   "to"    (str) — target node id
#   "label" (str) — relationship label (e.g. "foreign_key", "belongs_to")
Edge = Dict[str, Any]

# Full graph container
GraphDict = Dict[str, Any]  # {"nodes": List[Node], "edges": List[Edge]}


def make_node(id: str, label: str, content: Optional[str] = None, attributes: Optional[Dict[str, Any]] = None) -> Node:
    """Helper to construct a well-formed node dict."""
    node: Node = {"id": id, "label": label}
    if content is not None:
        node["content"] = content
    if attributes is not None:
        node["attributes"] = attributes
    return node


def make_edge(from_id: str, to_id: str, label: str) -> Edge:
    """Helper to construct a well-formed edge dict."""
    return {"from": from_id, "to": to_id, "label": label}
