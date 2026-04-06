"""Tests for the SchemaGraph class."""

import pytest
from graph2sql import SchemaGraph


def _ecommerce_graph() -> SchemaGraph:
    g = SchemaGraph()
    g.add_node("users",       "users",       content="id, name, email")
    g.add_node("orders",      "orders",      content="id, customer_id, total")
    g.add_node("products",    "products",    content="id, name, price")
    g.add_node("order_items", "order_items", content="id, order_id, product_id, quantity")
    g.add_edge("orders",      "users",    "belongs_to")
    g.add_edge("order_items", "orders",   "belongs_to")
    g.add_edge("order_items", "products", "references")
    return g


def test_repr():
    g = _ecommerce_graph()
    assert "SchemaGraph" in repr(g)
    assert "nodes=4" in repr(g)
    assert "edges=3" in repr(g)


def test_from_dict_roundtrip():
    g1 = _ecommerce_graph()
    d = g1.to_dict()
    g2 = SchemaGraph.from_dict(d)
    assert len(g2.to_dict()["nodes"]) == 4
    assert len(g2.to_dict()["edges"]) == 3


def test_duplicate_node_raises():
    g = SchemaGraph()
    g.add_node("1", "users")
    with pytest.raises(ValueError, match="already exists"):
        g.add_node("1", "orders")


def test_rank_returns_dict_with_expected_keys():
    g = _ecommerce_graph()
    result = g.rank("total revenue by customer")
    assert "nodes" in result
    assert "edges" in result


def test_rank_empty_graph_returns_empty():
    g = SchemaGraph()
    result = g.rank("anything")
    assert result == {"nodes": [], "edges": []}


def test_method_chaining():
    g = (
        SchemaGraph()
        .add_node("a", "alpha")
        .add_node("b", "beta")
        .add_edge("a", "b", "rel")
    )
    assert len(g.to_dict()["nodes"]) == 2
    assert len(g.to_dict()["edges"]) == 1


def test_rank_k_param():
    g = _ecommerce_graph()
    result = g.rank("orders products users", k=2)
    scored = [n for n in result["nodes"] if "score" in n]
    assert len(scored) <= 2


def test_node_content_preserved_in_rank():
    g = SchemaGraph()
    g.add_node("t", "transactions", content="id, amount, currency")
    result = g.rank("transactions")
    assert any(n["content"] == "id, amount, currency" for n in result["nodes"])
