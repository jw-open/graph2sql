"""Tests for the Personalized PageRank algorithm."""

import pytest
from graph2sql.ranking import personalized_page_rank


ECOMMERCE_GRAPH = {
    "nodes": [
        {"id": "users",       "label": "users",       "content": "id, name, email, country"},
        {"id": "orders",      "label": "orders",      "content": "id, customer_id, total, created_at"},
        {"id": "products",    "label": "products",    "content": "id, name, price, category"},
        {"id": "order_items", "label": "order_items", "content": "id, order_id, product_id, quantity"},
    ],
    "edges": [
        {"from": "orders",      "to": "users",    "label": "belongs_to"},
        {"from": "order_items", "to": "orders",   "label": "belongs_to"},
        {"from": "order_items", "to": "products", "label": "references"},
    ],
}


def test_returns_nodes_and_edges_keys():
    result = personalized_page_rank("revenue by customer", ECOMMERCE_GRAPH)
    assert "nodes" in result
    assert "edges" in result


def test_empty_graph_returns_empty():
    result = personalized_page_rank("anything", {"nodes": [], "edges": []})
    assert result == {"nodes": [], "edges": []}


def test_no_matching_tokens_returns_empty():
    result = personalized_page_rank("xyzzy frobnicator quux", ECOMMERCE_GRAPH)
    assert result["nodes"] == []
    assert result["edges"] == []


def test_top_k_nodes_have_score():
    result = personalized_page_rank("total orders", ECOMMERCE_GRAPH, k=2)
    scored = [n for n in result["nodes"] if "score" in n]
    assert len(scored) <= 2
    for node in scored:
        assert 0.0 <= node["score"] <= 1.0


def test_query_orders_surfaces_orders_node():
    result = personalized_page_rank("orders placed last month", ECOMMERCE_GRAPH, k=3)
    labels = [n["label"] for n in result["nodes"]]
    assert "orders" in labels


def test_query_products_surfaces_products_node():
    result = personalized_page_rank("most popular products", ECOMMERCE_GRAPH, k=3)
    labels = [n["label"] for n in result["nodes"]]
    assert "products" in labels


def test_edges_use_labels_not_ids():
    result = personalized_page_rank("orders by customer", ECOMMERCE_GRAPH, k=3)
    for edge in result["edges"]:
        # Edge from/to should be label strings, not raw IDs
        assert isinstance(edge["from"], str)
        assert isinstance(edge["to"], str)


def test_result_node_labels_are_strings():
    result = personalized_page_rank("total revenue", ECOMMERCE_GRAPH)
    for node in result["nodes"]:
        assert isinstance(node["label"], str)


def test_k_limits_top_scored_nodes():
    for k in [1, 2, 3]:
        result = personalized_page_rank("users orders products", ECOMMERCE_GRAPH, k=k)
        scored = [n for n in result["nodes"] if "score" in n]
        assert len(scored) <= k
