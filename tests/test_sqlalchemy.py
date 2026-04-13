"""Tests for SchemaGraph.from_sqlalchemy() using an in-memory SQLite database."""

import pytest

try:
    from sqlalchemy import (
        Column,
        ForeignKey,
        Integer,
        String,
        create_engine,
        text,
    )
    from sqlalchemy.orm import declarative_base

    HAS_SQLALCHEMY = True
except ImportError:
    HAS_SQLALCHEMY = False

from graph2sql import SchemaGraph

pytestmark = pytest.mark.skipif(
    not HAS_SQLALCHEMY, reason="sqlalchemy not installed"
)


def _make_engine():
    """In-memory SQLite engine with a small e-commerce schema."""
    engine = create_engine("sqlite:///:memory:")
    Base = declarative_base()

    class Customer(Base):
        __tablename__ = "customers"
        id = Column(Integer, primary_key=True)
        name = Column(String(100), nullable=False)
        email = Column(String(200), nullable=False)

    class Order(Base):
        __tablename__ = "orders"
        id = Column(Integer, primary_key=True)
        customer_id = Column(Integer, ForeignKey("customers.id"), nullable=False)
        total = Column(Integer, nullable=False)

    class OrderItem(Base):
        __tablename__ = "order_items"
        id = Column(Integer, primary_key=True)
        order_id = Column(Integer, ForeignKey("orders.id"), nullable=False)
        product = Column(String(200))
        quantity = Column(Integer)

    Base.metadata.create_all(engine)
    return engine


def test_nodes_created_for_each_table():
    engine = _make_engine()
    g = SchemaGraph.from_sqlalchemy(engine)
    labels = {n["label"] for n in g.to_dict()["nodes"]}
    assert "customers" in labels
    assert "orders" in labels
    assert "order_items" in labels


def test_column_content_included():
    engine = _make_engine()
    g = SchemaGraph.from_sqlalchemy(engine)
    nodes = {n["label"]: n for n in g.to_dict()["nodes"]}
    assert "name" in nodes["customers"]["content"]
    assert "email" in nodes["customers"]["content"]
    assert "total" in nodes["orders"]["content"]


def test_primary_key_flagged():
    engine = _make_engine()
    g = SchemaGraph.from_sqlalchemy(engine)
    nodes = {n["label"]: n for n in g.to_dict()["nodes"]}
    assert "PK" in nodes["customers"]["content"]


def test_foreign_key_edges_created():
    engine = _make_engine()
    g = SchemaGraph.from_sqlalchemy(engine)
    edges = g.to_dict()["edges"]
    edge_pairs = {(e["from"], e["to"]) for e in edges}
    assert ("orders", "customers") in edge_pairs
    assert ("order_items", "orders") in edge_pairs


def test_rank_returns_relevant_tables():
    engine = _make_engine()
    g = SchemaGraph.from_sqlalchemy(engine)
    context = g.rank("total revenue per customer", k=2)
    labels = [n["label"] for n in context["nodes"]]
    # customers or orders should surface for this query
    assert any(l in labels for l in ("customers", "orders"))


def test_missing_sqlalchemy_raises_import_error(monkeypatch):
    import builtins
    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "sqlalchemy":
            raise ImportError("mocked missing sqlalchemy")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)
    with pytest.raises(ImportError, match="pip install graph2sql"):
        SchemaGraph.from_sqlalchemy(None)
