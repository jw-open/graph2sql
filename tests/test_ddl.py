"""Tests for SchemaGraph.from_ddl() — pure-Python DDL parser."""

import pytest
from graph2sql import SchemaGraph

ECOMMERCE_DDL = """
CREATE TABLE customers (
    id    INT          PRIMARY KEY,
    name  VARCHAR(100) NOT NULL,
    email VARCHAR(200)
);

CREATE TABLE orders (
    id          INT          PRIMARY KEY,
    customer_id INT          NOT NULL REFERENCES customers(id),
    total       DECIMAL(10,2)
);

CREATE TABLE order_items (
    id         INT PRIMARY KEY,
    order_id   INT NOT NULL,
    product    VARCHAR(200),
    quantity   INT,
    FOREIGN KEY (order_id) REFERENCES orders(id)
);
"""

MYSQL_DDL = """
CREATE TABLE `users` (
  `id`       INT NOT NULL AUTO_INCREMENT,
  `username` VARCHAR(50) NOT NULL,
  `email`    VARCHAR(100),
  PRIMARY KEY (`id`)
);

CREATE TABLE `posts` (
  `id`      INT NOT NULL AUTO_INCREMENT,
  `user_id` INT NOT NULL,
  `title`   VARCHAR(255) NOT NULL,
  PRIMARY KEY (`id`),
  FOREIGN KEY (`user_id`) REFERENCES `users` (`id`)
);
"""


def test_tables_become_nodes():
    g = SchemaGraph.from_ddl(ECOMMERCE_DDL)
    labels = {n["label"] for n in g.to_dict()["nodes"]}
    assert labels == {"customers", "orders", "order_items"}


def test_column_content_present():
    g = SchemaGraph.from_ddl(ECOMMERCE_DDL)
    nodes = {n["label"]: n for n in g.to_dict()["nodes"]}
    assert "name" in nodes["customers"]["content"]
    assert "email" in nodes["customers"]["content"]
    assert "total" in nodes["orders"]["content"]


def test_inline_primary_key_flagged():
    g = SchemaGraph.from_ddl(ECOMMERCE_DDL)
    nodes = {n["label"]: n for n in g.to_dict()["nodes"]}
    assert "PK" in nodes["customers"]["content"]
    assert "PK" in nodes["orders"]["content"]


def test_not_null_flagged():
    g = SchemaGraph.from_ddl(ECOMMERCE_DDL)
    nodes = {n["label"]: n for n in g.to_dict()["nodes"]}
    assert "NOT NULL" in nodes["customers"]["content"]


def test_inline_fk_creates_edge():
    g = SchemaGraph.from_ddl(ECOMMERCE_DDL)
    edges = {(e["from"], e["to"]) for e in g.to_dict()["edges"]}
    assert ("orders", "customers") in edges


def test_block_fk_creates_edge():
    g = SchemaGraph.from_ddl(ECOMMERCE_DDL)
    edges = {(e["from"], e["to"]) for e in g.to_dict()["edges"]}
    assert ("order_items", "orders") in edges


def test_mysql_backtick_identifiers():
    g = SchemaGraph.from_ddl(MYSQL_DDL)
    labels = {n["label"] for n in g.to_dict()["nodes"]}
    assert "users" in labels
    assert "posts" in labels


def test_mysql_block_pk_flagged():
    g = SchemaGraph.from_ddl(MYSQL_DDL)
    nodes = {n["label"]: n for n in g.to_dict()["nodes"]}
    assert "PK" in nodes["users"]["content"]


def test_mysql_block_fk_edge():
    g = SchemaGraph.from_ddl(MYSQL_DDL)
    edges = {(e["from"], e["to"]) for e in g.to_dict()["edges"]}
    assert ("posts", "users") in edges


def test_rank_after_ddl():
    g = SchemaGraph.from_ddl(ECOMMERCE_DDL)
    context = g.rank("total revenue per customer", k=2)
    labels = [n["label"] for n in context["nodes"]]
    assert any(l in labels for l in ("customers", "orders"))


def test_empty_ddl_returns_empty_graph():
    g = SchemaGraph.from_ddl("")
    assert g.to_dict() == {"nodes": [], "edges": []}
