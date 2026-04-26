"""
Tests for multi-CTE aggregation builder (graph2sql.cte).
"""

import pytest

from graph2sql import SchemaGraph
from graph2sql.cte import Aggregation, CTEBuilder, CTEDefinition


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DDL = """
CREATE TABLE customers (
    id      INT PRIMARY KEY,
    name    VARCHAR(100) NOT NULL,
    region  VARCHAR(50)
);
CREATE TABLE orders (
    id          INT PRIMARY KEY,
    customer_id INT NOT NULL REFERENCES customers(id),
    amount      DECIMAL(10,2),
    order_date  DATE
);
CREATE TABLE products (
    id       INT PRIMARY KEY,
    name     VARCHAR(100) NOT NULL,
    category VARCHAR(50)
);
CREATE TABLE order_items (
    order_id   INT NOT NULL REFERENCES orders(id),
    product_id INT NOT NULL REFERENCES products(id),
    quantity   INT,
    unit_price DECIMAL(10,2)
);
"""


@pytest.fixture
def graph() -> SchemaGraph:
    return SchemaGraph.from_ddl(DDL)


# ---------------------------------------------------------------------------
# Aggregation tests
# ---------------------------------------------------------------------------

class TestAggregation:
    def test_normalises_function_to_uppercase(self):
        a = Aggregation("sum", "total")
        assert a.function == "SUM"

    def test_default_alias_built_from_function_and_column(self):
        a = Aggregation("AVG", "unit_price")
        assert a.alias == "avg_unit_price"

    def test_star_column_alias(self):
        a = Aggregation("COUNT", "*")
        assert a.alias == "count_all"

    def test_explicit_alias_preserved(self):
        a = Aggregation("SUM", "amount", alias="total_revenue")
        assert a.alias == "total_revenue"

    def test_count_distinct(self):
        a = Aggregation("COUNT_DISTINCT", "order_id")
        assert a.function == "COUNT_DISTINCT"
        assert a.alias == "count_distinct_order_id"

    def test_invalid_function_raises(self):
        with pytest.raises(ValueError, match="Unknown aggregation function"):
            Aggregation("MEDIAN", "score")

    def test_to_dict_shape(self):
        a = Aggregation("MIN", "price", alias="min_price")
        d = a.to_dict()
        assert d == {"function": "MIN", "column": "price", "alias": "min_price"}


# ---------------------------------------------------------------------------
# CTEDefinition tests
# ---------------------------------------------------------------------------

class TestCTEDefinition:
    def test_defaults_to_empty_lists(self):
        cte = CTEDefinition(name="foo", query="something")
        assert cte.aggregations == []
        assert cte.group_by == []

    def test_stores_fields(self):
        agg = Aggregation("SUM", "amount")
        cte = CTEDefinition("revenue", "revenue by region", aggregations=[agg], group_by=["region"])
        assert cte.name == "revenue"
        assert cte.group_by == ["region"]
        assert len(cte.aggregations) == 1


# ---------------------------------------------------------------------------
# CTEBuilder tests
# ---------------------------------------------------------------------------

class TestCTEBuilder:
    def test_empty_cte_list_returns_empty(self, graph):
        result = CTEBuilder(graph).build([])
        assert result == {"ctes": [], "shared_nodes": []}

    def test_single_cte_structure(self, graph):
        result = CTEBuilder(graph).build([
            CTEDefinition(
                name="revenue_by_region",
                query="total revenue by customer region",
                aggregations=[Aggregation("SUM", "amount", alias="total_revenue")],
                group_by=["region"],
            )
        ])
        assert len(result["ctes"]) == 1
        cte = result["ctes"][0]
        assert cte["name"] == "revenue_by_region"
        assert isinstance(cte["nodes"], list)
        assert isinstance(cte["edges"], list)
        assert cte["aggregations"] == [{"function": "SUM", "column": "amount", "alias": "total_revenue"}]
        assert cte["group_by"] == ["region"]
        assert cte["shared_with"] == []

    def test_multi_cte_names_preserved_in_order(self, graph):
        cte_defs = [
            CTEDefinition("a", "orders by customer"),
            CTEDefinition("b", "product category sales"),
            CTEDefinition("c", "monthly order summary"),
        ]
        result = CTEBuilder(graph).build(cte_defs)
        names = [c["name"] for c in result["ctes"]]
        assert names == ["a", "b", "c"]

    def test_shared_nodes_detected(self, graph):
        # Both CTEs should rank 'orders' as relevant
        result = CTEBuilder(graph).build([
            CTEDefinition("rev", "total revenue amount orders", aggregations=[Aggregation("SUM", "amount")]),
            CTEDefinition("cnt", "count of orders by customer", aggregations=[Aggregation("COUNT", "id")]),
        ])
        # orders table should appear in both → shared_nodes non-empty
        assert len(result["shared_nodes"]) >= 1

    def test_shared_with_is_symmetric(self, graph):
        result = CTEBuilder(graph).build([
            CTEDefinition("a", "total revenue amount orders", aggregations=[Aggregation("SUM", "amount")]),
            CTEDefinition("b", "count orders by customer", aggregations=[Aggregation("COUNT", "id")]),
        ])
        cte_a = next(c for c in result["ctes"] if c["name"] == "a")
        cte_b = next(c for c in result["ctes"] if c["name"] == "b")
        # If a shares with b, b must share with a
        if cte_a["shared_with"]:
            assert "a" in cte_b["shared_with"]

    def test_nodes_are_ranked_subsets(self, graph):
        result = CTEBuilder(graph).build([
            CTEDefinition("rev", "revenue by region", aggregations=[Aggregation("SUM", "amount")]),
        ], k=2)
        nodes = result["ctes"][0]["nodes"]
        scored = [n for n in nodes if "score" in n]
        assert len(scored) <= 2

    def test_group_by_preserved_exactly(self, graph):
        cte = CTEDefinition("x", "monthly revenue", group_by=["year", "month", "region"])
        result = CTEBuilder(graph).build([cte])
        assert result["ctes"][0]["group_by"] == ["year", "month", "region"]

    def test_multiple_aggregations_per_cte(self, graph):
        cte = CTEDefinition(
            "summary",
            "order statistics",
            aggregations=[
                Aggregation("COUNT", "*"),
                Aggregation("SUM", "amount"),
                Aggregation("AVG", "amount"),
            ],
        )
        result = CTEBuilder(graph).build([cte])
        aggs = result["ctes"][0]["aggregations"]
        assert len(aggs) == 3
        functions = {a["function"] for a in aggs}
        assert functions == {"COUNT", "SUM", "AVG"}

    def test_shared_nodes_sorted(self, graph):
        result = CTEBuilder(graph).build([
            CTEDefinition("a", "orders revenue amount"),
            CTEDefinition("b", "orders customer count"),
            CTEDefinition("c", "orders monthly summary"),
        ])
        shared = result["shared_nodes"]
        assert shared == sorted(shared)

    def test_no_self_reference_in_shared_with(self, graph):
        result = CTEBuilder(graph).build([
            CTEDefinition("a", "revenue by region"),
            CTEDefinition("b", "top customers by orders"),
        ])
        for cte in result["ctes"]:
            assert cte["name"] not in cte["shared_with"]


# ---------------------------------------------------------------------------
# SchemaGraph.rank_ctes convenience wrapper
# ---------------------------------------------------------------------------

class TestSchemaGraphRankCtes:
    def test_rank_ctes_delegates_to_builder(self, graph):
        ctes = [
            CTEDefinition("r", "revenue amount orders", aggregations=[Aggregation("SUM", "amount")]),
        ]
        direct = CTEBuilder(graph).build(ctes)
        via_method = graph.rank_ctes(ctes)
        assert direct["ctes"][0]["name"] == via_method["ctes"][0]["name"]
        assert direct["ctes"][0]["aggregations"] == via_method["ctes"][0]["aggregations"]

    def test_rank_ctes_accepts_k_parameter(self, graph):
        ctes = [CTEDefinition("x", "total revenue amount")]
        result = graph.rank_ctes(ctes, k=1)
        scored = [n for n in result["ctes"][0]["nodes"] if "score" in n]
        assert len(scored) <= 1
