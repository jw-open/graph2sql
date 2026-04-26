"""
Multi-CTE aggregation builder for graph2sql.

Lets you define several named Common Table Expressions (CTEs), each with its own
natural-language query, aggregation functions, and GROUP BY columns.
:class:`CTEBuilder` ranks the schema graph for each CTE independently and returns
structured context ready to pass to any LLM for WITH-clause SQL generation.

Example
-------
>>> from graph2sql import SchemaGraph
>>> from graph2sql.cte import CTEBuilder, CTEDefinition, Aggregation
>>> g = SchemaGraph.from_ddl(ddl)
>>> builder = CTEBuilder(g)
>>> result = builder.build([
...     CTEDefinition(
...         name="revenue_by_region",
...         query="total revenue grouped by customer region",
...         aggregations=[Aggregation("SUM", "total", alias="total_revenue")],
...         group_by=["region"],
...     ),
...     CTEDefinition(
...         name="top_customers",
...         query="top customers by number of orders",
...         aggregations=[Aggregation("COUNT", "order_id", alias="order_count")],
...         group_by=["customer_id", "customer_name"],
...     ),
... ])
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:  # avoid circular imports at runtime
    from .graph import SchemaGraph

_VALID_FUNCTIONS = {"SUM", "COUNT", "AVG", "MIN", "MAX", "COUNT_DISTINCT"}


@dataclass
class Aggregation:
    """
    One aggregation function applied to a column.

    Parameters
    ----------
    function : str
        SQL aggregation function.  One of: ``SUM``, ``COUNT``, ``AVG``,
        ``MIN``, ``MAX``, ``COUNT_DISTINCT``.  Case-insensitive.
    column : str
        Column name to aggregate (e.g. ``"total"``, ``"order_id"``, ``"*"``).
    alias : str, optional
        Output column alias.  Defaults to ``"<function>_<column>"``.

    Example
    -------
    >>> Aggregation("SUM", "total", alias="total_revenue")
    >>> Aggregation("COUNT", "*", alias="num_orders")
    >>> Aggregation("AVG", "unit_price")
    """

    function: str
    column: str
    alias: Optional[str] = None

    def __post_init__(self) -> None:
        fn = self.function.upper()
        if fn not in _VALID_FUNCTIONS:
            raise ValueError(
                f"Unknown aggregation function '{self.function}'. "
                f"Valid options: {', '.join(sorted(_VALID_FUNCTIONS))}"
            )
        self.function = fn
        if self.alias is None:
            safe_col = self.column.replace("*", "all")
            self.alias = f"{fn.lower()}_{safe_col}"

    def to_dict(self) -> Dict[str, str]:
        """Return a plain dict representation."""
        return {"function": self.function, "column": self.column, "alias": self.alias}


@dataclass
class CTEDefinition:
    """
    Describes one named CTE in a WITH clause.

    Parameters
    ----------
    name : str
        SQL identifier for this CTE (e.g. ``"revenue_by_region"``).
    query : str
        Natural-language description of what this CTE computes.
        Used to rank schema nodes via Personalized PageRank.
    aggregations : list[Aggregation]
        Aggregation functions to apply (e.g. ``SUM(total)``, ``COUNT(*)``).
        May be empty for filtering or deduplication CTEs.
    group_by : list[str]
        Column names to GROUP BY (e.g. ``["region", "year"]``).

    Example
    -------
    >>> CTEDefinition(
    ...     name="revenue_by_region",
    ...     query="total sales revenue grouped by region",
    ...     aggregations=[Aggregation("SUM", "amount", alias="total_revenue")],
    ...     group_by=["region"],
    ... )
    """

    name: str
    query: str
    aggregations: List[Aggregation] = field(default_factory=list)
    group_by: List[str] = field(default_factory=list)


class CTEBuilder:
    """
    Build ranked schema context for multiple named CTEs from a single SchemaGraph.

    Each CTE definition gets its own Personalized PageRank pass so the most
    relevant tables are identified independently per sub-query.  After ranking,
    the builder detects which table nodes are shared across CTEs — a useful hint
    for LLMs that need to understand join paths between the CTEs.

    Parameters
    ----------
    graph : SchemaGraph
        The schema graph to rank against.

    Example
    -------
    >>> from graph2sql import SchemaGraph
    >>> from graph2sql.cte import CTEBuilder, CTEDefinition, Aggregation
    >>>
    >>> g = SchemaGraph.from_ddl('''
    ...     CREATE TABLE customers (id INT PRIMARY KEY, name VARCHAR(100), region VARCHAR(50));
    ...     CREATE TABLE orders (id INT PRIMARY KEY, customer_id INT REFERENCES customers(id),
    ...                          amount DECIMAL(10,2), order_date DATE);
    ...     CREATE TABLE products (id INT PRIMARY KEY, name VARCHAR(100), category VARCHAR(50));
    ...     CREATE TABLE order_items (order_id INT REFERENCES orders(id),
    ...                               product_id INT REFERENCES products(id),
    ...                               quantity INT, unit_price DECIMAL(10,2));
    ... ''')
    >>>
    >>> result = CTEBuilder(g).build([
    ...     CTEDefinition(
    ...         name="revenue_by_region",
    ...         query="total revenue grouped by customer region",
    ...         aggregations=[Aggregation("SUM", "amount", alias="total_revenue")],
    ...         group_by=["region"],
    ...     ),
    ...     CTEDefinition(
    ...         name="top_products",
    ...         query="top selling products by quantity sold",
    ...         aggregations=[
    ...             Aggregation("SUM", "quantity", alias="total_qty"),
    ...             Aggregation("COUNT_DISTINCT", "order_id", alias="num_orders"),
    ...         ],
    ...         group_by=["product_id", "category"],
    ...     ),
    ...     CTEDefinition(
    ...         name="monthly_summary",
    ...         query="monthly order counts and average order value",
    ...         aggregations=[
    ...             Aggregation("COUNT", "order_id", alias="num_orders"),
    ...             Aggregation("AVG", "amount", alias="avg_order_value"),
    ...         ],
    ...         group_by=["month", "year"],
    ...     ),
    ... ])
    >>>
    >>> for cte in result["ctes"]:
    ...     print(cte["name"], "→ shared with:", cte["shared_with"])
    """

    def __init__(self, graph: "SchemaGraph") -> None:
        self._graph = graph

    def build(
        self,
        ctes: List[CTEDefinition],
        k: int = 3,
        alpha: float = 0.85,
        max_iter: int = 50,
        tol: float = 1e-6,
    ) -> Dict[str, Any]:
        """
        Rank schema context for each CTE and return a structured multi-CTE plan.

        Parameters
        ----------
        ctes : list[CTEDefinition]
            Ordered list of CTE definitions.  Order is preserved in the output.
        k : int
            Top-k seed nodes per PPR pass.  Default 3.
        alpha : float
            PPR damping factor.  Default 0.85.
        max_iter : int
            Maximum power-iteration steps.  Default 50.
        tol : float
            Convergence tolerance.  Default 1e-6.

        Returns
        -------
        dict
            A dict with two keys:

            ``"ctes"`` — list of CTE context dicts, each containing:

            * ``"name"`` — CTE identifier
            * ``"nodes"`` — ranked schema nodes (top-k carry a ``"score"`` field)
            * ``"edges"`` — edges between the ranked nodes
            * ``"aggregations"`` — list of ``{"function", "column", "alias"}`` dicts
            * ``"group_by"`` — list of column names
            * ``"shared_with"`` — names of other CTEs that share at least one table node

            ``"shared_nodes"`` — sorted list of node labels appearing in 2+ CTEs
            (useful for instructing the LLM about common join anchors).

        Example
        -------
        >>> result = builder.build(cte_list, k=4)
        >>> for cte in result["ctes"]:
        ...     nodes = [n["label"] for n in cte["nodes"]]
        ...     print(f"{cte['name']}: {nodes} | shared with {cte['shared_with']}")
        """
        if not ctes:
            return {"ctes": [], "shared_nodes": []}

        # ── Rank each CTE independently ──────────────────────────────────
        ranked: List[Dict[str, Any]] = []
        for cte in ctes:
            ctx = self._graph.rank(
                cte.query,
                k=k,
                alpha=alpha,
                max_iter=max_iter,
                tol=tol,
            )
            ranked.append({
                "name": cte.name,
                "nodes": ctx["nodes"],
                "edges": ctx["edges"],
                "aggregations": [a.to_dict() for a in cte.aggregations],
                "group_by": list(cte.group_by),
                "shared_with": [],  # filled below
            })

        # ── Detect shared nodes (labels appearing in 2+ CTEs) ────────────
        label_to_cte_names: Dict[str, List[str]] = {}
        for r in ranked:
            for node in r["nodes"]:
                lbl = node["label"]
                label_to_cte_names.setdefault(lbl, []).append(r["name"])

        shared_labels = {
            lbl for lbl, names in label_to_cte_names.items() if len(names) > 1
        }

        # ── Annotate each CTE with sibling names that share nodes ─────────
        for r in ranked:
            own_labels = {n["label"] for n in r["nodes"]}
            shared_with = [
                other["name"]
                for other in ranked
                if other["name"] != r["name"]
                and bool({n["label"] for n in other["nodes"]} & own_labels)
            ]
            r["shared_with"] = shared_with

        return {
            "ctes": ranked,
            "shared_nodes": sorted(shared_labels),
        }
