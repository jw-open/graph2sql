"""
SchemaGraph — the main public interface for graph2sql.

Build a schema graph from a dict, then call .rank() to extract
a ranked subgraph as context for LLM-based SQL generation.
"""

from typing import Any, Dict, List, Optional

from .ranking import personalized_page_rank
from .types import GraphDict, make_edge, make_node


class SchemaGraph:
    """
    A graph representation of a database (or any structured) schema.

    Nodes represent tables, columns, or any named schema entity.
    Edges represent relationships between them (foreign keys, ownership, etc.).

    The ``label`` of each node is used for natural language query matching,
    so use descriptive names (e.g. ``"orders"``, ``"customer_id"``).

    Example
    -------
    >>> graph = SchemaGraph()
    >>> graph.add_node("1", "orders", content="id, customer_id, total, created_at")
    >>> graph.add_node("2", "customers", content="id, name, email")
    >>> graph.add_edge("1", "2", "belongs_to")
    >>> context = graph.rank("total revenue per customer")
    >>> print(context)
    """

    def __init__(self) -> None:
        self._nodes: List[Dict[str, Any]] = []
        self._edges: List[Dict[str, Any]] = []
        self._node_ids: set = set()

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_dict(cls, graph: GraphDict) -> "SchemaGraph":
        """
        Build a SchemaGraph from a raw graph dict.

        Parameters
        ----------
        graph : dict
            ``{"nodes": [...], "edges": [...]}``
            See :mod:`graph2sql.types` for node/edge shapes.
        """
        instance = cls()
        instance._nodes = list(graph.get("nodes", []))
        instance._edges = list(graph.get("edges", []))
        instance._node_ids = {n["id"] for n in instance._nodes}
        return instance

    def add_node(
        self,
        id: str,
        label: str,
        content: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> "SchemaGraph":
        """
        Add a node to the graph.

        Parameters
        ----------
        id : str
            Unique node identifier.
        label : str
            Human-readable name (e.g. table or column name). Used for query matching.
        content : str, optional
            Description or schema definition (e.g. ``"id INT, name VARCHAR(255)"``).
        attributes : dict, optional
            Extra metadata (e.g. ``{"type": "table", "primary_key": "id"}``).

        Returns
        -------
        SchemaGraph
            Self, for method chaining.
        """
        if id in self._node_ids:
            raise ValueError(f"Node with id '{id}' already exists.")
        self._nodes.append(make_node(id, label, content, attributes))
        self._node_ids.add(id)
        return self

    def add_edge(self, from_id: str, to_id: str, label: str) -> "SchemaGraph":
        """
        Add a directed edge between two nodes.

        Parameters
        ----------
        from_id : str
            Source node id.
        to_id : str
            Target node id.
        label : str
            Relationship label (e.g. ``"foreign_key"``, ``"belongs_to"``).

        Returns
        -------
        SchemaGraph
            Self, for method chaining.
        """
        self._edges.append(make_edge(from_id, to_id, label))
        return self

    # ------------------------------------------------------------------
    # Query interface
    # ------------------------------------------------------------------

    def rank(
        self,
        query: str,
        k: int = 3,
        alpha: float = 0.85,
        max_iter: int = 50,
        tol: float = 1e-6,
    ) -> Dict[str, Any]:
        """
        Rank schema nodes by relevance to a natural language query using
        Personalized PageRank, and return the extended subgraph as context.

        Parameters
        ----------
        query : str
            Natural language question (e.g. ``"total revenue by customer last month"``).
        k : int
            Number of top-ranked seed nodes. Default 3.
        alpha : float
            PPR damping factor. Default 0.85.
        max_iter : int
            Max power-iteration steps. Default 50.
        tol : float
            Convergence tolerance. Default 1e-6.

        Returns
        -------
        dict
            ``{"nodes": [...], "edges": [...]}``
            Top-k nodes carry a ``"score"`` field. All nodes carry
            ``"label"``, ``"content"``, and ``"attributes"``.
            Returns empty lists if no labels match the query.
        """
        return personalized_page_rank(
            query=query,
            graph=self.to_dict(),
            alpha=alpha,
            tol=tol,
            max_iter=max_iter,
            k=k,
        )

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> GraphDict:
        """Return the raw graph dict ``{"nodes": [...], "edges": [...]}"``."""
        return {"nodes": self._nodes, "edges": self._edges}

    # ------------------------------------------------------------------
    # Database introspection (optional — requires SQLAlchemy)
    # ------------------------------------------------------------------

    @classmethod
    def from_sqlalchemy(cls, engine: Any) -> "SchemaGraph":
        """
        Build a SchemaGraph by introspecting a live database via SQLAlchemy.

        Each table becomes a node whose ``content`` lists every column with its
        type and key constraints.  Foreign-key relationships become directed
        edges between table nodes.

        Requires SQLAlchemy (any version ≥ 1.4)::

            pip install graph2sql[sqlalchemy]

        Parameters
        ----------
        engine : sqlalchemy.engine.Engine
            A connected SQLAlchemy engine (``create_engine(...)``).

        Returns
        -------
        SchemaGraph

        Example
        -------
        >>> from sqlalchemy import create_engine
        >>> from graph2sql import SchemaGraph
        >>> engine = create_engine("postgresql://user:pw@localhost/mydb")
        >>> g = SchemaGraph.from_sqlalchemy(engine)
        >>> context = g.rank("total revenue per customer last month")
        """
        try:
            from sqlalchemy import inspect as sa_inspect
        except ImportError:
            raise ImportError(
                "SQLAlchemy is required for from_sqlalchemy(). "
                "Install it with: pip install graph2sql[sqlalchemy]"
            )

        inspector = sa_inspect(engine)
        instance = cls()

        for table_name in inspector.get_table_names():
            columns = inspector.get_columns(table_name)
            pk_info = inspector.get_pk_constraint(table_name)
            pk_cols = set(pk_info.get("constrained_columns", []))

            col_defs = []
            for col in columns:
                col_type = str(col["type"])
                flags = []
                if col["name"] in pk_cols:
                    flags.append("PK")
                if not col.get("nullable", True):
                    flags.append("NOT NULL")
                flag_str = " " + " ".join(flags) if flags else ""
                col_defs.append(f"{col['name']} {col_type}{flag_str}")

            content = ", ".join(col_defs)
            instance.add_node(
                table_name,
                table_name,
                content=content,
                attributes={"type": "table"},
            )

        # Foreign-key edges (second pass so all nodes exist)
        for table_name in inspector.get_table_names():
            for fk in inspector.get_foreign_keys(table_name):
                ref_table = fk.get("referred_table", "")
                if ref_table and ref_table in instance._node_ids:
                    instance.add_edge(table_name, ref_table, "foreign_key")

        return instance

    def __repr__(self) -> str:
        return (
            f"SchemaGraph(nodes={len(self._nodes)}, edges={len(self._edges)})"
        )
