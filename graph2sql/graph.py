"""
SchemaGraph — the main public interface for graph2sql.

Build a schema graph from a dict, then call .rank() to extract
a ranked subgraph as context for LLM-based SQL generation.
"""

import re
from typing import Any, Dict, List, Optional

from .ranking import personalized_page_rank
from .types import GraphDict, make_edge, make_node


# ---------------------------------------------------------------------------
# DDL parsing helpers (module-level, no dependencies)
# ---------------------------------------------------------------------------

def _split_ddl_body(body: str) -> List[str]:
    """Split a CREATE TABLE body into individual column/constraint clauses."""
    clauses: List[str] = []
    depth = 0
    current: List[str] = []
    for char in body:
        if char == "(":
            depth += 1
            current.append(char)
        elif char == ")":
            depth -= 1
            current.append(char)
        elif char == "," and depth == 0:
            clauses.append("".join(current))
            current = []
        else:
            current.append(char)
    if current:
        clauses.append("".join(current))
    return clauses


def _unquote(name: str) -> str:
    """Strip SQL quoting characters from an identifier."""
    return name.strip().strip("`\"[]")


def _extract_table_blocks(ddl: str) -> List[tuple]:
    """
    Return list of (table_name, body) tuples extracted from CREATE TABLE DDL.
    Uses paren-depth tracking so column types like VARCHAR(100) don't confuse
    the parser.
    """
    create_re = re.compile(
        r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?([`\"\[]?\w+[`\"\]]?)\s*\(",
        re.IGNORECASE,
    )
    results = []
    for m in create_re.finditer(ddl):
        table_name = _unquote(m.group(1))
        start = m.end()  # position right after the opening '('
        depth = 1
        i = start
        while i < len(ddl) and depth > 0:
            if ddl[i] == "(":
                depth += 1
            elif ddl[i] == ")":
                depth -= 1
            i += 1
        body = ddl[start : i - 1]  # everything between the outer parens
        results.append((table_name, body))
    return results


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

    def rank_ctes(
        self,
        ctes: List[Any],
        k: int = 3,
        alpha: float = 0.85,
        max_iter: int = 50,
        tol: float = 1e-6,
    ) -> Dict[str, Any]:
        """
        Convenience wrapper around :class:`~graph2sql.cte.CTEBuilder`.

        Ranks the schema graph for each :class:`~graph2sql.cte.CTEDefinition`
        independently and returns a structured multi-CTE context plan suitable
        for passing to an LLM to generate a SQL ``WITH`` clause.

        Parameters
        ----------
        ctes : list[CTEDefinition]
            Ordered list of CTE definitions.
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
            See :meth:`~graph2sql.cte.CTEBuilder.build` for the full schema.

        Example
        -------
        >>> from graph2sql.cte import CTEDefinition, Aggregation
        >>> result = g.rank_ctes([
        ...     CTEDefinition(
        ...         name="revenue_by_region",
        ...         query="total revenue grouped by region",
        ...         aggregations=[Aggregation("SUM", "amount", alias="total_revenue")],
        ...         group_by=["region"],
        ...     ),
        ...     CTEDefinition(
        ...         name="top_customers",
        ...         query="customers with highest order count",
        ...         aggregations=[Aggregation("COUNT", "order_id", alias="num_orders")],
        ...         group_by=["customer_id"],
        ...     ),
        ... ])
        >>> print(result["shared_nodes"])
        """
        from .cte import CTEBuilder
        return CTEBuilder(self).build(
            ctes, k=k, alpha=alpha, max_iter=max_iter, tol=tol
        )

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
    # DDL parsing (pure Python, no dependencies)
    # ------------------------------------------------------------------

    @classmethod
    def from_ddl(cls, ddl: str) -> "SchemaGraph":
        """
        Build a SchemaGraph by parsing raw SQL DDL statements.

        Supports common SQL dialects (PostgreSQL, MySQL, SQLite).
        Handles ``CREATE TABLE`` statements including:

        * Column definitions with types, ``PRIMARY KEY``, and ``NOT NULL``
        * Block-level ``PRIMARY KEY (col, ...)`` constraints
        * Inline ``REFERENCES other_table`` and block-level ``FOREIGN KEY``

        No live database or extra dependencies required — pure Python.

        Parameters
        ----------
        ddl : str
            One or more ``CREATE TABLE`` SQL statements.

        Returns
        -------
        SchemaGraph

        Example
        -------
        >>> ddl = '''
        ...   CREATE TABLE customers (
        ...     id   INT PRIMARY KEY,
        ...     name VARCHAR(100) NOT NULL,
        ...     email VARCHAR(200)
        ...   );
        ...   CREATE TABLE orders (
        ...     id          INT PRIMARY KEY,
        ...     customer_id INT NOT NULL REFERENCES customers(id),
        ...     total       DECIMAL(10,2)
        ...   );
        ... '''
        >>> g = SchemaGraph.from_ddl(ddl)
        >>> context = g.rank("total revenue per customer")
        """
        instance = cls()

        # Collect tables first so FK edges can reference any table
        parsed: Dict[str, Any] = {}

        for table_name, body in _extract_table_blocks(ddl):
            clauses = _split_ddl_body(body)

            pk_cols: set = set()
            col_defs: List[str] = []
            fk_targets: List[str] = []

            for clause in clauses:
                clause = clause.strip()
                if not clause:
                    continue
                upper = clause.upper().lstrip()

                # Block PRIMARY KEY (col, ...)
                pk_m = re.match(
                    r"PRIMARY\s+KEY\s*\(([^)]+)\)", clause, re.IGNORECASE
                )
                if pk_m:
                    for col in pk_m.group(1).split(","):
                        pk_cols.add(_unquote(col))
                    continue

                # Block FOREIGN KEY (...) REFERENCES table(...)
                fk_m = re.match(
                    r"(?:CONSTRAINT\s+\S+\s+)?FOREIGN\s+KEY\s*\([^)]+\)\s*"
                    r"REFERENCES\s+([`\"\[]?\w+[`\"\]]?)",
                    clause, re.IGNORECASE,
                )
                if fk_m:
                    fk_targets.append(_unquote(fk_m.group(1)))
                    continue

                # Skip other table-level constraints
                if re.match(r"(UNIQUE|CHECK|INDEX|KEY|CONSTRAINT)\b", upper):
                    continue

                # Column definition
                col_m = re.match(
                    r"([`\"\[]?\w+[`\"\]]?)\s+(\S+.*)", clause, re.IGNORECASE
                )
                if not col_m:
                    continue

                col_name = _unquote(col_m.group(1))
                rest = col_m.group(2)

                # Type (may include size: VARCHAR(100))
                type_m = re.match(r"(\w+(?:\s*\([^)]*\))?)", rest)
                col_type = type_m.group(1) if type_m else rest.split()[0]

                flags: List[str] = []
                if re.search(r"\bPRIMARY\s+KEY\b", rest, re.IGNORECASE):
                    pk_cols.add(col_name)
                    flags.append("PK")
                if re.search(r"\bNOT\s+NULL\b", rest, re.IGNORECASE):
                    flags.append("NOT NULL")

                # Inline REFERENCES
                ref_m = re.search(
                    r"\bREFERENCES\s+([`\"\[]?\w+[`\"\]]?)", rest, re.IGNORECASE
                )
                if ref_m:
                    fk_targets.append(_unquote(ref_m.group(1)))

                flag_str = " " + " ".join(flags) if flags else ""
                col_defs.append(f"{col_name} {col_type}{flag_str}")

            # Apply block-level PK flags
            final_defs = []
            for col_def in col_defs:
                col_name = col_def.split()[0]
                if col_name in pk_cols and "PK" not in col_def:
                    final_defs.append(col_def + " PK")
                else:
                    final_defs.append(col_def)

            parsed[table_name] = {
                "col_defs": final_defs,
                "fk_targets": fk_targets,
            }

        # Add nodes
        for table_name, info in parsed.items():
            content = ", ".join(info["col_defs"])
            instance.add_node(
                table_name, table_name,
                content=content,
                attributes={"type": "table"},
            )

        # Add FK edges (second pass so all nodes exist)
        for table_name, info in parsed.items():
            for ref_table in info["fk_targets"]:
                if ref_table in instance._node_ids:
                    instance.add_edge(table_name, ref_table, "foreign_key")

        return instance

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
