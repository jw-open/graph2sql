# graph2sql

[![CI](https://github.com/jw-open/graph2sql/actions/workflows/ci.yml/badge.svg)](https://github.com/jw-open/graph2sql/actions/workflows/ci.yml)

Graph-based schema analysis for text-to-SQL.

Build a schema graph (tables and fields as nodes, relationships as edges), rank the most relevant nodes for a natural language question using **Personalized PageRank**, and extract structured context ready to pass to any LLM for SQL generation.

**No LLM included. Bring your own model.**

---

## How it works

```
natural language question
        +
  schema graph
        │
        ▼
Personalized PageRank
        │
        ▼
ranked subgraph (top-k nodes + 1-hop neighbours)
        │
        ▼
  structured context  ──►  your LLM  ──►  SQL
```

Instead of dumping the entire schema into an LLM prompt, graph2sql identifies which tables and relationships are most relevant to the question — reducing noise and improving SQL accuracy.

---

## Install

```bash
pip install graph2sql
```

Or from source:

```bash
git clone https://github.com/jw-open/graph2sql
cd graph2sql
pip install -e ".[dev]"
```

---

## Quick start

```python
from graph2sql import SchemaGraph

# Build the schema graph
graph = SchemaGraph()

graph.add_node("users",       "users",       content="id INT, name VARCHAR, email VARCHAR, country VARCHAR")
graph.add_node("orders",      "orders",      content="id INT, customer_id INT, total DECIMAL, created_at TIMESTAMP")
graph.add_node("products",    "products",    content="id INT, name VARCHAR, price DECIMAL, category VARCHAR")
graph.add_node("order_items", "order_items", content="id INT, order_id INT, product_id INT, quantity INT")

graph.add_edge("orders",      "users",    "belongs_to")
graph.add_edge("order_items", "orders",   "belongs_to")
graph.add_edge("order_items", "products", "references")

# Rank nodes for a natural language question
context = graph.rank("total revenue by customer last month", k=3)

# Pass context to your LLM
print(context)
# {
#   "nodes": [
#     {"label": "orders", "content": "...", "score": 0.312, ...},
#     {"label": "users",  "content": "...", "score": 0.198, ...},
#     ...
#   ],
#   "edges": [
#     {"from": "orders", "to": "users", "label": "belongs_to"},
#     ...
#   ]
# }
```

### Load from an existing dict

```python
graph = SchemaGraph.from_dict({
    "nodes": [
        {"id": "users",  "label": "users",  "content": "id, name, email"},
        {"id": "orders", "label": "orders", "content": "id, customer_id, total"},
    ],
    "edges": [
        {"from": "orders", "to": "users", "label": "belongs_to"}
    ]
})

context = graph.rank("how many orders per user")
```

---

## Graph schema

### Node

| Field | Type | Required | Description |
|---|---|---|---|
| `id` | str | yes | Unique identifier |
| `label` | str | yes | Table or column name — used for query token matching |
| `content` | str | no | Column definitions, constraints, DDL, or notes |
| `attributes` | dict | no | Typed metadata — see conventions below |

#### Node attribute conventions

Use `attributes` to describe what a node represents. The algorithm also checks attribute string values for query token matching (useful for aliases).

```python
# Table node
graph.add_node("orders", "orders",
    content="id INT PK, customer_id INT FK, total DECIMAL, created_at TIMESTAMP",
    attributes={
        "type": "table",
        "database": "mysql",          # mysql | postgres | sqlite | mongodb | etc.
        "schema": "public",           # schema/namespace if applicable
        "alias": "transactions",      # alternative names matched against queries
        "primary_key": "id",
    }
)

# Column node
graph.add_node("orders.total", "total",
    content="DECIMAL(10,2) — order grand total including tax",
    attributes={
        "type": "column",
        "table": "orders",
        "data_type": "DECIMAL",
        "nullable": "false",
        "alias": "revenue amount",    # matched against queries
    }
)

# View or virtual table
graph.add_node("monthly_revenue", "monthly_revenue",
    content="SELECT DATE_TRUNC('month', created_at), SUM(total) FROM orders GROUP BY 1",
    attributes={
        "type": "view",
        "database": "postgres",
    }
)
```

**Recognised `type` values** (convention, not enforced):

| Value | Meaning |
|---|---|
| `"table"` | A physical database table |
| `"column"` | A column within a table |
| `"view"` | A database view or virtual table |
| `"index"` | An index definition |
| `"schema"` | A database schema/namespace grouping |

**Recognised `database` values** (convention, not enforced):
`"mysql"`, `"postgres"`, `"sqlite"`, `"mongodb"`, `"bigquery"`, `"snowflake"`, `"redshift"`, `"mssql"`, `"oracle"`

Any additional attributes are valid — they are stored as-is and passed through to the LLM context.

### Edge

| Field | Type | Description |
|---|---|---|
| `"from"` | str | Source node id |
| `"to"` | str | Target node id |
| `"label"` | str | Relationship type — see conventions below |

#### Edge label conventions

| Label | Meaning |
|---|---|
| `"foreign_key"` | Standard FK relationship between tables |
| `"belongs_to"` | Child table → parent table |
| `"has_many"` | Parent → child (reverse of belongs_to) |
| `"column_of"` | Column node → its parent table |
| `"references"` | Looser reference between any two nodes |
| `"related_to"` | Semantic relationship (no strict FK) |

---

## API reference

### `SchemaGraph`

```python
SchemaGraph()
SchemaGraph.from_dict(graph: dict) -> SchemaGraph
graph.add_node(id, label, content=None, attributes=None) -> SchemaGraph
graph.add_edge(from_id, to_id, label) -> SchemaGraph
graph.rank(query, k=3, alpha=0.85) -> dict
graph.to_dict() -> dict
```

### `personalized_page_rank`

Low-level function used internally by `SchemaGraph.rank()`.

```python
from graph2sql import personalized_page_rank

result = personalized_page_rank(
    query="revenue by customer",
    graph={"nodes": [...], "edges": [...]},
    alpha=0.85,   # damping factor
    k=3,          # top-k seed nodes
)
```

---

## Run the example

```bash
python examples/ecommerce.py
```

---

## Run tests

```bash
pip install -e ".[dev]"
pytest tests/
```

---

## Design principles

- **No LLM dependency** — pure Python + numpy. Works with any model or no model at all.
- **No database connection required** — pass schema definitions as strings in `content`.
- **Bring your own LLM** — `rank()` returns a plain dict you can serialize and inject into any prompt.
- **Decoupled from infra** — no FastAPI, MongoDB, Redis, or cloud dependencies.

---

## Known limitations

**Token matching is exact.** The PPR algorithm matches query words against node labels using exact token overlap — it does not perform stemming, fuzzy matching, or semantic similarity.

For example, a query containing `"customer"` will not match a node labeled `"users"`.

**Workaround: use `attributes` for aliases.**

Add alternative names as attribute values on the node. The algorithm also checks all attribute values for token matches:

```python
graph.add_node(
    "users",
    "users",
    content="id, name, email",
    attributes={
        "alias": "customers",
        "also_known_as": "clients members",
    }
)

# Now "customers" and "clients" in a query will match this node
context = graph.rank("total revenue by customers")
```

Supported attribute patterns:
- `alias` — primary alternative name
- `also_known_as` — space-separated synonyms
- `related_to` — domain terms associated with this table
- `associated_with` — any custom terms relevant to queries

Any string attribute value is tokenized and matched — the key name is not significant.

---

## License

MIT
