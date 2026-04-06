# graph2sql

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
| `label` | str | yes | Table or field name — used for query token matching |
| `content` | str | no | Column definitions, constraints, notes |
| `attributes` | dict | no | Any extra metadata |

### Edge

| Field | Type | Description |
|---|---|---|
| `"from"` | str | Source node id |
| `"to"` | str | Target node id |
| `"label"` | str | Relationship type (e.g. `"foreign_key"`, `"belongs_to"`) |

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

## License

MIT
