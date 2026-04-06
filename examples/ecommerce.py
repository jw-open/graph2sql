"""
E-commerce schema example.

Demonstrates building a schema graph from an e-commerce database and
ranking nodes relevant to a natural language question.

Run:
    python examples/ecommerce.py
"""

import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from graph2sql import SchemaGraph

# Build the schema graph
graph = SchemaGraph()

# Tables as nodes (label = table name, content = column definitions)
graph.add_node("users",        "users",       content="id INT PK, name VARCHAR, email VARCHAR, country VARCHAR, created_at TIMESTAMP")
graph.add_node("orders",       "orders",      content="id INT PK, customer_id INT FK, total DECIMAL, status VARCHAR, created_at TIMESTAMP")
graph.add_node("products",     "products",    content="id INT PK, name VARCHAR, price DECIMAL, category VARCHAR, stock INT")
graph.add_node("order_items",  "order_items", content="id INT PK, order_id INT FK, product_id INT FK, quantity INT, unit_price DECIMAL")
graph.add_node("categories",   "categories",  content="id INT PK, name VARCHAR, parent_id INT")

# Relationships as edges
graph.add_edge("orders",      "users",       "customer_belongs_to")
graph.add_edge("order_items", "orders",      "item_belongs_to_order")
graph.add_edge("order_items", "products",    "item_references_product")
graph.add_edge("products",    "categories",  "product_in_category")

print(f"Graph: {graph}\n")

# Queries use the actual node label words for token matching.
# The PPR algorithm matches query tokens against node labels exactly —
# it does not perform stemming or semantic expansion.
# Downstream LLMs handle natural language variation after the subgraph is extracted.
queries = [
    "total revenue from orders by users",
    "most popular products by order_items quantity",
    "users without orders",
]

for query in queries:
    print(f"Query: {query!r}")
    context = graph.rank(query, k=3)
    print(f"Relevant nodes ({len(context['nodes'])}):")
    for node in context["nodes"]:
        score = f"  [score={node['score']:.4f}]" if "score" in node else ""
        print(f"  - {node['label']}{score}")
        print(f"    {node['content']}")
    print(f"Relevant edges ({len(context['edges'])}):")
    for edge in context["edges"]:
        print(f"  {edge['from']} --[{edge['label']}]--> {edge['to']}")
    print()

# Show full JSON context for the first query (what you'd pass to an LLM)
print("--- LLM context (JSON) for first query ---")
print(json.dumps(graph.rank(queries[0], k=3), indent=2))
