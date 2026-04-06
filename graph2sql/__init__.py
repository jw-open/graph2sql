"""
graph2sql — Graph-based schema analysis for text-to-SQL.

Build a schema graph (tables/fields as nodes, relationships as edges),
rank relevant nodes for a natural language query using Personalized PageRank,
and extract structured context ready to pass to any LLM for SQL generation.

No LLM dependency. Bring your own model.

Quick start
-----------
>>> from graph2sql import SchemaGraph
>>> g = SchemaGraph()
>>> g.add_node("1", "orders", content="id, customer_id, total, created_at")
>>> g.add_node("2", "customers", content="id, name, email, country")
>>> g.add_edge("1", "2", "belongs_to")
>>> context = g.rank("total revenue by customer last month")
"""

from .graph import SchemaGraph
from .ranking import personalized_page_rank
from .types import make_edge, make_node

__all__ = ["SchemaGraph", "personalized_page_rank", "make_node", "make_edge"]
__version__ = "0.1.0"
