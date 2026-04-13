"""
Example: build a SchemaGraph from a live database using SQLAlchemy.

Works with any SQLAlchemy-supported database:
  PostgreSQL:  postgresql://user:pw@localhost/mydb
  MySQL:       mysql+pymysql://user:pw@localhost/mydb
  SQLite:      sqlite:///path/to/db.sqlite
  MongoDB:     (use mongoengine or a SQL view layer)

Install:
  pip install graph2sql[sqlalchemy]
  pip install psycopg2-binary   # for PostgreSQL
  pip install pymysql           # for MySQL
"""

from sqlalchemy import create_engine
from graph2sql import SchemaGraph

# --- connect to your database -------------------------------------------
engine = create_engine("sqlite:///ecommerce.sqlite")

# --- auto-build the schema graph ----------------------------------------
g = SchemaGraph.from_sqlalchemy(engine)
print(g)  # SchemaGraph(nodes=N, edges=M)

# --- rank for a natural language query ----------------------------------
context = g.rank("total revenue per customer last month", k=3)

for node in context["nodes"]:
    print(f"[{node['score']:.4f}] {node['label']}: {node.get('content', '')}")
