"""
Spider benchmark evaluation for graph2sql.

Measures table recall: given a natural language question, does graph2sql
retrieve all the tables referenced in the gold SQL query?

Usage
-----
1. Download Spider dataset:
   https://yale-nlp.github.io/spider/
   Extract to a local directory, e.g. ./data/spider/

2. Run:
   python benchmarks/spider_eval.py --spider-dir ./data/spider --k 3

What this measures
------------------
Table Recall@k: proportion of gold tables that appear in the top-k
retrieved nodes (or their 1-hop neighbours).

This is different from exact SQL match — it only measures whether
graph2sql surfaces the right schema context, not whether the downstream
LLM generates correct SQL.

Metric definition
-----------------
  recall@k = |gold_tables ∩ retrieved_nodes| / |gold_tables|
  mean_recall@k = average recall across all questions

A score of 1.0 means every gold table was present in the retrieved
subgraph. Baseline (random k tables) ≈ k / total_tables.
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

# Allow running from repo root: python benchmarks/spider_eval.py
sys.path.insert(0, str(Path(__file__).parent.parent))

from graph2sql import SchemaGraph


# ---------------------------------------------------------------------------
# Spider schema loader
# ---------------------------------------------------------------------------

def load_spider_schemas(tables_path: Path) -> Dict[str, SchemaGraph]:
    """
    Parse Spider tables.json and return a dict of db_id → SchemaGraph.

    Spider table format:
      {
        "db_id": "concert_singer",
        "table_names_original": ["stadium", "singer", "concert", "singer_in_concert"],
        "column_names_original": [[-1, "*"], [0, "Stadium_ID"], [0, "Location"], ...],
        "foreign_keys": [[col_idx_a, col_idx_b], ...],
        "primary_keys": [col_idx, ...]
      }
    """
    with open(tables_path) as f:
        tables_data = json.load(f)

    schemas: Dict[str, SchemaGraph] = {}

    for db in tables_data:
        db_id: str = db["db_id"]
        table_names: List[str] = db["table_names_original"]
        column_names: List[Tuple[int, str]] = db["column_names_original"]  # [(table_idx, col_name), ...]
        foreign_keys: List[Tuple[int, int]] = db.get("foreign_keys", [])
        primary_keys: List[int] = db.get("primary_keys", [])

        # Build primary key lookup: table_idx → [col_name, ...]
        pk_cols: Dict[int, List[str]] = {}
        for pk_col_idx in primary_keys:
            t_idx, col_name = column_names[pk_col_idx]
            pk_cols.setdefault(t_idx, []).append(col_name)

        # Build column list per table: table_idx → [col_name, ...]
        table_cols: Dict[int, List[str]] = {i: [] for i in range(len(table_names))}
        for col_idx, (t_idx, col_name) in enumerate(column_names):
            if t_idx == -1:  # skip the wildcard column
                continue
            table_cols[t_idx].append(col_name)

        g = SchemaGraph()

        # Add a node per table
        for t_idx, tname in enumerate(table_names):
            cols = table_cols.get(t_idx, [])
            pks = pk_cols.get(t_idx, [])
            content = ", ".join(cols)
            attrs = {"primary_key": ", ".join(pks)} if pks else {}
            g.add_node(id=f"{db_id}__{tname}", label=tname, content=content, attributes=attrs)

        # Add edges for foreign keys: column_idx_a → column_idx_b
        seen_edges: Set[Tuple[str, str]] = set()
        for col_idx_a, col_idx_b in foreign_keys:
            t_a = column_names[col_idx_a][0]
            t_b = column_names[col_idx_b][0]
            if t_a == -1 or t_b == -1:
                continue
            from_id = f"{db_id}__{table_names[t_a]}"
            to_id = f"{db_id}__{table_names[t_b]}"
            edge_key = (from_id, to_id)
            if edge_key not in seen_edges:
                g.add_edge(from_id=from_id, to_id=to_id, label="foreign_key")
                seen_edges.add(edge_key)

        schemas[db_id] = g

    return schemas


# ---------------------------------------------------------------------------
# Gold table extraction from SQL
# ---------------------------------------------------------------------------

_TABLE_RE = re.compile(
    r"\b(?:FROM|JOIN)\s+([`\"\[]?[\w]+[`\"\]]?)",
    re.IGNORECASE,
)


def extract_gold_tables(sql: str) -> Set[str]:
    """
    Extract table names referenced in a SQL query.

    Uses a simple regex — handles most Spider queries (SELECT/FROM/JOIN).
    Does not handle subqueries perfectly, but coverage is high enough for eval.
    """
    return {m.group(1).strip('`"[]').lower() for m in _TABLE_RE.finditer(sql)}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    spider_dir: Path,
    split: str = "dev",
    k: int = 3,
    alpha: float = 0.85,
    limit: int = 0,
) -> Dict:
    """
    Run graph2sql table recall evaluation on Spider dev or train split.

    Parameters
    ----------
    spider_dir : Path
        Root of extracted Spider dataset.
    split : str
        "dev" or "train".
    k : int
        Number of top-k nodes for graph2sql.rank().
    alpha : float
        PPR damping factor.
    limit : int
        If > 0, evaluate only the first N questions (useful for quick tests).

    Returns
    -------
    dict
        {
          "split": "dev",
          "k": 3,
          "total_questions": 1034,
          "scored_questions": 1030,   # questions where gold tables were found
          "mean_recall": 0.82,
          "perfect_recall": 0.71,     # fraction with recall == 1.0
          "zero_recall": 0.05,        # fraction with recall == 0.0
        }
    """
    tables_path = spider_dir / "tables.json"
    questions_path = spider_dir / f"{split}.json"

    if not tables_path.exists():
        raise FileNotFoundError(f"tables.json not found at {tables_path}")
    if not questions_path.exists():
        raise FileNotFoundError(f"{split}.json not found at {questions_path}")

    print(f"Loading schemas from {tables_path}...")
    schemas = load_spider_schemas(tables_path)
    print(f"Loaded {len(schemas)} databases.")

    with open(questions_path) as f:
        questions = json.load(f)

    if limit > 0:
        questions = questions[:limit]

    print(f"Evaluating {len(questions)} questions (k={k}, split={split})...")

    recalls: List[float] = []
    skipped = 0

    for item in questions:
        db_id: str = item["db_id"]
        question: str = item["question"]
        gold_sql: str = item.get("query", item.get("SQL", ""))

        if db_id not in schemas:
            skipped += 1
            continue

        gold_tables = extract_gold_tables(gold_sql)
        if not gold_tables:
            skipped += 1
            continue

        g = schemas[db_id]
        result = g.rank(question, k=k, alpha=alpha)

        retrieved_labels = {n["label"].lower() for n in result["nodes"]}

        hits = len(gold_tables & retrieved_labels)
        recall = hits / len(gold_tables)
        recalls.append(recall)

    if not recalls:
        print("No questions scored.")
        return {}

    mean_recall = sum(recalls) / len(recalls)
    perfect = sum(1 for r in recalls if r == 1.0) / len(recalls)
    zero = sum(1 for r in recalls if r == 0.0) / len(recalls)

    result_dict = {
        "split": split,
        "k": k,
        "alpha": alpha,
        "total_questions": len(questions),
        "scored_questions": len(recalls),
        "skipped": skipped,
        "mean_recall": round(mean_recall, 4),
        "perfect_recall_fraction": round(perfect, 4),
        "zero_recall_fraction": round(zero, 4),
    }
    return result_dict


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="graph2sql Spider benchmark")
    parser.add_argument(
        "--spider-dir",
        type=Path,
        required=True,
        help="Path to extracted Spider dataset directory (contains tables.json, dev.json)",
    )
    parser.add_argument("--split", default="dev", choices=["dev", "train"])
    parser.add_argument("--k", type=int, default=3, help="Top-k nodes (default: 3)")
    parser.add_argument("--alpha", type=float, default=0.85, help="PPR damping factor (default: 0.85)")
    parser.add_argument("--limit", type=int, default=0, help="Evaluate only first N questions (0 = all)")
    args = parser.parse_args()

    results = evaluate(
        spider_dir=args.spider_dir,
        split=args.split,
        k=args.k,
        alpha=args.alpha,
        limit=args.limit,
    )

    print("\n=== graph2sql Spider Evaluation Results ===")
    for key, val in results.items():
        print(f"  {key:30s}: {val}")


if __name__ == "__main__":
    main()
