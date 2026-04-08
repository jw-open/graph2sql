# Benchmarks

## Spider — Table Recall

Measures whether graph2sql surfaces the right schema context for a natural language question.

**Metric:** Table Recall@k — fraction of gold tables (those in the correct SQL) that appear in graph2sql's top-k retrieved nodes.

| k | Baseline (random) | graph2sql target |
|---|---|---|
| 3 | ~15–30% | ≥ 70% |
| 5 | ~25–50% | ≥ 80% |

### Setup

```bash
# 1. Download Spider dataset
#    https://yale-nlp.github.io/spider/
#    Extract to data/spider/

# 2. Install dev deps
pip install -e ".[dev]"

# 3. Run eval (dev split, k=3)
python benchmarks/spider_eval.py --spider-dir ./data/spider --k 3

# Quick smoke test (first 50 questions)
python benchmarks/spider_eval.py --spider-dir ./data/spider --k 3 --limit 50
```

### Expected output

```
Loading schemas from data/spider/tables.json...
Loaded 166 databases.
Evaluating 1034 questions (k=3, split=dev)...

=== graph2sql Spider Evaluation Results ===
  split                         : dev
  k                             : 3
  alpha                         : 0.85
  total_questions               : 1034
  scored_questions              : 1030
  skipped                       : 4
  mean_recall                   : 0.XXXX
  perfect_recall_fraction       : 0.XXXX
  zero_recall_fraction          : 0.XXXX
```

### What this does NOT measure

- SQL correctness (that depends on the downstream LLM)
- Join correctness
- Column selection accuracy

Those require a full text-to-SQL pipeline. This eval is purely about schema context retrieval.

---

## BIRD-SQL (planned v0.2.0)

Harder benchmark — messier schemas, more ambiguous questions.

Download: https://bird-bench.github.io/
