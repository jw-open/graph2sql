"""
Token matching utilities for graph2sql.

Provides soft token matching to handle common natural language variations
(plurals, compound words, underscored names) without requiring external NLP
dependencies.

Strategy (in priority order):
1. Exact token match       — "orders" matches "orders"
2. Plural / suffix strip   — "customers" matches "customers", "customer"
3. Substring match         — "customer" matches "customer_id", "customers"

This is intentionally kept pure Python + stdlib — no sklearn, no spaCy.
"""

import re
from typing import List, Set


_SUFFIXES = ("ing", "tion", "ations", "ation", "ies", "es", "s", "ed")


def stem(token: str) -> str:
    """
    Minimal suffix-stripping stemmer for schema matching.

    Strips common English suffixes so that "customers" and "customer"
    both reduce to "customer", "ordering" → "order", etc.

    Not a proper linguistic stemmer — just enough for table/column names.
    """
    for suffix in _SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            return token[: len(token) - len(suffix)]
    return token


def tokenize(text: str) -> List[str]:
    """Split on non-word characters and underscores; lowercase."""
    return [t for t in re.split(r"[\W_]+", text.lower()) if t]


def stemmed_tokens(text: str) -> Set[str]:
    """Return a set of stemmed tokens from text."""
    return {stem(t) for t in tokenize(text)}


def soft_match_score(query: str, label: str, content: str = "", attributes: dict = None) -> float:
    """
    Compute a soft match score between a query and a schema node.

    Matching is done on stemmed tokens, checking:
    - Node label tokens
    - Node content tokens (column names, DDL)
    - Any string attribute values (aliases, etc.)

    Parameters
    ----------
    query : str
        Natural language question.
    label : str
        Node label (table or column name).
    content : str
        Node content (DDL, column list, description). Optional.
    attributes : dict
        Node attributes dict. Optional.

    Returns
    -------
    float
        Match score >= 0. Higher = more relevant.
        Label matches are weighted 2x content/attribute matches.
    """
    query_stems = stemmed_tokens(query)
    if not query_stems:
        return 0.0

    score = 0.0

    # Label match — weighted 2x (table names are the most signal-dense)
    label_stems = stemmed_tokens(label)
    label_hits = sum(1 for t in label_stems if t in query_stems)
    score += label_hits * 2.0

    # Content match (column names, DDL text) — weighted 1x
    if content:
        content_stems = stemmed_tokens(content)
        content_hits = sum(1 for t in content_stems if t in query_stems)
        score += content_hits * 1.0

    # Attribute string values (alias, type hints, etc.) — weighted 1x
    if attributes:
        for val in attributes.values():
            if isinstance(val, str):
                attr_stems = stemmed_tokens(val)
                attr_hits = sum(1 for t in attr_stems if t in query_stems)
                score += attr_hits * 1.0

    return score
