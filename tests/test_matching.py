"""Tests for graph2sql.matching — soft token matching."""

import pytest
from graph2sql.matching import stem, soft_match_score, stemmed_tokens


class TestStem:
    def test_plural_s(self):
        assert stem("customers") == "customer"

    def test_plural_es(self):
        assert stem("addresses") == "address"

    def test_plural_ies(self):
        # "ies" → strip "es" → "parti" — short but passes length check
        # main goal: "categories" → "categori" (close enough for matching)
        result = stem("categories")
        assert "categor" in result

    def test_ing(self):
        assert stem("ordering") == "order"

    def test_no_change_short(self):
        # Words shorter than stem threshold should not be changed
        assert stem("id") == "id"

    def test_no_suffix(self):
        assert stem("product") == "product"


class TestSoftMatchScore:
    def test_exact_label_match(self):
        score = soft_match_score("show all orders", label="orders")
        assert score > 0

    def test_plural_matches_singular(self):
        # "customers" in query should match "customer" label
        score_match = soft_match_score("show all customers", label="customer")
        score_no = soft_match_score("show all invoices", label="customer")
        assert score_match > score_no

    def test_singular_matches_plural_label(self):
        # "customer" in query should match "customers" label
        score = soft_match_score("total by customer", label="customers")
        assert score > 0

    def test_content_matching(self):
        # Query token appears in content but not label
        score = soft_match_score(
            "find revenue", label="orders", content="id, revenue, customer_id"
        )
        assert score > 0

    def test_attribute_alias_matching(self):
        score = soft_match_score(
            "find customers",
            label="users",
            attributes={"alias": "customers clients"},
        )
        assert score > 0

    def test_label_weighted_higher_than_content(self):
        # Label-only match (2x weight) should beat content-only match (1x weight)
        # when both have the same number of token hits
        label_score = soft_match_score("orders", label="orders", content="")
        content_only_score = soft_match_score("orders", label="invoices", content="orders total")
        # label: 1 hit * 2 = 2; content: 1 hit * 1 = 1
        assert label_score > content_only_score

    def test_zero_score_no_overlap(self):
        score = soft_match_score("weather forecast", label="orders", content="id, total")
        assert score == 0.0

    def test_empty_query(self):
        score = soft_match_score("", label="orders")
        assert score == 0.0
