"""Tests for odds conversion helpers."""

import pytest
from src.utils.helpers import implied_probability, decimal_from_probability, init_database


class TestOddsConversion:
    def test_implied_probability(self):
        """$2.00 odds → 50% implied probability."""
        assert abs(implied_probability(2.0) - 0.5) < 0.001

    def test_implied_probability_favourite(self):
        """$1.50 odds → 66.7% implied probability."""
        assert abs(implied_probability(1.5) - 0.6667) < 0.01

    def test_implied_probability_longshot(self):
        """$5.00 odds → 20% implied probability."""
        assert abs(implied_probability(5.0) - 0.2) < 0.001

    def test_decimal_from_probability(self):
        """50% probability → $2.00 odds."""
        assert abs(decimal_from_probability(0.5) - 2.0) < 0.001

    def test_decimal_from_probability_favourite(self):
        """80% probability → $1.25 odds."""
        assert abs(decimal_from_probability(0.8) - 1.25) < 0.001

    def test_round_trip(self):
        """Converting back and forth should be idempotent."""
        for odds in [1.5, 2.0, 3.0, 5.0, 10.0]:
            prob = implied_probability(odds)
            recovered = decimal_from_probability(prob)
            assert abs(recovered - odds) < 0.001

    def test_edge_cases(self):
        """Handle edge probabilities."""
        assert decimal_from_probability(1.0) == 1.0
        # Very small probability → very large odds
        odds = decimal_from_probability(0.01)
        assert odds == 100.0


class TestDatabase:
    def test_init_database(self):
        """init_database should create tables without error."""
        # This test just ensures the function runs without raising
        init_database()
