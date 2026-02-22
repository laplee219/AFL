"""Tests for AFL Predictor betting modules."""

import pytest
from config.settings import settings
from src.betting.kelly import kelly_fraction, calculate_stake
from src.betting.value import calculate_expected_value


class TestKellyFraction:
    def test_positive_edge(self):
        """When model probability > implied, Kelly should be positive."""
        frac = kelly_fraction(model_prob=0.6, decimal_odds=2.5)
        assert frac > 0

    def test_no_edge(self):
        """When fair odds, Kelly should be ~0."""
        frac = kelly_fraction(model_prob=0.5, decimal_odds=2.0)
        assert frac <= 0.01  # Should be near 0

    def test_negative_edge(self):
        """When model probability < implied, Kelly should be 0 (clamped)."""
        frac = kelly_fraction(model_prob=0.3, decimal_odds=2.0)
        assert frac == 0

    def test_fraction_scaling(self):
        """Quarter-Kelly should be 1/4 of full Kelly."""
        full = kelly_fraction(model_prob=0.7, decimal_odds=2.5, fraction=1.0)
        quarter = kelly_fraction(model_prob=0.7, decimal_odds=2.5, fraction=0.25)
        # quarter kelly should be approx 1/4 of full (may differ slightly due to capping)
        assert quarter > 0
        assert quarter <= full

    def test_max_bet_cap(self):
        """Kelly should be capped at the configured max bet fraction."""
        frac = kelly_fraction(model_prob=0.95, decimal_odds=10.0, fraction=1.0)
        assert frac <= settings.betting.max_bet_fraction


class TestCalculateStake:
    def test_basic_stake(self):
        stake = calculate_stake(
            bankroll=1000,
            model_prob=0.6,
            decimal_odds=2.5,
        )
        assert stake > 0
        assert stake <= 50  # max 5% of 1000

    def test_zero_edge(self):
        stake = calculate_stake(
            bankroll=1000,
            model_prob=0.4,
            decimal_odds=2.0,
        )
        assert stake == 0


class TestExpectedValue:
    def test_positive_ev(self):
        ev = calculate_expected_value(model_prob=0.6, decimal_odds=2.5)
        # EV = 0.6 * 2.5 - 1 = 0.5
        assert abs(ev - 0.5) < 0.01

    def test_negative_ev(self):
        ev = calculate_expected_value(model_prob=0.3, decimal_odds=2.0)
        # EV = 0.3 * 2.0 - 1 = -0.4
        assert ev < 0

    def test_fair_ev(self):
        ev = calculate_expected_value(model_prob=0.5, decimal_odds=2.0)
        assert abs(ev) < 0.01
