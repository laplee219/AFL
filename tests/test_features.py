"""Tests for AFL Predictor feature engineering."""

import pytest
import pandas as pd
import numpy as np
from src.preprocessing.features import EloSystem


class TestEloSystem:
    def setup_method(self):
        self.elo = EloSystem(k_factor=40, home_advantage=35)

    def test_initial_rating(self):
        """All teams start at 1500."""
        rating = self.elo.get_rating("Collingwood")
        assert rating == 1500

    def test_update_winner_gains(self):
        """Winner should gain Elo points."""
        before = self.elo.get_rating("Collingwood")
        self.elo.update("Collingwood", "Essendon", margin=30)
        after = self.elo.get_rating("Collingwood")
        assert after > before

    def test_update_loser_loses(self):
        """Loser should lose Elo points."""
        before = self.elo.get_rating("Essendon")
        self.elo.update("Collingwood", "Essendon", margin=30)
        after = self.elo.get_rating("Essendon")
        assert after < before

    def test_update_zero_sum(self):
        """Elo changes should be zero-sum."""
        before_a = self.elo.get_rating("Collingwood")
        before_b = self.elo.get_rating("Essendon")
        self.elo.update("Collingwood", "Essendon", margin=30)
        after_a = self.elo.get_rating("Collingwood")
        after_b = self.elo.get_rating("Essendon")

        change_a = after_a - before_a
        change_b = after_b - before_b
        assert abs(change_a + change_b) < 0.01

    def test_big_upset_large_change(self):
        """A big upset should produce a larger Elo change than expected."""
        # First make one team much stronger
        for _ in range(10):
            self.elo.update("Collingwood", "Essendon", margin=40)

        elo_before = self.elo.get_rating("Essendon")
        # Now Essendon (weaker) beats Collingwood (stronger) — upset
        self.elo.update("Essendon", "Collingwood", margin=50)
        change = self.elo.get_rating("Essendon") - elo_before
        assert change > 10  # Should be a meaningful gain for the upset

    def test_predict_returns_probability(self):
        """Predict should return a dict with home_win_prob between 0 and 1."""
        result = self.elo.predict("Collingwood", "Essendon")
        assert isinstance(result, dict)
        assert 0 <= result["home_win_prob"] <= 1

    def test_predict_equal_teams(self):
        """Equal teams should have ~0.5 win probability (with home advantage)."""
        result = self.elo.predict("Collingwood", "Essendon")
        prob = result["home_win_prob"]
        # With home advantage, should be slightly > 0.5
        assert 0.45 < prob < 0.7

    def test_regress_to_mean(self):
        """Season regression should move ratings toward 1500."""
        for _ in range(20):
            self.elo.update("Collingwood", "Essendon", margin=40)

        rating_before = self.elo.get_rating("Collingwood")
        assert rating_before > 1500

        self.elo.regress_to_mean()
        rating_after = self.elo.get_rating("Collingwood")

        assert 1500 < rating_after < rating_before

    def test_get_all_ratings(self):
        """Should return all teams that have been involved in a match."""
        self.elo.update("Sydney", "GWS", margin=10)
        ratings = self.elo.get_all_ratings()
        # Returns a DataFrame with 'team' column
        assert "Sydney" in ratings["team"].values
        assert "GWS" in ratings["team"].values
