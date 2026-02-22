"""
Feature Engineering Module

Builds predictive features from cleaned match data:
- Elo ratings (margin-adjusted, with home advantage)
- Rolling form stats (EWMA over multiple windows)
- Head-to-head records
- Venue-specific performance
- Travel/rest factors
- Scoring efficiency metrics

All features are computed as differentials (home minus away) for model input.
"""

import numpy as np
import pandas as pd

from config.settings import settings
from src.utils.constants import (
    get_team_state,
    get_travel_distance,
    get_venue_state,
    is_home_ground,
)
from src.utils.helpers import get_logger

logger = get_logger(__name__)


# ═════════════════════════════════════════════════════════════════════
# ELO RATING SYSTEM
# ═════════════════════════════════════════════════════════════════════

class EloSystem:
    """
    Margin-adjusted Elo rating system for AFL teams.

    Updates after each match using:
        R_new = R_old + K * (S - E)
    where S is a margin-adjusted actual score (sigmoid of margin).
    """

    def __init__(
        self,
        k_factor: float = None,
        home_advantage: float = None,
        initial_rating: float = None,
        season_regression: float = None,
        margin_sigma: float = 50.0,
    ):
        self.k_factor = k_factor or settings.model.elo_k_factor
        self.home_advantage = home_advantage or settings.model.elo_home_advantage
        self.initial_rating = initial_rating or settings.model.elo_initial_rating
        self.season_regression = season_regression or settings.model.elo_season_regression
        self.margin_sigma = margin_sigma
        self.ratings: dict[str, float] = {}

    def get_rating(self, team: str) -> float:
        """Get current rating for a team, initializing if needed."""
        if team not in self.ratings:
            self.ratings[team] = self.initial_rating
        return self.ratings[team]

    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """Calculate expected score for team A vs team B."""
        return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))

    def margin_to_score(self, margin: float) -> float:
        """Convert margin to a score between 0 and 1 using sigmoid."""
        return 1.0 / (1.0 + np.exp(-margin / self.margin_sigma))

    def update(self, home_team: str, away_team: str, margin: int) -> tuple[float, float]:
        """
        Update ratings after a match.

        Args:
            home_team: Home team name
            away_team: Away team name
            margin: Home score minus away score

        Returns:
            Tuple of (new_home_rating, new_away_rating)
        """
        home_rating = self.get_rating(home_team) + self.home_advantage
        away_rating = self.get_rating(away_team)

        expected_home = self.expected_score(home_rating, away_rating)
        actual_home = self.margin_to_score(margin)

        delta = self.k_factor * (actual_home - expected_home)

        self.ratings[home_team] = self.get_rating(home_team) + delta
        self.ratings[away_team] = self.get_rating(away_team) - delta

        return self.ratings[home_team], self.ratings[away_team]

    def predict(self, home_team: str, away_team: str) -> dict:
        """
        Predict match outcome using Elo ratings.

        Returns:
            Dict with home_win_prob, expected_margin, rating_diff
        """
        home_rating = self.get_rating(home_team) + self.home_advantage
        away_rating = self.get_rating(away_team)
        rating_diff = home_rating - away_rating

        home_prob = self.expected_score(home_rating, away_rating)
        expected_margin = rating_diff * 0.05  # Rough conversion: 20 Elo points ≈ 1 point margin

        return {
            "home_win_prob": home_prob,
            "expected_margin": expected_margin,
            "rating_diff": rating_diff,
            "home_rating": self.get_rating(home_team),
            "away_rating": self.get_rating(away_team),
        }

    def regress_to_mean(self):
        """Regress all ratings toward the mean at the start of a new season."""
        if not self.ratings:
            return
        mean_rating = np.mean(list(self.ratings.values()))
        for team in self.ratings:
            self.ratings[team] = (
                self.ratings[team] * (1 - self.season_regression)
                + mean_rating * self.season_regression
            )
        logger.info(f"Elo ratings regressed to mean ({self.season_regression:.0%})")

    def get_all_ratings(self) -> pd.DataFrame:
        """Get all current ratings as a DataFrame."""
        data = [
            {"team": team, "rating": rating}
            for team, rating in sorted(self.ratings.items(), key=lambda x: -x[1])
        ]
        return pd.DataFrame(data)


# ═════════════════════════════════════════════════════════════════════
# ROLLING FORM FEATURES
# ═════════════════════════════════════════════════════════════════════

def compute_team_rolling_stats(
    matches: pd.DataFrame,
    windows: list[int] = [3, 5, 10],
    ewm_span: int = 5,
) -> pd.DataFrame:
    """
    Compute rolling statistics for each team across the season.

    For each match, calculates the team's stats *before* that match (no look-ahead).

    Args:
        matches: Cleaned match DataFrame sorted by date
        windows: List of rolling window sizes
        ewm_span: Span for exponentially weighted moving avg

    Returns:
        DataFrame with per-match-per-team rolling features
    """
    if matches.empty:
        return pd.DataFrame()

    records = []

    # Build team-level match history
    for team in set(matches["home_team"].unique()) | set(matches["away_team"].unique()):
        # Get all matches for this team, in chronological order
        home_mask = matches["home_team"] == team
        away_mask = matches["away_team"] == team
        team_matches = matches[home_mask | away_mask].sort_values("date").copy()

        if team_matches.empty:
            continue

        # Normalize stats from team's perspective
        team_matches["is_home"] = (team_matches["home_team"] == team).astype(int)
        team_matches["team_score"] = np.where(
            team_matches["is_home"], team_matches["home_score"], team_matches["away_score"]
        )
        team_matches["opp_score"] = np.where(
            team_matches["is_home"], team_matches["away_score"], team_matches["home_score"]
        )
        team_matches["team_margin"] = team_matches["team_score"] - team_matches["opp_score"]
        team_matches["team_win"] = (team_matches["team_margin"] > 0).astype(float)

        # Conversion rate from team's perspective
        team_matches["team_goals"] = np.where(
            team_matches["is_home"], team_matches.get("home_goals", 0), team_matches.get("away_goals", 0)
        )
        team_matches["team_behinds"] = np.where(
            team_matches["is_home"], team_matches.get("home_behinds", 0), team_matches.get("away_behinds", 0)
        )
        team_shots = team_matches["team_goals"] + team_matches["team_behinds"]
        team_matches["team_conversion"] = np.where(
            team_shots > 0, team_matches["team_goals"] / team_shots, 0.0
        )

        # Compute rolling features (shifted by 1 to avoid look-ahead)
        for w in windows:
            prefix = f"rolling_{w}"
            team_matches[f"{prefix}_win_rate"] = (
                team_matches["team_win"].shift(1).rolling(w, min_periods=1).mean()
            )
            team_matches[f"{prefix}_avg_margin"] = (
                team_matches["team_margin"].shift(1).rolling(w, min_periods=1).mean()
            )
            team_matches[f"{prefix}_avg_score"] = (
                team_matches["team_score"].shift(1).rolling(w, min_periods=1).mean()
            )
            team_matches[f"{prefix}_avg_conceded"] = (
                team_matches["opp_score"].shift(1).rolling(w, min_periods=1).mean()
            )
            team_matches[f"{prefix}_conversion"] = (
                team_matches["team_conversion"].shift(1).rolling(w, min_periods=1).mean()
            )

        # EWMA features (more weight on recent games)
        team_matches["ewm_win_rate"] = (
            team_matches["team_win"].shift(1).ewm(span=ewm_span, min_periods=1).mean()
        )
        team_matches["ewm_margin"] = (
            team_matches["team_margin"].shift(1).ewm(span=ewm_span, min_periods=1).mean()
        )
        team_matches["ewm_score"] = (
            team_matches["team_score"].shift(1).ewm(span=ewm_span, min_periods=1).mean()
        )

        # Season-to-date stats
        team_matches["season_win_rate"] = (
            team_matches.groupby("year")["team_win"].shift(1)
            .expanding(min_periods=1).mean()
            .reset_index(level=0, drop=True)
        )
        team_matches["season_avg_margin"] = (
            team_matches.groupby("year")["team_margin"].shift(1)
            .expanding(min_periods=1).mean()
            .reset_index(level=0, drop=True)
        )

        # Games played this season (for weighting early-season uncertainty)
        team_matches["season_games_played"] = (
            team_matches.groupby("year").cumcount()
        )

        # Store with team identifier
        team_matches["team"] = team
        records.append(team_matches)

    if not records:
        return pd.DataFrame()

    result = pd.concat(records, ignore_index=True)
    logger.info(f"Computed rolling stats: {len(result)} team-match records")
    return result


# ═════════════════════════════════════════════════════════════════════
# HEAD-TO-HEAD FEATURES
# ═════════════════════════════════════════════════════════════════════

def compute_h2h_features(
    matches: pd.DataFrame, lookback_years: int = 5
) -> pd.DataFrame:
    """
    Compute head-to-head features for each matchup.

    For each match, calculates the H2H record between the two teams
    using only past data within the lookback window.
    """
    if matches.empty:
        return pd.DataFrame()

    records = []
    matches_sorted = matches.sort_values("date").reset_index(drop=True)

    for idx, row in matches_sorted.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        match_date = row["date"]
        cutoff_date = match_date - pd.DateOffset(years=lookback_years)

        # Past matches between these two teams
        past = matches_sorted[
            (matches_sorted["date"] < match_date)
            & (matches_sorted["date"] >= cutoff_date)
            & (
                ((matches_sorted["home_team"] == home) & (matches_sorted["away_team"] == away))
                | ((matches_sorted["home_team"] == away) & (matches_sorted["away_team"] == home))
            )
        ]

        if past.empty:
            records.append({
                "match_id": row.get("match_id", idx),
                "h2h_home_wins": 0,
                "h2h_away_wins": 0,
                "h2h_home_win_rate": 0.5,
                "h2h_avg_margin": 0.0,
                "h2h_n_matches": 0,
            })
            continue

        # Count wins from home team's perspective
        home_wins = 0
        total_margin = 0.0
        for _, h2h_row in past.iterrows():
            if h2h_row["home_team"] == home:
                if h2h_row["margin"] > 0:
                    home_wins += 1
                total_margin += h2h_row["margin"]
            else:
                if h2h_row["margin"] < 0:
                    home_wins += 1
                total_margin -= h2h_row["margin"]

        n = len(past)
        records.append({
            "match_id": row.get("match_id", idx),
            "h2h_home_wins": home_wins,
            "h2h_away_wins": n - home_wins,
            "h2h_home_win_rate": home_wins / n if n > 0 else 0.5,
            "h2h_avg_margin": total_margin / n if n > 0 else 0.0,
            "h2h_n_matches": n,
        })

    return pd.DataFrame(records)


# ═════════════════════════════════════════════════════════════════════
# VENUE & TRAVEL FEATURES
# ═════════════════════════════════════════════════════════════════════

def compute_venue_features(matches: pd.DataFrame) -> pd.DataFrame:
    """
    Compute venue-related and travel features for each match.
    """
    features = pd.DataFrame(index=matches.index)

    features["match_id"] = matches.get("match_id", matches.index)

    # Home ground advantage
    features["home_at_home_ground"] = matches.apply(
        lambda r: int(is_home_ground(r["home_team"], r.get("venue", ""))), axis=1
    )
    features["away_at_home_ground"] = matches.apply(
        lambda r: int(is_home_ground(r["away_team"], r.get("venue", ""))), axis=1
    )

    # Travel distance
    if "venue" in matches.columns:
        venue_states = matches["venue"].apply(get_venue_state)
        home_states = matches["home_team"].apply(get_team_state)
        away_states = matches["away_team"].apply(get_team_state)

        features["home_travel_km"] = [
            get_travel_distance(hs, vs)
            for hs, vs in zip(home_states, venue_states)
        ]
        features["away_travel_km"] = [
            get_travel_distance(as_, vs)
            for as_, vs in zip(away_states, venue_states)
        ]
        features["travel_diff_km"] = features["home_travel_km"] - features["away_travel_km"]

        # Interstate game flags
        features["home_interstate"] = (home_states != venue_states).astype(int)
        features["away_interstate"] = (away_states != venue_states).astype(int)
    else:
        features["home_travel_km"] = 0
        features["away_travel_km"] = 0
        features["travel_diff_km"] = 0
        features["home_interstate"] = 0
        features["away_interstate"] = 0

    return features


def compute_rest_features(matches: pd.DataFrame) -> pd.DataFrame:
    """
    Compute rest days between matches for each team.
    """
    if matches.empty or "date" not in matches.columns:
        return pd.DataFrame()

    matches_sorted = matches.sort_values("date").copy()
    team_last_game: dict[str, pd.Timestamp] = {}

    home_rest = []
    away_rest = []

    for _, row in matches_sorted.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        date = row["date"]

        # Home team rest days
        if home in team_last_game and pd.notna(date):
            home_rest.append((date - team_last_game[home]).days)
        else:
            home_rest.append(7)  # Default 7 days for first game

        # Away team rest days
        if away in team_last_game and pd.notna(date):
            away_rest.append((date - team_last_game[away]).days)
        else:
            away_rest.append(7)

        # Update last game dates
        if pd.notna(date):
            team_last_game[home] = date
            team_last_game[away] = date

    result = pd.DataFrame({
        "match_id": matches_sorted.get("match_id", matches_sorted.index),
        "home_rest_days": home_rest,
        "away_rest_days": away_rest,
        "rest_diff": [h - a for h, a in zip(home_rest, away_rest)],
        "home_short_rest": [int(d <= 6) for d in home_rest],
        "away_short_rest": [int(d <= 6) for d in away_rest],
    })

    return result


# ═════════════════════════════════════════════════════════════════════
# MASTER FEATURE BUILDER
# ═════════════════════════════════════════════════════════════════════

def build_feature_matrix(
    matches: pd.DataFrame,
    elo_system: EloSystem = None,
) -> pd.DataFrame:
    """
    Build the complete feature matrix for model training/prediction.

    Combines all feature groups into a single DataFrame with one row per match.
    All team-level features are expressed as differentials (home - away).

    Args:
        matches: Cleaned match DataFrame sorted by date
        elo_system: Existing EloSystem (or creates a new one)

    Returns:
        Feature matrix DataFrame ready for ML model input
    """
    if matches.empty:
        return pd.DataFrame()

    matches = matches.sort_values("date").reset_index(drop=True)
    elo = elo_system or EloSystem()

    # ── 1. Compute Elo features ──────────────────────────────────────
    elo_features = []
    current_year = None

    for _, row in matches.iterrows():
        # Season regression at year boundary
        if current_year is not None and row["year"] != current_year:
            elo.regress_to_mean()
        current_year = row["year"]

        # Pre-match prediction
        pred = elo.predict(row["home_team"], row["away_team"])
        elo_features.append({
            "match_id": row.get("match_id"),
            "elo_home_rating": pred["home_rating"],
            "elo_away_rating": pred["away_rating"],
            "elo_diff": pred["rating_diff"],
            "elo_home_win_prob": pred["home_win_prob"],
            "elo_expected_margin": pred["expected_margin"],
        })

        # Update Elo with actual result (only for completed matches)
        if row.get("is_complete", False) and pd.notna(row.get("margin")):
            elo.update(row["home_team"], row["away_team"], int(row["margin"]))

    elo_df = pd.DataFrame(elo_features)

    # ── 2. Compute rolling stats ─────────────────────────────────────
    completed = matches[matches.get("is_complete", True) == True].copy()
    rolling = compute_team_rolling_stats(completed)

    # Pivot rolling stats to per-match features (home vs away)
    rolling_cols = [c for c in rolling.columns if c.startswith(("rolling_", "ewm_", "season_"))]

    # Build look-up: for each match, get the team's latest rolling stats
    feature_rows = []
    for _, row in matches.iterrows():
        match_id = row.get("match_id")
        home = row["home_team"]
        away = row["away_team"]
        date = row.get("date")

        feat = {"match_id": match_id}

        # Get rolling stats for each team up to this match
        for team, prefix in [(home, "home"), (away, "away")]:
            team_rolling = rolling[
                (rolling["team"] == team) & (rolling["date"] <= date)
            ]
            if not team_rolling.empty:
                latest = team_rolling.iloc[-1]
                for col in rolling_cols:
                    if col in latest.index:
                        feat[f"{prefix}_{col}"] = latest[col]
            else:
                for col in rolling_cols:
                    feat[f"{prefix}_{col}"] = np.nan

        feature_rows.append(feat)

    rolling_features = pd.DataFrame(feature_rows)

    # ── 3. Compute H2H features ──────────────────────────────────────
    h2h_features = compute_h2h_features(matches)

    # ── 4. Compute venue & travel features ───────────────────────────
    venue_features = compute_venue_features(matches)

    # ── 5. Compute rest features ─────────────────────────────────────
    rest_features = compute_rest_features(matches)

    # ── 6. Merge all features ────────────────────────────────────────
    feature_matrix = matches[["match_id", "year", "round", "date",
                              "home_team", "away_team", "venue"]].copy()

    # Add target variables (only available for completed matches)
    if "margin" in matches.columns:
        feature_matrix["target_margin"] = matches["margin"]
        feature_matrix["target_home_win"] = matches.get("home_win", np.nan)

    # Merge feature groups
    for feat_df in [elo_df, rolling_features, h2h_features, venue_features, rest_features]:
        if not feat_df.empty and "match_id" in feat_df.columns:
            feature_matrix = feature_matrix.merge(
                feat_df, on="match_id", how="left", suffixes=("", "_dup")
            )
            # Drop duplicate columns
            dup_cols = [c for c in feature_matrix.columns if c.endswith("_dup")]
            feature_matrix = feature_matrix.drop(columns=dup_cols)

    # ── 7. Compute differential features ─────────────────────────────
    # For each pair of home/away stats, compute the difference
    home_cols = [c for c in feature_matrix.columns if c.startswith("home_") and not c.startswith("home_team")]
    for hcol in home_cols:
        acol = hcol.replace("home_", "away_", 1)
        if acol in feature_matrix.columns:
            diff_name = hcol.replace("home_", "diff_", 1)
            feature_matrix[diff_name] = feature_matrix[hcol] - feature_matrix[acol]

    # ── 8. Ladder position features ──────────────────────────────────
    feature_matrix["round_progress"] = feature_matrix["round"] / 24.0
    feature_matrix["is_final"] = matches.get("is_final", 0)

    n_features = len([c for c in feature_matrix.columns 
                      if c not in ["match_id", "year", "round", "date", 
                                   "home_team", "away_team", "venue",
                                   "target_margin", "target_home_win"]])
    logger.info(f"Built feature matrix: {len(feature_matrix)} matches × {n_features} features")

    return feature_matrix, elo
