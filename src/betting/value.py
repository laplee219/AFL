"""
Value Bet Identification Module

Identifies positive expected value bets by comparing model probabilities
with bookmaker implied probabilities.

A bet has value when: model_prob > bookmaker_implied_prob
Expected Value = (model_prob × payout) - 1
"""

import numpy as np
import pandas as pd

from config.settings import settings
from src.utils.helpers import get_logger, implied_probability

logger = get_logger(__name__)


def calculate_expected_value(model_prob: float, decimal_odds: float) -> float:
    """
    Calculate the expected value of a bet.

    EV = (model_prob × payout) - 1
    Positive EV means profitable in the long run.

    Args:
        model_prob: Model's estimated probability of the outcome
        decimal_odds: Bookmaker decimal odds (e.g., 1.80)

    Returns:
        Expected value as a fraction (e.g., 0.08 = +8% EV)
    """
    if decimal_odds <= 0:
        return -1.0
    return (model_prob * decimal_odds) - 1.0


def find_value_bets(
    predictions: pd.DataFrame,
    odds: pd.DataFrame,
    min_ev: float = None,
    min_model_prob: float = 0.30,
) -> pd.DataFrame:
    """
    Find value bets by comparing model predictions with bookmaker odds.

    Args:
        predictions: DataFrame from Predictor.predict_round() with ensemble_prob
        odds: DataFrame with bookmaker odds (home_odds, away_odds)
        min_ev: Minimum EV threshold (default from settings)
        min_model_prob: Min model probability to consider (avoid crazy bets)

    Returns:
        DataFrame of value bets with EV and Kelly sizing
    """
    min_ev = min_ev if min_ev is not None else settings.betting.min_ev_threshold

    if predictions.empty or odds.empty:
        logger.warning("Cannot find value bets: missing predictions or odds data")
        return pd.DataFrame()

    # Merge predictions with odds
    merged = predictions.merge(
        odds,
        on=["home_team", "away_team"],
        how="inner",
        suffixes=("_pred", "_odds"),
    )

    if merged.empty:
        logger.warning("No matches with both predictions and odds")
        return pd.DataFrame()

    value_bets = []

    for _, row in merged.iterrows():
        home_prob = row["ensemble_prob"]
        away_prob = 1.0 - home_prob
        home_odds = row.get("home_odds", row.get("best_home_odds", 0))
        away_odds = row.get("away_odds", row.get("best_away_odds", 0))

        if home_odds <= 0 or away_odds <= 0:
            continue

        home_implied = implied_probability(home_odds)
        away_implied = implied_probability(away_odds)

        # Check home team value bet
        home_ev = calculate_expected_value(home_prob, home_odds)
        if home_ev >= min_ev and home_prob >= min_model_prob:
            from src.betting.kelly import kelly_fraction
            kelly = kelly_fraction(home_prob, home_odds)
            value_bets.append({
                "match_id": row.get("match_id"),
                "year": row.get("year"),
                "round": row.get("round"),
                "home_team": row["home_team"],
                "away_team": row["away_team"],
                "bet_on": row["home_team"],
                "bet_type": "home_win",
                "model_prob": home_prob,
                "bookmaker_prob": home_implied,
                "decimal_odds": home_odds,
                "expected_value": home_ev,
                "edge": home_prob - home_implied,
                "kelly_fraction": kelly,
                "predicted_margin": row.get("ensemble_margin", 0),
                "confidence": row.get("confidence", 0),
                "venue": row.get("venue", ""),
            })

        # Check away team value bet
        away_ev = calculate_expected_value(away_prob, away_odds)
        if away_ev >= min_ev and away_prob >= min_model_prob:
            from src.betting.kelly import kelly_fraction
            kelly = kelly_fraction(away_prob, away_odds)
            value_bets.append({
                "match_id": row.get("match_id"),
                "year": row.get("year"),
                "round": row.get("round"),
                "home_team": row["home_team"],
                "away_team": row["away_team"],
                "bet_on": row["away_team"],
                "bet_type": "away_win",
                "model_prob": away_prob,
                "bookmaker_prob": away_implied,
                "decimal_odds": away_odds,
                "expected_value": away_ev,
                "edge": away_prob - away_implied,
                "kelly_fraction": kelly,
                "predicted_margin": -row.get("ensemble_margin", 0),
                "confidence": row.get("confidence", 0),
                "venue": row.get("venue", ""),
            })

    if not value_bets:
        logger.info("No value bets found")
        return pd.DataFrame()

    result = pd.DataFrame(value_bets)
    result = result.sort_values("expected_value", ascending=False).reset_index(drop=True)

    logger.info(f"Found {len(result)} value bets (min EV: {min_ev:.0%})")
    return result


def format_value_bets(value_bets: pd.DataFrame) -> str:
    """Format value bets as a readable string for CLI output."""
    if value_bets.empty:
        return "No value bets found for this round."

    lines = []
    lines.append("=" * 70)
    lines.append("  VALUE BET RECOMMENDATIONS")
    lines.append("=" * 70)

    for i, row in value_bets.iterrows():
        ev_pct = row["expected_value"] * 100
        edge_pct = row["edge"] * 100
        kelly_pct = row["kelly_fraction"] * 100

        # Rating stars based on EV
        stars = "★" * min(int(ev_pct / 3) + 1, 5) + "☆" * max(0, 5 - min(int(ev_pct / 3) + 1, 5))

        lines.append(f"\n  {stars}  {row['home_team']} vs {row['away_team']}")
        lines.append(f"  Bet:       {row['bet_on']} ({row['bet_type']})")
        lines.append(f"  Odds:      ${row['decimal_odds']:.2f}")
        lines.append(f"  Model:     {row['model_prob']:.1%} vs Bookmaker: {row['bookmaker_prob']:.1%}")
        lines.append(f"  Edge:      +{edge_pct:.1f}%")
        lines.append(f"  EV:        +{ev_pct:.1f}%")
        lines.append(f"  Kelly:     {kelly_pct:.1f}% of bankroll")
        lines.append(f"  Margin:    {row['predicted_margin']:+.0f} points")

    lines.append("\n" + "=" * 70)
    lines.append(f"  Total value bets: {len(value_bets)}")
    lines.append(f"  Avg EV: +{value_bets['expected_value'].mean() * 100:.1f}%")
    lines.append("=" * 70)

    return "\n".join(lines)
