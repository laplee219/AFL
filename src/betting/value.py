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
from src.utils.constants import normalize_team_name
from src.utils.helpers import get_logger, implied_probability

logger = get_logger(__name__)


def _normalize_teams(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace and apply canonical names to home_team / away_team."""
    df = df.copy()
    for col in ("home_team", "away_team"):
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().apply(normalize_team_name)
    return df


def _nearest_round_predictions(predictions: pd.DataFrame, odds: pd.DataFrame) -> pd.DataFrame:
    """
    Return only the predictions for the single round with the most odds coverage.

    Scores every upcoming round by counting how many of its (home, away) pairs
    appear in the live odds, then returns the top-scoring round.  Falls back to
    the soonest upcoming round if there is no overlap at all.
    """
    if predictions.empty:
        return predictions

    if not odds.empty and "home_team" in odds.columns:
        odds_pairs = set(
            zip(
                odds["home_team"].astype(str).str.strip().apply(normalize_team_name),
                odds["away_team"].astype(str).str.strip().apply(normalize_team_name),
            )
        )
        preds_norm = _normalize_teams(predictions)

        # Count how many pairs each round contributes to the odds set
        def _score_round(rnd):
            rows = preds_norm[preds_norm["round"] == rnd]
            return sum(
                1 for _, r in rows.iterrows()
                if (r["home_team"], r["away_team"]) in odds_pairs
            )

        rounds = sorted(predictions["round"].unique())
        scores = {rnd: _score_round(rnd) for rnd in rounds}
        best_score = max(scores.values())

        if best_score > 0:
            # Take the highest-scoring round (earliest if tied)
            best_round = next(rnd for rnd in rounds if scores[rnd] == best_score)
            return predictions[predictions["round"] == best_round]

    # Fallback: soonest upcoming round
    min_round = predictions["round"].min()
    return predictions[predictions["round"] == min_round]


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


def calculate_line_prob(
    predicted_margin: float,
    spread_line: float,
    sigma: float = 30.0,
) -> float:
    """
    Estimate the probability of the home team covering a spread line.

    Models actual margin as N(predicted_margin, sigma²) and computes
    P(actual_margin > cover_threshold).

    Args:
        predicted_margin: Model's predicted home margin (positive = home wins)
        spread_line: Bookmaker spread from home team's perspective (e.g., -12.5
                     means home team must win by >12.5 to cover)
        sigma: Std deviation of margin prediction error (~30 pts for AFL)

    Returns:
        Probability of home team covering the spread
    """
    try:
        from scipy.stats import norm
        # Home covers if actual_margin > -spread_line (e.g., spread=-12.5 → threshold=12.5)
        cover_threshold = -spread_line
        return float(1.0 - norm.cdf(cover_threshold, loc=predicted_margin, scale=sigma))
    except ImportError:
        # Fallback: numpy erf approximation
        import math
        cover_threshold = -spread_line
        z = (predicted_margin - cover_threshold) / (sigma * math.sqrt(2))
        return float(0.5 * (1.0 + math.erf(z)))


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

    # Restrict to the round(s) with odds coverage, then normalize names
    predictions = _nearest_round_predictions(predictions, odds)
    predictions = _normalize_teams(predictions)
    odds = _normalize_teams(odds)

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
    from src.betting.kelly import kelly_fraction

    for _, row in merged.iterrows():
        home_prob = row["ensemble_prob"]
        away_prob = 1.0 - home_prob
        home_odds = row.get("home_odds", row.get("best_home_odds", 0))
        away_odds = row.get("away_odds", row.get("best_away_odds", 0))
        predicted_margin = row.get("ensemble_margin", 0)

        # ── Head-to-Head (win/loss) bets ─────────────────────────────
        if home_odds > 0 and away_odds > 0:
            home_implied = implied_probability(home_odds)
            away_implied = implied_probability(away_odds)

            home_ev = calculate_expected_value(home_prob, home_odds)
            if home_ev >= min_ev and home_prob >= min_model_prob:
                value_bets.append({
                    "match_id": row.get("match_id"),
                    "year": row.get("year"),
                    "round": row.get("round"),
                    "home_team": row["home_team"],
                    "away_team": row["away_team"],
                    "bet_on": row["home_team"],
                    "market_type": "h2h",
                    "bet_type": "home_win",
                    "line": None,
                    "model_prob": home_prob,
                    "bookmaker_prob": home_implied,
                    "decimal_odds": home_odds,
                    "expected_value": home_ev,
                    "edge": home_prob - home_implied,
                    "kelly_fraction": kelly_fraction(home_prob, home_odds),
                    "predicted_margin": predicted_margin,
                    "confidence": row.get("confidence", 0),
                    "venue": row.get("venue", ""),
                })

            away_ev = calculate_expected_value(away_prob, away_odds)
            if away_ev >= min_ev and away_prob >= min_model_prob:
                value_bets.append({
                    "match_id": row.get("match_id"),
                    "year": row.get("year"),
                    "round": row.get("round"),
                    "home_team": row["home_team"],
                    "away_team": row["away_team"],
                    "bet_on": row["away_team"],
                    "market_type": "h2h",
                    "bet_type": "away_win",
                    "line": None,
                    "model_prob": away_prob,
                    "bookmaker_prob": away_implied,
                    "decimal_odds": away_odds,
                    "expected_value": away_ev,
                    "edge": away_prob - away_implied,
                    "kelly_fraction": kelly_fraction(away_prob, away_odds),
                    "predicted_margin": -predicted_margin,
                    "confidence": row.get("confidence", 0),
                    "venue": row.get("venue", ""),
                })

        # ── Line / Spread bets ────────────────────────────────────────
        home_spread = row.get("home_spread")
        away_spread = row.get("away_spread")
        home_spread_odds = row.get("best_home_spread_odds", row.get("home_spread_odds", 0))
        away_spread_odds = row.get("best_away_spread_odds", row.get("away_spread_odds", 0))

        if home_spread is not None and pd.notna(home_spread) and home_spread_odds > 0:
            # P(home covers spread), e.g. home_spread=-12.5 → must win by >12.5
            home_line_prob = calculate_line_prob(predicted_margin, float(home_spread))
            home_line_implied = implied_probability(home_spread_odds)
            home_line_ev = calculate_expected_value(home_line_prob, home_spread_odds)
            if home_line_ev >= min_ev and home_line_prob >= min_model_prob:
                value_bets.append({
                    "match_id": row.get("match_id"),
                    "year": row.get("year"),
                    "round": row.get("round"),
                    "home_team": row["home_team"],
                    "away_team": row["away_team"],
                    "bet_on": row["home_team"],
                    "market_type": "spread",
                    "bet_type": f"home_spread",
                    "line": float(home_spread),
                    "model_prob": home_line_prob,
                    "bookmaker_prob": home_line_implied,
                    "decimal_odds": home_spread_odds,
                    "expected_value": home_line_ev,
                    "edge": home_line_prob - home_line_implied,
                    "kelly_fraction": kelly_fraction(home_line_prob, home_spread_odds),
                    "predicted_margin": predicted_margin,
                    "confidence": row.get("confidence", 0),
                    "venue": row.get("venue", ""),
                })

        if away_spread is not None and pd.notna(away_spread) and away_spread_odds > 0:
            # P(away covers spread), e.g. away_spread=+12.5 → home must NOT win by >12.5
            # = complement of home covering when home_spread = -away_spread
            away_line_prob = calculate_line_prob(predicted_margin, -float(away_spread))
            away_line_prob = 1.0 - away_line_prob  # away covers = home does NOT cover
            away_line_implied = implied_probability(away_spread_odds)
            away_line_ev = calculate_expected_value(away_line_prob, away_spread_odds)
            if away_line_ev >= min_ev and away_line_prob >= min_model_prob:
                value_bets.append({
                    "match_id": row.get("match_id"),
                    "year": row.get("year"),
                    "round": row.get("round"),
                    "home_team": row["home_team"],
                    "away_team": row["away_team"],
                    "bet_on": row["away_team"],
                    "market_type": "spread",
                    "bet_type": f"away_spread",
                    "line": float(away_spread),
                    "model_prob": away_line_prob,
                    "bookmaker_prob": away_line_implied,
                    "decimal_odds": away_spread_odds,
                    "expected_value": away_line_ev,
                    "edge": away_line_prob - away_line_implied,
                    "kelly_fraction": kelly_fraction(away_line_prob, away_spread_odds),
                    "predicted_margin": -predicted_margin,
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


def format_odds_comparison(predictions: pd.DataFrame, odds: pd.DataFrame) -> str:
    """
    Format a full odds vs model probability comparison for every match.

    Works in two modes:
    - With predictions: shows model prob, edge and EV for each market
    - Without predictions (odds-only): shows current bookmaker odds and
      implied probabilities — useful when collect/features haven't been run yet
    """
    if odds.empty:
        return "No odds data available (check ODDS_API_KEY in .env)."

    def _fmt(val: float) -> str:
        return f"+{val*100:.1f}%" if val >= 0 else f"{val*100:.1f}%"

    # ── Odds-only mode (no model predictions available) ───────────────
    if predictions.empty:
        best = odds.copy()
        # Deduplicate to one row per match (use best_home_odds if available)
        if "best_home_odds" in best.columns:
            best = best.drop_duplicates(subset=["home_team", "away_team"])
        else:
            best = (
                best.groupby(["home_team", "away_team"], as_index=False)
                .agg(home_odds=("home_odds", "max"), away_odds=("away_odds", "max"),
                     home_spread=("home_spread", "first"), away_spread=("away_spread", "first"),
                     best_home_spread_odds=("home_spread_odds", "max"),
                     best_away_spread_odds=("away_spread_odds", "max"))
            )

        lines = []
        lines.append("=" * 72)
        lines.append("  CURRENT MARKET ODDS  (no model predictions — run collect + features)")
        lines.append("=" * 72)

        for _, row in best.iterrows():
            ho = float(row.get("best_home_odds", row.get("home_odds", 0)) or 0)
            ao = float(row.get("best_away_odds", row.get("away_odds", 0)) or 0)
            hs = row.get("home_spread")
            as_ = row.get("away_spread")
            hso = float(row.get("best_home_spread_odds", row.get("home_spread_odds", 0)) or 0)
            aso = float(row.get("best_away_spread_odds", row.get("away_spread_odds", 0)) or 0)

            lines.append(f"\n  {row['home_team']} vs {row['away_team']}")
            if ho > 0 and ao > 0:
                hi = implied_probability(ho)
                ai = implied_probability(ao)
                vig = (hi + ai - 1) * 100
                lines.append(
                    f"  H2H  Home: ${ho:.2f} (implied {hi:.0%})  "
                    f"Away: ${ao:.2f} (implied {ai:.0%})  Vig: {vig:.1f}%"
                )
            if hs is not None and pd.notna(hs) and hso > 0:
                lines.append(
                    f"  SPR  Home {float(hs):+.1f}: ${hso:.2f} (implied {implied_probability(hso):.0%})"
                )
            if as_ is not None and pd.notna(as_) and aso > 0:
                lines.append(
                    f"  SPR  Away {float(as_):+.1f}: ${aso:.2f} (implied {implied_probability(aso):.0%})"
                )

        lines.append("\n" + "=" * 72)
        lines.append(f"  {len(best)} match(es) with active odds")
        lines.append("  Run:  python main.py collect  →  python main.py features")
        lines.append("  to enable model probability comparisons and value bet detection.")
        lines.append("=" * 72)
        return "\n".join(lines)

    # ── Full mode: merge predictions with odds ────────────────────────
    # Determine which round(s) are being predicted
    requested_rounds = sorted(predictions["round"].unique().tolist())
    requested_label = (
        f"Round {requested_rounds[0]}"
        if len(requested_rounds) == 1
        else f"Rounds {requested_rounds[0]}–{requested_rounds[-1]}"
    )

    # Filter to the round(s) with best odds coverage, then normalize names
    predictions_filtered = _nearest_round_predictions(predictions, odds)
    preds_norm = _normalize_teams(predictions_filtered)
    odds_norm = _normalize_teams(odds)

    merged = preds_norm.merge(
        odds_norm,
        on=["home_team", "away_team"],
        how="inner",
        suffixes=("_pred", "_odds"),
    )

    if merged.empty:
        # The requested round has no bookmaker odds yet.
        # Show the available odds with a clear explanatory header.
        odds_rounds_label = "Opening Round" if 0 in requested_rounds else "an earlier round"
        lines = []
        lines.append("=" * 72)
        lines.append(f"  NO ODDS AVAILABLE FOR {requested_label.upper()} YET")
        lines.append(f"  Bookmakers currently cover: the upcoming fixtures below")
        lines.append("=" * 72)
        # Append the raw odds table inline (strip its own header, keep footer)
        raw = format_odds_comparison(pd.DataFrame(), odds)
        skip_header = True
        for line in raw.split("\n"):
            if skip_header:
                # Start including once we reach the first match line
                if "  " in line and " vs " in line:
                    skip_header = False
                    lines.append(line)
                # else: skip header lines (========, CURRENT MARKET ODDS, ========)
            else:
                # Filter the "run collect" instructions that don't apply here
                if "Run:  python main.py" in line or "to enable model" in line:
                    continue
                lines.append(line)
        return "\n".join(lines)


    lines = []
    lines.append("=" * 72)
    lines.append("  ODDS vs MODEL COMPARISON  (no value bets above threshold this round)")
    lines.append("=" * 72)

    for _, row in merged.iterrows():
        home_prob = float(row["ensemble_prob"])
        away_prob = 1.0 - home_prob
        margin = float(row.get("ensemble_margin", 0))
        home_odds = float(row.get("best_home_odds", row.get("home_odds", 0)) or 0)
        away_odds = float(row.get("best_away_odds", row.get("away_odds", 0)) or 0)

        lines.append(f"\n  {row['home_team']} vs {row['away_team']}")
        lines.append(f"  Predicted margin: {margin:+.0f} pts")

        # Head-to-Head
        if home_odds > 0 and away_odds > 0:
            home_impl = implied_probability(home_odds)
            away_impl = implied_probability(away_odds)
            home_ev = calculate_expected_value(home_prob, home_odds)
            away_ev = calculate_expected_value(away_prob, away_odds)
            lines.append(
                f"  H2H  Home: ${home_odds:.2f}  "
                f"Model {home_prob:.0%} vs Implied {home_impl:.0%}  "
                f"Edge {_fmt(home_prob - home_impl)}  EV {_fmt(home_ev)}"
            )
            lines.append(
                f"  H2H  Away: ${away_odds:.2f}  "
                f"Model {away_prob:.0%} vs Implied {away_impl:.0%}  "
                f"Edge {_fmt(away_prob - away_impl)}  EV {_fmt(away_ev)}"
            )

        # Spread / Line
        home_spread = row.get("home_spread")
        away_spread = row.get("away_spread")
        home_so = float(row.get("best_home_spread_odds", row.get("home_spread_odds", 0)) or 0)
        away_so = float(row.get("best_away_spread_odds", row.get("away_spread_odds", 0)) or 0)

        if home_spread is not None and pd.notna(home_spread) and home_so > 0:
            hlp = calculate_line_prob(margin, float(home_spread))
            hi = implied_probability(home_so)
            hev = calculate_expected_value(hlp, home_so)
            lines.append(
                f"  SPR  Home {float(home_spread):+.1f}: ${home_so:.2f}  "
                f"Model {hlp:.0%} vs Implied {hi:.0%}  "
                f"Edge {_fmt(hlp - hi)}  EV {_fmt(hev)}"
            )

        if away_spread is not None and pd.notna(away_spread) and away_so > 0:
            alp = 1.0 - calculate_line_prob(margin, -float(away_spread))
            ai = implied_probability(away_so)
            aev = calculate_expected_value(alp, away_so)
            lines.append(
                f"  SPR  Away {float(away_spread):+.1f}: ${away_so:.2f}  "
                f"Model {alp:.0%} vs Implied {ai:.0%}  "
                f"Edge {_fmt(alp - ai)}  EV {_fmt(aev)}"
            )

    lines.append("\n" + "=" * 72)
    lines.append(
        f"  {len(merged)} match(es) analysed — "
        f"no edge above threshold (MIN_EV_THRESHOLD={settings.betting.min_ev_threshold:.0%})"
    )
    lines.append("=" * 72)
    return "\n".join(lines)


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

        market = row.get("market_type", "h2h").upper()
        line = row.get("line")
        if line is not None and pd.notna(line):
            bet_label = f"{row['bet_on']} ({market} {line:+.1f})"
        else:
            bet_label = f"{row['bet_on']} ({market})"

        lines.append(f"\n  {stars}  {row['home_team']} vs {row['away_team']}")
        lines.append(f"  Bet:       {bet_label}")
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
