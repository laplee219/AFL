"""
Match Distribution Analysis Module

Provides three analytical views for each match prediction:

1. Margin Distribution  — full N(µ, σ²) probability distribution over
   outcome margins, including win/loss zone breakdown and key percentiles.

2. Spread Coverage Profile  — P(home covers) across a range of spread
   lines, helping identify which bookmaker lines carry structural value.

3. Closing Line Value (CLV)  — model implied probability vs bookmaker
   current/closing implied probability.  Positive CLV signals the model
   is sharper than the market consensus at time of record.
"""

import math
from typing import Optional

import numpy as np
import pandas as pd

from src.utils.helpers import get_logger, implied_probability

logger = get_logger(__name__)

# ── Helpers ──────────────────────────────────────────────────────────────────


def _norm_cdf(x: float, mu: float, sigma: float) -> float:
    """CDF of N(mu, sigma^2) evaluated at x."""
    z = (x - mu) / (sigma * math.sqrt(2))
    return 0.5 * (1.0 + math.erf(z))


def _norm_pdf(x: float, mu: float, sigma: float) -> float:
    """PDF of N(mu, sigma^2) evaluated at x."""
    return (1.0 / (sigma * math.sqrt(2 * math.pi))) * math.exp(-0.5 * ((x - mu) / sigma) ** 2)


def _quantile(p: float, mu: float, sigma: float) -> float:
    """Quantile function (inverse CDF) using rational approximation."""
    # Beasley-Springer-Moro approximation
    if p <= 0:
        return -math.inf
    if p >= 1:
        return math.inf
    # Use scipy if available, otherwise Newton-Raphson
    try:
        from scipy.stats import norm as _norm
        return float(_norm.ppf(p, loc=mu, scale=sigma))
    except ImportError:
        # Simple Newton-Raphson
        x = mu
        for _ in range(50):
            fx = _norm_cdf(x, mu, sigma) - p
            fpx = _norm_pdf(x, mu, sigma)
            if abs(fpx) < 1e-12:
                break
            x -= fx / fpx
            if abs(fx) < 1e-9:
                break
        return x


# ── 1. Margin Distribution ────────────────────────────────────────────────────


def margin_distribution(
    predicted_margin: float,
    sigma: float,
    zone_width: int = 12,
) -> dict:
    """
    Compute the full margin distribution for a match.

    The actual margin is modelled as N(predicted_margin, sigma²).
    Returns a breakdown by win/loss zones, key percentiles, and
    an ASCII-ready probability bar for each zone.

    Args:
        predicted_margin: Model's predicted home margin (+ = home wins)
        sigma: Std dev of margin prediction error (use model.margin_sigma)
        zone_width: Width of each zone in points (default 12)

    Returns:
        dict with keys:
          - "predicted_margin": float
          - "sigma": float
          - "home_win_prob": float          P(margin > 0)
          - "zones": list[dict]             per-zone probabilities
          - "percentiles": dict             {5, 10, 25, 50, 75, 90, 95} → margin
    """
    mu = predicted_margin

    home_win_prob = 1.0 - _norm_cdf(0.0, mu, sigma)

    # Build zones: each zone covers [lo, hi) margin
    zones = []
    # Determine range: cover ±4 sigma, rounded to zone_width
    extent = int(math.ceil(4 * sigma / zone_width)) * zone_width
    lo = -extent
    while lo < extent:
        hi = lo + zone_width
        prob = _norm_cdf(hi, mu, sigma) - _norm_cdf(lo, mu, sigma)
        centre = (lo + hi) / 2
        side = "Home" if centre > 0 else ("Away" if centre < 0 else "Draw")
        zones.append({
            "lo": lo,
            "hi": hi,
            "centre": centre,
            "prob": prob,
            "side": side,
            "label": (
                f"Home +{lo}–{hi}" if lo >= 0
                else (f"Away {abs(hi)}–{abs(lo)}" if hi <= 0
                      else f"Close  {lo}–{hi}")
            ),
        })
        lo = hi

    # Key percentiles
    percentiles = {p: _quantile(p / 100, mu, sigma) for p in (5, 10, 25, 50, 75, 90, 95)}

    return {
        "predicted_margin": predicted_margin,
        "sigma": sigma,
        "home_win_prob": home_win_prob,
        "zones": zones,
        "percentiles": percentiles,
    }


def format_margin_distribution(dist: dict, home_team: str, away_team: str) -> str:
    """Render margin_distribution() output as an ASCII block."""
    mu = dist["predicted_margin"]
    sigma = dist["sigma"]
    hw = dist["home_win_prob"]
    zones = dist["zones"]
    pct = dist["percentiles"]

    winner = home_team if mu >= 0 else away_team
    loser = away_team if mu >= 0 else home_team
    abs_mu = abs(mu)
    win_p = hw if mu >= 0 else (1.0 - hw)

    lines = []
    lines.append(f"  Predicted: {winner} by {abs_mu:.0f} pts  ({win_p:.0%} win)  σ={sigma:.0f} pts")
    lines.append(f"  P(home win) = {hw:.1%}   P(away win) = {1 - hw:.1%}")
    lines.append("")
    lines.append("  Margin Distribution")
    lines.append("  " + "─" * 52)

    max_prob = max(z["prob"] for z in zones)
    bar_width = 28

    # Only show zones with meaningful probability (avoid cluttering with tail zones)
    visible_zones = [z for z in zones if z["prob"] >= 0.003]

    for z in visible_zones:
        lo, hi, prob = z["lo"], z["hi"], z["prob"]
        bar_len = max(1, round(prob / max_prob * bar_width)) if prob > 0.001 else 0
        bar = "█" * bar_len
        if lo >= 0:
            label = f"Home  {lo:3d}–{hi:3d}"
        elif hi <= 0:
            label = f"Away  {abs(hi):3d}–{abs(lo):3d}"
        else:
            label = f"Close {lo:3d}–{hi:3d}"
        lines.append(f"  {label}  {bar:<{bar_width}}  {prob*100:5.1f}%")

    lines.append("  " + "─" * 52)
    lines.append(
        f"  Outcome percentiles (home margin):  "
        f"5%={pct[5]:+.0f}  25%={pct[25]:+.0f}  "
        f"50%={pct[50]:+.0f}  75%={pct[75]:+.0f}  95%={pct[95]:+.0f}"
    )
    return "\n".join(lines)


# ── 2. Spread Coverage Profile ───────────────────────────────────────────────


def cover_probability_profile(
    predicted_margin: float,
    sigma: float,
    book_line: Optional[float] = None,
    lines: Optional[list] = None,
) -> dict:
    """
    Compute P(home covers) across a range of spread lines.

    Args:
        predicted_margin: Model's predicted home margin
        sigma: Std dev of margin prediction error
        book_line: Actual bookmaker spread line (home perspective, e.g. -12.5)
        lines: List of lines to evaluate; if None uses a standard range

    Returns:
        dict with:
          - "predicted_margin": float
          - "fair_line": float         line where P(cover) = 50%
          - "book_line": float | None
          - "book_cover_prob": float | None
          - "profile": list[dict]      per-line {line, cover_prob, edge}
    """
    if lines is None:
        # Standard AFL spread lines from -36 to +36 in 6-pt steps
        lines = [x * 6.0 for x in range(-6, 7)]   # -36 to +36

    # The "fair line" is the line L where P(margin > -L) = 0.5
    # ↔ -L = mu  ↔  L = -mu
    fair_line = -predicted_margin

    profile = []
    for line in lines:
        cover_threshold = -line          # home covers if margin > cover_threshold
        cover_prob = 1.0 - _norm_cdf(cover_threshold, predicted_margin, sigma)
        profile.append({
            "line": line,
            "cover_prob": cover_prob,
            "away_cover_prob": 1.0 - cover_prob,
        })

    book_cover_prob = None
    if book_line is not None:
        cover_threshold = -book_line
        book_cover_prob = 1.0 - _norm_cdf(cover_threshold, predicted_margin, sigma)

    return {
        "predicted_margin": predicted_margin,
        "sigma": sigma,
        "fair_line": fair_line,
        "book_line": book_line,
        "book_cover_prob": book_cover_prob,
        "profile": profile,
    }


def format_cover_profile(
    profile: dict,
    home_team: str,
    away_team: str,
    home_spread_odds: float = 0.0,
    away_spread_odds: float = 0.0,
) -> str:
    """Render cover_probability_profile() output as an ASCII block."""
    book_line = profile["book_line"]
    fair_line = profile["fair_line"]
    book_cover = profile["book_cover_prob"]

    lines_out = []
    lines_out.append(f"  Fair spread line (P=50%): {fair_line:+.1f} pts")

    if book_line is not None and book_cover is not None:
        bm_implied = implied_probability(home_spread_odds) if home_spread_odds > 0 else None
        if bm_implied is not None:
            edge_pp = (book_cover - bm_implied) * 100
            flag = " ◄ VALUE" if edge_pp >= 5.0 else ""
            lines_out.append(
                f"  Bookmaker line {book_line:+.1f}: "
                f"model P(home cover) {book_cover:.1%}  "
                f"vs implied {bm_implied:.1%}  "
                f"edge {edge_pp:+.1f}pp{flag}"
            )
        else:
            lines_out.append(
                f"  Bookmaker line {book_line:+.1f}: model P(home cover) {book_cover:.1%}"
            )

    lines_out.append("")
    lines_out.append(f"  Coverage Profile  ({home_team} covering)")
    lines_out.append("  " + "─" * 50)
    lines_out.append(f"  {'Line':>7}  {'P(Home Cover)':>14}  {'P(Away Cover)':>14}  Bar")

    for entry in profile["profile"]:
        line = entry["line"]
        cp = entry["cover_prob"]
        acp = entry["away_cover_prob"]
        bar_len = round(cp * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        marker = " ◄ book" if (book_line is not None and abs(line - book_line) < 0.1) else (
            " ◄ fair" if abs(line - fair_line) < 3.0 else ""
        )
        lines_out.append(
            f"  {line:>+7.1f}  {cp:>13.1%}  {acp:>13.1%}  {bar}{marker}"
        )
    lines_out.append("  " + "─" * 50)
    return "\n".join(lines_out)


# ── 3. Closing Line Value ─────────────────────────────────────────────────────


def closing_line_value(
    model_prob: float,
    current_odds: float,
    open_odds: Optional[float] = None,
) -> dict:
    """
    Compute Closing Line Value (CLV) for a single market.

    CLV measures the quality of the model's probability estimate relative
    to what the market agrees on at closing time.  Positive CLV = the model
    was "sharper" than the market.

    Two formulations are returned:
      - Simple CLV:  model_prob − closing_implied_prob  (probability points)
      - Log CLV:     log(closing_odds / model_decimal_odds)
                     > 0 means closing odds are longer than model implies
                       i.e. the market moved toward the model after assessment

    Args:
        model_prob: Model's estimated win probability (post-calibration)
        current_odds: Current / closing bookmaker decimal odds
        open_odds: Opening bookmaker decimal odds (optional, for line movement)

    Returns:
        dict with clv_pp, log_clv, model_implied_odds, closing_implied_prob,
        line_movement (if open_odds given), and a qualitative grade
    """
    if model_prob <= 0 or model_prob >= 1:
        return {"error": "model_prob must be in (0, 1)"}
    if current_odds <= 1:
        return {"error": "current_odds must be > 1"}

    model_decimal_odds = 1.0 / model_prob
    closing_implied = implied_probability(current_odds)

    clv_pp = model_prob - closing_implied          # probability-point CLV
    log_clv = math.log(current_odds / model_decimal_odds)  # log CLV

    # Grade
    if clv_pp >= 0.08:
        grade = "A  (strong model edge)"
    elif clv_pp >= 0.04:
        grade = "B  (meaningful edge)"
    elif clv_pp >= 0.01:
        grade = "C  (small edge)"
    elif clv_pp >= -0.02:
        grade = "D  (neutral / noise)"
    else:
        grade = "F  (negative — market sharper)"

    result = {
        "model_prob": model_prob,
        "model_decimal_odds": model_decimal_odds,
        "closing_odds": current_odds,
        "closing_implied_prob": closing_implied,
        "clv_pp": clv_pp,
        "log_clv": log_clv,
        "grade": grade,
    }

    if open_odds is not None and open_odds > 1:
        open_implied = implied_probability(open_odds)
        line_movement_pp = closing_implied - open_implied   # positive = market shortened (favourite moved in)
        result["open_odds"] = open_odds
        result["open_implied_prob"] = open_implied
        result["line_movement_pp"] = line_movement_pp
        result["moved_toward_model"] = (
            (model_prob > closing_implied and line_movement_pp > 0) or
            (model_prob < closing_implied and line_movement_pp < 0)
        )

    return result


def format_clv(clv: dict, team: str) -> str:
    """Render closing_line_value() output as a single-line summary."""
    if "error" in clv:
        return f"  CLV: {clv['error']}"

    clv_pct = clv["clv_pp"] * 100
    log = clv["log_clv"]
    grade = clv["grade"]
    mp = clv["model_prob"]
    ci = clv["closing_implied_prob"]
    mo = clv["model_decimal_odds"]
    co = clv["closing_odds"]

    line = (
        f"  CLV  {team}: "
        f"model {mp:.1%} (${mo:.2f}) vs "
        f"closing {ci:.1%} (${co:.2f})  "
        f"CLV {clv_pct:+.1f}pp  logCLV {log:+.3f}  [{grade}]"
    )
    if "line_movement_pp" in clv:
        mv = clv["line_movement_pp"] * 100
        toward = "✓ toward model" if clv.get("moved_toward_model") else "✗ away from model"
        line += f"\n  Line movement: {mv:+.1f}pp ({toward})"
    return line


# ── Combined per-round report ─────────────────────────────────────────────────


def format_distribution_report(
    predictions: pd.DataFrame,
    odds: pd.DataFrame,
    sigma: float = 34.0,
) -> str:
    """
    Generate the full three-panel analysis for all matches in a round.

    Args:
        predictions: DataFrame from Predictor.predict_round()
        odds: DataFrame from OddsCollector.get_best_odds()
        sigma: Margin prediction std dev (use model.margin_sigma)

    Returns:
        Printable multi-match analysis report string
    """
    if predictions.empty:
        return "No predictions available for analysis."

    from src.utils.constants import normalize_team_name
    from src.betting.value import _normalize_teams, _nearest_round_predictions

    predictions = _nearest_round_predictions(predictions, odds)
    predictions = _normalize_teams(predictions)
    if not odds.empty:
        odds = _normalize_teams(odds)

    # Merge for odds columns
    if not odds.empty:
        merged = predictions.merge(
            odds,
            on=["home_team", "away_team"],
            how="left",
            suffixes=("_pred", "_odds"),
        )
    else:
        merged = predictions.copy()

    year = int(predictions["year"].iloc[0]) if "year" in predictions else "?"
    rnd = int(predictions["round"].iloc[0]) if "round" in predictions else "?"

    blocks = []
    blocks.append("=" * 72)
    blocks.append(f"  MATCH DISTRIBUTION ANALYSIS  —  {year} Round {rnd}")
    blocks.append("=" * 72)

    for _, row in merged.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        mu = float(row.get("ensemble_margin", 0))
        home_prob = float(row.get("ensemble_prob", 0.5))
        home_odds = float(row.get("best_home_odds", row.get("home_odds", 0)) or 0)
        away_odds = float(row.get("best_away_odds", row.get("away_odds", 0)) or 0)
        home_spread = row.get("home_spread")
        away_spread = row.get("away_spread")
        home_so = float(row.get("best_home_spread_odds", row.get("home_spread_odds", 0)) or 0)
        away_so = float(row.get("best_away_spread_odds", row.get("away_spread_odds", 0)) or 0)

        blocks.append(f"\n  ── {home} vs {away} ──")

        # ── 1. Margin Distribution
        blocks.append("\n  [1] Margin Distribution")
        dist = margin_distribution(mu, sigma)
        blocks.append(format_margin_distribution(dist, home, away))

        # ── 2. Spread Coverage Profile
        blocks.append("\n  [2] Spread Coverage Profile")
        book_line = float(home_spread) if (home_spread is not None and pd.notna(home_spread)) else None
        profile = cover_probability_profile(mu, sigma, book_line=book_line)
        blocks.append(format_cover_profile(profile, home, away, home_spread_odds=home_so, away_spread_odds=away_so))

        # ── 3. Closing Line Value
        blocks.append("\n  [3] Closing Line Value")
        if home_odds > 1:
            clv_home = closing_line_value(home_prob, home_odds)
            blocks.append(format_clv(clv_home, home))
        if away_odds > 1:
            away_prob = 1.0 - home_prob
            clv_away = closing_line_value(away_prob, away_odds)
            blocks.append(format_clv(clv_away, away))
        if home_odds <= 1 and away_odds <= 1:
            blocks.append("  CLV: no bookmaker odds available")

        blocks.append("")

    blocks.append("=" * 72)
    blocks.append(f"  σ = {sigma:.0f} pts  |  Predictions vs current market odds")
    blocks.append("=" * 72)
    return "\n".join(blocks)
