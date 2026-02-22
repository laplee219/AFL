"""
LLM Report Generator

Generates natural language reports for match predictions, 
value bets, and performance summaries.
"""

from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from config.settings import REPORTS_DIR
from src.llm.analyzer import _call_llm, SYSTEM_PROMPT
from src.utils.helpers import get_logger

logger = get_logger(__name__)


def _save_report(report: str, report_type: str, round_num: int = None, year: int = None) -> Path:
    """Save report to data/reports/ and return the file path."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parts = [report_type]
    if year:
        parts.append(str(year))
    if round_num:
        parts.append(f"R{round_num}")
    parts.append(timestamp)
    filename = "_".join(parts) + ".md"
    filepath = REPORTS_DIR / filename
    filepath.write_text(report, encoding="utf-8")
    logger.info(f"Report saved to {filepath}")
    return filepath


def generate_round_report(
    predictions: pd.DataFrame,
    value_bets: pd.DataFrame = None,
    model_metrics: dict = None,
    round_num: int = None,
    year: int = None,
) -> str:
    """
    Generate a comprehensive round preview report using LLM.

    Args:
        predictions: DataFrame from Predictor.predict_round()
        value_bets: DataFrame from find_value_bets()
        model_metrics: Dict of model evaluation metrics
        round_num: Round number
        year: Season year

    Returns:
        Natural language report string
    """
    if predictions.empty:
        return "No predictions available for this round."

    # Build match summaries
    match_lines = []
    for _, row in predictions.iterrows():
        margin = row.get("ensemble_margin", 0)
        prob = row.get("ensemble_prob", 0.5)
        winner = row["home_team"] if margin > 0 else row["away_team"]
        win_prob = prob if margin > 0 else (1 - prob)

        match_lines.append(
            f"- {row['home_team']} vs {row['away_team']}: "
            f"{winner} by {abs(margin):.0f} pts ({win_prob:.0%} confidence)"
        )

    # Build value bet summaries
    vb_lines = []
    if value_bets is not None and not value_bets.empty:
        for _, row in value_bets.iterrows():
            vb_lines.append(
                f"- {row['bet_on']} @ ${row['decimal_odds']:.2f} "
                f"(model: {row['model_prob']:.0%} vs bookmaker: {row['bookmaker_prob']:.0%}, "
                f"EV: +{row['expected_value']*100:.1f}%)"
            )

    prompt = f"""Generate a concise AFL Round {round_num or '?'} ({year or '?'}) preview report.

MATCH PREDICTIONS:
{chr(10).join(match_lines)}

{"VALUE BET OPPORTUNITIES:" + chr(10) + chr(10).join(vb_lines) if vb_lines else "No value bets identified this round."}

{f"MODEL PERFORMANCE: Accuracy {model_metrics.get('accuracy', 0):.0%}, margin MAE {model_metrics.get('margin_mae', 0):.1f} pts" if model_metrics else ""}

Write a report that includes:
1. Opening summary of the round's key matches
2. Predicted results with brief reasoning for the 2-3 most interesting matches
3. Value bet recommendations (if any) with risk assessment
4. Overall round outlook

Keep it concise and analytical. Use Australian English. 
Format with clear headings using markdown."""

    report = _call_llm(prompt, SYSTEM_PROMPT)
    saved = _save_report(report, "round_preview", round_num, year)
    logger.info(f"Round report saved to {saved}")
    return report, saved


def generate_performance_report(
    metrics: dict,
    bet_performance: dict,
    recent_predictions: pd.DataFrame = None,
) -> str:
    """
    Generate a model and betting performance report.
    """
    prompt = f"""Generate a concise AFL prediction model performance report.

MODEL METRICS:
- Accuracy: {metrics.get('accuracy', 0):.1%}
- Log Loss: {metrics.get('log_loss', 0):.4f}
- Brier Score: {metrics.get('brier_score', 0):.4f}
- Margin MAE: {metrics.get('margin_mae', 0):.1f} points
- Margin correlation: {metrics.get('margin_correlation', 0):.3f}

BETTING PERFORMANCE:
- Bets placed: {bet_performance.get('n_bets', 0)}
- Win rate: {bet_performance.get('win_rate', 0):.1%}
- ROI: {bet_performance.get('roi', 0):+.1%}
- Yield: {bet_performance.get('yield_pct', 0):+.1f}%
- Current bankroll: ${bet_performance.get('current_bankroll', 0):.2f}
- Max drawdown: {bet_performance.get('max_drawdown', 0):.1%}

Provide:
1. Overall assessment of model quality (is it performing well?)
2. Areas of concern or improvement
3. Betting strategy assessment
4. Recommendations for next steps

Keep it brief and actionable."""

    report = _call_llm(prompt, SYSTEM_PROMPT)
    saved = _save_report(report, "performance")
    logger.info(f"Performance report saved to {saved}")
    return report, saved


def generate_match_preview(
    home_team: str,
    away_team: str,
    prediction: dict,
    venue: str = "",
) -> str:
    """Generate a quick match preview (shorter than full analysis)."""
    margin = prediction.get("ensemble_margin", 0)
    prob = prediction.get("ensemble_prob", 0.5)
    winner = home_team if margin > 0 else away_team
    win_prob = prob if margin > 0 else (1 - prob)

    prompt = f"""Write a brief (100 words max) AFL match preview:

{home_team} (home) vs {away_team} at {venue or 'TBD'}
Prediction: {winner} by {abs(margin):.0f} points ({win_prob:.0%} confidence)

Include: key matchup, predicted winner rationale, one risk factor.
Use Australian English. Be concise."""

    return _call_llm(prompt, SYSTEM_PROMPT)
