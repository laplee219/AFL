"""
Kelly Criterion Module

Calculates optimal bet sizing using the Kelly Criterion,
with fractional Kelly for risk management.

Full Kelly: f* = (p*b - q) / b
where p = probability, b = net odds, q = 1-p

We use quarter-Kelly (0.25 × f*) by default to reduce variance.
"""

from config.settings import settings
from src.utils.helpers import get_logger

logger = get_logger(__name__)


def kelly_fraction(
    model_prob: float,
    decimal_odds: float,
    fraction: float = None,
) -> float:
    """
    Calculate the Kelly criterion fraction for bet sizing.

    Args:
        model_prob: Probability of winning (from our model)
        decimal_odds: Decimal odds offered by bookmaker
        fraction: Kelly fraction (default: quarter-Kelly from settings)

    Returns:
        Fraction of bankroll to wager (0 if no edge)
    """
    fraction = fraction if fraction is not None else settings.betting.kelly_fraction

    if model_prob <= 0 or model_prob >= 1 or decimal_odds <= 1:
        return 0.0

    b = decimal_odds - 1.0  # Net odds
    q = 1.0 - model_prob

    # Full Kelly formula: f* = (p*b - q) / b
    full_kelly = (model_prob * b - q) / b

    if full_kelly <= 0:
        return 0.0  # No edge — don't bet

    # Apply fractional Kelly
    adjusted = full_kelly * fraction

    # Cap at max bet fraction
    max_bet = settings.betting.max_bet_fraction
    adjusted = min(adjusted, max_bet)

    return round(adjusted, 4)


def calculate_stake(
    bankroll: float,
    model_prob: float,
    decimal_odds: float,
    fraction: float = None,
) -> float:
    """
    Calculate the dollar amount to stake.

    Args:
        bankroll: Current bankroll
        model_prob: Model's win probability
        decimal_odds: Bookmaker's decimal odds
        fraction: Kelly fraction (default from settings)

    Returns:
        Stake amount in dollars (0 if no edge)
    """
    kelly = kelly_fraction(model_prob, decimal_odds, fraction)
    if kelly <= 0:
        return 0.0

    stake = bankroll * kelly

    # Round to practical amount (nearest dollar)
    stake = round(max(stake, 0), 2)

    return stake


def calculate_expected_profit(
    model_prob: float,
    decimal_odds: float,
    stake: float,
) -> dict:
    """
    Calculate expected profit/loss scenarios.

    Returns:
        Dict with expected_profit, win_profit, loss_amount, expected_value
    """
    if stake <= 0 or decimal_odds <= 0:
        return {
            "expected_profit": 0.0,
            "win_profit": 0.0,
            "loss_amount": 0.0,
            "expected_value": 0.0,
        }

    win_profit = stake * (decimal_odds - 1.0)
    loss_amount = -stake
    expected_profit = model_prob * win_profit + (1 - model_prob) * loss_amount
    expected_value = expected_profit / stake if stake > 0 else 0

    return {
        "expected_profit": round(expected_profit, 2),
        "win_profit": round(win_profit, 2),
        "loss_amount": round(loss_amount, 2),
        "expected_value": round(expected_value, 4),
    }


def optimal_kelly_analysis(
    model_prob: float,
    decimal_odds: float,
    bankroll: float,
) -> dict:
    """
    Full Kelly analysis for a single bet opportunity.

    Returns detailed breakdown including different Kelly fractions.
    """
    analysis = {
        "model_prob": model_prob,
        "decimal_odds": decimal_odds,
        "implied_prob": 1.0 / decimal_odds if decimal_odds > 0 else 0,
        "edge": model_prob - (1.0 / decimal_odds if decimal_odds > 0 else 0),
        "bankroll": bankroll,
    }

    # Calculate for different Kelly fractions
    for frac_name, frac_val in [
        ("full", 1.0),
        ("half", 0.5),
        ("quarter", 0.25),
        ("eighth", 0.125),
    ]:
        kelly = kelly_fraction(model_prob, decimal_odds, frac_val)
        stake = bankroll * kelly
        profit_info = calculate_expected_profit(model_prob, decimal_odds, stake)

        analysis[f"{frac_name}_kelly"] = {
            "fraction": kelly,
            "stake": round(stake, 2),
            **profit_info,
        }

    # Recommended (from settings)
    rec_kelly = kelly_fraction(model_prob, decimal_odds)
    rec_stake = calculate_stake(bankroll, model_prob, decimal_odds)
    analysis["recommended"] = {
        "fraction": rec_kelly,
        "stake": rec_stake,
        **calculate_expected_profit(model_prob, decimal_odds, rec_stake),
    }

    return analysis
