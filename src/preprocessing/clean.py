"""
Data Cleaning Module

Standardizes raw match data from various sources into a clean, consistent format.
"""

import pandas as pd
import numpy as np

from src.utils.constants import normalize_team_name, normalize_venue_name
from src.utils.helpers import get_logger

logger = get_logger(__name__)


def clean_squiggle_games(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and standardize raw Squiggle API game data.

    Args:
        df: Raw DataFrame from SquiggleClient.get_games()

    Returns:
        Cleaned DataFrame with standardized column names and values.
    """
    if df.empty:
        return df

    df = df.copy()

    # ── Standardize column names ─────────────────────────────────────
    column_map = {
        "id": "match_id",
        "year": "year",
        "round": "round",
        "roundname": "round_name",
        "date": "date",
        "hteam": "home_team",
        "ateam": "away_team",
        "venue": "venue",
        "hscore": "home_score",
        "ascore": "away_score",
        "hgoals": "home_goals",
        "hbehinds": "home_behinds",
        "agoals": "away_goals",
        "abehinds": "away_behinds",
        "winner": "winner",
        "complete": "complete",
        "is_final": "is_final",
        "atten": "crowd",
    }

    # Only rename columns that exist
    rename_cols = {k: v for k, v in column_map.items() if k in df.columns}
    df = df.rename(columns=rename_cols)

    # ── Normalize team & venue names ─────────────────────────────────
    if "home_team" in df.columns:
        df["home_team"] = df["home_team"].apply(
            lambda x: normalize_team_name(x) if pd.notna(x) else ""
        )
    if "away_team" in df.columns:
        df["away_team"] = df["away_team"].apply(
            lambda x: normalize_team_name(x) if pd.notna(x) else ""
        )
    if "winner" in df.columns:
        df["winner"] = df["winner"].apply(
            lambda x: normalize_team_name(x) if pd.notna(x) and x else x
        )
    if "venue" in df.columns:
        df["venue"] = df["venue"].apply(
            lambda x: normalize_venue_name(x) if pd.notna(x) else ""
        )

    # ── Calculate derived fields ─────────────────────────────────────
    if "home_score" in df.columns and "away_score" in df.columns:
        df["margin"] = df["home_score"] - df["away_score"]
        df["total_score"] = df["home_score"] + df["away_score"]
        df["home_win"] = (df["margin"] > 0).astype(int)

    # Calculate scoring efficiency (goals / scoring shots)
    if "home_goals" in df.columns and "home_behinds" in df.columns:
        home_shots = df["home_goals"] + df["home_behinds"]
        df["home_conversion"] = np.where(
            home_shots > 0,
            df["home_goals"] / home_shots,
            0.0
        )
    if "away_goals" in df.columns and "away_behinds" in df.columns:
        away_shots = df["away_goals"] + df["away_behinds"]
        df["away_conversion"] = np.where(
            away_shots > 0,
            df["away_goals"] / away_shots,
            0.0
        )

    # ── Parse dates ──────────────────────────────────────────────────
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values(["date", "match_id"]).reset_index(drop=True)

    # ── Filter to completed games only (for historical data) ─────────
    if "complete" in df.columns:
        df["is_complete"] = df["complete"] == 100
    else:
        df["is_complete"] = df["home_score"].notna() & (df["home_score"] > 0)

    # ── Set is_final flag ────────────────────────────────────────────
    if "is_final" not in df.columns:
        if "round_name" in df.columns:
            finals_keywords = ["Final", "Preliminary", "Grand", "Elimination", "Qualifying"]
            df["is_final"] = df["round_name"].apply(
                lambda x: any(kw in str(x) for kw in finals_keywords) if pd.notna(x) else False
            ).astype(int)
        else:
            df["is_final"] = 0

    logger.info(f"Cleaned {len(df)} matches ({df['is_complete'].sum()} completed)")
    return df


def merge_odds_with_matches(
    matches: pd.DataFrame, odds: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge bookmaker odds data with match data.

    Links odds to matches based on team names and approximate timing.
    """
    if odds.empty:
        # Add empty odds columns
        matches["home_odds"] = np.nan
        matches["away_odds"] = np.nan
        matches["home_implied_prob"] = np.nan
        matches["away_implied_prob"] = np.nan
        return matches

    # Normalize team names in odds
    odds = odds.copy()
    odds["home_team"] = odds["home_team"].apply(normalize_team_name)
    odds["away_team"] = odds["away_team"].apply(normalize_team_name)

    # Merge on team names
    merged = matches.merge(
        odds[["home_team", "away_team", "home_odds", "away_odds",
              "home_implied_prob", "away_implied_prob"]].drop_duplicates(
            subset=["home_team", "away_team"], keep="last"
        ),
        on=["home_team", "away_team"],
        how="left",
    )

    n_matched = merged["home_odds"].notna().sum()
    logger.info(f"Matched odds for {n_matched}/{len(merged)} matches")
    return merged


def validate_data(df: pd.DataFrame) -> dict:
    """
    Validate cleaned match data for quality issues.

    Returns:
        Dictionary with validation results.
    """
    issues = {
        "total_rows": len(df),
        "missing_scores": 0,
        "duplicate_matches": 0,
        "invalid_scores": 0,
        "unknown_teams": [],
        "status": "OK",
    }

    from src.utils.constants import TEAMS

    # Check for missing scores in completed games
    if "is_complete" in df.columns and "home_score" in df.columns:
        completed = df[df["is_complete"]]
        issues["missing_scores"] = completed["home_score"].isna().sum()

    # Check for duplicates
    if "match_id" in df.columns:
        issues["duplicate_matches"] = df["match_id"].duplicated().sum()

    # Check for invalid scores
    if "home_score" in df.columns:
        invalid = df[
            (df["is_complete"]) &
            ((df["home_score"] < 0) | (df["away_score"] < 0))
        ]
        issues["invalid_scores"] = len(invalid)

    # Check for unknown teams
    if "home_team" in df.columns:
        all_teams = set(df["home_team"].unique()) | set(df["away_team"].unique())
        known_teams = set(TEAMS.keys())
        unknown = all_teams - known_teams
        if unknown:
            issues["unknown_teams"] = list(unknown)

    if issues["missing_scores"] > 0 or issues["invalid_scores"] > 0:
        issues["status"] = "WARNING"
    if issues["duplicate_matches"] > 0:
        issues["status"] = "ERROR"

    return issues
