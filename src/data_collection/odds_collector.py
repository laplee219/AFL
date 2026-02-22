"""
Odds Collector

Fetches bookmaker odds from The Odds API for AFL matches.
Calculates implied probabilities for value bet comparison.
"""

from typing import Optional

import pandas as pd
import requests

from config.settings import settings
from src.utils.constants import normalize_team_name
from src.utils.helpers import get_logger, implied_probability

logger = get_logger(__name__)


class OddsCollector:
    """Collects AFL betting odds from The Odds API."""

    SPORT_KEY = "aussierules_afl"

    def __init__(self):
        self.api_key = settings.odds.odds_api_key
        self.base_url = settings.odds.odds_api_base_url
        self.session = requests.Session()

    def _request(self, endpoint: str, params: Optional[dict] = None) -> dict:
        """Make a request to The Odds API."""
        if not self.api_key:
            logger.warning("Odds API key not configured. Set ODDS_API_KEY in .env")
            return {}

        url = f"{self.base_url}{endpoint}"
        default_params = {"apiKey": self.api_key}
        if params:
            default_params.update(params)

        try:
            response = self.session.get(url, params=default_params, timeout=30)
            response.raise_for_status()

            # Log remaining quota
            remaining = response.headers.get("x-requests-remaining", "?")
            logger.debug(f"Odds API requests remaining: {remaining}")

            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Odds API error: {e}")
            return {}

    def get_current_odds(
        self,
        markets: str = "h2h",
        regions: str = "au",
        odds_format: str = "decimal",
    ) -> pd.DataFrame:
        """
        Fetch current AFL odds from multiple bookmakers.

        Args:
            markets: Market type - 'h2h' (head-to-head/moneyline), 'spreads', 'totals'
            regions: Region filter - 'au' for Australian bookmakers
            odds_format: 'decimal' or 'american'

        Returns:
            DataFrame with columns: match_id, home_team, away_team, bookmaker,
            home_odds, away_odds, home_implied_prob, away_implied_prob, commence_time
        """
        endpoint = f"/sports/{self.SPORT_KEY}/odds"
        params = {
            "markets": markets,
            "regions": regions,
            "oddsFormat": odds_format,
        }

        data = self._request(endpoint, params)
        if not data:
            return pd.DataFrame()

        rows = []
        for event in data:
            home_team = event.get("home_team", "")
            away_team = event.get("away_team", "")
            commence_time = event.get("commence_time", "")
            event_id = event.get("id", "")

            for bookmaker in event.get("bookmakers", []):
                bk_name = bookmaker.get("title", "")
                for market in bookmaker.get("markets", []):
                    if market.get("key") == "h2h":
                        outcomes = {o["name"]: o["price"] for o in market.get("outcomes", [])}
                        home_odds = outcomes.get(home_team, 0)
                        away_odds = outcomes.get(away_team, 0)

                        rows.append({
                            "event_id": event_id,
                            "home_team": home_team,
                            "away_team": away_team,
                            "bookmaker": bk_name,
                            "home_odds": home_odds,
                            "away_odds": away_odds,
                            "home_implied_prob": implied_probability(home_odds),
                            "away_implied_prob": implied_probability(away_odds),
                            "overround": implied_probability(home_odds) + implied_probability(away_odds) - 1.0,
                            "commence_time": commence_time,
                        })

        if not rows:
            logger.warning("No odds data found for AFL")
            return pd.DataFrame()

        df = pd.DataFrame(rows)

        # Normalize team names to match our canonical short forms
        # e.g. "Sydney Swans" -> "Sydney", "Carlton Blues" -> "Carlton"
        df["home_team"] = df["home_team"].apply(normalize_team_name)
        df["away_team"] = df["away_team"].apply(normalize_team_name)

        logger.info(f"Fetched {len(df)} odds entries from {df['bookmaker'].nunique()} bookmakers")
        return df

    def get_best_odds(self) -> pd.DataFrame:
        """
        Get the best available odds for each AFL match (highest payout).

        Returns:
            DataFrame with best home/away odds across all bookmakers.
        """
        all_odds = self.get_current_odds()
        if all_odds.empty:
            return pd.DataFrame()

        # For each match, find the best home and away odds
        best = all_odds.groupby(["home_team", "away_team"]).agg(
            best_home_odds=("home_odds", "max"),
            best_away_odds=("away_odds", "max"),
            best_home_bookmaker=("home_odds", lambda x: all_odds.loc[x.idxmax(), "bookmaker"]),
            best_away_bookmaker=("away_odds", lambda x: all_odds.loc[x.idxmax(), "bookmaker"]),
            avg_home_odds=("home_odds", "mean"),
            avg_away_odds=("away_odds", "mean"),
            n_bookmakers=("bookmaker", "nunique"),
            commence_time=("commence_time", "first"),
        ).reset_index()

        best["best_home_implied_prob"] = best["best_home_odds"].apply(implied_probability)
        best["best_away_implied_prob"] = best["best_away_odds"].apply(implied_probability)

        return best

    def get_available_sports(self) -> list:
        """List all available sports on The Odds API (useful for checking AFL availability)."""
        data = self._request("/sports")
        return data if isinstance(data, list) else []


class ManualOddsManager:
    """
    Manager for manually entered odds when API is unavailable.
    Stores odds in the database for historical tracking.
    """

    def __init__(self):
        from src.utils.helpers import get_db_connection
        self.conn_factory = get_db_connection

    def add_odds(
        self,
        year: int,
        round_num: int,
        home_team: str,
        away_team: str,
        home_odds: float,
        away_odds: float,
        bookmaker: str = "manual",
    ):
        """Manually add odds for a match."""
        from src.utils.helpers import get_db_connection
        conn = get_db_connection()
        conn.execute("""
            INSERT OR REPLACE INTO match_odds 
            (year, round, home_team, away_team, home_odds, away_odds, 
             home_implied_prob, away_implied_prob, bookmaker, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
        """, (
            year, round_num, home_team, away_team,
            home_odds, away_odds,
            implied_probability(home_odds),
            implied_probability(away_odds),
            bookmaker,
        ))
        conn.commit()
        conn.close()
        logger.info(f"Added odds: {home_team} ${home_odds:.2f} vs {away_team} ${away_odds:.2f}")
