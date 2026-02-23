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
        markets: str = "h2h,spreads",
        regions: str = "au",
        odds_format: str = "decimal",
    ) -> pd.DataFrame:
        """
        Fetch current AFL odds from multiple bookmakers.

        Args:
            markets: Comma-separated market types: 'h2h' (head-to-head), 'spreads' (line bets)
            regions: Region filter - 'au' for Australian bookmakers
            odds_format: 'decimal' or 'american'

        Returns:
            DataFrame with h2h and spread (line) odds per bookmaker per match.
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

                # Parse all available markets in one pass
                h2h_prices = {}
                spread_prices = {}
                spread_points = {}

                for market in bookmaker.get("markets", []):
                    if market.get("key") == "h2h":
                        h2h_prices = {o["name"]: o["price"] for o in market.get("outcomes", [])}
                    elif market.get("key") == "spreads":
                        for o in market.get("outcomes", []):
                            spread_prices[o["name"]] = o["price"]
                            spread_points[o["name"]] = o.get("point", 0.0)

                home_odds = h2h_prices.get(home_team, 0)
                away_odds = h2h_prices.get(away_team, 0)

                if home_odds <= 0 and away_odds <= 0:
                    continue  # No usable data from this bookmaker

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
                    # Line / spread market
                    "home_spread": spread_points.get(home_team),        # e.g. -12.5
                    "away_spread": spread_points.get(away_team),        # e.g. +12.5
                    "home_spread_odds": spread_prices.get(home_team, 0),
                    "away_spread_odds": spread_prices.get(away_team, 0),
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
            # Best spread (line bet) odds
            best_home_spread_odds=("home_spread_odds", lambda x: x[x > 0].max() if (x > 0).any() else 0),
            best_away_spread_odds=("away_spread_odds", lambda x: x[x > 0].max() if (x > 0).any() else 0),
            # Consensus spread line (first non-null value)
            home_spread=("home_spread", lambda x: x.dropna().iloc[0] if x.notna().any() else None),
            away_spread=("away_spread", lambda x: x.dropna().iloc[0] if x.notna().any() else None),
        ).reset_index()

        best["best_home_implied_prob"] = best["best_home_odds"].apply(implied_probability)
        best["best_away_implied_prob"] = best["best_away_odds"].apply(implied_probability)

        return best

    def get_available_sports(self) -> list:
        """List all available sports on The Odds API (useful for checking AFL availability)."""
        data = self._request("/sports")
        return data if isinstance(data, list) else []

    # ── Odds Snapshot Persistence ────────────────────────────────────

    def save_odds_snapshot(
        self,
        year: int,
        round_num: int,
        snapshot_type: str = "closing",
        odds: pd.DataFrame = None,
    ) -> int:
        """
        Capture current best odds and persist them to the database.

        Args:
            year: Season year
            round_num: Round number
            snapshot_type: 'opening' or 'closing'
            odds: Pre-fetched best odds DataFrame; if None, will fetch live

        Returns:
            Number of match snapshots saved
        """
        from src.utils.helpers import get_db_connection, init_database

        if odds is None:
            odds = self.get_best_odds()

        if odds.empty:
            logger.warning(f"No odds available to snapshot ({snapshot_type})")
            return 0

        # Ensure table exists
        init_database()

        conn = get_db_connection()
        saved = 0
        for _, row in odds.iterrows():
            home = row.get("home_team", "")
            away = row.get("away_team", "")
            if not home or not away:
                continue

            home_odds = float(row.get("best_home_odds", 0) or 0)
            away_odds = float(row.get("best_away_odds", 0) or 0)

            try:
                conn.execute("""
                    INSERT OR REPLACE INTO odds_snapshots
                    (year, round, home_team, away_team, snapshot_type,
                     home_odds, away_odds, home_implied_prob, away_implied_prob,
                     home_spread, away_spread, home_spread_odds, away_spread_odds,
                     n_bookmakers, captured_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
                """, (
                    year, round_num, home, away, snapshot_type,
                    home_odds, away_odds,
                    implied_probability(home_odds), implied_probability(away_odds),
                    float(row.get("home_spread") or 0) if pd.notna(row.get("home_spread")) else None,
                    float(row.get("away_spread") or 0) if pd.notna(row.get("away_spread")) else None,
                    float(row.get("best_home_spread_odds", 0) or 0),
                    float(row.get("best_away_spread_odds", 0) or 0),
                    int(row.get("n_bookmakers", 0) or 0),
                ))
                saved += 1
            except Exception as e:
                logger.error(f"Failed to save odds snapshot for {home} vs {away}: {e}")

        conn.commit()
        conn.close()
        logger.info(
            f"Saved {saved} {snapshot_type} odds snapshots for {year} R{round_num}"
        )
        return saved

    def load_odds_snapshot(
        self,
        year: int,
        round_num: int,
        snapshot_type: str = "closing",
    ) -> pd.DataFrame:
        """
        Load historical odds snapshots from the database.

        Args:
            year: Season year
            round_num: Round number
            snapshot_type: 'opening', 'closing', or None for all

        Returns:
            DataFrame with saved odds (same column names as get_best_odds output)
        """
        from src.utils.helpers import df_from_db

        if snapshot_type:
            df = df_from_db(
                "SELECT * FROM odds_snapshots WHERE year = ? AND round = ? AND snapshot_type = ?",
                (year, round_num, snapshot_type),
            )
        else:
            df = df_from_db(
                "SELECT * FROM odds_snapshots WHERE year = ? AND round = ?",
                (year, round_num),
            )

        if df.empty:
            return pd.DataFrame()

        # Rename columns to match get_best_odds() output for interoperability
        rename_map = {
            "home_odds": "best_home_odds",
            "away_odds": "best_away_odds",
            "home_spread_odds": "best_home_spread_odds",
            "away_spread_odds": "best_away_spread_odds",
            "home_implied_prob": "best_home_implied_prob",
            "away_implied_prob": "best_away_implied_prob",
        }
        df = df.rename(columns=rename_map)
        return df


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
