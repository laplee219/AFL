"""
Squiggle API Client

Wrapper for the Squiggle API (api.squiggle.com.au) — the primary structured
data source for AFL match data, tips, standings, and power rankings.

API Docs: https://api.squiggle.com.au
"""

import time
from typing import Optional

import pandas as pd
import requests

from config.settings import settings
from src.utils.helpers import get_logger

logger = get_logger(__name__)

# Rate limit: be respectful — max 1 request per second
_last_request_time = 0.0
_MIN_REQUEST_INTERVAL = 1.0


class SquiggleClient:
    """Client for the Squiggle AFL API."""

    def __init__(self):
        self.base_url = settings.data.squiggle_base_url
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": settings.data.squiggle_user_agent,
            "Accept": "application/json",
        })

    def _request(self, params: dict) -> dict:
        """Make a rate-limited request to the Squiggle API."""
        global _last_request_time

        # Rate limiting
        elapsed = time.time() - _last_request_time
        if elapsed < _MIN_REQUEST_INTERVAL:
            time.sleep(_MIN_REQUEST_INTERVAL - elapsed)

        url = self.base_url
        logger.debug(f"Squiggle request: {params}")

        try:
            response = self.session.get(url, params=params, timeout=30)
            _last_request_time = time.time()
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Squiggle API error: {e}")
            raise

    # ── Games (Match Results & Fixtures) ──────────────────────────────

    def get_games(
        self,
        year: Optional[int] = None,
        round_num: Optional[int] = None,
        team: Optional[str] = None,
        game_id: Optional[int] = None,
        complete: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Fetch match data from Squiggle.

        Args:
            year: Season year (e.g., 2025)
            round_num: Round number (1-24 for home&away, 25+ for finals)
            team: Team name filter
            game_id: Specific game ID
            complete: 100 for completed games only, 0 for upcoming

        Returns:
            DataFrame with columns: id, year, round, date, hteam, ateam, 
            venue, hscore, ascore, hgoals, hbehinds, agoals, abehinds,
            winner, margin, complete, etc.
        """
        params = {"q": "games"}
        if year is not None:
            params["year"] = year
        if round_num is not None:
            params["round"] = round_num
        if team is not None:
            params["team"] = team
        if game_id is not None:
            params["game"] = game_id
        if complete is not None:
            params["complete"] = complete

        data = self._request(params)
        games = data.get("games", [])

        if not games:
            logger.warning(f"No games returned for params: {params}")
            return pd.DataFrame()

        df = pd.DataFrame(games)
        logger.info(f"Fetched {len(df)} games from Squiggle")
        return df

    def get_completed_games(self, year: int, round_num: Optional[int] = None) -> pd.DataFrame:
        """Fetch only completed games for a year/round."""
        return self.get_games(year=year, round_num=round_num, complete=100)

    def get_upcoming_games(self, year: Optional[int] = None) -> pd.DataFrame:
        """Fetch upcoming (unplayed) games."""
        year = year or settings.data.current_season
        return self.get_games(year=year, complete=0)

    def get_season_games(self, year: int) -> pd.DataFrame:
        """Fetch all games for a complete season."""
        return self.get_games(year=year)

    # ── Tips (Predictions from various models) ────────────────────────

    def get_tips(
        self,
        year: Optional[int] = None,
        round_num: Optional[int] = None,
        source: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Fetch tips (predictions) from Squiggle's aggregated models.

        Useful for benchmarking our model against others.

        Args:
            year: Season year
            round_num: Round number
            source: Source ID (specific prediction model)

        Returns:
            DataFrame with tips including predicted margins and confidence.
        """
        params = {"q": "tips"}
        if year is not None:
            params["year"] = year
        if round_num is not None:
            params["round"] = round_num
        if source is not None:
            params["source"] = source

        data = self._request(params)
        tips = data.get("tips", [])

        if not tips:
            return pd.DataFrame()

        df = pd.DataFrame(tips)
        logger.info(f"Fetched {len(df)} tips from Squiggle")
        return df

    # ── Standings (Ladder) ────────────────────────────────────────────

    def get_standings(
        self,
        year: Optional[int] = None,
        round_num: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Fetch ladder/standings.

        Args:
            year: Season year
            round_num: Round number (for historical ladder at that point)

        Returns:
            DataFrame with ladder positions, wins, losses, percentage, etc.
        """
        params = {"q": "standings"}
        if year is not None:
            params["year"] = year
        if round_num is not None:
            params["round"] = round_num

        data = self._request(params)
        standings = data.get("standings", [])

        if not standings:
            return pd.DataFrame()

        df = pd.DataFrame(standings)
        logger.info(f"Fetched standings: {len(df)} teams")
        return df

    # ── Power Rankings ────────────────────────────────────────────────

    def get_power_rankings(
        self,
        year: Optional[int] = None,
        round_num: Optional[int] = None,
        source: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Fetch power rankings from various prediction models.

        Args:
            year: Season year
            round_num: Round number
            source: Source ID

        Returns:
            DataFrame with power rankings per team per source.
        """
        params = {"q": "pav"}
        if year is not None:
            params["year"] = year
        if round_num is not None:
            params["round"] = round_num
        if source is not None:
            params["source"] = source

        data = self._request(params)
        pav = data.get("pav", [])

        if not pav:
            return pd.DataFrame()

        df = pd.DataFrame(pav)
        logger.info(f"Fetched {len(df)} power ranking entries")
        return df

    # ── Sources (Prediction Model Info) ───────────────────────────────

    def get_sources(self) -> pd.DataFrame:
        """Fetch info about all prediction sources/models on Squiggle."""
        data = self._request({"q": "sources"})
        sources = data.get("sources", [])
        return pd.DataFrame(sources) if sources else pd.DataFrame()

    # ── Bulk Data Collection ──────────────────────────────────────────

    def collect_historical_data(
        self,
        start_year: Optional[int] = None,
        end_year: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Collect all match data for a range of years.

        Args:
            start_year: First year to collect (default: from settings)
            end_year: Last year to collect (default: current season)

        Returns:
            Combined DataFrame of all matches.
        """
        start = start_year or settings.data.data_start_year
        end = end_year or settings.data.current_season

        all_games = []
        for year in range(start, end + 1):
            logger.info(f"Collecting games for {year}...")
            df = self.get_season_games(year)
            if not df.empty:
                all_games.append(df)

        if not all_games:
            return pd.DataFrame()

        combined = pd.concat(all_games, ignore_index=True)
        logger.info(f"Total historical matches collected: {len(combined)}")
        return combined
