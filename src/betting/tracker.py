"""
Bet Tracker Module

Tracks all bets placed, calculates running ROI, yield, and bankroll.
Stores bet history in the database for performance analysis.
"""

from typing import Optional

import numpy as np
import pandas as pd

from config.settings import settings
from src.utils.helpers import (
    current_timestamp,
    df_from_db,
    execute_db,
    get_db_connection,
    get_logger,
)

logger = get_logger(__name__)


class BetTracker:
    """
    Tracks betting performance and manages bankroll.
    """

    def __init__(self, initial_bankroll: float = None):
        self.initial_bankroll = initial_bankroll or settings.betting.initial_bankroll
        self._bankroll = self.initial_bankroll

    @property
    def bankroll(self) -> float:
        """Current bankroll (loaded from DB if available)."""
        try:
            df = df_from_db(
                "SELECT bankroll_after FROM bets ORDER BY id DESC LIMIT 1"
            )
            if not df.empty:
                return float(df.iloc[0]["bankroll_after"])
        except Exception:
            pass
        return self._bankroll

    @property
    def stop_loss_triggered(self) -> bool:
        """Check if bankroll has dropped below stop-loss threshold."""
        return self.bankroll < self.initial_bankroll * settings.betting.stop_loss_fraction

    def place_bet(
        self,
        match_id: int,
        year: int,
        round_num: int,
        team: str,
        bet_type: str,
        model_prob: float,
        bookmaker_odds: float,
        stake: float,
    ) -> dict:
        """
        Record a bet placement.

        Args:
            match_id: Match identifier
            year: Season year
            round_num: Round number
            team: Team bet on
            bet_type: Type of bet (e.g., 'home_win', 'away_win')
            model_prob: Model's estimated probability
            bookmaker_odds: Decimal odds
            stake: Amount staked

        Returns:
            Dict with bet details
        """
        if self.stop_loss_triggered:
            logger.warning(
                f"STOP LOSS: Bankroll (${self.bankroll:.2f}) below threshold. Bet rejected."
            )
            return {"status": "rejected", "reason": "stop_loss"}

        bookmaker_prob = 1.0 / bookmaker_odds if bookmaker_odds > 0 else 0
        ev = (model_prob * bookmaker_odds) - 1.0

        bet = {
            "match_id": match_id,
            "year": year,
            "round": round_num,
            "team": team,
            "bet_type": bet_type,
            "model_prob": model_prob,
            "bookmaker_prob": bookmaker_prob,
            "bookmaker_odds": bookmaker_odds,
            "expected_value": ev,
            "kelly_fraction": stake / self.bankroll if self.bankroll > 0 else 0,
            "stake": stake,
            "result": None,
            "profit_loss": None,
            "bankroll_after": None,
        }

        # Save to database
        conn = get_db_connection()
        conn.execute("""
            INSERT INTO bets 
            (match_id, year, round, team, bet_type, model_prob, bookmaker_prob,
             bookmaker_odds, expected_value, kelly_fraction, stake, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            match_id, year, round_num, team, bet_type,
            model_prob, bookmaker_prob, bookmaker_odds,
            ev, bet["kelly_fraction"], stake, current_timestamp(),
        ))
        conn.commit()
        conn.close()

        logger.info(
            f"Bet placed: {team} @ ${bookmaker_odds:.2f}, "
            f"stake=${stake:.2f}, EV={ev:+.1%}"
        )
        return {"status": "placed", **bet}

    def settle_bet(self, bet_id: int, won: bool):
        """
        Settle an open bet with the actual result.

        Args:
            bet_id: Database ID of the bet
            won: Whether the bet won
        """
        conn = get_db_connection()
        cursor = conn.cursor()

        # Get bet details
        cursor.execute("SELECT * FROM bets WHERE id = ?", (bet_id,))
        row = cursor.fetchone()
        if not row:
            logger.error(f"Bet {bet_id} not found")
            conn.close()
            return

        stake = row["stake"]
        odds = row["bookmaker_odds"]

        if won:
            profit = stake * (odds - 1.0)
            result = "won"
        else:
            profit = -stake
            result = "lost"

        current_bankroll = self.bankroll + profit

        cursor.execute("""
            UPDATE bets 
            SET result = ?, profit_loss = ?, bankroll_after = ?
            WHERE id = ?
        """, (result, profit, current_bankroll, bet_id))

        conn.commit()
        conn.close()

        self._bankroll = current_bankroll
        logger.info(
            f"Bet {bet_id} {result}: P/L=${profit:+.2f}, "
            f"bankroll=${current_bankroll:.2f}"
        )

    def settle_round(self, year: int, round_num: int, results: pd.DataFrame):
        """
        Settle all bets for a completed round.

        Args:
            year: Season year
            round_num: Round number
            results: DataFrame with actual match results (home_team, away_team, winner)
        """
        conn = get_db_connection()
        open_bets = pd.read_sql_query(
            "SELECT * FROM bets WHERE year = ? AND round = ? AND result IS NULL",
            conn, params=(year, round_num),
        )
        conn.close()

        if open_bets.empty:
            logger.info(f"No open bets for {year} R{round_num}")
            return

        for _, bet in open_bets.iterrows():
            # Find the matching result
            match = results[
                ((results["home_team"] == bet["team"]) | (results["away_team"] == bet["team"]))
            ]

            if match.empty:
                logger.warning(f"No result found for bet {bet['id']} on {bet['team']}")
                continue

            match = match.iloc[0]
            actual_winner = match.get("winner", "")

            if bet["bet_type"] == "home_win":
                won = match["home_team"] == actual_winner
            elif bet["bet_type"] == "away_win":
                won = match["away_team"] == actual_winner
            else:
                won = bet["team"] == actual_winner

            self.settle_bet(int(bet["id"]), won)

    # ── Performance Analytics ────────────────────────────────────────

    def get_performance(self) -> dict:
        """Get overall betting performance metrics."""
        try:
            bets = df_from_db("SELECT * FROM bets WHERE result IS NOT NULL")
        except Exception:
            return self._empty_performance()

        if bets.empty:
            return self._empty_performance()

        total_staked = bets["stake"].sum()
        total_profit = bets["profit_loss"].sum()
        n_bets = len(bets)
        n_wins = (bets["result"] == "won").sum()

        return {
            "n_bets": n_bets,
            "n_wins": int(n_wins),
            "n_losses": n_bets - int(n_wins),
            "win_rate": n_wins / n_bets if n_bets > 0 else 0,
            "total_staked": round(total_staked, 2),
            "total_profit": round(total_profit, 2),
            "roi": total_profit / total_staked if total_staked > 0 else 0,
            "yield_pct": (total_profit / total_staked * 100) if total_staked > 0 else 0,
            "current_bankroll": self.bankroll,
            "initial_bankroll": self.initial_bankroll,
            "bankroll_growth": (self.bankroll - self.initial_bankroll) / self.initial_bankroll,
            "avg_stake": total_staked / n_bets if n_bets > 0 else 0,
            "avg_odds": bets["bookmaker_odds"].mean(),
            "avg_ev": bets["expected_value"].mean(),
            "max_drawdown": self._calculate_max_drawdown(bets),
        }

    def _calculate_max_drawdown(self, bets: pd.DataFrame) -> float:
        """Calculate maximum drawdown from bankroll history."""
        if "bankroll_after" not in bets.columns or bets.empty:
            return 0.0

        bankroll_series = bets["bankroll_after"].dropna()
        if bankroll_series.empty:
            return 0.0

        peak = bankroll_series.expanding().max()
        drawdown = (bankroll_series - peak) / peak
        return float(drawdown.min()) if len(drawdown) > 0 else 0.0

    def _empty_performance(self) -> dict:
        return {
            "n_bets": 0, "n_wins": 0, "n_losses": 0, "win_rate": 0,
            "total_staked": 0, "total_profit": 0, "roi": 0, "yield_pct": 0,
            "current_bankroll": self.bankroll, "initial_bankroll": self.initial_bankroll,
            "bankroll_growth": 0, "avg_stake": 0, "avg_odds": 0, "avg_ev": 0,
            "max_drawdown": 0,
        }

    def get_bet_history(self, limit: int = 50) -> pd.DataFrame:
        """Get recent bet history."""
        try:
            return df_from_db(
                "SELECT * FROM bets ORDER BY id DESC LIMIT ?", (limit,)
            )
        except Exception:
            return pd.DataFrame()

    def get_clv_summary(self) -> dict:
        """
        Compute aggregate CLV (Closing Line Value) across settled bets.

        Compares each bet's bookmaker_odds at placement time with the closing
        odds snapshot stored for that round.  Positive CLV = model was sharper
        than the market at close.

        Returns:
            dict with avg_clv_pp, n_clv_bets, pct_positive_clv, clv_by_round
        """
        try:
            bets = df_from_db(
                "SELECT * FROM bets WHERE result IS NOT NULL"
            )
        except Exception:
            return {}

        if bets.empty:
            return {}

        try:
            snapshots = df_from_db(
                "SELECT * FROM odds_snapshots WHERE snapshot_type = 'closing'"
            )
        except Exception:
            return {}

        if snapshots.empty:
            return {}

        clv_records = []
        for _, bet in bets.iterrows():
            yr = int(bet["year"])
            rnd = int(bet["round"])
            team = bet["team"]
            placement_odds = float(bet["bookmaker_odds"])

            # Find matching closing odds snapshot
            snap = snapshots[
                (snapshots["year"] == yr) & (snapshots["round"] == rnd)
            ]
            if snap.empty:
                continue

            # Determine if team was home or away
            home_snap = snap[snap["home_team"] == team]
            away_snap = snap[snap["away_team"] == team]

            if not home_snap.empty:
                closing_odds = float(home_snap.iloc[0].get("home_odds", 0) or 0)
            elif not away_snap.empty:
                closing_odds = float(away_snap.iloc[0].get("away_odds", 0) or 0)
            else:
                continue

            if closing_odds <= 1 or placement_odds <= 1:
                continue

            # CLV = 1/closing_odds - 1/placement_odds
            # Positive = we got better odds than closing
            placement_implied = 1.0 / placement_odds
            closing_implied = 1.0 / closing_odds
            clv_pp = placement_implied - closing_implied  # negative if closing tightened (we got better price early)
            # Alternative: odds-based CLV (did our placement odds beat closing?)
            clv_odds = (placement_odds - closing_odds) / closing_odds

            clv_records.append({
                "year": yr,
                "round": rnd,
                "team": team,
                "placement_odds": placement_odds,
                "closing_odds": closing_odds,
                "clv_pp": clv_pp,
                "clv_odds_pct": clv_odds,
                "got_better_price": placement_odds > closing_odds,
            })

        if not clv_records:
            return {}

        import pandas as pd
        clv_df = pd.DataFrame(clv_records)
        return {
            "n_clv_bets": len(clv_df),
            "avg_clv_odds_pct": float(clv_df["clv_odds_pct"].mean()),
            "pct_better_price": float(clv_df["got_better_price"].mean()),
            "avg_placement_odds": float(clv_df["placement_odds"].mean()),
            "avg_closing_odds": float(clv_df["closing_odds"].mean()),
        }

    def format_performance(self) -> str:
        """Format performance as a readable string."""
        p = self.get_performance()

        lines = [
            "=" * 50,
            "  BETTING PERFORMANCE",
            "=" * 50,
            f"  Bankroll:   ${p['current_bankroll']:.2f} (started: ${p['initial_bankroll']:.2f})",
            f"  Growth:     {p['bankroll_growth']:+.1%}",
            f"  Bets:       {p['n_bets']} ({p['n_wins']}W / {p['n_losses']}L)",
            f"  Win Rate:   {p['win_rate']:.1%}",
            f"  ROI:        {p['roi']:+.1%}",
            f"  Yield:      {p['yield_pct']:+.1f}%",
            f"  Total P/L:  ${p['total_profit']:+.2f}",
            f"  Avg Stake:  ${p['avg_stake']:.2f}",
            f"  Avg Odds:   ${p['avg_odds']:.2f}",
            f"  Avg EV:     {p['avg_ev']:+.1%}",
            f"  Max DD:     {p['max_drawdown']:.1%}",
        ]

        # Append CLV summary if closing odds data exists
        clv = self.get_clv_summary()
        if clv:
            lines.append("  " + "─" * 48)
            lines.append("  CLOSING LINE VALUE (CLV)")
            lines.append(f"  CLV Bets:   {clv['n_clv_bets']}")
            lines.append(f"  Avg CLV:    {clv['avg_clv_odds_pct']:+.1%} (odds improvement)")
            lines.append(f"  Beat Close: {clv['pct_better_price']:.0%} of bets got better price")
            lines.append(f"  Avg Placed: ${clv['avg_placement_odds']:.2f}  →  Avg Close: ${clv['avg_closing_odds']:.2f}")

        lines.append("=" * 50)
        return "\n".join(lines)
