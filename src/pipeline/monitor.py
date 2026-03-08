"""
Model Performance Monitor

Tracks prediction accuracy over time, detects performance degradation,
and triggers retraining when needed.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

from config.settings import settings
from src.utils.helpers import (
    current_timestamp,
    df_from_db,
    get_db_connection,
    get_logger,
)

logger = get_logger(__name__)


class ModelMonitor:
    """
    Monitors model performance and detects when retraining is needed.
    """

    def __init__(self):
        self.baseline_logloss: float = 0.69  # ln(2) — coin flip baseline
        self.baseline_accuracy: float = 0.50

    def log_round_performance(
        self,
        year: int,
        round_num: int,
        predictions: pd.DataFrame,
        actuals: pd.DataFrame,
        model_version: str = "",
    ):
        """
        Log model performance for a completed round.

        Args:
            year: Season year
            round_num: Round number
            predictions: DataFrame with predicted probabilities
            actuals: DataFrame with actual results
        """
        # Merge predictions with actual results
        merged = predictions.merge(
            actuals[["match_id", "home_win", "margin"]].rename(
                columns={"home_win": "actual_home_win", "margin": "actual_margin"}
            ),
            on="match_id",
            how="inner",
        )

        if merged.empty:
            logger.warning(f"No matches to evaluate for {year} R{round_num}")
            return

        pred_probs = merged["ensemble_prob"].values
        actual = merged["actual_home_win"].astype(int).values
        pred_cls = (pred_probs > 0.5).astype(int)

        # Calculate metrics
        n_correct = int((pred_cls == actual).sum())
        n_total = len(merged)
        accuracy = accuracy_score(actual, pred_cls)

        try:
            ll = log_loss(actual, pred_probs)
        except ValueError:
            ll = None

        try:
            brier = brier_score_loss(actual, pred_probs)
        except ValueError:
            brier = None

        margin_mae = None
        if "ensemble_margin" in merged.columns and "actual_margin" in merged.columns:
            margin_mae = float(
                np.mean(np.abs(merged["ensemble_margin"] - merged["actual_margin"]))
            )

        # Save to database
        conn = get_db_connection()
        conn.execute("""
            INSERT OR REPLACE INTO monitoring_metrics
            (year, round, model_version, accuracy, log_loss, brier_score,
             margin_mae, n_predictions, n_correct, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            year, round_num, model_version,
            accuracy, ll, brier, margin_mae,
            n_total, n_correct, current_timestamp(),
        ))
        conn.commit()
        conn.close()

        logger.info(
            f"R{round_num} performance: {n_correct}/{n_total} correct ({accuracy:.0%}), "
            f"log_loss={ll:.4f}" if ll else f"R{round_num}: {n_correct}/{n_total} ({accuracy:.0%})"
        )

    def check_retrain_needed(
        self,
        year: int,
        current_round: int,
    ) -> dict:
        """
        Check if model retraining is needed based on recent performance.

        Returns:
            Dict with 'should_retrain', 'reason', and diagnostic info
        """
        result = {
            "should_retrain": False,
            "reason": "",
            "diagnostics": {},
        }

        try:
            metrics = df_from_db(
                "SELECT * FROM monitoring_metrics WHERE year = ? ORDER BY round DESC",
                (year,),
            )
        except Exception:
            return result

        if metrics.empty:
            result["reason"] = "Insufficient data for monitoring"
            return result

        if len(metrics) < 2:
            result["reason"] = "Preliminary monitoring (1 evaluated round)"
            result["diagnostics"].update({
                "rounds_evaluated": len(metrics),
                "overall_accuracy": float(metrics["accuracy"].mean()),
                "recent_3_accuracy": float(metrics.head(3)["accuracy"].mean()),
                "best_round": int(metrics.loc[metrics["accuracy"].idxmax(), "round"]),
                "worst_round": int(metrics.loc[metrics["accuracy"].idxmin(), "round"]),
                "avg_margin_mae": float(metrics["margin_mae"].mean()) if "margin_mae" in metrics.columns else None,
            })
            return result

        # ── Check 1: Consecutive poor rounds ─────────────────────────
        consecutive_threshold = settings.model.consecutive_poor_rounds
        accuracy_threshold = settings.model.accuracy_alert_threshold

        recent = metrics.head(consecutive_threshold)
        if len(recent) >= consecutive_threshold:
            if all(recent["accuracy"] < accuracy_threshold):
                result["should_retrain"] = True
                result["reason"] = (
                    f"Accuracy below {accuracy_threshold:.0%} for "
                    f"{consecutive_threshold} consecutive rounds"
                )
                result["diagnostics"]["recent_accuracies"] = recent["accuracy"].tolist()

        # ── Check 2: Log loss degradation ────────────────────────────
        if not result["should_retrain"] and "log_loss" in metrics.columns:
            recent_ll = metrics.head(5)["log_loss"].mean()
            if pd.notna(recent_ll) and recent_ll > self.baseline_logloss * settings.model.logloss_alert_multiplier:
                result["should_retrain"] = True
                result["reason"] = (
                    f"Log loss ({recent_ll:.4f}) exceeds {settings.model.logloss_alert_multiplier}× "
                    f"baseline ({self.baseline_logloss:.4f})"
                )

        # ── Check 3: Periodic retrain schedule ───────────────────────
        if not result["should_retrain"]:
            rounds_since_train = current_round % settings.model.retrain_every_n_rounds
            if rounds_since_train == 0 and current_round > 0:
                result["should_retrain"] = True
                result["reason"] = f"Scheduled retrain every {settings.model.retrain_every_n_rounds} rounds"

        # ── Diagnostics ──────────────────────────────────────────────
        result["diagnostics"].update({
            "rounds_evaluated": len(metrics),
            "overall_accuracy": float(metrics["accuracy"].mean()),
            "recent_3_accuracy": float(metrics.head(3)["accuracy"].mean()),
            "best_round": int(metrics.loc[metrics["accuracy"].idxmax(), "round"]),
            "worst_round": int(metrics.loc[metrics["accuracy"].idxmin(), "round"]),
            "avg_margin_mae": float(metrics["margin_mae"].mean()) if "margin_mae" in metrics.columns else None,
        })

        if result["should_retrain"]:
            logger.warning(f"RETRAIN TRIGGERED: {result['reason']}")
        else:
            logger.info(f"Model OK — accuracy: {result['diagnostics']['overall_accuracy']:.0%}")

        return result

    def get_performance_trend(self, year: int = None) -> pd.DataFrame:
        """Get performance metrics trend over time."""
        query = "SELECT * FROM monitoring_metrics"
        params = ()
        if year:
            query += " WHERE year = ?"
            params = (year,)
        query += " ORDER BY year, round"

        try:
            return df_from_db(query, params)
        except Exception:
            return pd.DataFrame()

    def format_status(self, year: int, current_round: int) -> str:
        """Format monitoring status as a readable string."""
        check = self.check_retrain_needed(year, current_round)
        diag = check["diagnostics"]

        status_icon = "🔴" if check["should_retrain"] else "🟢"

        lines = [
            "=" * 50,
            "  MODEL HEALTH MONITOR",
            "=" * 50,
            f"  Status: {status_icon} {'RETRAIN NEEDED' if check['should_retrain'] else 'OK'}",
        ]

        if check["reason"]:
            lines.append(f"  Reason: {check['reason']}")

        if diag:
            lines.extend([
                "",
                f"  Rounds evaluated:     {diag.get('rounds_evaluated', 0)}",
                f"  Overall accuracy:     {diag.get('overall_accuracy', 0):.1%}",
                f"  Recent 3-round avg:   {diag.get('recent_3_accuracy', 0):.1%}",
                f"  Best round:           R{diag.get('best_round', '?')}",
                f"  Worst round:          R{diag.get('worst_round', '?')}",
            ])
            if diag.get("avg_margin_mae"):
                lines.append(f"  Avg margin MAE:       {diag['avg_margin_mae']:.1f} pts")

        lines.append("=" * 50)
        return "\n".join(lines)
