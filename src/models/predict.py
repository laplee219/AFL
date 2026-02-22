"""
Model Prediction Module

Generates predictions using the trained model ensemble.
Combines individual model outputs into calibrated probabilities.
"""

import numpy as np
import pandas as pd

from src.models.train import AFLModel
from src.utils.helpers import get_logger

logger = get_logger(__name__)


class Predictor:
    """
    Generate match predictions using the ensemble model.

    Combines XGBoost, LightGBM, and Logistic Regression outputs
    with configurable weights.
    """

    def __init__(
        self,
        model: AFLModel,
        xgb_weight: float = 0.4,
        lgb_weight: float = 0.4,
        lr_weight: float = 0.2,
    ):
        self.model = model
        self.weights = {
            "xgb": xgb_weight,
            "lgb": lgb_weight,
            "lr": lr_weight,
        }

    def predict_match(self, features: np.ndarray) -> dict:
        """
        Predict a single match outcome.

        Args:
            features: 1D array of feature values (same order as feature_names)

        Returns:
            Dict with predicted margin, win probability, and individual model outputs.
        """
        X = features.reshape(1, -1).astype(np.float32)
        results = {}

        # ── XGBoost predictions ──────────────────────────────────────
        if self.model.xgb_margin is not None:
            results["xgb_margin"] = float(self.model.xgb_margin.predict(X)[0])
        if self.model.xgb_cls is not None:
            results["xgb_prob"] = float(self.model.xgb_cls.predict_proba(X)[0, 1])

        # ── LightGBM predictions ─────────────────────────────────────
        if self.model.lgb_margin is not None:
            if hasattr(self.model.lgb_margin, "predict"):
                pred = self.model.lgb_margin.predict(X)
                results["lgb_margin"] = float(pred[0]) if len(pred) > 0 else 0.0
            else:
                results["lgb_margin"] = 0.0

        if self.model.lgb_cls is not None:
            if hasattr(self.model.lgb_cls, "predict"):
                pred = self.model.lgb_cls.predict(X)
                # LightGBM Booster.predict returns probability directly
                results["lgb_prob"] = float(pred[0]) if len(pred) > 0 else 0.5
            else:
                results["lgb_prob"] = 0.5

        # ── Logistic Regression predictions ──────────────────────────
        if self.model.lr_cls is not None and self.model.scaler is not None:
            X_scaled = self.model.scaler.transform(X)
            results["lr_prob"] = float(self.model.lr_cls.predict_proba(X_scaled)[0, 1])

        if self.model.ridge_margin is not None and self.model.scaler is not None:
            X_scaled = self.model.scaler.transform(X)
            results["ridge_margin"] = float(self.model.ridge_margin.predict(X_scaled)[0])

        # ── Ensemble combination ─────────────────────────────────────
        # Margin ensemble
        margins = []
        margin_weights = []
        for key, w_key in [("xgb_margin", "xgb"), ("lgb_margin", "lgb"), ("ridge_margin", "lr")]:
            if key in results:
                margins.append(results[key])
                margin_weights.append(self.weights[w_key])

        if margins:
            w = np.array(margin_weights)
            w = w / w.sum()
            results["ensemble_margin"] = float(np.average(margins, weights=w))
        else:
            results["ensemble_margin"] = 0.0

        # Probability ensemble
        probs = []
        prob_weights = []
        for key, w_key in [("xgb_prob", "xgb"), ("lgb_prob", "lgb"), ("lr_prob", "lr")]:
            if key in results:
                probs.append(results[key])
                prob_weights.append(self.weights[w_key])

        if probs:
            w = np.array(prob_weights)
            w = w / w.sum()
            results["ensemble_prob"] = float(np.average(probs, weights=w))
        else:
            results["ensemble_prob"] = 0.5

        # Predicted winner
        results["predicted_home_win"] = results["ensemble_prob"] > 0.5
        results["confidence"] = abs(results["ensemble_prob"] - 0.5) * 2  # 0-1 scale

        return results

    def predict_round(
        self,
        feature_matrix: pd.DataFrame,
        year: int,
        round_num: int,
    ) -> pd.DataFrame:
        """
        Generate predictions for all matches in a round.

        Args:
            feature_matrix: Full feature matrix (must contain the target round)
            year: Season year
            round_num: Round number

        Returns:
            DataFrame with predictions for each match.
        """
        from src.preprocessing.dataset import get_feature_columns, NON_FEATURE_COLS

        round_matches = feature_matrix[
            (feature_matrix["year"] == year)
            & (feature_matrix["round"] == round_num)
        ].copy()

        if round_matches.empty:
            logger.warning(f"No matches found for {year} round {round_num}")
            return pd.DataFrame()

        feature_cols = [c for c in self.model.feature_names if c in round_matches.columns]
        missing_cols = set(self.model.feature_names) - set(feature_cols)
        if missing_cols:
            logger.warning(f"Missing {len(missing_cols)} features, filling with medians")
            for col in missing_cols:
                round_matches[col] = 0.0

        predictions = []
        for idx, row in round_matches.iterrows():
            features = row[self.model.feature_names].values.astype(np.float32)

            # Fill NaN values with column medians
            if self.model.col_medians is not None:
                nan_mask = np.isnan(features)
                features[nan_mask] = self.model.col_medians[nan_mask]

            pred = self.predict_match(features)
            pred["match_id"] = row.get("match_id")
            pred["year"] = year
            pred["round"] = round_num
            pred["home_team"] = row["home_team"]
            pred["away_team"] = row["away_team"]
            pred["venue"] = row.get("venue", "")

            predictions.append(pred)

        result = pd.DataFrame(predictions)

        # Sort by confidence (most confident first)
        result = result.sort_values("confidence", ascending=False).reset_index(drop=True)

        logger.info(
            f"Predictions for {year} R{round_num}: "
            f"{len(result)} matches, avg confidence: {result['confidence'].mean():.1%}"
        )
        return result

    def predict_upcoming(self, feature_matrix: pd.DataFrame) -> pd.DataFrame:
        """
        Predict all upcoming (incomplete) matches in the feature matrix.
        """
        upcoming = feature_matrix[
            feature_matrix.get("target_margin", pd.Series(dtype=float)).isna()
            | ~feature_matrix.get("is_complete", pd.Series(True, index=feature_matrix.index))
        ]

        if upcoming.empty:
            logger.info("No upcoming matches to predict")
            return pd.DataFrame()

        all_preds = []
        for (year, round_num), group in upcoming.groupby(["year", "round"]):
            preds = self.predict_round(feature_matrix, int(year), int(round_num))
            all_preds.append(preds)

        if all_preds:
            return pd.concat(all_preds, ignore_index=True)
        return pd.DataFrame()


def format_predictions(predictions: pd.DataFrame) -> str:
    """Format predictions as a readable string for CLI output."""
    if predictions.empty:
        return "No predictions available."

    lines = []
    lines.append("=" * 70)
    lines.append(f"  AFL MATCH PREDICTIONS — {predictions['year'].iloc[0]} Round {predictions['round'].iloc[0]}")
    lines.append("=" * 70)

    for _, row in predictions.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        margin = row["ensemble_margin"]
        prob = row["ensemble_prob"]
        conf = row["confidence"]

        if margin > 0:
            winner = home
            win_prob = prob
        else:
            winner = away
            win_prob = 1 - prob

        bar_len = int(conf * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)

        lines.append(f"\n  {home} vs {away}")
        lines.append(f"  Predicted: {winner} by {abs(margin):.0f} points")
        lines.append(f"  Win Prob:  {win_prob:.1%}  [{bar}]")
        lines.append(f"  Venue:     {row.get('venue', 'TBD')}")

    lines.append("\n" + "=" * 70)
    return "\n".join(lines)
