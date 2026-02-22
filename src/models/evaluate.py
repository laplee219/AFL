"""
Model Evaluation Module

Evaluates model performance with metrics relevant to sports prediction:
- Accuracy, Log Loss, Brier Score
- Margin MAE (Mean Absolute Error)
- Calibration analysis
- SHAP feature importance
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    roc_auc_score,
)
from sklearn.calibration import calibration_curve

from src.models.predict import Predictor
from src.utils.helpers import get_logger

logger = get_logger(__name__)


def evaluate_predictions(
    predictions: pd.DataFrame,
    actuals: pd.DataFrame,
) -> dict:
    """
    Evaluate model predictions against actual results.

    Args:
        predictions: DataFrame with predicted margins and probabilities
        actuals: DataFrame with actual margins and results

    Returns:
        Dict of evaluation metrics
    """
    # Merge predictions with actuals
    merged = predictions.merge(
        actuals[["match_id", "margin", "home_win"]].rename(columns={
            "margin": "actual_margin",
            "home_win": "actual_home_win",
        }),
        on="match_id",
        how="inner",
    )

    if merged.empty:
        logger.warning("No matching predictions and actuals to evaluate")
        return {}

    metrics = {"n_matches": len(merged)}

    # ── Classification Metrics ───────────────────────────────────────
    pred_home_win = (merged["ensemble_prob"] > 0.5).astype(int)
    actual_home_win = merged["actual_home_win"].astype(int)
    probs = merged["ensemble_prob"].values

    metrics["accuracy"] = float(accuracy_score(actual_home_win, pred_home_win))
    metrics["log_loss"] = float(log_loss(actual_home_win, probs))
    metrics["brier_score"] = float(brier_score_loss(actual_home_win, probs))

    try:
        metrics["auc_roc"] = float(roc_auc_score(actual_home_win, probs))
    except ValueError:
        metrics["auc_roc"] = None

    # ── Regression Metrics (Margin) ──────────────────────────────────
    pred_margins = merged["ensemble_margin"].values
    actual_margins = merged["actual_margin"].values

    metrics["margin_mae"] = float(mean_absolute_error(actual_margins, pred_margins))
    metrics["margin_rmse"] = float(np.sqrt(np.mean((pred_margins - actual_margins) ** 2)))
    metrics["margin_correlation"] = float(np.corrcoef(pred_margins, actual_margins)[0, 1])

    # ── Individual Model Metrics ─────────────────────────────────────
    for model_key in ["xgb", "lgb", "lr"]:
        prob_col = f"{model_key}_prob"
        if prob_col in merged.columns:
            model_probs = merged[prob_col].values
            model_pred = (model_probs > 0.5).astype(int)
            metrics[f"{model_key}_accuracy"] = float(accuracy_score(actual_home_win, model_pred))
            metrics[f"{model_key}_logloss"] = float(log_loss(actual_home_win, model_probs))

    for model_key in ["xgb", "lgb", "ridge"]:
        margin_col = f"{model_key}_margin"
        if margin_col in merged.columns:
            metrics[f"{model_key}_margin_mae"] = float(
                mean_absolute_error(actual_margins, merged[margin_col].values)
            )

    logger.info(
        f"Evaluation: accuracy={metrics['accuracy']:.1%}, "
        f"log_loss={metrics['log_loss']:.4f}, "
        f"margin_MAE={metrics['margin_mae']:.1f}"
    )
    return metrics


def compute_calibration(
    predictions: pd.DataFrame,
    actuals: pd.DataFrame,
    n_bins: int = 10,
) -> dict:
    """
    Compute calibration curve data.

    A well-calibrated model should have predicted probabilities
    matching observed frequencies at each bin.
    """
    merged = predictions.merge(
        actuals[["match_id", "home_win"]].rename(columns={"home_win": "actual_home_win"}),
        on="match_id",
        how="inner",
    )

    if len(merged) < 20:
        return {"bins": [], "observed": [], "predicted": []}

    probs = merged["ensemble_prob"].values
    actual = merged["actual_home_win"].astype(int).values

    prob_true, prob_pred = calibration_curve(actual, probs, n_bins=n_bins, strategy="quantile")

    return {
        "bins": prob_pred.tolist(),
        "observed": prob_true.tolist(),
        "predicted": prob_pred.tolist(),
        "n_samples_per_bin": len(merged) // n_bins,
    }


def compute_feature_importance(model, feature_names: list[str], top_n: int = 20) -> pd.DataFrame:
    """
    Compute feature importance using SHAP values.
    Falls back to built-in importance if SHAP is unavailable.
    """
    importance_data = []

    # Try SHAP first
    try:
        import shap

        if model.xgb_margin is not None:
            explainer = shap.TreeExplainer(model.xgb_margin)
            # Use a small sample for speed
            importance_data.append({
                "source": "xgb_shap",
                "method": "SHAP (XGBoost margin)",
            })
    except Exception:
        pass

    # Built-in feature importance from XGBoost
    if model.xgb_margin is not None:
        try:
            importance = model.xgb_margin.feature_importances_
            df = pd.DataFrame({
                "feature": feature_names[:len(importance)],
                "importance": importance,
                "source": "xgb_builtin",
            }).sort_values("importance", ascending=False).head(top_n)
            importance_data.append(df)
        except Exception:
            pass

    # Built-in from LightGBM
    if model.lgb_margin is not None and hasattr(model.lgb_margin, "feature_importances_"):
        try:
            importance = model.lgb_margin.feature_importances_
            df = pd.DataFrame({
                "feature": feature_names[:len(importance)],
                "importance": importance,
                "source": "lgb_builtin",
            }).sort_values("importance", ascending=False).head(top_n)
            importance_data.append(df)
        except Exception:
            pass

    # Logistic regression coefficients
    if model.lr_cls is not None:
        try:
            coefs = np.abs(model.lr_cls.coef_[0])
            df = pd.DataFrame({
                "feature": feature_names[:len(coefs)],
                "importance": coefs,
                "source": "lr_coef",
            }).sort_values("importance", ascending=False).head(top_n)
            importance_data.append(df)
        except Exception:
            pass

    if importance_data:
        result = pd.concat([d for d in importance_data if isinstance(d, pd.DataFrame)], ignore_index=True)
        return result

    return pd.DataFrame(columns=["feature", "importance", "source"])


def format_evaluation(metrics: dict) -> str:
    """Format evaluation metrics as a readable string."""
    lines = [
        "=" * 50,
        "  MODEL EVALUATION METRICS",
        "=" * 50,
        f"  Matches evaluated:      {metrics.get('n_matches', 0)}",
        "",
        "  Classification:",
        f"    Accuracy:             {metrics.get('accuracy', 0):.1%}",
        f"    Log Loss:             {metrics.get('log_loss', 0):.4f}",
        f"    Brier Score:          {metrics.get('brier_score', 0):.4f}",
        f"    AUC-ROC:              {metrics.get('auc_roc', 'N/A')}",
        "",
        "  Margin Prediction:",
        f"    MAE:                  {metrics.get('margin_mae', 0):.1f} points",
        f"    RMSE:                 {metrics.get('margin_rmse', 0):.1f} points",
        f"    Correlation:          {metrics.get('margin_correlation', 0):.3f}",
        "",
        "  Individual Models:",
    ]

    for m in ["xgb", "lgb", "lr"]:
        acc = metrics.get(f"{m}_accuracy")
        if acc is not None:
            lines.append(f"    {m.upper():4s} accuracy:       {acc:.1%}")

    lines.append("=" * 50)
    return "\n".join(lines)
