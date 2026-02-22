"""
Dataset Builder

Constructs train/validation/test splits from the feature matrix,
handling time-series splitting and sample weighting.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from config.settings import settings
from src.utils.helpers import get_logger

logger = get_logger(__name__)

# Columns that are metadata / targets, not features
NON_FEATURE_COLS = {
    "match_id", "year", "round", "date",
    "home_team", "away_team", "venue",
    "target_margin", "target_home_win",
    "is_complete", "winner", "home_score", "away_score",
    "complete", "round_name", "margin", "total_score", "home_win",
}


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Get the list of feature columns (excluding metadata and targets)."""
    return [c for c in df.columns if c not in NON_FEATURE_COLS]


def compute_sample_weights(
    df: pd.DataFrame,
    decay_per_season: float = None,
    current_year: int = None,
) -> np.ndarray:
    """
    Compute sample weights with exponential decay by season recency.

    More recent seasons get higher weight.

    Args:
        df: Feature matrix with 'year' column
        decay_per_season: Decay factor per season (default from settings)
        current_year: The reference year (default from settings)

    Returns:
        Array of sample weights
    """
    decay = decay_per_season or settings.data.sample_weight_decay
    ref_year = current_year or settings.data.current_season

    years_ago = ref_year - df["year"]
    weights = decay ** years_ago.values

    # Normalize so mean weight = 1
    weights = weights / weights.mean()

    return weights


def build_train_test_split(
    feature_matrix: pd.DataFrame,
    test_year: int = None,
    val_year: int = None,
) -> dict:
    """
    Build chronological train/validation/test splits.

    Args:
        feature_matrix: Full feature matrix from build_feature_matrix()
        test_year: Year to use as test set (default: current season)
        val_year: Year to use as validation set (default: test_year - 1)

    Returns:
        Dict with keys: X_train, y_train_margin, y_train_cls,
        X_val, y_val_margin, y_val_cls, X_test, y_test_margin, y_test_cls,
        sample_weights, feature_names, meta_train, meta_val, meta_test
    """
    test_yr = test_year or settings.data.current_season
    val_yr = val_year or (test_yr - 1)

    # Only use completed matches with valid targets
    completed = feature_matrix[
        feature_matrix["target_margin"].notna()
    ].copy()

    feature_cols = get_feature_columns(completed)

    # Remove any remaining non-numeric columns from features
    numeric_features = []
    for col in feature_cols:
        if completed[col].dtype in [np.float64, np.int64, float, int]:
            numeric_features.append(col)
        else:
            try:
                completed[col] = pd.to_numeric(completed[col], errors="coerce")
                numeric_features.append(col)
            except (ValueError, TypeError):
                logger.debug(f"Dropping non-numeric feature: {col}")

    feature_cols = numeric_features

    # Split by year
    train_mask = completed["year"] < val_yr
    val_mask = completed["year"] == val_yr
    test_mask = completed["year"] == test_yr

    train = completed[train_mask]
    val = completed[val_mask]
    test = completed[test_mask]

    logger.info(
        f"Data split: train={len(train)} ({train['year'].min()}-{train['year'].max()}), "
        f"val={len(val)} ({val_yr}), test={len(test)} ({test_yr})"
    )

    # Build feature matrices
    X_train = train[feature_cols].values.astype(np.float32)
    X_val = val[feature_cols].values.astype(np.float32) if len(val) > 0 else np.array([])
    X_test = test[feature_cols].values.astype(np.float32) if len(test) > 0 else np.array([])

    # Target: margin (regression)
    y_train_margin = train["target_margin"].values.astype(np.float32)
    y_val_margin = val["target_margin"].values.astype(np.float32) if len(val) > 0 else np.array([])
    y_test_margin = test["target_margin"].values.astype(np.float32) if len(test) > 0 else np.array([])

    # Target: home win (classification)
    y_train_cls = train["target_home_win"].values.astype(np.float32)
    y_val_cls = val["target_home_win"].values.astype(np.float32) if len(val) > 0 else np.array([])
    y_test_cls = test["target_home_win"].values.astype(np.float32) if len(test) > 0 else np.array([])

    # Sample weights (recency weighting for training data)
    weights = compute_sample_weights(train)

    # Handle NaN values — fill with column median from training set
    col_medians = np.nanmedian(X_train, axis=0)
    for i in range(X_train.shape[1]):
        nan_mask = np.isnan(X_train[:, i])
        if nan_mask.any():
            X_train[nan_mask, i] = col_medians[i]

    # Apply same medians to val/test
    for X in [X_val, X_test]:
        if len(X) > 0:
            for i in range(X.shape[1]):
                nan_mask = np.isnan(X[:, i])
                if nan_mask.any():
                    X[nan_mask, i] = col_medians[i] if not np.isnan(col_medians[i]) else 0.0

    result = {
        "X_train": X_train,
        "y_train_margin": y_train_margin,
        "y_train_cls": y_train_cls,
        "X_val": X_val,
        "y_val_margin": y_val_margin,
        "y_val_cls": y_val_cls,
        "X_test": X_test,
        "y_test_margin": y_test_margin,
        "y_test_cls": y_test_cls,
        "sample_weights": weights,
        "feature_names": feature_cols,
        "col_medians": col_medians,
        "meta_train": train[["match_id", "year", "round", "home_team", "away_team"]].reset_index(drop=True),
        "meta_val": val[["match_id", "year", "round", "home_team", "away_team"]].reset_index(drop=True),
        "meta_test": test[["match_id", "year", "round", "home_team", "away_team"]].reset_index(drop=True),
    }

    logger.info(f"Features: {len(feature_cols)}, NaN columns filled: {np.isnan(col_medians).sum()} all-NaN")
    return result


def build_leave_one_season_out_splits(
    feature_matrix: pd.DataFrame,
    min_train_years: int = 3,
) -> list[dict]:
    """
    Build leave-one-season-out cross-validation splits.

    Each split uses one season as test, previous seasons as train.

    Args:
        feature_matrix: Full feature matrix
        min_train_years: Minimum number of training years

    Returns:
        List of split dicts (same format as build_train_test_split)
    """
    years = sorted(feature_matrix["year"].unique())
    splits = []

    for i, test_year in enumerate(years):
        if i < min_train_years:
            continue

        val_year = years[i - 1] if i > 0 else test_year
        split = build_train_test_split(
            feature_matrix,
            test_year=test_year,
            val_year=val_year,
        )
        split["fold_test_year"] = test_year
        splits.append(split)

    logger.info(f"Created {len(splits)} leave-one-season-out splits")
    return splits
