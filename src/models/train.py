"""
Model Training Module

Trains XGBoost, LightGBM, and Logistic Regression models for:
1. Margin prediction (regression)
2. Win probability (classification)

Supports full training and warm-start incremental updates.
"""

import json
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import brier_score_loss
from sklearn.preprocessing import StandardScaler

from config.settings import MODELS_DIR, settings
from src.utils.helpers import current_timestamp, get_logger

logger = get_logger(__name__)


class AFLModel:
    """
    Wrapper for the AFL prediction model ensemble.

    Contains:
    - XGBoost (margin regression + classification)
    - LightGBM (margin regression + classification)
    - Logistic Regression (baseline classification)
    - Ridge Regression (baseline margin)
    - StandardScaler for logistic/ridge
    """

    def __init__(self, version: str = None):
        self.version = version or f"v_{current_timestamp().replace(':', '-')}"
        self.xgb_margin = None
        self.xgb_cls = None
        self.lgb_margin = None
        self.lgb_cls = None
        self.lr_cls = None
        self.ridge_margin = None
        self.scaler = StandardScaler()
        self.feature_names: list[str] = []
        self.col_medians: Optional[np.ndarray] = None
        self.training_info: dict = {}
        self.calibrator = None          # fitted IsotonicRegression or LogisticRegression
        self.calibration_method: str = "platt"  # "isotonic" or "platt"
        self.margin_sigma: float = 30.0  # std dev of margin prediction error (computed from val residuals)
        self.calibration_temperature: float = 1.5  # logit-space temperature > 1 shrinks probability extremes toward 0.5

    # ── Full Training ────────────────────────────────────────────────

    def train(self, data: dict, use_optuna: bool = False):
        """
        Full model training pipeline.

        Args:
            data: Dict from build_train_test_split() with X_train, y_train_*, etc.
            use_optuna: Whether to use Optuna for hyperparameter optimization
        """
        X_train = data["X_train"]
        y_margin = data["y_train_margin"]
        y_cls = data["y_train_cls"]
        weights = data["sample_weights"]
        self.feature_names = data["feature_names"]
        self.col_medians = data.get("col_medians")

        X_val = data.get("X_val", np.array([]))
        y_val_margin = data.get("y_val_margin", np.array([]))
        y_val_cls = data.get("y_val_cls", np.array([]))

        has_val = len(X_val) > 0

        logger.info(f"Training models on {len(X_train)} samples, {len(self.feature_names)} features")

        # ── Hyperparameters ──────────────────────────────────────────
        if use_optuna:
            xgb_params, lgb_params = self._optimize_hyperparams(
                X_train, y_margin, y_cls, weights, X_val, y_val_margin, y_val_cls
            )
        else:
            xgb_params = {
                "max_depth": settings.model.max_depth,
                "learning_rate": settings.model.full_train_lr,
                "n_estimators": settings.model.n_estimators,
                "min_child_weight": 5,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_alpha": 1.0,
                "reg_lambda": 1.0,
                "random_state": 42,
            }
            lgb_params = {
                "max_depth": settings.model.max_depth,
                "learning_rate": settings.model.full_train_lr,
                "n_estimators": settings.model.n_estimators,
                "min_child_samples": 20,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_alpha": 1.0,
                "reg_lambda": 1.0,
                "random_state": 42,
                "verbose": -1,
            }

        # ── XGBoost Margin (Regression) ──────────────────────────────
        logger.info("Training XGBoost margin model...")
        self.xgb_margin = xgb.XGBRegressor(
            objective="reg:squarederror",
            eval_metric="mae",
            early_stopping_rounds=settings.model.early_stopping_rounds if has_val else None,
            **xgb_params,
        )
        eval_set = [(X_val, y_val_margin)] if has_val else None
        self.xgb_margin.fit(
            X_train, y_margin,
            sample_weight=weights,
            eval_set=eval_set,
            verbose=False,
        )

        # ── XGBoost Classification ───────────────────────────────────
        logger.info("Training XGBoost classification model...")
        self.xgb_cls = xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            early_stopping_rounds=settings.model.early_stopping_rounds if has_val else None,
            **xgb_params,
        )
        eval_set_cls = [(X_val, y_val_cls)] if has_val else None
        self.xgb_cls.fit(
            X_train, y_cls,
            sample_weight=weights,
            eval_set=eval_set_cls,
            verbose=False,
        )

        # ── LightGBM Margin (Regression) ─────────────────────────────
        logger.info("Training LightGBM margin model...")
        self.lgb_margin = lgb.LGBMRegressor(
            objective="regression",
            metric="mae",
            **lgb_params,
        )
        lgb_eval = [(X_val, y_val_margin)] if has_val else None
        self.lgb_margin.fit(
            X_train, y_margin,
            sample_weight=weights,
            eval_set=lgb_eval,
            callbacks=[lgb.early_stopping(settings.model.early_stopping_rounds)] if has_val else None,
        )

        # ── LightGBM Classification ──────────────────────────────────
        logger.info("Training LightGBM classification model...")
        self.lgb_cls = lgb.LGBMClassifier(
            objective="binary",
            metric="binary_logloss",
            **lgb_params,
        )
        lgb_eval_cls = [(X_val, y_val_cls)] if has_val else None
        self.lgb_cls.fit(
            X_train, y_cls,
            sample_weight=weights,
            eval_set=lgb_eval_cls,
            callbacks=[lgb.early_stopping(settings.model.early_stopping_rounds)] if has_val else None,
        )

        # ── Logistic Regression (Baseline) ───────────────────────────
        logger.info("Training Logistic Regression baseline...")
        X_scaled = self.scaler.fit_transform(X_train)
        self.lr_cls = LogisticRegression(
            max_iter=1000, C=1.0, random_state=42, solver="lbfgs"
        )
        self.lr_cls.fit(X_scaled, y_cls, sample_weight=weights)

        # ── Ridge Regression (Baseline margin) ───────────────────────
        logger.info("Training Ridge Regression baseline...")
        self.ridge_margin = Ridge(alpha=1.0)
        self.ridge_margin.fit(X_scaled, y_margin, sample_weight=weights)

        # ── Store training info ──────────────────────────────────────
        self.training_info = {
            "version": self.version,
            "n_train": len(X_train),
            "n_features": len(self.feature_names),
            "trained_at": current_timestamp(),
            "xgb_params": xgb_params,
            "lgb_params": lgb_params,
        }

        # ── Margin prediction sigma (from validation residuals) ─────────
        if has_val and len(X_val) >= 10:
            try:
                xgb_m = self.xgb_margin.predict(X_val.astype(np.float32))
                if hasattr(self.lgb_margin, "predict"):
                    lgb_m = np.array(self.lgb_margin.predict(X_val.astype(np.float32)), dtype=np.float64)
                else:
                    lgb_m = xgb_m
                X_val_sc = self.scaler.transform(X_val)
                ridge_m = self.ridge_margin.predict(X_val_sc)
                pred_margins = 0.4 * xgb_m + 0.4 * lgb_m + 0.2 * ridge_m
                residuals = y_val_margin.astype(np.float64) - pred_margins.astype(np.float64)
                self.margin_sigma = float(max(12.0, np.std(residuals)))
                logger.info(f"Margin sigma (val residuals): {self.margin_sigma:.1f} pts  "
                            f"(MAE: {np.mean(np.abs(residuals)):.1f} pts)")
            except Exception as e:
                logger.warning(f"Could not compute margin_sigma from residuals: {e}")

        # ── Probability Calibration (walk-forward OOF + val) ──────────────
        if has_val and len(X_val) >= 10:
            meta_train = data.get("meta_train")
            oof_probs, oof_labels = self._collect_oof_probs(
                X_train, y_cls.astype(int), weights, meta_train, n_folds=4
            ) if meta_train is not None else (np.array([]), np.array([]))

            val_probs = self._predict_ensemble_probs_batch(X_val)
            val_labels = y_val_cls.astype(int)

            if len(oof_probs) >= 50:
                all_probs = np.concatenate([oof_probs, val_probs])
                all_labels = np.concatenate([oof_labels, val_labels])
                logger.info(
                    f"Calibration: {len(oof_probs)} OOF + {len(val_probs)} val "
                    f"= {len(all_probs)} total points"
                )
            else:
                all_probs = val_probs
                all_labels = val_labels
                logger.info("Calibration: val set only (OOF collection skipped)")

            self._fit_calibrator_on_probs(all_probs, all_labels)

        logger.info(f"All models trained successfully (version: {self.version})")

    # ── Probability Calibration ──────────────────────────────────────

    def _predict_ensemble_probs_batch(self, X: np.ndarray) -> np.ndarray:
        """Compute raw weighted-ensemble win probability for a batch of samples."""
        probs = np.zeros(len(X), dtype=np.float64)
        total_w = 0.0
        xgb_w, lgb_w, lr_w = 0.4, 0.4, 0.2

        if self.xgb_cls is not None:
            probs += xgb_w * self.xgb_cls.predict_proba(X)[:, 1]
            total_w += xgb_w

        if self.lgb_cls is not None:
            try:
                raw = self.lgb_cls.predict(X)
                probs += lgb_w * np.asarray(raw, dtype=np.float64)
                total_w += lgb_w
            except Exception:
                pass

        if self.lr_cls is not None and self.scaler is not None:
            X_scaled = self.scaler.transform(X)
            probs += lr_w * self.lr_cls.predict_proba(X_scaled)[:, 1]
            total_w += lr_w

        if total_w > 0:
            probs /= total_w
        return np.clip(probs, 1e-6, 1 - 1e-6)

    def _collect_oof_probs(
        self,
        X: np.ndarray,
        y: np.ndarray,
        weights: np.ndarray,
        meta: "pd.DataFrame",
        n_folds: int = 4,
    ) -> tuple:
        """
        Walk-forward out-of-fold probability collection for calibration.

        Trains lightweight mini-models on expanding windows, predicts each
        held-out year, and returns concatenated OOF probs + labels.
        This gives a larger, unbiased calibration set than a single val year.
        """
        import pandas as pd
        years = sorted(meta["year"].unique())
        if len(years) < n_folds + 2:
            return np.array([]), np.array([])

        oof_years = years[-(n_folds):]
        oof_probs_list, oof_labels_list = [], []

        for test_yr in oof_years:
            tr_mask = (meta["year"] < test_yr).values
            te_mask = (meta["year"] == test_yr).values
            if tr_mask.sum() < 80 or te_mask.sum() < 8:
                continue

            X_tr, y_tr, w_tr = X[tr_mask], y[tr_mask], weights[tr_mask]
            X_te, y_te = X[te_mask], y[te_mask]

            # Fast mini-ensemble (fewer trees, no early stopping)
            mini_xgb = xgb.XGBClassifier(
                n_estimators=60, max_depth=4, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8,
                objective="binary:logistic", random_state=42, verbosity=0,
            )
            mini_lgb = lgb.LGBMClassifier(
                n_estimators=60, max_depth=4, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8,
                objective="binary", verbose=-1, random_state=42,
            )
            mini_scaler = StandardScaler()
            X_tr_sc = mini_scaler.fit_transform(X_tr)
            mini_lr = LogisticRegression(max_iter=500, C=1.0, random_state=42, solver="lbfgs")

            try:
                mini_xgb.fit(X_tr, y_tr, sample_weight=w_tr, verbose=False)
                mini_lgb.fit(X_tr, y_tr, sample_weight=w_tr)
                mini_lr.fit(X_tr_sc, y_tr, sample_weight=w_tr)

                xgb_p = mini_xgb.predict_proba(X_te)[:, 1]
                lgb_p = mini_lgb.predict_proba(X_te)[:, 1]
                lr_p = mini_lr.predict_proba(mini_scaler.transform(X_te))[:, 1]
                fold_probs = np.clip(0.4 * xgb_p + 0.4 * lgb_p + 0.2 * lr_p, 1e-6, 1 - 1e-6)

                oof_probs_list.append(fold_probs)
                oof_labels_list.append(y_te)
                logger.info(
                    f"  OOF fold {test_yr}: {te_mask.sum()} games, "
                    f"Brier={brier_score_loss(y_te, fold_probs):.4f}, "
                    f"Acc={np.mean((fold_probs > 0.5) == y_te):.3f}"
                )
            except Exception as e:
                logger.warning(f"  OOF fold {test_yr} failed: {e}")

        if not oof_probs_list:
            return np.array([]), np.array([])
        return np.concatenate(oof_probs_list), np.concatenate(oof_labels_list)

    def _fit_calibrator_on_probs(
        self,
        raw_probs: np.ndarray,
        y: np.ndarray,
        method: str = None,
    ):
        """
        Fit calibrator directly on pre-computed raw ensemble probabilities.

        Args:
            raw_probs: 1-D array of raw ensemble win probabilities
            y:         True binary labels (0/1)
            method:    "platt" (default) or "isotonic"
        """
        method = method or self.calibration_method
        self.calibration_method = method
        raw_probs = np.clip(raw_probs, 1e-6, 1 - 1e-6)

        if method == "isotonic":
            cal = IsotonicRegression(out_of_bounds="clip")
            cal.fit(raw_probs, y)
            cal_probs = cal.predict(raw_probs)
        else:  # platt / sigmoid
            cal = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
            cal.fit(raw_probs.reshape(-1, 1), y)
            cal_probs = cal.predict_proba(raw_probs.reshape(-1, 1))[:, 1]

        self.calibrator = cal
        raw_brier = brier_score_loss(y, raw_probs)
        cal_brier = brier_score_loss(y, cal_probs)
        logger.info(
            f"Calibration ({method}, n={len(y)}): "
            f"Brier {raw_brier:.4f} → {cal_brier:.4f} "
            f"({'improved' if cal_brier < raw_brier else 'no change'})"
        )

    def _fit_calibrator(self, X_val: np.ndarray, y_val: np.ndarray, method: str = None):
        """Convenience wrapper: compute raw probs from X_val then calibrate."""
        raw_probs = self._predict_ensemble_probs_batch(X_val)
        self._fit_calibrator_on_probs(raw_probs, y_val.astype(int), method)

    # ── Warm-Start Update ────────────────────────────────────────────

    def warm_start_update(self, data: dict):
        """
        Incrementally update models with new data via warm-starting.

        XGBoost: Adds a few new trees using the existing model as base.
        LightGBM: Uses refit to adjust leaf values.

        Args:
            data: Dict with updated X_train, y_train_* (full dataset including new matches)
        """
        X_train = data["X_train"]
        y_margin = data["y_train_margin"]
        y_cls = data["y_train_cls"]
        weights = data["sample_weights"]

        n_trees = settings.model.warmstart_trees
        lr = settings.model.warmstart_lr

        logger.info(f"Warm-start update: {len(X_train)} samples, adding {n_trees} trees at lr={lr}")

        # ── XGBoost warm-start ───────────────────────────────────────
        if self.xgb_margin is not None:
            # Save current model to temp, then reload with new training
            model_path = MODELS_DIR / "_temp_xgb_margin.json"
            self.xgb_margin.save_model(str(model_path))

            self.xgb_margin = xgb.XGBRegressor(
                n_estimators=n_trees,
                learning_rate=lr,
                max_depth=3,  # Shallower trees for incremental update
                min_child_weight=10,
                objective="reg:squarederror",
            )
            self.xgb_margin.fit(
                X_train, y_margin,
                sample_weight=weights,
                xgb_model=str(model_path),
                verbose=False,
            )
            model_path.unlink(missing_ok=True)

        if self.xgb_cls is not None:
            model_path = MODELS_DIR / "_temp_xgb_cls.json"
            self.xgb_cls.save_model(str(model_path))

            self.xgb_cls = xgb.XGBClassifier(
                n_estimators=n_trees,
                learning_rate=lr,
                max_depth=3,
                min_child_weight=10,
                objective="binary:logistic",
            )
            self.xgb_cls.fit(
                X_train, y_cls,
                sample_weight=weights,
                xgb_model=str(model_path),
                verbose=False,
            )
            model_path.unlink(missing_ok=True)

        # ── LightGBM refit ───────────────────────────────────────────
        if self.lgb_margin is not None:
            try:
                self.lgb_margin = self.lgb_margin.booster_.refit(
                    X_train, y_margin, decay_rate=0.9
                )
            except Exception as e:
                logger.warning(f"LightGBM margin refit failed: {e}, skipping")

        if self.lgb_cls is not None:
            try:
                self.lgb_cls = self.lgb_cls.booster_.refit(
                    X_train, y_cls.astype(int), decay_rate=0.9
                )
            except Exception as e:
                logger.warning(f"LightGBM cls refit failed: {e}, skipping")

        # ── Re-fit linear models ─────────────────────────────────────
        X_scaled = self.scaler.fit_transform(X_train)
        if self.lr_cls is not None:
            self.lr_cls.fit(X_scaled, y_cls, sample_weight=weights)
        if self.ridge_margin is not None:
            self.ridge_margin.fit(X_scaled, y_margin, sample_weight=weights)

        self.training_info["last_warmstart"] = current_timestamp()
        self.training_info["n_train"] = len(X_train)

        logger.info("Warm-start update complete")

    # ── Hyperparameter Optimization ──────────────────────────────────

    def _optimize_hyperparams(self, X_train, y_margin, y_cls, weights, X_val, y_val_margin, y_val_cls):
        """Use Optuna for hyperparameter search."""
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def xgb_objective(trial):
            params = {
                "max_depth": trial.suggest_int("max_depth", 3, 7),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "min_child_weight": trial.suggest_int("min_child_weight", 3, 20),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 10.0, log=True),
            }
            model = xgb.XGBRegressor(objective="reg:squarederror", random_state=42, **params)
            model.fit(X_train, y_margin, sample_weight=weights,
                      eval_set=[(X_val, y_val_margin)] if len(X_val) > 0 else None,
                      verbose=False)
            pred = model.predict(X_val if len(X_val) > 0 else X_train)
            target = y_val_margin if len(X_val) > 0 else y_margin
            return np.mean(np.abs(pred - target))

        logger.info("Running Optuna hyperparameter optimization (50 trials)...")
        study = optuna.create_study(direction="minimize")
        study.optimize(xgb_objective, n_trials=50, show_progress_bar=False)

        best = study.best_params
        best["random_state"] = 42
        logger.info(f"Best XGBoost params: MAE={study.best_value:.2f}")

        lgb_params = {
            "max_depth": best.get("max_depth", 5),
            "learning_rate": best.get("learning_rate", 0.1),
            "n_estimators": best.get("n_estimators", 300),
            "min_child_samples": best.get("min_child_weight", 10),
            "subsample": best.get("subsample", 0.8),
            "colsample_bytree": best.get("colsample_bytree", 0.8),
            "reg_alpha": best.get("reg_alpha", 1.0),
            "reg_lambda": best.get("reg_lambda", 1.0),
            "random_state": 42,
            "verbose": -1,
        }

        return best, lgb_params

    # ── Save / Load ──────────────────────────────────────────────────

    def save(self, directory: Path = None):
        """Save all models and metadata to disk."""
        save_dir = directory or MODELS_DIR
        save_dir.mkdir(parents=True, exist_ok=True)

        model_dir = save_dir / self.version
        model_dir.mkdir(parents=True, exist_ok=True)

        # Save XGBoost models
        if self.xgb_margin:
            self.xgb_margin.save_model(str(model_dir / "xgb_margin.json"))
        if self.xgb_cls:
            self.xgb_cls.save_model(str(model_dir / "xgb_cls.json"))

        # Save LightGBM models
        if self.lgb_margin:
            if hasattr(self.lgb_margin, 'save_model'):
                self.lgb_margin.save_model(str(model_dir / "lgb_margin.txt"))
            else:
                with open(model_dir / "lgb_margin.pkl", "wb") as f:
                    pickle.dump(self.lgb_margin, f)
        if self.lgb_cls:
            if hasattr(self.lgb_cls, 'save_model'):
                self.lgb_cls.save_model(str(model_dir / "lgb_cls.txt"))
            else:
                with open(model_dir / "lgb_cls.pkl", "wb") as f:
                    pickle.dump(self.lgb_cls, f)

        # Save sklearn models
        with open(model_dir / "lr_cls.pkl", "wb") as f:
            pickle.dump(self.lr_cls, f)
        with open(model_dir / "ridge_margin.pkl", "wb") as f:
            pickle.dump(self.ridge_margin, f)
        with open(model_dir / "scaler.pkl", "wb") as f:
            pickle.dump(self.scaler, f)

        # Save metadata
        meta = {
            "version": self.version,
            "feature_names": self.feature_names,
            "training_info": self.training_info,
        }
        with open(model_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2, default=str)

        # Save column medians
        if self.col_medians is not None:
            np.save(model_dir / "col_medians.npy", self.col_medians)

        # Save calibrator
        if self.calibrator is not None:
            with open(model_dir / "calibrator.pkl", "wb") as f:
                pickle.dump(
                    {"calibrator": self.calibrator, "method": self.calibration_method,
                     "margin_sigma": self.margin_sigma,
                     "calibration_temperature": self.calibration_temperature}, f
                )

        logger.info(f"Models saved to {model_dir}")

    @classmethod
    def load(cls, version: str, directory: Path = None) -> "AFLModel":
        """Load saved models from disk."""
        load_dir = (directory or MODELS_DIR) / version
        if not load_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {load_dir}")

        model = cls(version=version)

        # Load metadata
        meta_path = load_dir / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            model.feature_names = meta.get("feature_names", [])
            model.training_info = meta.get("training_info", {})

        # Load XGBoost
        xgb_margin_path = load_dir / "xgb_margin.json"
        if xgb_margin_path.exists():
            model.xgb_margin = xgb.XGBRegressor()
            model.xgb_margin.load_model(str(xgb_margin_path))

        xgb_cls_path = load_dir / "xgb_cls.json"
        if xgb_cls_path.exists():
            model.xgb_cls = xgb.XGBClassifier()
            model.xgb_cls.load_model(str(xgb_cls_path))

        # Load LightGBM
        lgb_margin_path = load_dir / "lgb_margin.txt"
        if lgb_margin_path.exists():
            model.lgb_margin = lgb.Booster(model_file=str(lgb_margin_path))
        else:
            pkl_path = load_dir / "lgb_margin.pkl"
            if pkl_path.exists():
                with open(pkl_path, "rb") as f:
                    model.lgb_margin = pickle.load(f)

        lgb_cls_path = load_dir / "lgb_cls.txt"
        if lgb_cls_path.exists():
            model.lgb_cls = lgb.Booster(model_file=str(lgb_cls_path))
        else:
            pkl_path = load_dir / "lgb_cls.pkl"
            if pkl_path.exists():
                with open(pkl_path, "rb") as f:
                    model.lgb_cls = pickle.load(f)

        # Load sklearn models
        lr_path = load_dir / "lr_cls.pkl"
        if lr_path.exists():
            with open(lr_path, "rb") as f:
                model.lr_cls = pickle.load(f)

        ridge_path = load_dir / "ridge_margin.pkl"
        if ridge_path.exists():
            with open(ridge_path, "rb") as f:
                model.ridge_margin = pickle.load(f)

        scaler_path = load_dir / "scaler.pkl"
        if scaler_path.exists():
            with open(scaler_path, "rb") as f:
                model.scaler = pickle.load(f)

        # Load column medians
        medians_path = load_dir / "col_medians.npy"
        if medians_path.exists():
            model.col_medians = np.load(medians_path)

        # Load calibrator
        cal_path = load_dir / "calibrator.pkl"
        if cal_path.exists():
            with open(cal_path, "rb") as f:
                cal_data = pickle.load(f)
            model.calibrator = cal_data["calibrator"]
            model.calibration_method = cal_data["method"]
            model.margin_sigma = float(cal_data.get("margin_sigma", 30.0))
            model.calibration_temperature = float(cal_data.get("calibration_temperature", 1.5))
            logger.info(
                f"  Calibrator loaded: {model.calibration_method}, "
                f"margin_sigma={model.margin_sigma:.1f} pts, "
                f"temp={model.calibration_temperature:.2f}"
            )

        logger.info(f"Model loaded: {version} ({len(model.feature_names)} features)")
        return model

    @classmethod
    def load_latest(cls, directory: Path = None) -> "AFLModel":
        """Load the most recently saved model."""
        model_dir = directory or MODELS_DIR
        if not model_dir.exists():
            raise FileNotFoundError(f"Models directory not found: {model_dir}")

        versions = sorted(
            [d.name for d in model_dir.iterdir() if d.is_dir() and d.name.startswith("v_")],
            reverse=True,
        )
        if not versions:
            raise FileNotFoundError("No saved models found")

        return cls.load(versions[0], directory)
