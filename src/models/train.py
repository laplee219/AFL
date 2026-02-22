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
from sklearn.linear_model import LogisticRegression, Ridge
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

        logger.info(f"All models trained successfully (version: {self.version})")

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
