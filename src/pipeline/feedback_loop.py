"""
Feedback Loop Pipeline

Orchestrates the end-to-end flow after each round:
1. Ingest new match results
2. Update Elo ratings and rolling features
3. Compare predictions vs actuals
4. Check if retraining is needed
5. Generate predictions for next round
6. Identify value bets

Can be run manually or scheduled (e.g., weekly via cron/Task Scheduler).
"""

import pickle
from pathlib import Path
from typing import Optional

import pandas as pd

from config.settings import MODELS_DIR, PROCESSED_DATA_DIR, settings
from src.data_collection.squiggle_client import SquiggleClient
from src.data_collection.odds_collector import OddsCollector
from src.preprocessing.clean import clean_squiggle_games
from src.preprocessing.features import EloSystem, build_feature_matrix
from src.preprocessing.dataset import build_train_test_split
from src.models.train import AFLModel
from src.models.predict import Predictor
from src.models.evaluate import evaluate_predictions
from src.betting.value import find_value_bets
from src.betting.tracker import BetTracker
from src.pipeline.monitor import ModelMonitor
from src.utils.helpers import (
    current_timestamp,
    get_db_connection,
    get_logger,
    init_database,
)

logger = get_logger(__name__)


class Pipeline:
    """
    Main pipeline for the AFL prediction system.

    Coordinates data collection, model training, prediction, and betting.
    """

    def __init__(self, model_version: str = None):
        self.squiggle = SquiggleClient()
        self.odds_collector = OddsCollector()
        self.monitor = ModelMonitor()
        self.tracker = BetTracker()
        self.model: Optional[AFLModel] = None
        self.elo: Optional[EloSystem] = None
        self.feature_matrix: Optional[pd.DataFrame] = None
        self._model_version = model_version  # If set, load this specific version

        # Ensure database exists
        init_database()

    # ── Step 1a: Refresh Upcoming Fixtures ──────────────────────────

    def refresh_upcoming_fixtures(self, year: int = None) -> bool:
        """
        Fetch upcoming (unplayed) matches for the current season from Squiggle
        and merge them into the processed CSV so predict_upcoming can find them.

        Called automatically by predict() when no upcoming matches are in the
        feature matrix.

        Returns:
            True if new fixtures were added, False otherwise.
        """
        year = year or settings.data.current_season
        logger.info(f"Refreshing upcoming fixtures for {year}...")

        upcoming_raw = self.squiggle.get_upcoming_games(year=year)
        if upcoming_raw.empty:
            logger.info("No upcoming fixtures found on Squiggle")
            return False

        upcoming = clean_squiggle_games(upcoming_raw)
        upcoming = upcoming[~upcoming["is_complete"]].copy()

        if upcoming.empty:
            return False

        # Merge with existing CSV (update-or-append)
        matches_path = PROCESSED_DATA_DIR / "matches_all.csv"
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

        if matches_path.exists():
            existing = pd.read_csv(matches_path, parse_dates=["date"])
            # Remove any old placeholder rows for the same match_ids
            if "match_id" in existing.columns and "match_id" in upcoming.columns:
                existing = existing[~existing["match_id"].isin(upcoming["match_id"])]
            merged = pd.concat([existing, upcoming], ignore_index=True)
        else:
            merged = upcoming

        merged = merged.sort_values(["date", "match_id"]).reset_index(drop=True)
        merged.to_csv(matches_path, index=False)
        logger.info(f"Added {len(upcoming)} upcoming fixture(s) to {matches_path}")

        # Rebuild feature matrix so upcoming rows are available for prediction
        self.build_features(merged)
        return True

    # ── Step 1: Data Collection ──────────────────────────────────────

    def collect_data(
        self,
        start_year: int = None,
        end_year: int = None,
    ) -> pd.DataFrame:
        """
        Collect and store historical match data.

        Args:
            start_year: First year to collect
            end_year: Last year to collect

        Returns:
            Cleaned match DataFrame
        """
        start = start_year or settings.data.data_start_year
        end = end_year or settings.data.current_season

        logger.info(f"Collecting data from {start} to {end}...")

        # Fetch from Squiggle API
        raw_data = self.squiggle.collect_historical_data(start, end)

        if raw_data.empty:
            logger.error("No data collected")
            return pd.DataFrame()

        # Clean and standardize
        matches = clean_squiggle_games(raw_data)

        # Store completed matches in database
        completed = matches[matches["is_complete"]].copy()
        self._store_matches(completed)

        # Save raw data as backup
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        matches.to_csv(PROCESSED_DATA_DIR / "matches_all.csv", index=False)

        logger.info(f"Collected {len(matches)} matches ({len(completed)} completed)")
        return matches

    def _store_matches(self, matches: pd.DataFrame):
        """Store matches in the database."""
        conn = get_db_connection()
        for _, row in matches.iterrows():
            try:
                conn.execute("""
                    INSERT OR REPLACE INTO matches
                    (match_id, year, round, round_name, date, home_team, away_team,
                     venue, home_score, away_score, home_goals, home_behinds,
                     away_goals, away_behinds, margin, winner, is_final, crowd)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    row.get("match_id"), row.get("year"), row.get("round"),
                    row.get("round_name"), str(row.get("date", "")),
                    row.get("home_team"), row.get("away_team"), row.get("venue"),
                    row.get("home_score"), row.get("away_score"),
                    row.get("home_goals"), row.get("home_behinds"),
                    row.get("away_goals"), row.get("away_behinds"),
                    row.get("margin"), row.get("winner"),
                    row.get("is_final", 0), row.get("crowd"),
                ))
            except Exception as e:
                logger.debug(f"Error storing match: {e}")
        conn.commit()
        conn.close()

    def _store_predictions(self, predictions: pd.DataFrame, model_version: str = ""):
        """Persist a prediction snapshot for later round evaluation."""
        if predictions.empty:
            return

        captured_at = current_timestamp()
        conn = get_db_connection()
        for _, row in predictions.iterrows():
            try:
                predicted_winner = (
                    row.get("home_team")
                    if float(row.get("ensemble_margin", 0.0)) > 0
                    else row.get("away_team")
                )
                conn.execute(
                    """
                    INSERT INTO predictions
                    (match_id, year, round, home_team, away_team, predicted_home_prob,
                     predicted_margin, predicted_winner, model_version, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        row.get("match_id"),
                        row.get("year"),
                        row.get("round"),
                        row.get("home_team"),
                        row.get("away_team"),
                        row.get("ensemble_prob"),
                        row.get("ensemble_margin"),
                        predicted_winner,
                        model_version,
                        captured_at,
                    ),
                )
            except Exception as e:
                logger.debug(f"Error storing prediction snapshot: {e}")
        conn.commit()
        conn.close()

    def _load_prediction_snapshot(self, year: int, round_num: int) -> pd.DataFrame:
        """Load the latest saved prediction snapshot for a round."""
        try:
            latest = get_db_connection()
            ts_df = pd.read_sql_query(
                """
                SELECT created_at
                FROM predictions
                WHERE year = ? AND round = ?
                ORDER BY created_at DESC, id DESC
                LIMIT 1
                """,
                latest,
                params=(year, round_num),
            )
            latest.close()
        except Exception:
            return pd.DataFrame()

        if ts_df.empty:
            return pd.DataFrame()

        captured_at = ts_df.iloc[0]["created_at"]
        try:
            conn = get_db_connection()
            snapshot = pd.read_sql_query(
                """
                SELECT match_id, year, round, home_team, away_team,
                       predicted_home_prob AS ensemble_prob,
                       predicted_margin AS ensemble_margin,
                       model_version, created_at
                FROM predictions
                WHERE year = ? AND round = ? AND created_at = ?
                ORDER BY id
                """,
                conn,
                params=(year, round_num, captured_at),
            )
            conn.close()
            return snapshot
        except Exception:
            return pd.DataFrame()

    # ── Step 2: Feature Engineering ──────────────────────────────────

    def build_features(self, matches: pd.DataFrame = None) -> pd.DataFrame:
        """
        Build the feature matrix from match data.

        Args:
            matches: Match DataFrame (loads from file if not provided)

        Returns:
            Feature matrix DataFrame
        """
        if matches is None:
            matches_path = PROCESSED_DATA_DIR / "matches_all.csv"
            if matches_path.exists():
                matches = pd.read_csv(matches_path, parse_dates=["date"])
            else:
                logger.error("No match data found. Run 'collect' first.")
                return pd.DataFrame()

        logger.info("Building feature matrix...")
        self.feature_matrix, self.elo = build_feature_matrix(matches)

        # Save feature matrix and Elo system
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        if isinstance(self.feature_matrix, pd.DataFrame):
            self.feature_matrix.to_csv(
                PROCESSED_DATA_DIR / "feature_matrix.csv", index=False
            )

        if self.elo:
            with open(PROCESSED_DATA_DIR / "elo_system.pkl", "wb") as f:
                pickle.dump(self.elo, f)

        logger.info(f"Feature matrix: {self.feature_matrix.shape}")
        return self.feature_matrix

    # ── Step 3: Model Training ───────────────────────────────────────

    def train_model(
        self,
        feature_matrix: pd.DataFrame = None,
        use_optuna: bool = False,
    ) -> AFLModel:
        """
        Train the prediction model.

        Args:
            feature_matrix: Feature matrix (loads from file if not provided)
            use_optuna: Use Optuna for hyperparameter optimization

        Returns:
            Trained AFLModel
        """
        if feature_matrix is None:
            feature_matrix = self._load_feature_matrix()

        if feature_matrix.empty:
            logger.error("No feature matrix available. Run 'features' first.")
            return None

        # Build train/val/test splits
        data = build_train_test_split(feature_matrix)

        # Train model
        self.model = AFLModel()
        self.model.train(data, use_optuna=use_optuna)

        # Save model
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        self.model.save()

        # Evaluate on validation set
        if len(data["X_val"]) > 0:
            predictor = Predictor(self.model)
            # Quick evaluation summary
            logger.info(f"Model saved as version: {self.model.version}")

        return self.model

    # ── Step 4: Predict ──────────────────────────────────────────────

    def predict(
        self,
        year: int = None,
        round_num: int = None,
    ) -> pd.DataFrame:
        """
        Generate predictions for a specific round.

        Args:
            year: Season year
            round_num: Round number

        Returns:
            DataFrame of predictions
        """
        year = year or settings.data.current_season

        # Load model if needed
        if self.model is None:
            try:
                if self._model_version:
                    self.model = AFLModel.load(self._model_version)
                    logger.info(f"Using specified model: {self._model_version}")
                else:
                    self.model = AFLModel.load_latest()
            except FileNotFoundError:
                logger.error("No trained model found. Run 'train' first.")
                return pd.DataFrame()

        # Load feature matrix — rebuild from CSV if missing but data exists
        feature_matrix = self._load_feature_matrix()
        if feature_matrix.empty:
            matches_path = PROCESSED_DATA_DIR / "matches_all.csv"
            if matches_path.exists():
                logger.info("Feature matrix missing; rebuilding from existing match data...")
                self.build_features()
                feature_matrix = self._load_feature_matrix()
            if feature_matrix.empty:
                logger.error("No feature matrix available. Run 'collect' then 'features' first.")
                return pd.DataFrame()

        predictor = Predictor(self.model)

        if round_num is not None:
            result = predictor.predict_round(feature_matrix, year, round_num)
            if not result.empty:
                self._store_predictions(result, self.model.version)
            return result
        else:
            result = predictor.predict_upcoming(feature_matrix)
            if result.empty:
                # No upcoming matches in the feature matrix — try to pull them
                # from Squiggle and rebuild features automatically.
                logger.info("No upcoming matches found; refreshing fixtures from Squiggle...")
                if self.refresh_upcoming_fixtures(year):
                    feature_matrix = self._load_feature_matrix()
                    result = predictor.predict_upcoming(feature_matrix)
            return result

    # ── Step 5: Value Bets ───────────────────────────────────────────

    def find_bets(
        self,
        year: int = None,
        round_num: int = None,
    ) -> pd.DataFrame:
        """
        Find value bets for a round.

        Args:
            year: Season year
            round_num: Round number

        Returns:
            DataFrame of value bets
        """
        predictions = self.predict(year, round_num)
        if predictions.empty:
            return pd.DataFrame()

        # Get current odds
        odds = self.odds_collector.get_best_odds()
        if odds.empty:
            logger.warning("No odds data available — cannot identify value bets")
            return pd.DataFrame()

        margin_sigma = getattr(self.model, "margin_sigma", 30.0) if self.model else 30.0
        return find_value_bets(predictions, odds, spread_sigma=margin_sigma,
                               min_edge=settings.betting.min_edge)

    # ── Step 6: Ingest Results & Update ──────────────────────────────

    def ingest_results(self, year: int = None, round_num: int = None):
        """
        Ingest completed match results and update the system.

        This is the core feedback loop:
        1. Fetch latest results
        2. Update database
        3. Update Elo ratings and features
        4. Log prediction performance
        5. Check if retraining is needed
        6. Settle any open bets
        """
        year = year or settings.data.current_season

        logger.info(
            f"Ingesting results for {year} R{round_num if round_num is not None else 'latest'}..."
        )

        predictions_for_eval = pd.DataFrame()
        prediction_model_version = ""
        if round_num is not None:
            predictions_for_eval = self._load_prediction_snapshot(year, round_num)
            if not predictions_for_eval.empty:
                logger.info(
                    f"Loaded saved prediction snapshot for {year} R{round_num} "
                    f"({len(predictions_for_eval)} matches)"
                )
                if (
                    "model_version" in predictions_for_eval.columns
                    and predictions_for_eval["model_version"].notna().any()
                ):
                    prediction_model_version = str(
                        predictions_for_eval["model_version"].dropna().iloc[0]
                    )

        # Fetch results
        if round_num is not None:
            results = self.squiggle.get_completed_games(year, round_num)
        else:
            results = self.squiggle.get_completed_games(year)

        if results.empty:
            logger.warning("No new results to ingest")
            return

        results = clean_squiggle_games(results)
        self._store_matches(results)

        # Rebuild features with new data
        all_matches = pd.read_csv(
            PROCESSED_DATA_DIR / "matches_all.csv", parse_dates=["date"]
        ) if (PROCESSED_DATA_DIR / "matches_all.csv").exists() else pd.DataFrame()

        if not all_matches.empty:
            # Upsert by match_id so completed results replace earlier fixture placeholders
            if "match_id" in results.columns and "match_id" in all_matches.columns:
                all_matches = all_matches[
                    ~all_matches["match_id"].isin(results["match_id"])
                ]
                all_matches = pd.concat([all_matches, results], ignore_index=True)
                all_matches = all_matches.sort_values(["date", "match_id"]).reset_index(drop=True)
                all_matches.to_csv(PROCESSED_DATA_DIR / "matches_all.csv", index=False)

            self.build_features(all_matches)

        # Log performance for the round
        if round_num is not None:
            # Ingest is often run as a standalone command where no model is loaded yet.
            # Load the latest model so this round can still be evaluated and monitored.
            if self.model is None:
                try:
                    self.model = AFLModel.load_latest()
                except FileNotFoundError:
                    logger.warning("No trained model found; skipping monitoring log for this round")

            if self.model is not None:
                if predictions_for_eval.empty:
                    predictions_for_eval = self.predict(year, round_num)
                    if not predictions_for_eval.empty:
                        prediction_model_version = self.model.version

                if not predictions_for_eval.empty:
                    self.monitor.log_round_performance(
                        year,
                        round_num,
                        predictions_for_eval,
                        results,
                        prediction_model_version or self.model.version,
                    )

        # Settle open bets
        if round_num is not None:
            # Capture closing odds before settling (for post-match CLV tracking)
            try:
                n_snaps = self.odds_collector.save_odds_snapshot(
                    year, round_num, snapshot_type="closing"
                )
                if n_snaps > 0:
                    logger.info(f"Captured {n_snaps} closing odds snapshots")
            except Exception as e:
                logger.warning(f"Could not capture closing odds: {e}")

            self.tracker.settle_round(year, round_num, results)

        # Check if retraining needed
        if round_num is not None:
            check = self.monitor.check_retrain_needed(year, round_num)
            if check["should_retrain"]:
                logger.warning(f"Retraining recommended: {check['reason']}")
                # Auto-retrain or just warn based on config
                self._handle_retrain(year, round_num, check["reason"])

        logger.info("Results ingested successfully")

    def _handle_retrain(self, year: int, round_num: int, reason: str):
        """Handle retraining decision."""
        rounds_since_last = round_num % settings.model.retrain_every_n_rounds

        if "Scheduled" in reason or rounds_since_last == 0:
            # Full retrain
            logger.info("Executing scheduled full retrain...")
            feature_matrix = self._load_feature_matrix()
            if not feature_matrix.empty:
                self.train_model(feature_matrix)
        else:
            # Warm-start update
            logger.info("Executing warm-start update...")
            if self.model is None:
                try:
                    self.model = AFLModel.load_latest()
                except FileNotFoundError:
                    logger.error("No model to warm-start from")
                    return

            feature_matrix = self._load_feature_matrix()
            if not feature_matrix.empty:
                data = build_train_test_split(feature_matrix)
                self.model.warm_start_update(data)
                self.model.save()

    # ── Utility Methods ──────────────────────────────────────────────

    def _load_feature_matrix(self) -> pd.DataFrame:
        """Load feature matrix from file."""
        path = PROCESSED_DATA_DIR / "feature_matrix.csv"
        if path.exists():
            return pd.read_csv(path, parse_dates=["date"])
        return pd.DataFrame()

    def get_status(self) -> dict:
        """Get current system status."""
        status = {
            "model_loaded": self.model is not None,
            "model_version": self.model.version if self.model else None,
            "feature_matrix_loaded": self.feature_matrix is not None,
            "bankroll": self.tracker.bankroll,
            "stop_loss_triggered": self.tracker.stop_loss_triggered,
        }

        # Check for data files
        status["has_match_data"] = (PROCESSED_DATA_DIR / "matches_all.csv").exists()
        status["has_feature_matrix"] = (PROCESSED_DATA_DIR / "feature_matrix.csv").exists()
        status["has_elo"] = (PROCESSED_DATA_DIR / "elo_system.pkl").exists()

        # Check for saved models
        if MODELS_DIR.exists():
            versions = [d.name for d in MODELS_DIR.iterdir() if d.is_dir() and d.name.startswith("v_")]
            status["n_saved_models"] = len(versions)
            status["latest_model"] = sorted(versions, reverse=True)[0] if versions else None
        else:
            status["n_saved_models"] = 0
            status["latest_model"] = None

        return status

    def run_full_pipeline(
        self,
        year: int = None,
        round_num: int = None,
        use_optuna: bool = False,
    ):
        """
        Run the complete pipeline from data collection to predictions.

        Args:
            year: Season year
            round_num: Round to predict
            use_optuna: Use Optuna optimization for training
        """
        year = year or settings.data.current_season

        logger.info("=" * 60)
        logger.info("  AFL PREDICTOR — FULL PIPELINE")
        logger.info("=" * 60)

        # Step 1: Collect data
        logger.info("\n[1/5] Collecting data...")
        matches = self.collect_data()
        if matches.empty:
            return

        # Step 2: Build features
        logger.info("\n[2/5] Building features...")
        feature_matrix = self.build_features(matches)
        if feature_matrix.empty:
            return

        # Step 3: Train model
        logger.info("\n[3/5] Training model...")
        self.train_model(feature_matrix, use_optuna=use_optuna)

        # Step 4: Generate predictions
        logger.info("\n[4/5] Generating predictions...")
        predictions = self.predict(year, round_num)

        # Step 5: Find value bets
        logger.info("\n[5/5] Finding value bets...")
        value_bets = self.find_bets(year, round_num)

        logger.info("\n" + "=" * 60)
        logger.info("  PIPELINE COMPLETE")
        logger.info("=" * 60)

        return {
            "predictions": predictions,
            "value_bets": value_bets,
            "model_version": self.model.version if self.model else None,
        }
