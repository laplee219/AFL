"""
AFL Predictor - Helper Utilities

Common utility functions used across the project.
"""

import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from config.settings import DB_PATH, settings


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Get a configured logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(name)-25s | %(levelname)-7s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def get_db_connection(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """Get a SQLite database connection."""
    path = db_path or DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_database(db_path: Optional[Path] = None):
    """Initialize the database with required tables."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    # Matches table - raw match data
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS matches (
            match_id INTEGER PRIMARY KEY,
            year INTEGER NOT NULL,
            round INTEGER NOT NULL,
            round_name TEXT,
            date TEXT,
            home_team TEXT NOT NULL,
            away_team TEXT NOT NULL,
            venue TEXT,
            home_score INTEGER,
            away_score INTEGER,
            home_goals INTEGER,
            home_behinds INTEGER,
            away_goals INTEGER,
            away_behinds INTEGER,
            margin INTEGER,
            winner TEXT,
            is_final INTEGER DEFAULT 0,
            crowd INTEGER,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(year, round, home_team, away_team)
        )
    """)

    # Elo ratings history
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS elo_ratings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            team TEXT NOT NULL,
            rating REAL NOT NULL,
            match_id INTEGER,
            year INTEGER,
            round INTEGER,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (match_id) REFERENCES matches(match_id)
        )
    """)

    # Predictions log
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            match_id INTEGER,
            year INTEGER NOT NULL,
            round INTEGER NOT NULL,
            home_team TEXT NOT NULL,
            away_team TEXT NOT NULL,
            predicted_home_prob REAL,
            predicted_margin REAL,
            predicted_winner TEXT,
            model_version TEXT,
            actual_winner TEXT,
            actual_margin INTEGER,
            correct INTEGER,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (match_id) REFERENCES matches(match_id)
        )
    """)

    # Bets tracking
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS bets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            match_id INTEGER,
            year INTEGER NOT NULL,
            round INTEGER NOT NULL,
            team TEXT NOT NULL,
            bet_type TEXT NOT NULL,
            model_prob REAL NOT NULL,
            bookmaker_prob REAL NOT NULL,
            bookmaker_odds REAL NOT NULL,
            expected_value REAL NOT NULL,
            kelly_fraction REAL NOT NULL,
            stake REAL NOT NULL,
            result TEXT,
            profit_loss REAL,
            bankroll_after REAL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (match_id) REFERENCES matches(match_id)
        )
    """)

    # Model registry
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS model_registry (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            version TEXT NOT NULL UNIQUE,
            model_type TEXT NOT NULL,
            training_rounds TEXT,
            training_years TEXT,
            n_features INTEGER,
            validation_accuracy REAL,
            validation_logloss REAL,
            validation_brier REAL,
            validation_mae REAL,
            model_path TEXT,
            notes TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Monitoring metrics per round
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS monitoring_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            year INTEGER NOT NULL,
            round INTEGER NOT NULL,
            model_version TEXT,
            accuracy REAL,
            log_loss REAL,
            brier_score REAL,
            margin_mae REAL,
            n_predictions INTEGER,
            n_correct INTEGER,
            retrain_triggered INTEGER DEFAULT 0,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(year, round, model_version)
        )
    """)

    conn.commit()
    conn.close()


def df_from_db(query: str, params: tuple = (), db_path: Optional[Path] = None) -> pd.DataFrame:
    """Execute a query and return results as a DataFrame."""
    conn = get_db_connection(db_path)
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return df


def execute_db(query: str, params: tuple = (), db_path: Optional[Path] = None):
    """Execute a write query on the database."""
    conn = get_db_connection(db_path)
    conn.execute(query, params)
    conn.commit()
    conn.close()


def current_timestamp() -> str:
    """Get current timestamp as ISO format string."""
    return datetime.now().isoformat()


def season_progress(year: int, round_num: int, total_rounds: int = 24) -> float:
    """Calculate season progress as a fraction (0.0 to 1.0)."""
    return min(round_num / total_rounds, 1.0)


def format_odds(decimal_odds: float) -> str:
    """Format decimal odds for display."""
    return f"${decimal_odds:.2f}"


def implied_probability(decimal_odds: float) -> float:
    """Convert decimal odds to implied probability."""
    if decimal_odds <= 0:
        return 0.0
    return 1.0 / decimal_odds


def decimal_from_probability(prob: float) -> float:
    """Convert probability to decimal odds."""
    if prob <= 0:
        return float("inf")
    return 1.0 / prob
