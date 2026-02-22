"""
AFL Predictor - Configuration Settings

Uses pydantic-settings to manage configuration from .env file and environment variables.
"""

import os
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field


# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models"
REPORTS_DIR = DATA_DIR / "reports"
DB_PATH = DATA_DIR / "afl_predictor.db"


class LLMSettings(BaseSettings):
    """LLM API configuration."""
    openai_api_key: str = Field(default="", description="OpenAI API key")
    anthropic_api_key: str = Field(default="", description="Anthropic API key")
    anthropic_base_url: str = Field(default="", description="Custom Anthropic API base URL (e.g. proxy)")
    llm_provider: str = Field(default="anthropic", description="Primary LLM provider: 'openai' or 'anthropic'")
    llm_model: str = Field(default="claude-sonnet-4-20250514", description="LLM model name")
    llm_max_tokens: int = Field(default=2048, description="Max tokens for LLM responses")

    model_config = {"env_file": PROJECT_ROOT / ".env", "extra": "ignore"}


class OddsSettings(BaseSettings):
    """Betting odds API configuration."""
    odds_api_key: str = Field(default="", description="The Odds API key")
    odds_api_base_url: str = Field(
        default="https://api.the-odds-api.com/v4",
        description="The Odds API base URL"
    )

    model_config = {"env_file": PROJECT_ROOT / ".env", "extra": "ignore"}


class BettingSettings(BaseSettings):
    """Betting and bankroll configuration."""
    initial_bankroll: float = Field(default=1000.0, description="Starting bankroll")
    kelly_fraction: float = Field(default=0.25, description="Kelly criterion fraction (0.25 = quarter-Kelly)")
    max_bet_fraction: float = Field(default=0.05, description="Maximum fraction of bankroll per bet")
    min_ev_threshold: float = Field(default=0.05, description="Minimum expected value to place a bet")
    stop_loss_fraction: float = Field(default=0.5, description="Stop betting if bankroll drops below this fraction of initial")

    model_config = {"env_file": PROJECT_ROOT / ".env", "extra": "ignore"}


class ModelSettings(BaseSettings):
    """ML model configuration."""
    retrain_every_n_rounds: int = Field(default=4, description="Full retrain every N rounds")
    warmstart_trees: int = Field(default=5, description="Number of trees to add during warm-start")
    warmstart_lr: float = Field(default=0.01, description="Learning rate for warm-start updates")
    full_train_lr: float = Field(default=0.1, description="Learning rate for full training")
    max_depth: int = Field(default=5, description="Max tree depth")
    n_estimators: int = Field(default=300, description="Number of trees for full training")
    early_stopping_rounds: int = Field(default=20, description="Early stopping patience")

    # Elo configuration
    elo_k_factor: float = Field(default=40.0, description="Elo K-factor for update magnitude")
    elo_home_advantage: float = Field(default=35.0, description="Elo home advantage in rating points")
    elo_season_regression: float = Field(default=0.3, description="Fraction to regress Elo toward mean between seasons")
    elo_initial_rating: float = Field(default=1500.0, description="Initial Elo rating for new teams")

    # Monitoring
    accuracy_alert_threshold: float = Field(default=0.55, description="Alert if rolling accuracy drops below this")
    logloss_alert_multiplier: float = Field(default=1.5, description="Alert if log loss exceeds baseline * this multiplier")
    consecutive_poor_rounds: int = Field(default=3, description="Number of poor rounds before triggering retrain")

    model_config = {"env_file": PROJECT_ROOT / ".env", "extra": "ignore"}


class DataSettings(BaseSettings):
    """Data collection and processing configuration."""
    data_start_year: int = Field(default=2018, description="Earliest year to collect data from")
    current_season: int = Field(default=2026, description="Current AFL season year")
    squiggle_base_url: str = Field(
        default="https://api.squiggle.com.au",
        description="Squiggle API base URL"
    )
    squiggle_user_agent: str = Field(
        default="AFL-Predictor/0.1 (github.com/afl-predictor)",
        description="User-Agent for Squiggle API requests"
    )
    sample_weight_decay: float = Field(default=0.85, description="Per-season decay factor for training sample weights")

    model_config = {"env_file": PROJECT_ROOT / ".env", "extra": "ignore"}


class Settings:
    """Aggregated settings container."""

    def __init__(self):
        self.llm = LLMSettings()
        self.odds = OddsSettings()
        self.betting = BettingSettings()
        self.model = ModelSettings()
        self.data = DataSettings()

    def ensure_directories(self):
        """Create necessary data directories if they don't exist."""
        for d in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR]:
            d.mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()
