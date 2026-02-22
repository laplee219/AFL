# AFL Match Prediction & Value Betting System

An AI-powered AFL match prediction system that combines ensemble machine learning models with LLM-based qualitative analysis to predict match outcomes and identify value betting opportunities.

## Features

- **Match Predictions**: Ensemble of XGBoost, LightGBM, and Logistic Regression models
- **Margin-Adjusted Elo Ratings**: Custom Elo system for AFL with home advantage and season regression
- **Feature Engineering**: 50+ features covering rolling stats, H2H records, venue effects, travel, rest days
- **Value Bet Detection**: Compares model probabilities against bookmaker odds to find +EV bets
- **Kelly Criterion Sizing**: Quarter-Kelly bet sizing with bankroll management and stop-loss
- **LLM Analysis**: AI-generated match previews, bet justification, and round reports
- **Model Feedback Loop**: Three-layer update strategy with warm-start and automatic retrain triggers
- **Web Dashboard**: Streamlit-based interactive dashboard with charts and monitoring

## Quick Start

### 1. Install Dependencies

```bash
cd AFL
pip install -r requirements.txt
```

### 2. Configure API Keys

Copy `.env.example` to `.env` and fill in your API keys:

```bash
cp .env.example .env
```

| Key | Required | Source |
|-----|----------|--------|
| `ODDS_API_KEY` | Optional | [the-odds-api.com](https://the-odds-api.com) (free tier: 500 req/month) |
| `ANTHROPIC_API_KEY` | Optional | [anthropic.com](https://www.anthropic.com) |
| `OPENAI_API_KEY` | Optional | [openai.com](https://openai.com) |

> The system works without API keys — odds and LLM features will be disabled.

### 3. Run the Pipeline

```bash
# Collect data → build features → train model → predict
python main.py pipeline --round 1 --year 2025

# Or step by step
python main.py collect
python main.py features
python main.py train
python main.py predict --round 1
```

### 4. Launch Dashboard

```bash
streamlit run app/streamlit_app.py
```

The dashboard includes a **Reports** page where you can browse, read, and download all saved LLM reports, or generate a new one directly from the UI.

## CLI Commands

| Command | Description |
|---------|-------------|
| `python main.py collect` | Collect match data from Squiggle API |
| `python main.py features` | Build feature matrix |
| `python main.py train [--optuna]` | Train models (optionally with Optuna) |
| `python main.py predict --round N` | Predict match outcomes |
| `python main.py bet --round N` | Find value bet opportunities |
| `python main.py ingest --round N` | Ingest completed round results |
| `python main.py monitor` | Show model health status |
| `python main.py report --round N` | Generate LLM analysis report (saved to `data/reports/`) |
| `python main.py status` | Show system status |
| `python main.py performance` | Show betting performance |
| `python main.py models` | List all saved model versions |
| `python main.py update --round N --mode warmstart` | Update model incrementally |
| `python main.py backtest --season 2024` | Backtest against historical season |
| `python main.py pipeline --round N` | Run full pipeline end-to-end |

### Model Selection

Use `python main.py models` to list all saved model versions, then pass `--model` to use a specific one:

```bash
python main.py models                                          # List available models
python main.py predict --round 1 --model v_20260222_130000     # Predict with a specific model
python main.py bet --round 1 --model v_20260222_130000         # Find value bets with a specific model
python main.py pipeline --round 1 --model v_20260222_130000    # Run pipeline with a specific model
```

If `--model` is omitted, the latest model is used automatically.

## Architecture

```
AFL/
├── main.py                     # CLI entry point (Click)
├── config/
│   └── settings.py             # Centralized config (pydantic-settings)
├── src/
│   ├── data_collection/
│   │   ├── squiggle_client.py  # Squiggle API wrapper
│   │   └── odds_collector.py   # The Odds API + manual odds
│   ├── preprocessing/
│   │   ├── clean.py            # Data standardization
│   │   ├── features.py         # Feature engineering + Elo system
│   │   └── dataset.py          # Train/test splits + sample weights
│   ├── models/
│   │   ├── train.py            # Model training (XGB/LGB/LR) + Optuna
│   │   ├── predict.py          # Ensemble predictions
│   │   └── evaluate.py         # Evaluation metrics + SHAP
│   ├── betting/
│   │   ├── value.py            # Value bet detection
│   │   ├── kelly.py            # Kelly criterion sizing
│   │   └── tracker.py          # Bet tracking + bankroll management
│   ├── llm/
│   │   ├── analyzer.py         # LLM match analysis
│   │   └── reporter.py         # Report generation
│   ├── pipeline/
│   │   ├── feedback_loop.py    # Pipeline orchestration
│   │   └── monitor.py          # Model health monitoring
│   └── utils/
│       ├── constants.py        # Teams, venues, distances
│       └── helpers.py          # DB utilities, logging, odds conversion
├── app/
│   └── streamlit_app.py        # Web dashboard
├── tests/                      # Test suite
├── data/                       # Data storage (auto-created)
│   ├── raw/
│   ├── processed/
│   ├── models/
│   └── reports/                # LLM-generated reports (Markdown)
└── afl_predictor.db            # SQLite database (auto-created)
```

## Model Details

### Features (50+)
- **Team Rolling Stats**: Points scored/conceded, margins, win rate (3/5/10 game windows + EWMA)
- **Elo Ratings**: Margin-adjusted Elo for both teams, Elo difference, expected win probability
- **Head-to-Head**: Win rate, average margin over 5-year lookback
- **Venue**: Home ground advantage, travel distance, interstate flag
- **Rest**: Days since last match, rest advantage differential
- **Derived**: All features computed as home-away differentials

### Ensemble
| Model | Weight | Type |
|-------|--------|------|
| XGBoost | 40% | Gradient boosting (margin + classification) |
| LightGBM | 40% | Gradient boosting (margin + classification) |
| Logistic Regression | 20% | Linear baseline |

### Update Strategy
1. **Real-time**: Elo ratings update after each match
2. **Per-round**: Warm-start XGBoost/LightGBM with recent data
3. **Periodic**: Full retrain every 4 rounds or when performance degrades

## Data Sources

- **[Squiggle API](https://api.squiggle.com.au)**: Match results, tips, standings, power rankings (free, no key needed)
- **[The Odds API](https://the-odds-api.com)**: Bookmaker odds aggregation (free tier: 500 req/month)

## Configuration

All settings are in `config/settings.py` and can be overridden via environment variables:

```env
# LLM
LLM_MODEL=claude-sonnet-4-20250514
ANTHROPIC_BASE_URL=              # Optional: custom API proxy URL

# Betting
INITIAL_BANKROLL=1000
KELLY_FRACTION=0.25
MIN_EV_THRESHOLD=0.05

# Model
ELO_K_FACTOR=40
ELO_HOME_ADVANTAGE=35
RETRAIN_EVERY_N_ROUNDS=4

# Data
DATA_START_YEAR=2018
CURRENT_SEASON=2025
```

## License

MIT
