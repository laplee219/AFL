# AFL Match Prediction & Value Betting System

An AI-powered AFL match prediction system that combines ensemble machine learning models with LLM-based qualitative analysis to predict match outcomes and identify value betting opportunities.

## Features

- **Match Predictions**: Ensemble of XGBoost, LightGBM, and Logistic Regression models
- **Margin-Adjusted Elo Ratings**: Custom Elo system for AFL with home advantage and season regression
- **Feature Engineering**: 50+ features covering rolling stats, H2H records, venue effects, travel, rest days
- **Probability Calibration**: Two-stage pipeline — Platt scaling (logistic regression on OOF walk-forward probabilities across 4 time-folds) followed by logit-space temperature scaling (T=1.5) to prevent overconfidence; no retraining required when tuning temperature
- **Empirical Spread Sigma**: Standard deviation of margin prediction errors computed from held-out validation residuals (~34 pts) — replaces hardcoded 30 pt assumption in the Gaussian spread model
- **Value Bet Detection**: Finds +EV bets across two markets per match with a two-layer filter:
  - **Head-to-Head (H2H)**: Compares model win probability against bookmaker implied probability
  - **Line / Spread**: Models $P(\text{cover})$ as $N(\text{predicted margin}, \sigma^2)$ using empirical $\sigma$ and compares against bookmaker spread odds
  - Bets must clear both `MIN_EV_THRESHOLD` *and* `MIN_EDGE` (minimum gap over bookmaker implied) to avoid marginal false-positives
- **Odds Comparison Report**: When no bets clear the EV threshold the `bet` command prints a full market comparison table — odds, model probability, bookmaker implied probability, edge and EV for every H2H and spread market — so you can see exactly how far each match is from having an edge
- **Auto-Fixture Refresh**: `predict` and `bet` automatically fetch upcoming fixtures from Squiggle when none are in the feature matrix, rebuild features on the fly, then retry — no manual `collect` step needed for a new season
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

> **Shortcut for a new season**: `python main.py bet` will automatically pull upcoming fixtures from Squiggle, rebuild the feature matrix, and show current bookmaker odds with model comparisons — no manual `collect` or `features` run required.

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
| `python main.py bet --round N` | Find value bet opportunities; shows full odds vs model comparison when no edge exists |
| `python main.py ingest --round N` | Ingest completed round results |
| `python main.py monitor` | Show model health status |
| `python main.py report --round N` | Generate LLM analysis report (saved to `data/reports/`) |
| `python main.py status` | Show system status |
| `python main.py performance` | Show betting performance |
| `python main.py models` | List all saved model versions |
| `python main.py update --round N --mode warmstart` | Update model incrementally |
| `python main.py backtest --season 2024` | Backtest against historical season |
| `python main.py pipeline --round N` | Run full pipeline end-to-end |

### `bet` Output Modes

| Situation | Output |
|-----------|--------|
| Value bets found (EV ≥ threshold) | Recommendation cards — odds, model prob, edge, EV, Kelly % for each bet |
| No edge this round (predictions + odds available) | Full odds vs model comparison — EV and edge for every H2H and spread market |
| Specific `--round N` requested, no odds for that round yet | `NO ODDS AVAILABLE FOR ROUND N YET` header + currently available bookmaker odds |
| No predictions yet (new season / no `--round`) | Raw bookmaker odds table showing prices, implied probabilities, and vig per match |

The threshold is controlled by `MIN_EV_THRESHOLD` in `.env` (default `5%`). An additional `MIN_EDGE` guard (default `10%`) requires the model probability to exceed the bookmaker-implied probability by at least that margin.

> **Round 0 = Opening Round**: Use `--round 0` to target the Opening Round specifically.

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
│   │   ├── value.py            # Value bet detection + odds comparison report (H2H + line/spread markets)
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

### Probability Calibration Pipeline

Raw ensemble probabilities pass through two calibration stages before being used for betting:

1. **OOF Platt Scaling** — The full dataset is split into 4 walk-forward time folds (2021–2024). For each fold, a lightweight 60-tree ensemble is trained on the preceding years and used to produce out-of-fold (OOF) probabilities. These ~846 OOF predictions plus the 216-game validation set (2025) are used to fit a logistic regression calibrator, avoiding any data leakage from the final validation year.

2. **Temperature Scaling** — After Platt scaling, the logit of the calibrated probability is divided by a temperature parameter T (default `1.5`) before converting back to probability. This symmetrically shrinks extremes toward 0.5, reducing overconfidence without retraining:

   | Raw prob | After T=1.5 |
   |----------|-------------|
   | 55% | 53% |
   | 65% | 60% |
   | 75% | 68% |
   | 85% | 76% |

   Temperature is stored in `calibrator.pkl` and adjustable without retraining.

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
LLM_MODEL=claude-opus-4-6
ANTHROPIC_BASE_URL=              # Optional: custom API proxy URL

# Betting
INITIAL_BANKROLL=1000
KELLY_FRACTION=0.25
MAX_BET_FRACTION=0.05
MIN_EV_THRESHOLD=0.05            # Minimum EV to flag a value bet (5%)
MIN_EDGE=0.10                    # Minimum gap over bookmaker implied probability (10pp)

# Model
ELO_K_FACTOR=40
ELO_HOME_ADVANTAGE=35
RETRAIN_EVERY_N_ROUNDS=4

# Data
DATA_START_YEAR=2018
CURRENT_SEASON=2026
```

## License

MIT
