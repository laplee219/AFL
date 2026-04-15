# Feature Engineering Architecture

This document explains how the AFL predictor engineers its model features, where those features come from, and how they are refreshed over time.

Primary implementation files:

- [`src/preprocessing/clean.py`](src/preprocessing/clean.py)
- [`src/preprocessing/features.py`](src/preprocessing/features.py)
- [`src/preprocessing/dataset.py`](src/preprocessing/dataset.py)
- [`src/pipeline/feedback_loop.py`](src/pipeline/feedback_loop.py)
- [`src/utils/constants.py`](src/utils/constants.py)

## High-Level Flow

The feature pipeline is built around one rule: every feature should reflect what was knowable **before** the match being predicted.

The end-to-end flow is:

1. Raw match data is fetched from Squiggle.
2. `clean_squiggle_games()` standardizes names, scores, dates, and completion flags.
3. `build_feature_matrix()` sorts matches chronologically and constructs feature families.
4. Team-level statistics are converted into match-level features by comparing the home and away sides.
5. The final matrix is written to `data/processed/feature_matrix.csv`.
6. The live Elo state is also saved to `data/processed/elo_system.pkl`.

For upcoming fixtures, the same pipeline still runs, but targets such as margin and winner stay empty while pre-match features are populated.

## Stage 1: Input Standardization

Before feature engineering starts, the raw Squiggle payload is normalized in [`clean.py`](src/preprocessing/clean.py).

Key transformations:

- API fields such as `hteam`, `ateam`, `hscore`, and `ascore` are renamed into stable internal columns like `home_team`, `away_team`, `home_score`, and `away_score`.
- Team and venue names are normalized through lookup tables in [`constants.py`](src/utils/constants.py), so aliases from different sources collapse into one canonical label.
- Core targets are derived:
  - `margin = home_score - away_score`
  - `total_score = home_score + away_score`
  - `home_win = 1 if margin > 0 else 0`
- Scoring efficiency is computed as conversion rate:
  - `home_conversion = home_goals / (home_goals + home_behinds)`
  - `away_conversion = away_goals / (away_goals + away_behinds)`
- Dates are parsed and the dataset is sorted chronologically.
- `is_complete` is inferred so incomplete fixtures never leak placeholder scores into training targets.
- Finals are flagged via `is_final`.

This cleaning step is important because the feature builder assumes a consistent schema and reliable chronological ordering.

## Stage 2: Elo Features

The first feature family is a custom AFL-specific Elo system implemented by `EloSystem` in [`features.py`](src/preprocessing/features.py).

### Design

Each team begins at an initial rating from config:

- `elo_initial_rating = 1500`
- `elo_home_advantage = 35`
- `elo_k_factor = 40`
- `elo_season_regression = 0.3`

The expected score is the standard Elo expectation:

```text
E_home = 1 / (1 + 10^((R_away - R_home) / 400))
```

The actual result is not a binary win/loss. Instead, the margin is converted into a smooth score with a sigmoid:

```text
S_home = 1 / (1 + exp(-margin / 50))
```

That means:

- a narrow win produces a score just above `0.5`
- a big win produces a score closer to `1.0`
- a big loss produces a score closer to `0.0`

Ratings are then updated with:

```text
delta = K * (S_home - E_home)
R_home_new = R_home_old + delta
R_away_new = R_away_old - delta
```

### Leakage control

For every row, Elo features are recorded **before** the current match result is used to update the ratings. Only completed matches update the state.

At each season boundary, ratings are regressed partway back toward the population mean to avoid carrying full prior-season strength indefinitely.

### Output columns

- `elo_home_rating`
- `elo_away_rating`
- `elo_diff`
- `elo_home_win_prob`
- `elo_expected_margin`

`elo_expected_margin` is a rough linear conversion from rating gap:

```text
expected_margin = rating_diff * 0.05
```

## Stage 3: Rolling Form Features

Rolling form is engineered in `compute_team_rolling_stats()` and is the largest feature family.

### Team-centric reshaping

The function rewrites each match from each team's perspective:

- `team_score`
- `opp_score`
- `team_margin`
- `team_win`
- `team_goals`
- `team_behinds`
- `team_conversion`

This allows the same rolling logic to work for both home and away appearances.

### Lookback windows

The current implementation builds rolling features over:

- 3 matches
- 5 matches
- 10 matches

For each window it computes:

- win rate
- average margin
- average score
- average score conceded
- average conversion rate

Example column patterns:

- `rolling_3_win_rate`
- `rolling_5_avg_margin`
- `rolling_10_avg_score`
- `rolling_3_avg_conceded`
- `rolling_5_conversion`

### Exponentially weighted form

To emphasize recency, the pipeline also computes exponential moving averages with `span=5`:

- `ewm_win_rate`
- `ewm_margin`
- `ewm_score`

### Season-context features

The builder also carries season-context fields:

- `season_win_rate`
- `season_avg_margin`
- `season_games_played`

`season_games_played` is effectively a prior-games count within the season and helps the model understand early-season uncertainty.

### Leakage control

All rolling and EWM statistics use `.shift(1)` before aggregation, so the current match never contributes to its own pre-match feature values.

### Match-level projection

Rolling stats are first built on completed matches only. Then, for every scheduled match in the master table, the builder looks up the latest available rolling snapshot for:

- the home team
- the away team

Those snapshots are attached as:

- `home_rolling_*`, `home_ewm_*`, `home_season_*`
- `away_rolling_*`, `away_ewm_*`, `away_season_*`

This is what makes the same matrix usable for both historical backtesting and future fixture prediction.

## Stage 4: Head-to-Head Features

Head-to-head context is engineered by `compute_h2h_features()`.

### Lookback logic

For each match, the code searches only prior meetings between the two clubs within a 5-year window.

It then computes the matchup from the **current home team's perspective**:

- `h2h_home_wins`
- `h2h_away_wins`
- `h2h_home_win_rate`
- `h2h_avg_margin`
- `h2h_n_matches`

If there is no recent head-to-head history, the defaults are neutral:

- win rate defaults to `0.5`
- average margin defaults to `0.0`
- match count defaults to `0`

This gives the model matchup context without forcing it to overreact when the sample is small.

## Stage 5: Venue and Travel Features

Venue and travel effects are engineered in `compute_venue_features()` using reference data from [`constants.py`](src/utils/constants.py).

### Reference data

The repository defines:

- canonical team home states
- canonical venue states
- home-ground mappings
- approximate interstate travel distances

### Output columns

- `home_at_home_ground`
- `away_at_home_ground`
- `home_travel_km`
- `away_travel_km`
- `travel_diff_km`
- `home_interstate`
- `away_interstate`

These features capture a few distinct ideas:

- whether a team is truly at one of its home venues
- how far each side had to travel
- whether the match is interstate for either club

This supplements the generic Elo home-advantage constant with venue-specific context.

## Stage 6: Rest and Scheduling Features

Rest features are engineered in `compute_rest_features()`.

For each match, the pipeline tracks the last game date seen for every team and computes:

- `home_rest_days`
- `away_rest_days`
- `rest_diff`
- `home_short_rest`
- `away_short_rest`

The first observed match for a club defaults to `7` rest days so the feature space stays defined even when there is no prior history.

`home_short_rest` and `away_short_rest` are binary flags for rest periods of 6 days or fewer.

## Stage 7: Differential Features

Once all feature groups are merged, the builder creates difference features for every paired `home_*` and `away_*` column.

The transformation is:

```text
diff_feature = home_feature - away_feature
```

Examples:

- `diff_rolling_3_win_rate`
- `diff_rolling_10_avg_margin`
- `diff_ewm_score`
- `diff_season_games_played`
- `diff_at_home_ground`
- `diff_travel_km`
- `diff_interstate`
- `diff_rest_days`
- `diff_short_rest`

This is a key modeling choice. Instead of making the model independently infer the relationship between home and away values, the pipeline explicitly encodes the matchup gap.

Not every feature family uses this pattern:

- Elo already has dedicated matchup-level features such as `elo_diff` and `elo_home_win_prob`
- Head-to-head is already expressed from the current home side's perspective

## Stage 8: Match Context Features

Two final contextual features are added directly in the master builder:

- `round_progress = round / 24.0`
- `is_final`

`round_progress` gives the model a normalized sense of where the season sits. `is_final` distinguishes finals from regular-season matches.

## What the Final Matrix Looks Like

The builder starts with match metadata:

- `match_id`
- `year`
- `round`
- `date`
- `home_team`
- `away_team`
- `venue`
- `is_complete`

For completed matches it also carries the targets:

- `target_margin`
- `target_home_win`

Everything else becomes model input.

In the current generated CSV, `data/processed/feature_matrix.csv` contains:

- `102` total columns
- `92` model/input columns after excluding identifiers and targets

That total comes from the feature families above plus the differential expansion.

## Missing Values and Early-Season Behavior

Early in a season, some rolling fields naturally have little or no history behind them. The feature builder leaves those values as missing where appropriate.

The next stage, [`dataset.py`](src/preprocessing/dataset.py), handles this by:

- selecting only completed matches for supervised training
- excluding metadata and target columns from the feature list
- converting remaining feature columns to numeric form
- filling `NaN` values with the training-set column median

That means the feature engineering code can stay faithful to the real information available at prediction time, while the dataset builder handles model-ready imputation.

## Temporal Integrity Rules

Several engineering choices are explicitly there to avoid data leakage:

- matches are sorted by date before features are built
- Elo values are recorded before each match is used for updating
- rolling features are shifted by one game
- head-to-head uses only strictly earlier matches
- incomplete future fixtures keep null targets
- training uses chronological year-based splits rather than random shuffles

This is one of the most important properties of the pipeline. The model is always trained on features that mimic live prediction conditions.

## How the Feedback Loop Keeps Features Fresh

The orchestration layer in [`feedback_loop.py`](src/pipeline/feedback_loop.py) makes feature engineering part of the weekly operating cycle.

### Initial build

`Pipeline.build_features()`:

- loads match data from `matches_all.csv`
- calls `build_feature_matrix()`
- saves the resulting matrix and Elo object

### Upcoming fixtures

`Pipeline.refresh_upcoming_fixtures()`:

- fetches unplayed fixtures from Squiggle
- cleans them with the same schema
- merges them into `matches_all.csv`
- rebuilds the matrix so predictions can be generated immediately

### Post-round updates

`Pipeline.ingest_results()`:

- fetches newly completed matches
- upserts them into the historical match store
- rebuilds the full feature matrix
- evaluates predictions against actuals
- checks whether retraining is needed

So feature engineering is not a one-time preprocessing script. It is a persistent stateful part of the production loop.

## Why This Design Works Well

The current design combines three useful properties:

- **Stable long-term strength** from Elo
- **Short-term form** from rolling and EWM statistics
- **Contextual matchup effects** from head-to-head, venue, travel, rest, and season timing

Just as importantly, it does this in a way that supports both:

- historical model training
- forward-looking prediction for upcoming rounds

That makes the same feature system usable across backtesting, live forecasting, monitoring, and retraining.
