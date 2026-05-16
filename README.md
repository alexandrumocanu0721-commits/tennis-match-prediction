# ATP Tennis Match Prediction System

## Overview

This repository contains a quantitative ATP match prediction pipeline built around one practical objective: producing pre-match probabilities that beat the closing market, specifically Betfair Exchange closing prices.

The project does not treat headline accuracy as the primary success criterion. Accuracy and log loss still matter, but the real test is whether the model's probabilities are systematically better than the final market consensus. In betting-market terms, that means positive closing line value (CLV).

That is a materially harder bar than simply classifying winners on historical data. Closing prices already aggregate large amounts of public information, sharp action, and market corrections. Consistently outperforming them, even by a few probability points on average, is a meaningful result.

## Objective

The system is designed to:

- reconstruct player state strictly through time
- generate leakage-free pre-match features
- train a calibrated binary classifier for ATP matches
- score future matches and historical holdout matches
- evaluate whether the model's probabilities improve on Betfair closing odds

In short: this is a probability estimation project with a market benchmark, not a generic sports-prediction classifier.

## Pipeline

Run the project in this order.

### 1. `scripts/build_features.py`

Chronologically replays the ATP match history, updates player state after each completed match, and writes the supervised training matrix to `data/processed/features.csv`.

Key behaviors:

- loads all historical ATP results from `data/raw/atp_matches_*.csv`
- removes retirements, walkovers, and similar non-standard endings
- pulls ATP rankings and ranking points from `data/raw/atp_rankings_20s.csv`
- snapshots each player's state before the current match
- writes two symmetric rows per match:
  - player A = actual winner, `result = 1`
  - player A = actual loser, `result = 0`

This script is the core state engine. Everything downstream depends on it being temporally correct.

### 2. `scripts/train_model.py`

Loads `features.csv`, applies a temporal split, tunes XGBoost with Optuna on a late slice of the pre-2026 training window, retrains on the full pre-2026 sample, and saves the model to `models/xgboost_model.pkl`.

Key behaviors:

- no random shuffle
- train set: matches before `2026-01-01`
- test set: matches on or after `2026-01-01`
- Optuna minimizes validation log loss on the tail of the training period
- best hyperparameters are refit on the full training window before serialization

### 3. `scripts/evaluate_model.py`

Reloads the saved model and evaluates it on the held-out 2026+ window using the same temporal split used in training.

Outputs:

- accuracy
- log loss
- calibration curve
- XGBoost feature importance plot
- SHAP summary and local explanation views

This is the offline model-diagnostics step. It measures predictive quality, calibration, and feature behavior on unseen forward data.

### 4. `scripts/predict_match.py`

Replays the full historical match dataset to rebuild current player state, then scores rows from `data/predict/today_matches.csv` and writes `data/predict/predictions.csv`.

Key behaviors:

- uses the same shared feature logic as training
- supports per-row ranking reference dates through `date`, `match_date`, or `tourney_date`
- performs one batched `predict_proba` call after feature assembly

This is the live inference path.

### 5. `scripts/backtest_predictions.py`

Applies the trained model to the historical 2026+ holdout window and writes canonical one-row-per-match predictions to `data/processed/backtest_predictions.csv`.

Key behaviors:

- starts from `features.csv`
- keeps only the holdout period
- collapses the mirrored training rows into a stable single orientation per real match
- stores predicted probability for player A alongside the realized outcome

This artifact is the input for CLV benchmarking.

### 6. `scripts/clv_calculator.py`

Joins historical model predictions with archived market odds from `data/backtest/real_2026_odds.csv`, removes margin from the two-way prices, and computes CLV relative to multiple benchmarks.

Key behaviors:

- fuzzy player-name normalization for cross-dataset matching
- date matching within a small window
- surface normalization as a sanity check
- no-vig conversion from odds to implied true probabilities
- CLV outputs for Betfair, Pinnacle, and max market odds
- monthly, surface-level, bucketed, and cumulative CLV summaries

Primary output:

- `data/processed/clv_results.csv`

Auxiliary output:

- `data/processed/clv_cumulative.png`

## Features

The model currently uses 14 pre-match differential features. All are expressed from the perspective of player A minus player B.

1. `elo_diff`  
   Difference in overall Elo rating.

2. `surface_elo_diff`  
   Difference in surface-specific Elo on Hard, Clay, or Grass.

3. `rank_diff`  
   ATP ranking advantage, encoded so better rank for player A produces a positive value.

4. `points_diff`  
   Difference in ATP ranking points as of the match date.

5. `recent_form_diff`  
   Difference in trailing overall win rate over the most recent results window.

6. `recent_surface_form_diff`  
   Difference in trailing win rate on the current surface.

7. `win_pct_diff`  
   Difference in cumulative historical win percentage.

8. `matches_played_diff`  
   Difference in total historical ATP matches played in the replayed sample.

9. `serve_rating_diff`  
   Difference in rolling serve quality. The per-match serve rating is a weighted composite:
   `0.40 * first-serve-in rate + 0.35 * first-serve points-won rate + 0.25 * second-serve points-won rate`,
   then averaged over recent history.

10. `return_rating_diff`  
    Difference in rolling return quality, derived from the opponent's serve rating and averaged through time.

11. `bp_save_rate_diff`  
    Difference in rolling break-point save rate.

12. `bp_convert_rate_diff`  
    Difference in rolling break-point conversion rate.

13. `h2h_win_rate_diff`  
    Difference in overall head-to-head win rate, with time-decay weighting by recency.

14. `h2h_surface_win_rate_diff`  
    Difference in surface-specific head-to-head win rate, also time-decayed.

## Engineering Decisions

Several design choices matter more here than the specific model family.

### Temporal train/test split

The dataset is split by date, not randomly shuffled.

- train: `date < 2026-01-01`
- test: `date >= 2026-01-01`

That keeps evaluation aligned with the real deployment problem: predict the future using only the past.

### Symmetric duplicate rows for training

Each historical match is written twice in `features.csv`, once from each player perspective. This makes the training set symmetric and forces the model to learn relative strength rather than memorize positional bias.

### Chronological state replay

Player ratings, form, rankings, serve metrics, and head-to-head history are built by replaying the raw ATP matches in date order. Every row is generated from information available before that match started.

### H2H time-decay weighting

Head-to-head is not treated as a flat lifetime count. More recent meetings receive higher weight:

- within 1 year: `1.00`
- 1 to 3 years: `0.75`
- 3 to 5 years: `0.50`
- older than 5 years: `0.25`

If the weighted head-to-head sample is too thin, the feature falls back to a neutral prior instead of forcing a noisy edge.

### Serve rating weighted composite

Serve strength is not pulled directly from one raw stat. It is built from a weighted composite of first-serve-in rate, first-serve effectiveness, and second-serve effectiveness, then rolled through recent history. That gives a more stable skill proxy than any single component on its own.

### Optuna hyperparameter tuning

Hyperparameters are tuned with Optuna inside the pre-2026 sample only, using a temporal validation tail. The 2026+ holdout remains untouched until final evaluation.

## Model Performance

Held-out performance on the current evaluation window:

- Accuracy: `66.02%`
- Log Loss: `0.6049`

These numbers are useful, but they are secondary to the market benchmark. A sports model can post respectable accuracy and still fail where it matters if its probabilities do not beat the close.

## CLV Results

Average CLV versus Betfair Exchange closing probabilities:

- Overall: `+0.0217`
- Clay: `+0.0357`
- Hard: `+0.0166`

Monthly average Betfair CLV:

- January: `+0.0168`
- February: `+0.0172`
- March: `+0.0240`
- April: `+0.0383`

Interpretation:

- the aggregate Betfair CLV is positive
- the result is positive across both major sampled surfaces
- the monthly path is also positive and improving over the observed period

That does not prove long-run tradability on its own, but it is strong evidence that the model is extracting information the closing market does not fully price in ahead of time.

## Technologies

- Python
- pandas
- numpy
- scikit-learn
- XGBoost
- Optuna
- SHAP
- matplotlib
- joblib
- Jupyter ecosystem for analysis and iteration

See [requirements.txt](/Users/alexmocanu/Documents/Proiecte/Proiect Tenis/requirements.txt) for the exact package versions currently pinned in the project.

## Data

The pipeline expects local CSV inputs in these directories:

- `data/raw/`
  - `atp_matches_2020.csv` through `atp_matches_2026.csv`
  - `atp_rankings_20s.csv`
- `data/backtest/`
  - `real_2026_odds.csv`

The historical ATP data and rankings are structured around Jeff Sackmann-style files. The CLV benchmark uses archived odds snapshots aligned against the model holdout period.

## Project Structure

```text
.
├── README.md
├── requirements.txt
├── data/
│   ├── backtest/
│   │   └── real_2026_odds.csv
│   ├── predict/
│   │   ├── predictions.csv
│   │   └── today_matches.csv
│   ├── processed/
│   │   ├── backtest_predictions.csv
│   │   ├── clv_cumulative.png
│   │   ├── clv_results.csv
│   │   └── features.csv
│   └── raw/
│       ├── atp_matches_2020.csv
│       ├── atp_matches_2021.csv
│       ├── atp_matches_2022.csv
│       ├── atp_matches_2023.csv
│       ├── atp_matches_2024.csv
│       ├── atp_matches_2025.csv
│       ├── atp_matches_2026.csv
│       └── atp_rankings_20s.csv
├── models/
│   └── xgboost_model.pkl
├── notebooks/
├── scripts/
│   ├── backtest_predictions.py
│   ├── build_features.py
│   ├── clv_calculator.py
│   ├── evaluate_model.py
│   ├── predict_match.py
│   ├── tennis_pipeline.py
│   └── train_model.py
└── tests/
    └── test_h2h.py
```

## Typical Run Sequence

```bash
python scripts/build_features.py
python scripts/train_model.py
python scripts/evaluate_model.py
python scripts/predict_match.py
python scripts/backtest_predictions.py
python scripts/clv_calculator.py
```

## Disclaimer

This repository is a research system for quantitative sports modeling. It is not betting advice, not an execution engine, and not a guarantee of future edge. Market conditions, data quality, liquidity, and implementation details all matter. Positive historical CLV is encouraging, but any claim of durable profitability requires continued out-of-sample validation and live process discipline.
