# Tennis Match Prediction System

## Project Overview

This repository contains an end-to-end ATP match prediction pipeline built around chronological player-state reconstruction and an XGBoost classifier.

The project replays historical ATP matches in strict date order, continuously updates player strength signals, and creates pre-match features that are later used for classification and probability estimation.

The core forecasting setup is intentionally time-aware:

- train on history before `2026-01-01`
- evaluate on `2026-01-01` onward

To keep training and inference consistent, shared logic in `scripts/tennis_pipeline.py` is reused across feature generation, model training, evaluation, live prediction, and backtesting.

## Current Pipeline

Run scripts in this order:

1. `scripts/build_features.py`
2. `scripts/train_model.py`
3. `scripts/evaluate_model.py`
4. `scripts/predict_match.py`
5. `scripts/backtest_predictions.py`
6. `scripts/clv_calculator.py`

### What each stage does

- `build_features.py`: replays ATP history chronologically, updates player state (global Elo, surface Elo, form, win rate, matches played, rank, points), and writes `data/processed/features.csv`. Each match is stored in mirrored form (player-A winner row + player-A loser row).
- `train_model.py`: loads features, applies temporal split, tunes XGBoost with Optuna on a pre-2026 temporal validation tail (to avoid test leakage), then retrains on full pre-2026 data and saves `models/xgboost_model.pkl`.
- `evaluate_model.py`: evaluates the saved model on the held-out 2026+ window and generates calibration/importance/SHAP diagnostics.
- `predict_match.py`: reconstructs current player state from raw history and predicts probabilities for rows in `data/predict/today_matches.csv`, saving output to `data/predict/predictions.csv`.
- `backtest_predictions.py`: generates historical predictions on the 2026+ period and writes one canonical row per match to `data/processed/backtest_predictions.csv`.
- `clv_calculator.py`: joins backtest predictions with historical odds snapshots and computes CLV benchmarks in `data/processed/clv_results.csv`.

### Engineered Features

The model currently uses eight pre-match differential features:

- `elo_diff`
- `surface_elo_diff`
- `rank_diff`
- `points_diff`
- `recent_form_diff`
- `recent_surface_form_diff`
- `win_pct_diff`
- `matches_played_diff`

## Train/Test Split

- Split rule: `date < 2026-01-01` for training, `date >= 2026-01-01` for testing.
- Current processed dataset:
  - Total rows: `34,468`
  - Train rows: `32,130`
  - Test rows: `2,338`
  - Unique test matches: `1,169` (features are mirrored, so 2 rows per match before canonicalization)

## Current Model Performance

Using the latest saved model in `models/xgboost_model.pkl` evaluated on the 2026+ test split:

- Accuracy: `65.57%`
- Log loss: `0.6109`

Notes:
- Hyperparameter search is now done only inside the pre-2026 training window.
- Final test metrics above are from the untouched 2026+ holdout.

This is a stricter and more realistic evaluation setup than tuning directly on the test period.

## CLV Backtesting (2026 Odds Snapshot)

From `data/processed/clv_results.csv`:

- Matched matches: `557`
- Average Betfair CLV: `+0.0197`

### CLV by surface (average `clv_betfair`)

- Clay: `+0.0324`
- Hard: `+0.0150`

### CLV by month (average `clv_betfair`)

- 2026-01: `+0.0117` (`160` matches)
- 2026-02: `+0.0178` (`182` matches)
- 2026-03: `+0.0214` (`138` matches)
- 2026-04: `+0.0376` (`77` matches)

Interpretation:
- CLV is positive overall across the matched sample.
- Clay currently shows stronger average CLV than Hard in this snapshot.
- Monthly CLV trends upward across the observed January-April 2026 window.

## Data Sources

- Historical ATP results and rankings: Jeff Sackmann ATP datasets (stored locally under `data/raw/atp_matches_*.csv` and `data/raw/atp_rankings_20s.csv`).
- Historical bookmaker/market odds for backtesting: tennis-data.co.uk exports (local file `data/backtest/real_2026_odds.csv`).

The pipeline assumes these local files are present, even if they are ignored by git.

## Repository Structure

Git-tracked structure:

```text
.
├── data/
│   ├── processed/
│   │   ├── features.csv
│   │   ├── backtest_predictions.csv
│   │   ├── clv_results.csv
│   │   └── clv_cumulative.png
│   ├── predict/
│   │   ├── today_matches.csv
│   │   └── predictions.csv
├── notebooks/
├── scripts/
│   ├── tennis_pipeline.py
│   ├── build_features.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   ├── predict_match.py
│   ├── backtest_predictions.py
│   └── clv_calculator.py
├── requirements.txt
└── README.md
```

Local directories used by the pipeline but ignored by git (see `.gitignore`):

- `data/raw/` (Sackmann match/ranking inputs)
- `data/backtest/` (odds snapshots such as `real_2026_odds.csv`)
- `models/` (trained model artifact)
- `env/` (local virtual environment)

## Running the Pipeline

Typical run sequence:

```bash
./env/bin/python scripts/build_features.py
./env/bin/python scripts/train_model.py
./env/bin/python scripts/evaluate_model.py
./env/bin/python scripts/predict_match.py
./env/bin/python scripts/backtest_predictions.py
./env/bin/python scripts/clv_calculator.py
```

Input/output checkpoints:

- Prediction input: `data/predict/today_matches.csv`
- Prediction output: `data/predict/predictions.csv`
- Backtest output: `data/processed/backtest_predictions.csv`
- CLV output: `data/processed/clv_results.csv`

## Technical Notes

- Feature generation is chronological by construction, so each row only uses information available before that match.
- Rank/points logic is aligned between training and inference through shared ranking-history lookups.
- Hyperparameter tuning uses a temporal validation slice inside pre-2026 data, leaving 2026+ for final holdout testing.
- CLV matching uses fuzzy player-name keys plus a date window and surface sanity checks.

## Planned Future Improvements

- Replace single holdout tuning with rolling time-based cross-validation.
- Add robust ID-level matching for CLV joins (instead of fuzzy name/date matching).
- Add richer features: rest/fatigue windows, tournament round/context, and travel/surface transition effects.
- Explicitly model uncertainty and calibration drift over time.
- Add automated data validation checks (date ranges, duplicate detection, missing odds coverage).

## Disclaimer

This project is for research and educational use. It is not financial or betting advice.
