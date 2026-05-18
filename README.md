# ATP Tennis Match Prediction System

## 1. Project Overview

Hobby quant sports betting pipeline for predicting ATP tennis match outcomes with the specific objective of beating closing line value (CLV), especially against Betfair Exchange closing prices.

Context:

- Romanian bettor
- Betfair Exchange access
- Superbet account
- Built in Python
- Historical match data from Jeff Sackmann's `tennis_atp`
- Historical odds from `tennis-data.co.uk`

The project is not framed as a generic winner-picking classifier. The practical benchmark is whether pre-match probabilities are strong enough to outperform the market close.

## 2. Pipeline Scripts

Execution order:

1. `scripts/tennis_pipeline.py`
2. `scripts/build_features.py`
3. `scripts/train_model.py`
4. `scripts/evaluate_model.py`
5. `scripts/predict_match.py`
6. `scripts/backtest_predictions.py`
7. `scripts/clv_calculator.py`

### `scripts/tennis_pipeline.py`

Role: shared pipeline module used by feature generation, training, evaluation, inference, and backtest scripts.

Inputs:

- `data/raw/atp_matches_*.csv`
- `data/raw/atp_rankings_20s.csv`

Outputs:

- No direct artifact; provides shared constants, paths, replay logic, feature definitions, and helper functions.

Key logic:

- Defines the canonical `FEATURES` list and label column.
- Defines the temporal cutoff:

```python
MODEL_EVAL_CUTOFF_DATE = "2026-01-01"
```

- Implements chronological player-state replay.
- Implements Elo and surface Elo updates.
- Implements ATP rank and points lookups as of match date.
- Implements rank momentum, rolling form, serve/return, break-point, deciding-set, and H2H features.
- Implements shared train/test and train/validation temporal split functions.
- Keeps training and inference feature logic aligned.

### `scripts/build_features.py`

Role: replay ATP history chronologically and build the supervised training matrix.

Inputs:

- `data/raw/atp_matches_2020.csv` through `data/raw/atp_matches_2026.csv`
- `data/raw/atp_rankings_20s.csv`

Output:

- `data/processed/features.csv`

Key logic:

- Loads and concatenates all ATP match CSVs.
- Parses and sorts matches by `tourney_date`.
- Filters out non-standard results and unsupported event classes:

```python
RET | W/O | Walkover | DEF
tourney_level in {"D", "O"}
```

- Initializes player state on first appearance.
- Refreshes both players' ranking snapshot before each match.
- Computes pre-match feature differences from the winner perspective.
- Writes two symmetric rows per match:
  - player A = actual winner, `result = 1`
  - player A = actual loser, `result = 0`
- Updates player state only after features for that match are written.

### `scripts/train_model.py`

Role: train the XGBoost model with Optuna hyperparameter tuning.

Input:

- `data/processed/features.csv`

Output:

- `models/xgboost_model.pkl`

Key logic:

- Loads `features.csv` and parses `date`.
- Splits data temporally:

```python
train: date < 2026-01-01
test:  date >= 2026-01-01
```

- Creates a temporal validation split from the last 20% of unique pre-2026 dates.
- Runs Optuna for 50 trials with a TPE sampler.
- Minimizes validation log loss only on the pre-2026 window.
- Refits the best XGBoost configuration on the full pre-2026 training set.
- Prints held-out test accuracy and log loss.

### `scripts/evaluate_model.py`

Role: offline diagnostics for the saved model on the held-out 2026+ window.

Inputs:

- `data/processed/features.csv`
- `models/xgboost_model.pkl`

Outputs:

- Printed metrics
- Interactive plots

Key logic:

- Reuses the same temporal split as `train_model.py`.
- Computes accuracy and log loss on the held-out test set.
- Plots calibration curve.
- Plots XGBoost feature importance.
- Generates SHAP summary and local waterfall explanations.

### `scripts/predict_match.py`

Role: rebuild current player state from history, then score upcoming matches.

Inputs:

- `data/raw/atp_matches_2020.csv` through `data/raw/atp_matches_2026.csv`
- `data/raw/atp_rankings_20s.csv`
- `data/predict/today_matches.csv`
- `models/xgboost_model.pkl`

Output:

- `data/predict/predictions.csv`

Key logic:

- Replays the full historical ATP dataset without writing training rows.
- Uses the same filtering rules and state update order as `build_features.py`.
- Resolves prediction-row reference date from:

```python
date
match_date
tourney_date
```

- Refreshes rankings for each prediction row as of that row's reference date.
- Computes player-A-versus-player-B features using the same shared logic as training.
- Runs one batched `predict_proba` call.
- Writes a compact output with match label, predicted winner, and winner probability.

### `scripts/backtest_predictions.py`

Role: create one-row-per-match historical predictions for the held-out evaluation period.

Inputs:

- `data/processed/features.csv`
- `models/xgboost_model.pkl`

Output:

- `data/processed/backtest_predictions.csv`

Key logic:

- Keeps only rows from `2026-01-01` onward.
- Deduplicates the mirrored training rows by building a canonical `match_key` from date, surface, and sorted player IDs.
- Scores the canonical test rows with the trained model.
- Stores predicted probability for player A together with the realized outcome.

### `scripts/clv_calculator.py`

Role: benchmark historical predictions against archived closing odds and compute CLV.

Inputs:

- `data/processed/backtest_predictions.csv`
- `data/backtest/real_2026_odds.csv`

Outputs:

- `data/processed/clv_results.csv`
- `data/processed/clv_cumulative.png`

Key logic:

- Normalizes names and surfaces for cross-dataset matching.
- Fuzzy-matches players using a `surname|first_initial` key.
- Searches odds rows within `+/- 3` days of prediction date.
- Accepts only unique matched candidates.
- Removes margin from two-way prices for Betfair, Pinnacle, and Max odds.
- Maps market probabilities back to player A orientation.
- Computes:

```python
clv_betfair  = predicted_prob_a - betfair_true_prob_a
clv_pinnacle = predicted_prob_a - pinnacle_true_prob_a
clv_max      = predicted_prob_a - max_true_prob_a
```

- Prints aggregate, surface, monthly, bucketed, and cumulative CLV summaries.

## 3. Current Feature Set

The current shared `FEATURES` list contains `23` model features. They are all differential features from player A minus player B, except the three tournament one-hot flags which are match-level context variables applied identically to both mirrored rows.

Canonical feature order:

```python
[
    "elo_diff",
    "surface_elo_diff",
    "is_grand_slam",
    "is_masters",
    "is_atp_open",
    "rank_diff",
    "points_diff",
    "rank_momentum_diff",
    "recent_form_diff",
    "recent_surface_form_diff",
    "win_pct_diff",
    "matches_played_diff",
    "deciding_set_win_rate_diff",
    "serve_rating_diff",
    "return_rating_diff",
    "bp_save_rate_diff",
    "bp_convert_rate_diff",
    "serve_rating_surface_diff",
    "return_rating_surface_diff",
    "bp_save_rate_surface_diff",
    "bp_convert_rate_surface_diff",
    "h2h_win_rate_diff",
    "h2h_surface_win_rate_diff",
]
```

### Elo Features

#### `elo_diff`

Captures overall rating edge. Computed as player A overall Elo minus player B overall Elo, with Elo updated sequentially after each historical match.

#### `surface_elo_diff`

Captures surface-specific rating edge. Computed as player A surface Elo minus player B surface Elo on the current surface.

### Ranking Features

#### `rank_diff`

Captures ATP ranking advantage. Computed as:

```python
rank_diff = player_b_rank - player_a_rank
```

This makes a better ranking for player A produce a positive feature.

#### `points_diff`

Captures raw ATP ranking-points edge. Computed as player A ranking points minus player B ranking points, using the latest weekly ranking snapshot on or before match date.

#### `rank_momentum_diff`

Captures recent ranking-points trend. For each player:

```python
(current_points - points_12_weeks_ago) / (points_12_weeks_ago + 1)
```

The feature is player A momentum minus player B momentum.

### Form Features

#### `recent_form_diff`

Captures recent overall form. Computed as the difference in decay-weighted win rate over each player's last 10 matches.

#### `recent_surface_form_diff`

Captures recent form on the current surface. Computed as the difference in decay-weighted win rate over each player's recent surface-specific results.

#### `win_pct_diff`

Captures cumulative career-in-sample win rate edge. Computed as player A historical wins divided by player A historical matches, minus the same value for player B.

#### `matches_played_diff`

Captures relative sample depth and match experience. Computed as player A historical ATP matches played minus player B historical ATP matches played.

#### `deciding_set_win_rate_diff`

Captures pressure performance in deciding sets. Computed as the difference in each player's decay-weighted deciding-set win rate over the last 20 deciding-set results, with a neutral fallback until a player has at least 3 such matches.

### Serve/Return Features

#### `serve_rating_diff`

Captures rolling serve quality. Each match-level serve rating is:

```text
1stServe% * 0.40 + 1stWon% * 0.35 + 2ndWon% * 0.25
```

The feature is the difference in each player's decay-weighted rolling serve rating history.

#### `return_rating_diff`

Captures rolling return quality. It is derived from opponent serve quality:

```python
return_rating = 1 - opponent_serve_rating
```

The feature is player A rolling return rating minus player B rolling return rating.

#### `bp_save_rate_diff`

Captures rolling break-point save ability. Computed from historical `bpSaved / bpFaced`, then decay-averaged through time, and differenced between player A and player B.

#### `bp_convert_rate_diff`

Captures rolling break-point conversion ability. It is derived from the opponent's break-point save rate:

```python
bp_convert_rate = 1 - opponent_bp_save_rate
```

The feature is player A rolling conversion rate minus player B rolling conversion rate.

#### `serve_rating_surface_diff`

Captures surface-specific serve strength. Computed from the current surface history if a player has at least 5 surface matches; otherwise it falls back to global serve history. The feature is player A minus player B.

#### `return_rating_surface_diff`

Captures surface-specific return strength. Computed from current surface return history if at least 5 surface matches exist; otherwise it falls back to global return history. The feature is player A minus player B.

#### `bp_save_rate_surface_diff`

Captures surface-specific break-point saving ability, with the same surface-history threshold and global fallback. The feature is player A minus player B.

#### `bp_convert_rate_surface_diff`

Captures surface-specific break-point conversion ability, with the same surface-history threshold and global fallback. The feature is player A minus player B.

### H2H Features

#### `h2h_win_rate_diff`

Captures overall head-to-head edge. For each player, the win rate is computed from prior meetings only, using age-based time decay, and defaults to neutral if fewer than `2.0` weighted matches exist. The feature is player A H2H win rate minus player B H2H win rate.

#### `h2h_surface_win_rate_diff`

Captures surface-specific head-to-head edge using the same time-decay schedule and weighted-match threshold, but only on the current surface.

### Tournament Features

#### `is_grand_slam`

Match-level one-hot flag:

```python
int(tourney_level == "G")
```

#### `is_masters`

Match-level one-hot flag:

```python
int(tourney_level == "M")
```

#### `is_atp_open`

Match-level one-hot flag:

```python
int(tourney_level == "A")
```

Unknown or other levels are treated as baseline zeros across all three tournament flags.

## 4. Key Engineering Decisions

### Temporal train/test split

No random shuffle. The model is evaluated strictly forward in time.

```python
train: date < 2026-01-01
test:  date >= 2026-01-01
```

### Symmetric duplicate rows

Every historical match appears twice in `features.csv`, once from each player perspective. This enforces training symmetry and avoids positional bias. `backtest_predictions.py` deduplicates those mirrored rows back to one canonical row per real match.

### Exponential decay weighting

All rolling form, serve, return, and break-point statistics use exponential decay with:

```python
alpha = 0.85
```

The most recent match receives weight `1.0`.

### H2H time decay

H2H win rates use age-based weights:

```text
<= 1 year  -> 1.00
<= 3 years -> 0.75
<= 5 years -> 0.50
>  5 years -> 0.25
```

H2H defaults to neutral until at least `2.0` weighted matches exist.

### Serve rating composite formula

Per-match serve rating is:

```text
1stServe% * 0.40 + 1stWon% * 0.35 + 2ndWon% * 0.25
```

### Return/BP features derived from opponent stats

Return and break-point conversion are not read directly from separate raw columns. They are derived from opponent serve and break-point prevention:

```python
return_rating   = 1 - opponent_serve_rating
bp_convert_rate = 1 - opponent_bp_save_rate
```

### Serve validity guard

Serve/return and break-point histories are only appended when serve stats are valid:

```python
w_svpt > 0
```

### Surface-specific serve stats fallback

Surface-specific serve, return, and break-point features use global history when a player has fewer than `5` matches on that surface.

### Deciding set fallback

Deciding-set win rate defaults to `0.5` until a player has at least `3` deciding-set matches in history.

### Rank momentum lookback

Rank momentum uses a `12`-week ranking-points lookback.

### Retirement/Davis Cup/Olympics filter

Rows are removed before feature generation and inference replay if they are:

```text
RET, W/O, Walkover, DEF
tourney_level D or O
```

### Optuna hyperparameter tuning

Training uses:

- `50` Optuna trials
- `TPESampler(seed=42)`
- temporal validation split from the last `20%` of unique pre-2026 dates

### Train/inference parity

`predict_match.py` replays history with the same shared helper functions and the same state update order used by `build_features.py`, so feature generation is aligned between training and live inference.

## 5. Current Performance (Jan-Apr 2026 Test Set)

```text
Accuracy:    65.76%
Log Loss:    0.6056
Betfair CLV: +0.0248
Matched:     557 matches
```

By surface:

```text
Clay: +0.0345
Hard: +0.0213
```

Monthly Betfair CLV trend:

```text
Jan: +0.0275
Feb: +0.0190
Mar: +0.0235
Apr: +0.0355
```

## 6. Raw Data Sources

Match CSVs:

```text
atp_matches_2020.csv
atp_matches_2021.csv
atp_matches_2022.csv
atp_matches_2023.csv
atp_matches_2024.csv
atp_matches_2025.csv
atp_matches_2026.csv
```

Rankings:

```text
atp_rankings_20s.csv
```

Odds:

```text
real_2026_odds.csv
```

Odds file contents referenced in the project:

```text
Betfair Exchange
Pinnacle
Max odds
Bet365
```

Sources:

- Jeff Sackmann `tennis_atp` GitHub repository
- `tennis-data.co.uk`

## 7. What Was Tried and Abandoned

### Fatigue features

Abandoned because `tourney_date` is tournament-level, not true per-match date, which caused leakage and an unrealistic accuracy jump to `77.9%`.

### Time decay training weights

Tried at model-training level and dropped because there was no clear benefit with only 2020-2026 data.

### Ace rate + double fault rate

Dropped as redundant with `serve_rating_diff` and weak in SHAP importance.

### Game dominance

Game-win-share style features were dropped because they showed no meaningful SHAP signal and overlapped with the current feature set.

### Opponent-adjusted stats

Dropped for now because complexity was high relative to the apparent incremental value on the current dataset.

## 8. Deferred Features Memo

These are explicitly deferred until the data foundation changes.

### Fatigue features

Revisit only when true per-match dates are available instead of `tourney_date`.

### Time decay training weights

Revisit when expanding the history before 2020.

### Ace rate + double fault rate

Revisit with a larger dataset.

### Opponent-adjusted stats

Revisit at Challenger expansion.

### Surface transition/familiarity

Revisit when per-match dates become available.

### Best-of-five experience

Deferred because current per-player sample depth is too thin.

### Tournament-level performance splits

Revisit with a larger dataset.

### Tournament level encoding

Recheck value counts when switching databases.

## 9. Tech Stack

```text
Python
XGBoost
Optuna
pandas
scikit-learn
SHAP
joblib
```

## 10. GitHub

```text
alexandrumocanu0721-commits/tennis-match-prediction
```
