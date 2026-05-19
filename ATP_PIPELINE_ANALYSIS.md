# ATP Pipeline Analysis

## Scope

This report analyzes the current ATP main-tour prediction system as implemented in:

- `scripts/tennis_pipeline.py`
- `scripts/build_features.py`
- `scripts/train_model.py`
- `scripts/evaluate_model.py`
- `scripts/predict_match.py`
- `scripts/backtest_predictions.py`
- `scripts/clv_calculator.py`
- `tests/test_h2h.py`
- `README.md`

The Challenger data and SofaScore scraper exist, but they are not integrated into the ATP model pipeline yet.

## 1. Data Ingestion Pipeline

Raw ATP match history is loaded from `data/raw/atp_matches_*.csv` via `load_match_history_csvs()` in `scripts/tennis_pipeline.py`.

Rankings are loaded from `data/raw/atp_rankings_20s.csv` via `load_rankings_csv()` in `scripts/tennis_pipeline.py`.

Current observed raw main-tour files:

- `atp_matches_2020.csv` through `atp_matches_2026.csv`
- Total raw ATP rows: `17,287`
- Processed `features.csv`: `30,736` rows, meaning `15,368` accepted matches after filtering and symmetric duplication

The raw Challenger files under `data/raw_chal/` are not used by the current ATP pipeline.

## 2. Match Loading Process

`build_features.py` loads all ATP match files, parses `tourney_date`, sorts by `tourney_date`, and filters out:

- `RET`
- `W/O`
- `Walkover`
- `DEF`
- `tourney_level` in `D`, `O`

Important assumption: sorting is only by `tourney_date`, which is tournament-level date, not true match date. Same-tournament match order is therefore dependent on source row order after sorting and may not represent true chronological match order.

## 3. Chronological Replay System

The core architecture is a replay simulator.

For each match:

1. Initialize both players if unseen.
2. Refresh both players' rankings as of match date.
3. Compute pre-match features.
4. Write feature rows.
5. Update form, serve/return histories, Elo, H2H, and fatigue state after the row is captured.

This order prevents direct same-match leakage.

Training replay is in `scripts/build_features.py`.

Inference replay is duplicated in `scripts/predict_match.py`, but uses the same shared helper functions.

## 4. Player-State Storage Structure

Player state is a mutable nested dict initialized in `initialize_player()` in `scripts/tennis_pipeline.py`.

Each player stores:

- `name`
- `elo`
- `rank`
- `points`
- `surface_elo`: `Hard`, `Clay`, `Grass`
- `recent_results`
- `surface_recent_results`
- serve/return/BP global histories
- serve/return/BP surface histories
- `deciding_set_results`
- `matches_played`
- `wins`
- `last_match_date`
- `recent_minutes`
- `tournament_round`

Some stored state is currently unused as model features, especially fatigue and tournament round.

## 5. Feature Engineering Flow

Canonical feature order is defined once in `FEATURES` in `scripts/tennis_pipeline.py`.

Training creates two rows per match in `build_symmetric_training_rows()`:

- Winner as `player_a`, `result = 1`
- Loser as `player_a`, `result = 0`
- Differential features are sign-flipped for the mirrored row

Inference uses `compute_player_a_perspective_features()` to generate the same feature set from arbitrary `player_a` / `player_b` orientation.

## 6. Stateful Variables and Decay Logic

Decay helper: `_exponential_decay_mean()`.

Default decay:

- `alpha = 0.85`
- Most recent item receives weight `1.0`
- Empty histories return neutral `0.5`

Rolling windows:

- Recent form: last `10` matches
- Serve/return/BP stats: last `20` matches
- Surface serve/return/BP stats: surface-specific if at least `5` surface matches, else global fallback
- Deciding-set rate: last `20`, neutral until at least `3` deciding-set matches

Rank momentum uses a `12` week ATP points lookback.

## 7. Elo Implementation

Elo constants:

- `ELO_K = 32`
- Initial Elo: `1500`
- Initial surface Elo: `1500` per `Hard`, `Clay`, `Grass`

Both global Elo and surface Elo use the standard logistic expected score formula:

```text
expected = 1 / (1 + 10 ** ((loser_elo - winner_elo) / 400))
```

Winner gains `K * (1 - expected)`, loser loses the corresponding amount.

## 8. Serve / Return Calculations

Serve/return stats are computed in `compute_serve_return_stats()`.

Serve rating:

```text
1st serve in rate * 0.40
+ 1st serve won rate * 0.35
+ 2nd serve won rate * 0.25
```

Return rating is derived, not directly read:

```text
return_rating = 1 - opponent_serve_rating
```

Break-point save:

```text
bpSaved / bpFaced
```

Break-point conversion is also derived:

```text
bp_convert_rate = 1 - opponent_bp_save_rate
```

Serve histories are only appended if winner serve points exist and `w_svpt > 0`.

## 9. Momentum Calculations

There are two momentum concepts:

- Ranking momentum: percentage ATP points change over 12 weeks.
- Recent form momentum: decay-weighted recent win rate.

Rank momentum:

```text
(current_points - past_points) / (past_points + 1)
```

Recent form uses `recent_win_rate()` with window `10` and `alpha = 0.85`.

## 10. H2H Implementation

H2H state is a dict keyed by `frozenset({player_a, player_b})`.

Each record stores:

```text
(match_date, surface, winner_id)
```

H2H decay weights:

- `<= 1 year`: `1.00`
- `<= 3 years`: `0.75`
- `<= 5 years`: `0.50`
- `> 5 years`: `0.25`

If weighted total is below `2.0`, H2H returns neutral `0.5`.

Tests in `tests/test_h2h.py` validate neutrality, surface filters, antisymmetry, age decay, and exclusion of the current match.

## 11. Tournament Context Features

Tournament level is encoded in `encode_tourney_level()`.

Features:

- `is_grand_slam`: `tourney_level == "G"`
- `is_masters`: `tourney_level == "M"`
- `is_atp_open`: `tourney_level == "A"`

Everything else is baseline zero, including Finals `F` and Challenger `C`.

Round encoding exists in `encode_round()`, but round is not currently part of `FEATURES`.

## 12. Training Dataset Creation

`build_features.py` produces `data/processed/features.csv`.

Output columns include metadata, 23 model features, and `result`.

The system writes two symmetric rows per match, so the classifier learns "probability that player A wins" without positional bias.

Feature rows are created before Elo/form/H2H updates, preserving pre-match snapshots.

## 13. Inference Pipeline

`predict_match.py`:

1. Loads `models/xgboost_model.pkl`.
2. Replays full ATP history to rebuild current player state.
3. Reads `data/predict/today_matches.csv`.
4. Resolves names to player IDs using exact historical player names.
5. Uses optional row date from `date`, `match_date`, or `tourney_date`.
6. Refreshes rankings as of that reference date.
7. Computes player-A perspective features.
8. Runs batched `predict_proba`.
9. Writes `data/predict/predictions.csv`.

Main fragility: player-name matching is exact and based on historical names.

## 14. Backtesting System

`backtest_predictions.py` evaluates the held-out test window.

It loads `features.csv`, keeps rows with:

```text
date >= 2026-01-01
```

Then deduplicates mirrored rows into one canonical row per real match using:

```text
(date, surface, sorted(player IDs))
```

Observed current output:

- `data/processed/backtest_predictions.csv`
- Shape: `1009` rows

## 15. CLV Calculations

`clv_calculator.py` compares backtest probabilities to historical odds from `data/backtest/real_2026_odds.csv`.

Matching logic:

- Normalize names to ASCII.
- Build `surname|first_initial` keys.
- Search odds rows within `+/- 3` days.
- Require exactly one name match.
- Prefer surface match when deduplicating.

CLV formulas:

```text
clv_betfair  = predicted_prob_a - betfair_true_prob_a
clv_pinnacle = predicted_prob_a - pinnacle_true_prob_a
clv_max      = predicted_prob_a - max_true_prob_a
```

Observed current CLV output:

- `data/processed/clv_results.csv`
- Shape: `557` matched rows

## 16. Optuna / XGBoost Configuration

Outer split:

```text
train: date < 2026-01-01
test:  date >= 2026-01-01
```

Validation split:

- Last `20%` of unique pre-cutoff dates
- Implemented in `temporal_train_validation_split()`

Optuna:

- Direction: minimize
- Objective: validation log loss
- Sampler: `TPESampler(seed=42)`
- Trials: `50`

XGBoost search space:

- `n_estimators`: `50` to `300`
- `max_depth`: `3` to `10`
- `learning_rate`: `0.01` to `0.3`, log scale
- `subsample`: `0.5` to `1.0`
- `colsample_bytree`: `0.5` to `1.0`
- `random_state`: `42`

The final model is refit on all pre-2026 rows and saved to `models/xgboost_model.pkl`.

## 17. Feature List

Canonical model features, in order:

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

There are 23 model features.

## 18. Assumptions and ATP-Specific Hardcoding

Key ATP-specific assumptions:

- Match files must live in `data/raw/` and match `atp_matches_*.csv`.
- Rankings file must be `data/raw/atp_rankings_20s.csv`.
- Ranking schema assumes `ranking_date`, `rank`, `player`, `points`.
- Missing rank defaults to `2000`; missing points defaults to `0`.
- Supported surfaces are hardcoded as `Hard`, `Clay`, `Grass`.
- Tournament level encoding assumes ATP main-tour levels `G`, `M`, `A`.
- Davis Cup and Olympics are excluded via levels `D`, `O`.
- Challenger level `C` would currently be baseline tournament context, not a distinct feature.
- Grand Slam deciding-set logic assumes best-of-five only for `tourney_level == "G"`.
- Inference requires exact player-name matches against historical ATP names.
- Backtest cutoff is hardcoded to `2026-01-01`.
- CLV odds schema is hardcoded to tennis-data style columns like `BFEW`, `BFEL`, `PSW`, `PSL`, `MaxW`, `MaxL`.
- Current chronological replay depends on `tourney_date`, not actual match date.

## 19. Existing Challenger-Related Code

`scripts/scrape_sofascore_challenger_odds.py` is separate from the ATP model pipeline.

It scrapes SofaScore Challenger men's singles events and odds, caches JSON under `data/sofascore/events/` and `data/sofascore/odds/`, and writes CSV outputs under `data/sofascore/`.

It does not currently feed:

- `build_features.py`
- `train_model.py`
- `predict_match.py`
- `backtest_predictions.py`
- `clv_calculator.py`

Current Challenger raw match files under `data/raw_chal/` use the same 49-column Jeff Sackmann-style schema and have `tourney_level == "C"`.

## 20. Important Implementation Notes for Challenger Mirroring

To mirror ATP cleanly for Challengers later, the likely reusable core is:

- replay order
- player state structure
- Elo and surface Elo
- rolling form
- serve/return/BP formulas
- H2H structure
- symmetric row generation
- temporal split structure
- XGBoost/Optuna training loop

The parts that should be parameterized before Challenger expansion are:

- raw data directory
- match glob
- rankings source
- tournament level encoding
- supported surfaces
- output paths
- model path
- prediction input/output paths
- backtest odds source
- exact-name player resolution
- hardcoded 2026 cutoff
- ATP-only rank defaults and ranking availability assumptions
