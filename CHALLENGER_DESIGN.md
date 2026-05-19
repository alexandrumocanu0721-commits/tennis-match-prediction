# Challenger Integration Design

## Goal

Build an ATP Challenger prediction pipeline that mirrors the existing ATP main-tour architecture as closely as possible while keeping Version 1 fully isolated.

Version 1 rule:

```text
Challenger model state = Challenger-only match history.
ATP model state = ATP-only match history.
No ATP player state, form, Elo, H2H, ranking-derived state, or serve/return history may enter the Challenger pipeline.
```

Player IDs must remain globally compatible with future ATP merging. Challenger rows should keep the original Jeff Sackmann `winner_id` and `loser_id` values without remapping to a local Challenger-only ID space.

## Version Plan

### V1: Isolated Challenger Ecosystem

Training:

- `2020` through `2025` Challenger matches
- Challenger-only player state
- Challenger-only Elo
- Challenger-only surface Elo
- Challenger-only form
- Challenger-only serve/return histories
- Challenger-only H2H

Backtesting:

- `2026` Challenger matches
- No random split
- Forward temporal evaluation, matching ATP approach

Odds benchmark:

- bet365 odds scraped from SofaScore
- Source currently represented as `bet365_via_sofascore`
- Closing or latest pre-match odds are treated as the benchmark unless a better timestamped close becomes available

### V2: Shared ATP + Challenger State

Future work can combine ATP and Challenger history into a shared state model. V1 should not block that direction.

To preserve the path to V2:

- Keep global player IDs unchanged.
- Avoid Challenger-only synthetic IDs.
- Parameterize data roots and output paths instead of copying hardcoded ATP paths.
- Keep state object shape compatible with the ATP `players` dict.
- Keep feature names and definitions aligned where possible.

## 1. Existing Modules Reusable Without Changes

These parts of the current ATP architecture can be reused conceptually and, after import path decisions, should not need behavior changes for V1.

### Core Math and Rolling Helpers

Reusable:

- `_exponential_decay_mean()`
- `recent_win_rate()`
- `rolling_serve_stat()`
- `surface_or_global_stat()`
- `count_completed_sets()`
- `deciding_set_win_rate()`

Reason:

These functions do not depend on ATP-specific files. They operate on in-memory histories and are valid for Challenger-only state.

### Elo Logic

Reusable:

- `update_elo_and_match_counts()`

Reason:

The Elo algorithm is tour-agnostic as long as it is applied to Challenger-only player state.

Important V1 condition:

The `players` dict passed to Elo updates must be built only from Challenger matches.

### Serve / Return Calculation

Reusable:

- `compute_serve_return_stats()`
- `append_recent_result_lists()`

Reason:

The Challenger CSV schema matches the ATP match schema, including serve and break-point columns.

### H2H Logic

Reusable:

- `initialize_h2h_records()`
- `update_h2h_records()`
- `compute_h2h_win_rate()`
- `compute_h2h_diffs()`

Reason:

The H2H structure is independent of tour level. V1 must build H2H from Challenger matches only.

### Feature Row Symmetry

Reusable:

- `build_symmetric_training_rows()`
- `compute_player_a_perspective_features()`

Reason:

The player-A orientation and mirrored-row design should remain identical to ATP.

Potential caveat:

Tournament-level flags may need Challenger-specific handling before these functions can be reused unchanged.

### Temporal Split Concept

Reusable concept:

- forward-only temporal split
- Optuna validation split from late training window
- held-out future backtest

The current `temporal_train_test_split_for_modeling()` can be reused only if the Challenger cutoff is the same hardcoded `2026-01-01`, which matches V1.

## 2. Modules Needing Parameterization

The following ATP modules should not be copied wholesale with hardcoded paths. They need configuration parameters so ATP and Challenger can use the same architecture without state mixing.

### `tennis_pipeline.py`

Needs parameterization for:

- raw match directory
- raw match glob
- rankings path or ranking source
- processed features output path
- prediction input path
- prediction output path
- backtest prediction output path
- CLV output path
- model path
- backtest odds path
- tournament-level encoding
- model evaluation cutoff
- supported tournament levels
- optional pipeline namespace, e.g. `atp` vs `challenger`

Current ATP hardcoding to isolate:

- `data/raw`
- `atp_matches_*.csv`
- `atp_rankings_20s.csv`
- `data/processed/features.csv`
- `models/xgboost_model.pkl`
- `data/backtest/real_2026_odds.csv`
- tournament flags for `G`, `M`, `A`

Recommended V1 design:

Create a small pipeline configuration object, for example:

```text
PipelineConfig(
    name="challenger",
    raw_dir="data/raw_chal",
    match_glob="atp_matches_qual_chall_*.csv",
    rankings_path=None or "data/raw/atp_rankings_20s.csv",
    features_path="data/processed_chal/features.csv",
    model_path="models/challenger_xgboost_model.pkl",
    prediction_input_path="data/predict_chal/today_matches.csv",
    prediction_output_path="data/predict_chal/predictions.csv",
    backtest_predictions_path="data/processed_chal/backtest_predictions.csv",
    clv_results_path="data/processed_chal/clv_results.csv",
    odds_path="data/sofascore/march-april_sofascore_odds.csv",
    eval_cutoff_date="2026-01-01",
)
```

### `build_features.py`

Needs parameterization or a Challenger wrapper.

Current ATP role:

- load ATP files
- filter ATP events
- replay state
- write `data/processed/features.csv`

Challenger V1 needs:

- load `data/raw_chal/atp_matches_qual_chall_2020.csv` through `2026`
- include only `tourney_level == "C"` or implicitly load only Challenger files
- train rows can include all years, but model training should train pre-2026 and backtest 2026
- write to `data/processed_chal/features.csv`

### `train_model.py`

Needs parameterization for:

- feature file
- model output path
- cutoff date
- optional study name/output logs

The training logic should remain the same.

### `evaluate_model.py`

Needs parameterization for:

- feature file
- model file
- report labels

The diagnostic logic can remain the same.

### `predict_match.py`

Needs parameterization for:

- model path
- raw Challenger history path
- Challenger prediction input
- Challenger prediction output
- ranking source behavior

Important:

For V1, replay must load only Challenger history.

### `backtest_predictions.py`

Needs parameterization for:

- feature file
- model file
- output file
- cutoff date

Deduplication by date, surface, and sorted player IDs remains valid.

### `clv_calculator.py`

Needs parameterization and Challenger-specific odds schema support.

Current ATP CLV expects tennis-data columns:

- `Winner`
- `Loser`
- `Date`
- `Surface`
- `BFEW`
- `BFEL`
- `PSW`
- `PSL`
- `MaxW`
- `MaxL`

Challenger V1 odds source is SofaScore bet365 scrape, which uses different columns:

- `home_player`
- `away_player`
- `match_start_date_utc`
- `surface`
- `winner_code`
- `home_latest_decimal`
- `away_latest_decimal`
- `home_latest_no_vig`
- `away_latest_no_vig`
- `source_bookmaker`

The CLV module should support an odds adapter instead of assuming tennis-data format.

## 3. Modules Needing Challenger-Specific Code

### Challenger Feature Builder Entry Point

Needed file, later:

```text
scripts/build_challenger_features.py
```

Role:

- use Challenger pipeline config
- load Challenger raw matches
- replay Challenger-only state
- write Challenger feature matrix

This can be thin if the shared replay engine is parameterized correctly.

### Challenger Trainer Entry Point

Needed file, later:

```text
scripts/train_challenger_model.py
```

Role:

- load `data/processed_chal/features.csv`
- train on `date < 2026-01-01`
- validate inside 2020-2025 using temporal validation
- save `models/challenger_xgboost_model.pkl`

### Challenger Backtest Entry Point

Needed file, later:

```text
scripts/backtest_challenger_predictions.py
```

Role:

- load Challenger features
- keep `date >= 2026-01-01`
- dedupe symmetric rows
- score with Challenger model
- write `data/processed_chal/backtest_predictions.csv`

### Challenger CLV Calculator

Needed file or adapter, later:

```text
scripts/clv_challenger_calculator.py
```

Preferred design:

- reuse generic CLV matching and no-vig logic
- add a SofaScore bet365 odds adapter
- compare model probability for player A against bet365 no-vig probability for player A

### SofaScore Odds Adapter

Needed logic:

- map `home_player` / `away_player` to model `player_a` / `player_b`
- use `winner_code` only for result verification, not probability mapping
- use `home_latest_no_vig` and `away_latest_no_vig`
- preserve `event_id` for debugging
- preserve `source_bookmaker = bet365_via_sofascore`
- track odds freshness assumption via `odds_source_confidence`

### Challenger Prediction Input Format

Needed file, later:

```text
data/predict_chal/today_matches.csv
```

Recommended columns:

```text
player_a,player_b,surface,date,tourney_level
```

For Challenger V1, `tourney_level` should be `C`.

## 4. Risks

### Ranking Leakage or Tour Mixing

The biggest V1 risk is accidentally using ATP-derived ranking history or ATP match state.

Allowed:

- global player IDs
- optional ranking snapshots if treated as external public rankings, not state

Not allowed:

- ATP Elo
- ATP recent form
- ATP serve/return history
- ATP H2H
- ATP match counts
- ATP wins

Decision needed:

Whether Challenger V1 should use ATP rankings as an external feature source. The current ATP model uses `atp_rankings_20s.csv`. If strict isolation means no ATP information at all, then rank/points features must either be removed, neutralized, or sourced only from Challenger-available data. If public ATP rankings are allowed as external context, they can be used without merging player state. This must be decided before implementation.

### Tournament Date Is Not Match Date

Jeff Sackmann match files use `tourney_date`, not true match date. This weakens:

- same-tournament replay order
- rest/fatigue features
- daily odds matching

Current ATP system already avoids using fatigue features in `FEATURES`, but it still stores fatigue state. Challenger V1 should not add fatigue features unless true match dates are available.

### SofaScore Odds Are Latest Pre-Match Assumed

The scraper labels odds confidence as `latest_pre_match_assumed`. If odds are not true closing odds, CLV should be described as CLV-like benchmark versus scraped latest bet365 price, not guaranteed market close.

### Name Matching

ATP inference uses exact names. CLV uses fuzzy `surname|first_initial`.

For Challengers, name mismatches are likely more frequent because:

- SofaScore spellings may differ from Jeff Sackmann names.
- Accents and initials may differ.
- Challenger players are less consistently represented.

The design should include a skipped-match report and unmatched-name diagnostics.

### Sparse Player Histories

Challenger players may have:

- fewer matches
- unstable serve/return histories
- many new entrants
- partial stats coverage

Neutral fallbacks will fire more often. This is acceptable for V1 but should be measured.

### Surface Coverage

Observed Challenger 2026 raw file currently has `Clay` and `Hard`; ATP state supports `Grass` too. The shared state should keep all three surfaces for future compatibility.

### H2H Sample Size

Challenger H2H may be more common than ATP for some players, but still sparse globally. Existing weighted-total threshold of `2.0` is reasonable for V1.

### Feature Compatibility

If rank/points are removed or neutralized for Challenger V1, the feature list will diverge from ATP. That reduces architecture mirroring but may be necessary for strict isolation.

Preferred V1 approach:

- keep the same feature list if a valid non-leaking ranking source is accepted
- otherwise keep columns but set rank-derived features to neutral values until Challenger-specific ranking logic exists

## 5. Expected Data Flow

### Challenger Training Flow

Input:

```text
data/raw_chal/atp_matches_qual_chall_2020.csv
data/raw_chal/atp_matches_qual_chall_2021.csv
data/raw_chal/atp_matches_qual_chall_2022.csv
data/raw_chal/atp_matches_qual_chall_2023.csv
data/raw_chal/atp_matches_qual_chall_2024.csv
data/raw_chal/atp_matches_qual_chall_2025.csv
data/raw_chal/atp_matches_qual_chall_2026.csv
```

Replay:

```text
sort by tourney_date
filter invalid outcomes
initialize Challenger-only state
compute pre-match features
write symmetric rows
update Challenger-only state
```

Feature artifact:

```text
data/processed_chal/features.csv
```

Training:

```text
train rows: date < 2026-01-01
test rows:  date >= 2026-01-01
```

Model artifact:

```text
models/challenger_xgboost_model.pkl
```

### Challenger Backtest Flow

Input:

```text
data/processed_chal/features.csv
models/challenger_xgboost_model.pkl
```

Steps:

```text
keep 2026 rows
dedupe mirrored rows
score player_a probability
write backtest predictions
```

Output:

```text
data/processed_chal/backtest_predictions.csv
```

### Challenger Odds / CLV Flow

Input:

```text
data/processed_chal/backtest_predictions.csv
data/sofascore/march-april_sofascore_odds.csv
```

Or a future broader SofaScore Challenger odds file:

```text
data/sofascore/challenger_2026_odds.csv
```

Steps:

```text
normalize names
match model player_a/player_b to SofaScore home/away
match date using match_start_date_utc or endpoint date
map home/away no-vig probability to player_a
compute model probability minus bet365 no-vig probability
write CLV output and diagnostics
```

Output:

```text
data/processed_chal/clv_results.csv
data/processed_chal/clv_cumulative.png
data/processed_chal/clv_unmatched.csv
```

## 6. Folder Structure

Recommended V1 folder structure:

```text
data/
  raw/
    atp_matches_2020.csv
    atp_matches_2021.csv
    ...
    atp_rankings_20s.csv

  raw_chal/
    atp_matches_qual_chall_2020.csv
    atp_matches_qual_chall_2021.csv
    atp_matches_qual_chall_2022.csv
    atp_matches_qual_chall_2023.csv
    atp_matches_qual_chall_2024.csv
    atp_matches_qual_chall_2025.csv
    atp_matches_qual_chall_2026.csv

  processed/
    features.csv
    backtest_predictions.csv
    clv_results.csv
    clv_cumulative.png

  processed_chal/
    features.csv
    backtest_predictions.csv
    clv_results.csv
    clv_cumulative.png
    clv_unmatched.csv

  predict/
    today_matches.csv
    predictions.csv

  predict_chal/
    today_matches.csv
    predictions.csv

  sofascore/
    events/
    odds/
    challenger_2026_odds.csv
    sofascore_challenger_scrape_report.txt
    scrape_errors.log

models/
  xgboost_model.pkl
  challenger_xgboost_model.pkl

scripts/
  tennis_pipeline.py
  build_features.py
  train_model.py
  evaluate_model.py
  predict_match.py
  backtest_predictions.py
  clv_calculator.py
  scrape_sofascore_challenger_odds.py

  build_challenger_features.py
  train_challenger_model.py
  evaluate_challenger_model.py
  predict_challenger_match.py
  backtest_challenger_predictions.py
  clv_challenger_calculator.py
```

## Recommended Implementation Strategy

1. Extract pipeline configuration from ATP hardcoded paths without changing ATP behavior.
2. Add Challenger config pointing to isolated Challenger folders.
3. Reuse the existing replay/state functions with Challenger-only match input.
4. Add Challenger feature builder and model trainer as thin wrappers.
5. Add SofaScore odds adapter for Challenger CLV.
6. Validate that no ATP match files are read during Challenger feature building, training, inference, or backtesting.
7. Add tests that fail if Challenger replay loads `data/raw/atp_matches_*.csv`.

## Non-Negotiable V1 Invariants

- No ATP match rows in Challenger replay.
- No ATP Elo in Challenger state.
- No ATP serve/return history in Challenger state.
- No ATP H2H in Challenger state.
- No ATP form or match-count history in Challenger state.
- Player IDs remain original global IDs.
- Challenger artifacts are written outside ATP artifact folders.
- ATP pipeline outputs remain unchanged.
