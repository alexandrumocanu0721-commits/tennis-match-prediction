# Tennis Quant Betting Research

## What This Repo Is

This repository is a research codebase for pre-match tennis pricing, market comparison, and post-model betting diagnostics. It currently contains:

- An ATP main-tour feature/training/backtest pipeline.
- An ATP Challenger feature/training/backtest pipeline.
- CLV and calibration analysis on both tours.
- Several ROI and bankroll-style strategy studies layered on top of the model outputs.
- Two separate underdog research tracks:
  `underdog_rescue` for general underdogs and `near_dog_specialist` for the 2.20-3.00 odds band.

The project is still research-only. The current artifacts do not justify live betting.

## Current Bottom Line

The repo now contains a lot of encouraging model-vs-market signal, but the generated artifacts still argue for caution:

- ATP and Challenger broad flat-stake ROI summaries are negative in the current saved CSVs even when mean model edge / CLV is positive.
- Challenger favorite diagnostics are more nuanced than the earlier broad-Challenger results, but the saved `roi_favorites_summary.csv` is still negative across thresholds.
- ATP underdog rescue ends in `BLOCK_UNDERDOGS`.
- Challenger underdog rescue ends in `BLOCK_UNDERDOGS`.
- The near-underdog specialist ends in `BLOCK_NEAR_DOGS`.
- Several interesting slices exist, but concentration, calibration, and sample-size issues still dominate the final recommendations.

If you use this repo, treat it as a model-research and market-diagnostics project, not as production betting infrastructure.

## Repository Map

```text
project_root/
  data/
    raw/
      atp/                      Jeff Sackmann ATP match/ranking CSVs.
      challenger/               Jeff Sackmann Challenger/qualifying match CSVs.
    backtest/
      real_2026_odds.csv        ATP market benchmark file used by the ATP CLV flow.
    sofascore/
      events/                   Cached SofaScore event JSON.
      odds/                     Cached SofaScore odds JSON.
      challenger/               Challenger odds scrape outputs/logs.
    processed/
      atp/                      ATP engineered features, backtests, CLV, ROI outputs.
      challenger/               Challenger engineered features, CLV, calibration, ROI outputs.
      underdog_specialist/      Near-dog dataset, training results, backtests, recommendation.
      diagnostic_player_state_temporal.txt
      strategy_backtest_*.csv   Bankroll/equity simulations built from saved outputs.
    predict/
      atp/
        today_matches.csv       ATP prediction input template / active input file.
        predictions.csv         ATP model predictions for the current input file.
      challenger/
        today_matches.csv       Placeholder prediction input path.
        predictions.csv         Placeholder prediction output path.
  models/
    atp/
      xgboost_model.pkl
    challenger/
      challenger_xgboost_model.pkl
    underdog_specialist/
      near_dog_specialist.pkl
      near_dog_specialist_metadata.json
  scripts/
    atp/                        ATP pipeline, evaluation, prediction, CLV logic.
    challenger/                 Challenger pipeline, config, safety helpers, CLV, scraping.
    analysis/                   Calibration, ROI studies, underdog workflows, strategy backtests.
  tests/                        Unit/safety tests for H2H, Challenger isolation, underdog logic.
  notebooks/                    Ad hoc exploration notebooks.
  requirements.txt              Python dependencies.
```

## Core Data Sources

### ATP

- Match history: `data/raw/atp/atp_matches_*.csv`
- Rankings: `data/raw/atp/atp_rankings_20s.csv`
- CLV / market benchmark: `data/backtest/real_2026_odds.csv`

### Challenger

- Match history: `data/raw/challenger/atp_matches_qual_chall_*.csv`
- Public ranking context: `data/raw/atp/atp_rankings_20s.csv`
- Challenger odds benchmark: SofaScore-scraped Bet365-style market data under `data/sofascore/challenger/`

The Challenger pipeline is explicitly designed to keep Challenger match-derived state separate from ATP match-derived state. ATP rankings are allowed only as external ranking context.

## Modeling Approach

Both tour pipelines are built around pre-match engineered state features and XGBoost classifiers.

Important feature families used across the repo include:

- Elo and surface Elo
- Ranking, points, and rank momentum
- Recent form and recent surface form
- Win percentage and match-volume state
- Deciding-set win rate
- Serve/return rolling strength
- Break-point save/convert rates
- Head-to-head and surface head-to-head

The feature engines replay historical matches chronologically and update player state over time. The saved temporal diagnostic in `data/processed/diagnostic_player_state_temporal.txt` confirms the state is live by match date rather than frozen at the 2025-01-01 model cutoff.

## Date Split Convention

Both main modeling branches use a temporal cutoff at `2025-01-01`:

- Pre-cutoff data is used for model fitting / tuning.
- Post-cutoff data is used for held-out evaluation, backtests, and downstream diagnostics.

The near-dog and underdog analysis branches then add their own walk-forward train/tune/final splits inside the downstream research workflow.

## ATP Pipeline

### Main scripts

- `scripts/atp/build_features.py`
  Replays ATP match history and writes the ATP engineered feature table.
- `scripts/atp/train_model.py`
  Runs an Optuna-tuned XGBoost training flow and saves `models/atp/xgboost_model.pkl`.
- `scripts/atp/evaluate_model.py`
  Evaluates the saved ATP model on the held-out split.
- `scripts/atp/backtest_predictions.py`
  Generates backtest predictions on the held-out sample.
- `scripts/atp/clv_calculator.py`
  Matches predictions to market odds and computes ATP CLV outputs.
- `scripts/atp/predict_match.py`
  Scores upcoming ATP rows from `data/predict/atp/today_matches.csv`.
- `scripts/atp/tennis_pipeline.py`
  Shared ATP feature/state engine, path helpers, and split logic.

### Typical ATP run order

```bash
python3 scripts/atp/build_features.py
python3 scripts/atp/train_model.py
python3 scripts/atp/evaluate_model.py
python3 scripts/atp/backtest_predictions.py
python3 scripts/atp/clv_calculator.py
python3 scripts/analysis/roi_simulation_atp.py
```

### Main ATP artifacts

- `data/processed/atp/features.csv`
- `models/atp/xgboost_model.pkl`
- `data/processed/atp/backtest_predictions.csv`
- `data/processed/atp/clv_results.csv`
- `data/processed/atp/clv_cumulative.png`
- `data/processed/atp/roi_summary_atp.csv`
- `data/processed/atp/roi_by_surface_atp.csv`
- `data/processed/atp/roi_by_bucket_atp.csv`
- `data/predict/atp/predictions.csv`

### Current ATP read-through

The saved ATP ROI summary still shows negative flat-stake ROI across thresholds in both 2025 and 2026, despite positive mean edge in the saved tables. That means the current ATP model may still be learning something useful relative to market prices, but the saved backtest evidence is not strong enough to claim a monetizable signal.

## Challenger Pipeline

### Main scripts

- `scripts/challenger/build_challenger_features.py`
  Builds Challenger features from Challenger match history.
- `scripts/challenger/train_challenger_model.py`
  Trains the Challenger XGBoost model and saves `models/challenger/challenger_xgboost_model.pkl`.
- `scripts/challenger/backtest_challenger_predictions.py`
  Produces held-out Challenger predictions.
- `scripts/challenger/clv_challenger_calculator.py`
  Matches predictions to Challenger market odds and computes CLV outputs.
- `scripts/challenger/challenger_config.py`
  Defines Challenger-specific paths, guardrails, defaults, and the `2025-01-01` cutoff.
- `scripts/challenger/challenger_utils.py`
  Challenger helpers for player/ranking state.
- `scripts/challenger/challenger_feature_ablation.py`
  Runs feature-family ablation experiments on the Challenger model.
- `scripts/challenger/scrape_sofascore_challenger_odds.py`
  Scrapes Challenger odds from SofaScore for a chosen date range.

### Typical Challenger run order

```bash
python3 scripts/challenger/build_challenger_features.py
python3 scripts/challenger/train_challenger_model.py
python3 scripts/challenger/backtest_challenger_predictions.py
python3 scripts/challenger/clv_challenger_calculator.py
python3 scripts/analysis/calibrate_challenger.py
python3 scripts/analysis/roi_simulation.py
python3 scripts/analysis/roi_favorites_only.py
python3 scripts/analysis/roi_monthly_fix.py
python3 scripts/analysis/underdog_diagnosis.py
python3 scripts/analysis/underdog_rescue.py --tour all
python3 scripts/analysis/build_near_dog_dataset.py --tour all
python3 scripts/analysis/train_near_dog_model.py
python3 scripts/analysis/backtest_near_dog_model.py
```

### Main Challenger artifacts

- `data/processed/challenger/features.csv`
- `models/challenger/challenger_xgboost_model.pkl`
- `data/processed/challenger/backtest_predictions.csv`
- `data/processed/challenger/clv_results.csv`
- `data/processed/challenger/clv_results_calibrated.csv`
- `data/processed/challenger/calibration_reliability_by_year.csv`
- `data/processed/challenger/calibration_bucket_by_year.csv`
- `data/processed/challenger/roi_summary.csv`
- `data/processed/challenger/roi_by_surface.csv`
- `data/processed/challenger/roi_by_bucket.csv`
- `data/processed/challenger/roi_favorites_summary.csv`
- `data/processed/challenger/roi_favorites_by_surface.csv`
- `data/processed/challenger/roi_favorites_by_tier.csv`
- `data/processed/challenger/roi_favorites_by_month.csv`
- `data/processed/challenger/favorite_sleeve_diagnostic.txt`
- `data/processed/challenger/clv_unmatched.csv`
- `data/processed/challenger/clv_ambiguous.csv`
- `data/processed/challenger/clv_placebo.csv`
- `data/processed/challenger/clv_placebo_diagnostic.txt`

### Current Challenger read-through

The broad saved Challenger ROI summary is negative across thresholds in both 2025 and 2026. The follow-up favorite-only analysis is more selective and much more informative, but the saved summary file is still negative across thresholds. The favorite diagnostic text does show some more intuitive sub-slices, especially around certain rank-diff and confidence bands, but the repo’s own latest outputs still stop short of endorsing a stable deployable edge.

## Calibration, ROI, and Diagnostic Scripts

### `scripts/analysis/calibrate_challenger.py`

- Fits post-model isotonic calibration for Challenger outputs.
- Falls back from `clv_results.csv` to `backtest_predictions.csv` if the CLV file has too few labeled rows.
- Writes:
  `clv_results_calibrated.csv`,
  `calibration_reliability_by_year.csv`,
  `calibration_bucket_by_year.csv`

### `scripts/analysis/roi_simulation.py`

- Summarizes broad Challenger flat-stake ROI by year, surface, and odds bucket.
- Writes:
  `roi_summary.csv`,
  `roi_by_surface.csv`,
  `roi_by_bucket.csv`

### `scripts/analysis/roi_simulation_atp.py`

- ATP equivalent of the Challenger ROI summary script.
- Writes:
  `roi_summary_atp.csv`,
  `roi_by_surface_atp.csv`,
  `roi_by_bucket_atp.csv`

### `scripts/analysis/roi_favorites_only.py`

- Focuses specifically on Challenger favorite sleeves.
- Produces by-threshold, by-surface, by-tier, and by-month summaries.
- Also writes a narrative diagnostic text file discussing where the apparent favorite edge is concentrated.

### `scripts/analysis/roi_monthly_fix.py`

- Monthly Challenger ROI helper / repair script for calibrated outputs.

### `scripts/analysis/underdog_diagnosis.py`

- Diagnostic work focused on why underdog performance is structurally failing.

### `data/processed/diagnostic_player_state_temporal.txt`

- A saved textual diagnostic showing that both ATP and Challenger player-state engines continue updating through post-cutoff dates.

## Underdog Rescue Workflow

The `underdog_rescue` branch is an explicit attempt to rescue underdog betting rather than simply report that it fails.

### Main script

- `scripts/analysis/underdog_rescue.py`

### What it tries

- Odds-bucket-specific isotonic calibration
- Underdog-only calibration
- Probability caps on underdogs
- Market-anchored shrinkage
- Separate underdog logistic models
- Feature-family ablation on those underdog models
- Walk-forward train/tune/final selection logic

### Output directories

- `data/processed/atp/underdog_rescue/`
- `data/processed/challenger/underdog_rescue/`

### Key output files per tour

- `regime_summary.csv`
- `underdog_odds_bins.csv`
- `underdog_calibration.csv`
- `feature_family_ablation.csv`
- `rescue_experiments.csv`
- `final_concentration.csv`
- `final_recommendation.txt`

### Current underdog status

The latest saved final recommendation files block underdogs on both tours:

- ATP selected `dog_logit_no_recent_form` on tune, but final-period ROI/calibration/concentration checks failed.
- Challenger selected `dog_market_delta_shrink_25` on tune, but final-period ROI/calibration/concentration checks failed.

This is one of the clearest repo-level conclusions right now: underdogs remain broken in the current setup.

## Near-Underdog Specialist Workflow

This is a separate research branch, not just another variant of the broad underdog rescue.

### Purpose

It builds a dedicated dataset for two-sided candidates in the `2.20-3.00` odds band and tests whether a narrower, market-aware specialist can behave better than the broad underdog logic.

### Main scripts

- `scripts/analysis/build_near_dog_dataset.py`
- `scripts/analysis/train_near_dog_model.py`
- `scripts/analysis/backtest_near_dog_model.py`
- `scripts/analysis/near_dog_specialist_common.py`

### Generated artifacts

- `data/processed/underdog_specialist/near_dog_dataset.csv`
- `data/processed/underdog_specialist/near_dog_dataset_summary.csv`
- `data/processed/underdog_specialist/near_dog_train_results.csv`
- `data/processed/underdog_specialist/near_dog_backtest_predictions.csv`
- `data/processed/underdog_specialist/near_dog_backtest_summary.csv`
- `data/processed/underdog_specialist/near_dog_backtest_by_tour.csv`
- `data/processed/underdog_specialist/near_dog_backtest_calibration.csv`
- `data/processed/underdog_specialist/near_dog_final_recommendation.txt`
- `models/underdog_specialist/near_dog_specialist.pkl`
- `models/underdog_specialist/near_dog_specialist_metadata.json`

### Current near-dog status

The latest saved recommendation selects candidate `near_dog_market_only` with shrinkage factor `0.75` and edge threshold `0.10`, but the final recommendation is still `BLOCK_NEAR_DOGS`. The saved final-period ROI is positive, but calibration error and concentration fail the workflow’s pass criteria. So this branch is promising enough to keep researching, but not strong enough to approve.

## Strategy Backtest Scripts

These scripts are downstream staking studies built on top of existing saved predictions / CLV outputs. They do not create the core model artifacts.

### `scripts/analysis/strategy_backtest.py`

- Mixed ATP + Challenger post-`2025-01-01` bankroll simulation.
- Uses edge tiers and variable staking.
- Writes:
  `data/processed/strategy_backtest_equity.csv`,
  `data/processed/strategy_backtest_monthly.csv`

### `scripts/analysis/strategy_backtest_focused.py`

- Challenger favorite-style focused strategy.
- Restricts to Hard/Clay, `odds < 1.50`, edge between `0.03` and `0.08`, and 1% flat stake with daily exposure cap.
- Writes:
  `data/processed/strategy_backtest_focused_equity.csv`,
  `data/processed/strategy_backtest_focused_monthly.csv`

### `scripts/analysis/strategy_backtest_focused_clay_small.py`

- Even narrower Challenger clay-only favorite strategy.
- Restricts to Clay, `odds < 1.50`, edge between `0.03` and `0.05`.
- Writes:
  `data/processed/strategy_backtest_focused_clay_small_equity.csv`,
  `data/processed/strategy_backtest_focused_clay_small_monthly.csv`

These scripts are useful for stress-testing bankroll behavior, drawdowns, and concentration, but they should not be mistaken for independent evidence that the model is production-ready.

## Tests

The current test suite covers the most failure-prone logic:

- `tests/test_h2h.py`
  Verifies H2H rate behavior, recency weighting, surface filtering, and antisymmetry.
- `tests/test_challenger_safety.py`
  Enforces Challenger path isolation so Challenger code does not accidentally overwrite ATP artifacts or import ATP match-state flows improperly.
- `tests/test_underdog_rescue.py`
  Covers odds-bucket boundaries, walk-forward splitting, flat-stake ROI helpers, and tune-only candidate selection.
- `tests/test_near_dog_specialist.py`
  Covers two-sided candidate construction, market shrinkage, and specialist signal filtering.

## Environment and Dependencies

Dependencies in `requirements.txt`:

- `ipykernel`
- `jupyterlab`
- `joblib`
- `matplotlib`
- `notebook`
- `numpy`
- `optuna`
- `pandas`
- `scikit-learn`
- `scipy`
- `shap`
- `xgboost`

Recommended setup:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Use `python3` for scripts in this repo.

## Common Commands

### Train and evaluate ATP

```bash
python3 scripts/atp/build_features.py
python3 scripts/atp/train_model.py
python3 scripts/atp/evaluate_model.py
python3 scripts/atp/backtest_predictions.py
python3 scripts/atp/clv_calculator.py
python3 scripts/analysis/roi_simulation_atp.py
```

### Predict ATP matches

```bash
python3 scripts/atp/predict_match.py
```

### Train and evaluate Challenger

```bash
python3 scripts/challenger/build_challenger_features.py
python3 scripts/challenger/train_challenger_model.py
python3 scripts/challenger/backtest_challenger_predictions.py
python3 scripts/challenger/clv_challenger_calculator.py
python3 scripts/analysis/calibrate_challenger.py
python3 scripts/analysis/roi_simulation.py
python3 scripts/analysis/roi_favorites_only.py
```

### Run underdog research

```bash
python3 scripts/analysis/underdog_rescue.py --tour all
python3 scripts/analysis/build_near_dog_dataset.py --tour all
python3 scripts/analysis/train_near_dog_model.py --rebuild-dataset --tour all
python3 scripts/analysis/backtest_near_dog_model.py
```

### Scrape Challenger odds

```bash
python3 scripts/challenger/scrape_sofascore_challenger_odds.py --start-date 2026-03-01 --end-date 2026-04-28 --output march-april_sofascore_odds.csv
```

### Run tests

```bash
python3 -m unittest discover -s tests
```

## Practical Reading Guide

If you are new to the repo, the fastest way to understand it is:

1. Read `scripts/atp/tennis_pipeline.py` for the shared state-engine ideas.
2. Read `scripts/challenger/challenger_config.py` and `scripts/challenger/challenger_utils.py` for the Challenger isolation model.
3. Inspect the latest outputs under `data/processed/challenger/` and `data/processed/underdog_specialist/`.
4. Read the final recommendation text files before trusting any slice that looks profitable in a CSV.

## Final Research Status

The repo is much more mature than a single-model prototype. It now has separate tour pipelines, calibration, CLV, ROI slicing, underdog rescue experiments, a dedicated near-dog specialist, safety tests, and bankroll studies.

But the latest saved outputs still point to the same operational conclusion:

- Useful research signal exists.
- Some slices look interesting.
- The evidence is still not strong enough for live deployment.

That conclusion is worth preserving, because the repo is most valuable right now as a disciplined falsification and diagnostics framework for tennis pricing ideas.
