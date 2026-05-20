# Tennis Quant Betting Research

## Project Overview

This repository contains a tennis quant betting research project focused on estimating pre-match win probabilities and evaluating whether those probabilities beat market closing prices. It has two separate pipelines: ATP Main Tour and ATP Challenger. Both pipelines use engineered tennis state features and XGBoost classifiers, then evaluate results with closing line value (CLV) and small-sample ROI simulations. The project is research-only at this stage; neither model has enough live or out-of-sample evidence to justify real-money betting.

## Repo Structure

```text
project_root/
  data/
    raw/
      atp/                  Jeff Sackmann ATP main-tour match and ranking CSVs.
      challenger/           Jeff Sackmann ATP Challenger qualifying/challenger match CSVs.
    processed/
      atp/                  ATP features, backtest predictions, CLV, ROI summaries, plots.
      challenger/           Challenger features, backtest predictions, CLV, ROI summaries.
    backtest/               Shared/ATP historical odds snapshot, including real_2026_odds.csv.
    predict/
      atp/                  ATP upcoming match input and prediction output CSVs.
      challenger/           Reserved for Challenger prediction inputs/outputs.
    sofascore/
      events/               Cached SofaScore event JSON responses, unchanged.
      odds/                 Cached SofaScore odds JSON responses, unchanged.
      challenger/           Challenger SofaScore odds CSVs, scrape logs, scrape report.
  models/
    atp/                    ATP XGBoost model artifacts.
    challenger/             Challenger XGBoost model artifacts.
  scripts/
    atp/                    ATP feature, training, evaluation, prediction, backtest, CLV scripts.
    challenger/             Challenger feature, training, backtest, CLV, scraping, config scripts.
    analysis/               ROI, calibration, and diagnostic analysis scripts.
  tests/                    Safety and H2H tests.
  notebooks/                Exploratory notebooks.
  README.md                 Project documentation.
  requirements.txt          Python dependencies.
```

## Pipelines

### ATP Main Tour

Data sources:

- Match data: Jeff Sackmann ATP raw CSVs in `data/raw/atp/`.
- Rankings: Jeff Sackmann ATP ranking CSV in `data/raw/atp/atp_rankings_20s.csv`.
- Odds and CLV benchmark: `data/backtest/real_2026_odds.csv`, using Betfair as the primary CLV benchmark with Pinnacle and max odds also calculated.

Execution order:

```bash
python3 scripts/atp/build_features.py
python3 scripts/atp/train_model.py
python3 scripts/atp/evaluate_model.py
python3 scripts/atp/backtest_predictions.py
python3 scripts/atp/clv_calculator.py
python3 scripts/analysis/roi_simulation_atp.py
```

Prediction flow:

```bash
python3 scripts/atp/predict_match.py
```

Key outputs:

- `data/processed/atp/features.csv`
- `models/atp/xgboost_model.pkl`
- `data/processed/atp/backtest_predictions.csv`
- `data/processed/atp/clv_results.csv`
- `data/processed/atp/clv_cumulative.png`
- `data/processed/atp/roi_summary_atp.csv`
- `data/processed/atp/roi_by_surface_atp.csv`
- `data/processed/atp/roi_by_bucket_atp.csv`
- `data/predict/atp/predictions.csv`

### ATP Challenger

Data sources:

- Match data: Jeff Sackmann Challenger CSVs in `data/raw/challenger/`.
- Ranking context: ATP rankings in `data/raw/atp/atp_rankings_20s.csv`, used only as external public ranking context.
- Odds and CLV benchmark: SofaScore-scraped Bet365 odds in `data/sofascore/challenger/march-april_sofascore_odds.csv`.

Execution order:

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
```

Optional SofaScore scrape:

```bash
python3 scripts/challenger/scrape_sofascore_challenger_odds.py --start-date YYYY-MM-DD --end-date YYYY-MM-DD
```

Key outputs:

- `data/processed/challenger/features.csv`
- `models/challenger/challenger_xgboost_model.pkl`
- `data/processed/challenger/backtest_predictions.csv`
- `data/processed/challenger/clv_results.csv`
- `data/processed/challenger/clv_results_calibrated.csv`
- `data/processed/challenger/roi_summary.csv`
- `data/processed/challenger/roi_by_surface.csv`
- `data/processed/challenger/roi_by_bucket.csv`
- `data/processed/challenger/roi_favorites_summary.csv`
- `data/processed/challenger/roi_favorites_by_surface.csv`
- `data/processed/challenger/roi_favorites_by_month.csv`

## Models

Both pipelines use XGBoost classifiers trained on pre-match differential features. Core features include Elo and surface Elo, ranking and ranking momentum, recent form, surface form, win percentage, match volume, deciding-set history, serve/return rolling stats, break-point stats, head-to-head stats, and tournament-level flags.

The ATP model is saved at `models/atp/xgboost_model.pkl` and is benchmarked primarily against Betfair closing prices from `data/backtest/real_2026_odds.csv`. The Challenger model is saved at `models/challenger/challenger_xgboost_model.pkl` and is benchmarked against Bet365 prices scraped via SofaScore. Challenger feature generation uses Challenger match history for match-derived state and ATP rankings only as external public ranking context.

## Analysis Results

Challenger:

- Mean CLV is positive at approximately `+0.0165` versus Bet365/SofaScore closing odds.
- Hard-court favorites showed approximately `9%` ROI over `130` bets from February to April 2026.
- Clay favorites were close to breakeven in the available sample.
- Underdog betting is structurally broken in the current Challenger setup and should not be used as a live signal.

ATP Main Tour:

- CLV is positive versus Betfair in the current backtest artifacts.
- The balanced odds bucket shows promising ROI around `+2%` to `+5%` depending on threshold and sample slice.
- The sample is still too small to conclude that the edge is durable.

Research status:

- These are early research signals, not production betting evidence.
- Both models need more data, stricter forward testing, and live-paper tracking before any real-money use.
- Current ROI summaries should be treated as diagnostics, not profitability claims.

## How To Run

Install dependencies first:

```bash
pip install -r requirements.txt
```

Run the ATP Main Tour pipeline:

```bash
python3 scripts/atp/build_features.py
python3 scripts/atp/train_model.py
python3 scripts/atp/evaluate_model.py
python3 scripts/atp/backtest_predictions.py
python3 scripts/atp/clv_calculator.py
python3 scripts/analysis/roi_simulation_atp.py
```

Run ATP prediction for rows in `data/predict/atp/today_matches.csv`:

```bash
python3 scripts/atp/predict_match.py
```

Run the Challenger pipeline:

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
```

Scrape Challenger SofaScore odds when needed:

```bash
python3 scripts/challenger/scrape_sofascore_challenger_odds.py --start-date 2026-03-01 --end-date 2026-04-28 --output march-april_sofascore_odds.csv
```

## Requirements

Dependencies are listed in `requirements.txt`.

Use `python3` in this environment; `python` may not be available on PATH.
