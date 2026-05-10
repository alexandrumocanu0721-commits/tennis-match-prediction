# Tennis Match Prediction Model

## Objective

Build a machine learning pipeline capable of predicting ATP singles match outcomes using historical tennis data and pre-match features.

The model outputs:

- probability that Player A wins against Player B
- comparison against bookmaker implied probabilities

---

# Dataset

Source:

- Jeff Sackmann ATP Match Data

Data used:

- ATP singles matches
- Historical data from 2020–2024 for training
- 2025 for testing
- Future deployment for real 2026 matches

---

# Core Principles

## 1. Time-Aware Modeling

For every match, only information available BEFORE the match date may be used.

No future information leakage is allowed.

---

## 2. Probabilistic Prediction

The model predicts probabilities, not only winners.

Example:

- Player A win probability = 0.73

---

## 3. Sequential Feature Generation

Features are generated chronologically:

1. compute features before match
2. store training row
3. update ratings/history
4. move to next match

---

# Initial Feature Set

## Player Strength Features

- elo_diff
- surface_elo_diff
- rank_diff

## Recent Form Features

- recent_form_diff
- recent_surface_form_diff

## Surface Features

- hard_win_pct_diff
- clay_win_pct_diff
- grass_win_pct_diff

---

# Feature Definitions

## Elo Rating

Dynamic rating representing overall player strength.

## Surface Elo

Separate Elo rating per surface:

- hard
- clay
- grass

## Rank Difference

ATP ranking difference between players.

## Recent Form

Win percentage over recent matches.

## Surface Win Percentage

Win percentage on a specific surface only.

---

# Dataset Structure

Each row represents one pre-match snapshot.

Example:


| date | player_a | player_b | elo_diff | surface_elo_diff | rank_diff | recent_form_diff | result |
| ---- | -------- | -------- | -------- | ---------------- | --------- | ---------------- | ------ |


Target:

- result = 1 if Player A wins
- result = 0 otherwise

---

# Project Architecture

```text
tennis-ml/

data/
    raw/
    processed/

notebooks/

scripts/
    build_features.py
    train_model.py
    evaluate_model.py
    predict_today.py

models/
```

---

# Modeling Pipeline

## Step 1 — Raw Data

Load historical ATP match data.

## Step 2 — Feature Generation

Generate chronological pre-match features.

## Step 3 — Training

Train models on historical matches.

## Step 4 — Evaluation

Evaluate:

- accuracy
- log loss
- calibration

## Step 5 — Prediction

Generate probabilities for future ATP matches.

---

# Models

## Baseline

- Elo system

## First ML Model

- Logistic Regression

## Advanced Model

- XGBoost / LightGBM

---

# Validation Strategy

Chronological split only.

Example:

- Train: 2020–2024
- Test: 2025

No random shuffling allowed.

---

# Long-Term Goal

Create an updatable prediction pipeline capable of:

- ingesting new ATP matches
- updating features
- retraining models
- predicting future matches in real time
- comparing predictions against bookmaker odds

