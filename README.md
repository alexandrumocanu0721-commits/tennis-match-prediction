# Tennis Match Prediction System using Machine Learning

## Overview

This project is an end-to-end machine learning pipeline designed to predict professional tennis match outcomes using historical ATP data.

The system dynamically reconstructs player strength over time using Elo ratings, surface-specific Elo, rankings, recent form, and other engineered features, then predicts match win probabilities using gradient boosted decision trees (XGBoost).

The project was built as a practical introduction to:

* data science
* machine learning
* feature engineering
* probabilistic prediction
* explainable AI
* model evaluation
* ML pipelines

---

## Main Features

### Dynamic Player State Reconstruction

The system chronologically replays historical ATP matches and continuously updates:

* global Elo rating
* surface-specific Elo ratings
* recent form
* surface-specific recent form
* win percentage
* experience (matches played)
* ATP ranking and points

This allows the model to simulate realistic player strength at any point in time.

---

## Feature Engineering

The final model uses the following features:

| Feature                  | Description                                |
| ------------------------ | ------------------------------------------ |
| elo_diff                 | Difference in global Elo ratings           |
| surface_elo_diff         | Difference in surface-specific Elo ratings |
| rank_diff                | ATP ranking difference                     |
| points_diff              | ATP points difference                      |
| recent_form_diff         | Recent win-rate difference                 |
| recent_surface_form_diff | Recent surface win-rate difference         |
| win_pct_diff             | Overall historical win-rate difference     |
| matches_played_diff      | Difference in career experience            |

---

## Machine Learning Models

The project explored several machine learning approaches and model refinements throughout development.

The project explored multiple approaches:

### Logistic Regression

Used as the baseline probabilistic classifier.

### XGBoost

Main production model used for final predictions.

The final system uses a tuned XGBoost classifier trained on engineered tennis features extracted chronologically from ATP match history.

### Optuna Hyperparameter Optimization

Used automated hyperparameter tuning to improve model performance.

Parameters explored included:

* number of estimators
* tree depth
* learning rate
* subsampling
* column sampling

Optuna was used to minimize log loss on a temporally separated validation set.

### DART Booster Experiments

Explored dropout-based boosted trees for robustness and generalization.

Although DART slightly improved classification accuracy, the final selected model prioritized lower log loss and cleaner probability estimates for bookmaker probability comparison.

---

## Model Evaluation

The project intentionally used a temporal split rather than random shuffling in order to simulate real-world forecasting conditions and avoid future information leakage.

The project uses:

* temporal train/test split
* accuracy
* log loss
* calibration analysis

Final tuned XGBoost model achieved approximately:

| Metric   | Result |
| -------- | ------ |
| Accuracy | 64.41%  |
| Log Loss | 0.623  |

The project also experimented with DART boosting, calibration analysis, and probability consistency validation.

---

## Calibration Analysis

Calibration curves were used to verify whether predicted probabilities matched real-world outcome frequencies.

This was particularly important because the project focuses on probability estimation rather than only winner prediction.

---

## Explainable AI (SHAP)

The project uses SHAP values to explain:

* global feature importance
* individual match predictions
* feature contribution to predicted probabilities

This allowed inspection of how factors such as Elo, recent form, and rankings influence model decisions.

---

## Live Prediction System

The project includes a standalone prediction pipeline for forecasting future ATP matches using the latest reconstructed player states.

A batch prediction workflow was implemented.

Users can provide a CSV file containing upcoming matches:

```csv
player_a,player_b,surface
Novak Djokovic,Carlos Alcaraz,Clay
Jannik Sinner,Daniil Medvedev,Hard
```

The system outputs predicted win probabilities for all matches.

Predictions are generated using only historical information available before the match date, preserving realistic forecasting conditions.

---

## Scripts

| Script            | Purpose                                                                   |
| ----------------- | ------------------------------------------------------------------------- |
| build_features.py | Chronologically reconstructs player states and creates ML features        |
| train_model.py    | Trains and saves the XGBoost model                                        |
| evaluate_model.py  | Generates calibration curves, SHAP analysis, and feature importance plots |
| predict_match.py  | Predicts future match probabilities from CSV input                        |

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* Optuna
* SHAP
* Matplotlib

---

## Project Structure

```text
project/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── predict/
│
├── models/
│   └── xgboost_model.pkl
│
├── scripts/
│   ├── build_features.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   └── predict_match.py
│
├── README.md
├── requirements.txt
└── .gitignore
````

---

## Engineering Challenges Solved

Throughout development, the project addressed several real-world ML engineering problems:

* chronological feature generation
* avoiding temporal leakage
* maintaining train/inference consistency
* symmetric feature handling for mirrored predictions
* calibration of probabilistic outputs
* model explainability using SHAP
* automated hyperparameter optimization

---

## Key Concepts Learned

This project involved practical experience with:

* feature engineering
* time-aware validation
* Elo systems
* probabilistic prediction
* calibration
* explainable AI
* hyperparameter optimization
* train/inference consistency
* machine learning experimentation
* predictive pipelines

---

## Future Improvements

Potential future additions:

* injury/fatigue information
* head-to-head statistics
* tournament context
* automated odds comparison
* web deployment
* real-time ATP data ingestion

---

## Disclaimer

This project was built primarily for educational purposes and as a practical introduction to machine learning and predictive systems.

It is not intended as financial or betting advice.
