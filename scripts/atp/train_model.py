# =============================================================================
# train_model.py — Optuna-tuned XGBoost on engineered tennis features
# =============================================================================
# Flow: load features.csv → temporal split (see tennis_pipeline.MODEL_EVAL_CUTOFF_DATE) →
# Optuna minimizes log loss on a late slice of pre-cutoff data (no test leakage) →
# refit best params on full pre-cutoff train → print held-out test metrics →
# save models/atp/xgboost_model.pkl.
# =============================================================================

from functools import partial

import joblib
import optuna
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss
from xgboost import XGBClassifier

from tennis_pipeline import (
    FEATURES,
    path_processed_features_csv,
    path_trained_model_pkl,
    project_root,
    temporal_train_validation_split,
    temporal_train_test_split_for_modeling,
)


def objective(
    trial: optuna.Trial,
    X_fit,
    y_fit,
    X_val,
    y_val,
) -> float:
    """One Optuna trial: suggest hyperparams, fit on fit window, return validation log loss."""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float(
            "learning_rate", 0.01, 0.3, log=True
        ),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "random_state": 42,
    }
    model = XGBClassifier(**params)
    model.fit(X_fit, y_fit)
    pred_probs = model.predict_proba(X_val)[:, 1]
    return log_loss(y_val, pred_probs)


root = project_root()
df = pd.read_csv(path_processed_features_csv(root))
df["date"] = pd.to_datetime(df["date"])

# Same outer split as evaluate_model: train date < cutoff, test date >= cutoff.
train_df, test_df = temporal_train_test_split_for_modeling(df, date_column="date")
fit_df, val_df = temporal_train_validation_split(train_df, date_column="date")

X_train = train_df[FEATURES]
y_train = train_df["result"]
X_fit = fit_df[FEATURES]
y_fit = fit_df["result"]
X_val = val_df[FEATURES]
y_val = val_df["result"]
X_test = test_df[FEATURES]
y_test = test_df["result"]

study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(
    partial(
        objective,
        X_fit=X_fit,
        y_fit=y_fit,
        X_val=X_val,
        y_val=y_val,
    ),
    n_trials=50,
)

best_params = study.best_params
validation_model = XGBClassifier(
    n_estimators=best_params["n_estimators"],
    max_depth=best_params["max_depth"],
    learning_rate=best_params["learning_rate"],
    subsample=best_params["subsample"],
    colsample_bytree=best_params["colsample_bytree"],
    random_state=42,
)
validation_model.fit(X_fit, y_fit)
val_probs = validation_model.predict_proba(X_val)[:, 1]
val_log_loss = log_loss(y_val, val_probs)

model = XGBClassifier(
    n_estimators=best_params["n_estimators"],
    max_depth=best_params["max_depth"],
    learning_rate=best_params["learning_rate"],
    subsample=best_params["subsample"],
    colsample_bytree=best_params["colsample_bytree"],
    random_state=42,
)
model.fit(X_train, y_train)

print(
    "Validation Log Loss (Optuna split before full-train refit):",
    val_log_loss,
)

pred_probs = model.predict_proba(X_test)[:, 1]
preds = (pred_probs >= 0.5).astype(int)
print("Accuracy:", accuracy_score(y_test, preds))
print("Log Loss:", log_loss(y_test, pred_probs))

joblib.dump(model, path_trained_model_pkl(root))
