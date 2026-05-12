# =============================================================================
# train_model.py — Optuna-tuned XGBoost on engineered tennis features
# =============================================================================
# Flow: load features.csv → temporal split (see tennis_pipeline.MODEL_EVAL_CUTOFF_DATE) →
# Optuna minimizes validation log loss on the test year → refit best params on
# full train → print metrics → save models/xgboost_model.pkl.
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
    temporal_train_test_split_for_modeling,
)


def objective(
    trial: optuna.Trial,
    X_train,
    y_train,
    X_test,
    y_test,
) -> float:
    """One Optuna trial: suggest hyperparams, fit on train, return test log loss."""
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
    model.fit(X_train, y_train)
    pred_probs = model.predict_proba(X_test)[:, 1]
    return log_loss(y_test, pred_probs)


root = project_root()
df = pd.read_csv(path_processed_features_csv(root))
df["date"] = pd.to_datetime(df["date"])

# Same split as evaluate_model: train date < cutoff, test date >= MODEL_EVAL_CUTOFF_DATE.
train_df, test_df = temporal_train_test_split_for_modeling(df, date_column="date")

X_train = train_df[FEATURES]
y_train = train_df["result"]
X_test = test_df[FEATURES]
y_test = test_df["result"]

study = optuna.create_study(direction="minimize")
study.optimize(
    partial(
        objective,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    ),
    n_trials=30,
)

best_params = study.best_params
model = XGBClassifier(
    n_estimators=best_params["n_estimators"],
    max_depth=best_params["max_depth"],
    learning_rate=best_params["learning_rate"],
    subsample=best_params["subsample"],
    colsample_bytree=best_params["colsample_bytree"],
    random_state=42,
)
model.fit(X_train, y_train)

pred_probs = model.predict_proba(X_test)[:, 1]
preds = (pred_probs >= 0.5).astype(int)
print("Accuracy:", accuracy_score(y_test, preds))
print("Log Loss:", log_loss(y_test, pred_probs))

joblib.dump(model, path_trained_model_pkl(root))
