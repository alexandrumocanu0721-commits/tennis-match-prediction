import pandas as pd
from pathlib import Path

import optuna
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss

import joblib

df = pd.read_csv("./data/processed/features.csv")

df["date"] = pd.to_datetime(df["date"])
train_df = df[df["date"].dt.year <= 2024]
test_df = df[df["date"].dt.year == 2025]

def objective(trial):

    params = {
        "n_estimators": trial.suggest_int(
            "n_estimators",
            50,
            300
        ),

        "max_depth": trial.suggest_int(
            "max_depth",
            3,
            10
        ),

        "learning_rate": trial.suggest_float(
            "learning_rate",
            0.01,
            0.3,
            log=True
        ),

        "subsample": trial.suggest_float(
            "subsample",
            0.5,
            1.0
        ),

        "colsample_bytree": trial.suggest_float(
            "colsample_bytree",
            0.5,
            1.0
        ),

        "random_state": 42
    }

    model = XGBClassifier(**params)

    model.fit(X_train, y_train)

    pred_probs = model.predict_proba(X_test)[:, 1]

    loss = log_loss(y_test, pred_probs)

    return loss

FEATURES = [
    "elo_diff",
    "surface_elo_diff",
    "rank_diff",
    "points_diff",

    "recent_form_diff", 
    "recent_surface_form_diff", 
    "win_pct_diff", 
    "matches_played_diff",
]

X_train = train_df[FEATURES]
y_train = train_df["result"]
X_test = test_df[FEATURES]
y_test = test_df["result"]

study = optuna.create_study(
    direction="minimize"
)

study.optimize(
    objective,
    n_trials=30
)

best_params = study.best_params

model = XGBClassifier(
    n_estimators=best_params["n_estimators"],
    max_depth=best_params["max_depth"],
    learning_rate=best_params["learning_rate"],
    subsample=best_params["subsample"],
    colsample_bytree=best_params["colsample_bytree"],
    random_state=42
)
model.fit(X_train, y_train)

pred_probs = model.predict_proba(X_test)[:, 1]
preds = (pred_probs >= 0.5).astype(int)

accuracy = accuracy_score(y_test, preds)
loss = log_loss(y_test, pred_probs)

print("Accuracy:", accuracy)
print("Log Loss:", loss)

joblib.dump(model, "./models/xgboost_model.pkl")