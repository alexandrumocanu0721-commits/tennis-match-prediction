from __future__ import annotations

from functools import partial

import joblib
import optuna
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss
from xgboost import XGBClassifier

from challenger_config import (
    CHALLENGER_MODEL_EVAL_CUTOFF_DATE,
    assert_challenger_model_path,
    path_challenger_model_pkl,
    path_challenger_processed_features_csv,
    project_root,
)
from tennis_pipeline import FEATURES, temporal_train_validation_split

TARGET = "result"


def dedupe_match_rows(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["match_key"] = out.apply(
        lambda row: (
            row["date"],
            row["surface"],
            row["tourney_id"] if "tourney_id" in out.columns else "",
            tuple(sorted((row["player_a_id"], row["player_b_id"]))),
        ),
        axis=1,
    )
    sort_cols = ["date", "surface", "player_a_id", "player_b_id"]
    if "tourney_id" in out.columns:
        sort_cols.insert(1, "tourney_id")
    out = out.sort_values(sort_cols, kind="mergesort")
    return out.drop_duplicates(subset=["match_key"], keep="first").copy()


def objective(
    trial: optuna.Trial,
    X_fit: pd.DataFrame,
    y_fit: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> float:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "random_state": 42,
        "eval_metric": "logloss",
    }
    model = XGBClassifier(**params)
    model.fit(X_fit, y_fit)
    pred_probs = model.predict_proba(X_val)[:, 1]
    return log_loss(y_val, pred_probs)


def main() -> None:
    root = project_root()
    features_path = path_challenger_processed_features_csv(root)
    df = pd.read_csv(features_path)

    print(f"Feature file shape: {df.shape}", flush=True)
    df["date"] = pd.to_datetime(df["date"])

    nan_counts = df[FEATURES].isna().sum()
    print("NaN counts for FEATURES:", flush=True)
    print(nan_counts.to_string(), flush=True)
    missing_features = nan_counts[nan_counts > 0]
    if not missing_features.empty:
        raise ValueError(
            "Found NaNs in FEATURES; refusing to train:\n"
            f"{missing_features.to_string()}"
        )

    cutoff = pd.Timestamp(CHALLENGER_MODEL_EVAL_CUTOFF_DATE)
    train_df = df[df["date"] < cutoff].copy()
    test_df = df[df["date"] >= cutoff].copy()

    if train_df.empty:
        raise ValueError("Training split is empty (date < 2026-01-01).")
    if test_df.empty:
        raise ValueError("Test split is empty (date >= 2026-01-01).")

    print(f"Train size: {len(train_df)}", flush=True)
    print(f"Test size: {len(test_df)}", flush=True)
    print(
        "Train date range: "
        f"{train_df['date'].min().date()} -> {train_df['date'].max().date()}",
        flush=True,
    )
    print(
        "Test date range: "
        f"{test_df['date'].min().date()} -> {test_df['date'].max().date()}",
        flush=True,
    )
    print(
        f"Train result balance:\n{train_df[TARGET].value_counts(normalize=True).sort_index()}",
        flush=True,
    )
    print(
        f"Test result balance:\n{test_df[TARGET].value_counts(normalize=True).sort_index()}",
        flush=True,
    )

    fit_df, val_df = temporal_train_validation_split(train_df, date_column="date")

    X_train = train_df[FEATURES]
    y_train = train_df[TARGET]
    X_fit = fit_df[FEATURES]
    y_fit = fit_df[TARGET]
    X_val = val_df[FEATURES]
    y_val = val_df[TARGET]
    X_test = test_df[FEATURES]
    y_test = test_df[TARGET]

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(
        partial(objective, X_fit=X_fit, y_fit=y_fit, X_val=X_val, y_val=y_val),
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
        eval_metric="logloss",
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
        eval_metric="logloss",
    )
    model.fit(X_train, y_train)

    test_probs = model.predict_proba(X_test)[:, 1]
    test_preds = (test_probs >= 0.5).astype(int)
    test_match_df = dedupe_match_rows(test_df)
    test_match_probs = model.predict_proba(test_match_df[FEATURES])[:, 1]
    test_match_preds = (test_match_probs >= 0.5).astype(int)
    test_match_y = test_match_df[TARGET].astype(int)

    model_path = path_challenger_model_pkl(root)
    assert_challenger_model_path(model_path, root)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)

    importances = (
        pd.Series(model.feature_importances_, index=FEATURES)
        .sort_values(ascending=False)
    )

    print(f"Best Optuna params: {best_params}")
    print(f"Validation log loss before full-train refit: {val_log_loss:.6f}")
    print(f"2026 mirrored-row test accuracy: {accuracy_score(y_test, test_preds):.6f}")
    print(f"2026 mirrored-row test log loss: {log_loss(y_test, test_probs):.6f}")
    print(
        "2026 one-row-per-match test accuracy: "
        f"{accuracy_score(test_match_y, test_match_preds):.6f}"
    )
    print(
        "2026 one-row-per-match test log loss: "
        f"{log_loss(test_match_y, test_match_probs, labels=[0, 1]):.6f}"
    )
    print("Feature importance (descending):")
    print(importances.to_string())
    print(f"Saved model: {model_path}")


if __name__ == "__main__":
    main()
