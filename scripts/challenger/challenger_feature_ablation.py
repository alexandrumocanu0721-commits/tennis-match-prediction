from __future__ import annotations

import os
from functools import partial

import optuna
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss
from xgboost import XGBClassifier

from challenger_config import (
    CHALLENGER_MODEL_EVAL_CUTOFF_DATE,
    path_challenger_processed_features_csv,
    project_root,
)
from clv_challenger_calculator import (
    SOFASCORE_ODDS_RELATIVE_PATH,
    map_row_to_player_a_market,
    name_key,
    prepare_sofascore_odds,
    resolve_candidate,
    surface_key,
)
from tennis_pipeline import FEATURES, temporal_train_validation_split

TARGET = "result"

RANK_FEATURES = ["rank_diff", "points_diff", "rank_momentum_diff"]
SERVE_RETURN_FEATURES = [
    "serve_rating_diff",
    "return_rating_diff",
    "bp_save_rate_diff",
    "bp_convert_rate_diff",
    "serve_rating_surface_diff",
    "return_rating_surface_diff",
    "bp_save_rate_surface_diff",
    "bp_convert_rate_surface_diff",
]
H2H_FEATURES = ["h2h_win_rate_diff", "h2h_surface_win_rate_diff"]
CORE_STATE_ONLY_KEEP = [
    "elo_diff",
    "surface_elo_diff",
    "recent_form_diff",
    "recent_surface_form_diff",
    "win_pct_diff",
    "matches_played_diff",
    "deciding_set_win_rate_diff",
]
CORE_STATE_ONLY_REMOVE = [feature for feature in FEATURES if feature not in CORE_STATE_ONLY_KEEP]

EXPERIMENTS = [
    ("full_model", []),
    ("no_rank_family", RANK_FEATURES),
    ("no_serve_return_family", SERVE_RETURN_FEATURES),
    ("no_h2h_family", H2H_FEATURES),
    ("core_state_only", CORE_STATE_ONLY_REMOVE),
]


def _build_model_params(trial: optuna.Trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "random_state": 42,
        "eval_metric": "logloss",
    }


def _objective(
    trial: optuna.Trial,
    X_fit: pd.DataFrame,
    y_fit: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> float:
    model = XGBClassifier(**_build_model_params(trial))
    model.fit(X_fit, y_fit)
    pred_probs = model.predict_proba(X_val)[:, 1]
    return log_loss(y_val, pred_probs, labels=[0, 1])


def _deduplicate_test_rows(df: pd.DataFrame) -> pd.DataFrame:
    test_df = df.copy()
    test_df["match_key"] = test_df.apply(
        lambda row: (
            row["date"],
            row["surface"],
            tuple(sorted((row["player_a_id"], row["player_b_id"]))),
        ),
        axis=1,
    )
    test_df = test_df.sort_values(
        ["date", "surface", "player_a_id", "player_b_id"], kind="mergesort"
    )
    return test_df.drop_duplicates(subset=["match_key"], keep="first").copy()


def _build_backtest_like_predictions(
    test_df: pd.DataFrame, probs_a: pd.Series | list[float]
) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "date": test_df["date"].dt.strftime("%Y-%m-%d"),
            "player_a": test_df["player_a_name"],
            "player_b": test_df["player_b_name"],
            "surface": test_df["surface"],
            "predicted_prob_a": probs_a,
            "actual_winner": test_df[TARGET].astype(int),
        }
    )
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    out["player_a_key"] = out["player_a"].map(name_key)
    out["player_b_key"] = out["player_b"].map(name_key)
    out["surface_key"] = out["surface"].map(surface_key)
    out["tournament_key"] = ""
    return out


def _compute_clv_mean(
    predictions_df: pd.DataFrame,
    valid_odds_df: pd.DataFrame,
    invalid_odds_df: pd.DataFrame,
) -> tuple[float, int]:
    clv_values: list[float] = []
    for _, prediction_row in predictions_df.iterrows():
        matched_row, _, _, _ = resolve_candidate(
            prediction_row, valid_odds_df, invalid_odds_df
        )
        if matched_row is None:
            continue
        market_prob_a, _, _, clv_bet365 = map_row_to_player_a_market(
            prediction_row, matched_row
        )
        if pd.notna(market_prob_a) and pd.notna(clv_bet365):
            clv_values.append(float(clv_bet365))

    if not clv_values:
        return float("nan"), 0
    return float(pd.Series(clv_values).mean()), len(clv_values)


def _feature_set(remove_features: list[str]) -> list[str]:
    missing = [feat for feat in remove_features if feat not in FEATURES]
    if missing:
        raise ValueError(f"Requested to remove unknown features: {missing}")
    return [feat for feat in FEATURES if feat not in set(remove_features)]


def run_experiment(
    name: str,
    remove_features: list[str],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    valid_odds_df: pd.DataFrame,
    invalid_odds_df: pd.DataFrame,
    n_trials: int,
) -> dict:
    feature_cols = _feature_set(remove_features)
    fit_df, val_df = temporal_train_validation_split(train_df, date_column="date")

    for split_name, split_df in [("train", train_df), ("fit", fit_df), ("val", val_df), ("test", test_df)]:
        nan_counts = split_df[feature_cols].isna().sum()
        bad = nan_counts[nan_counts > 0]
        if not bad.empty:
            raise ValueError(
                f"NaNs found for experiment '{name}' in {split_name} split:\n{bad.to_string()}"
            )

    X_fit = fit_df[feature_cols]
    y_fit = fit_df[TARGET]
    X_val = val_df[feature_cols]
    y_val = val_df[TARGET]

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(
        partial(_objective, X_fit=X_fit, y_fit=y_fit, X_val=X_val, y_val=y_val),
        n_trials=n_trials,
    )

    best_params = study.best_params
    model = XGBClassifier(
        n_estimators=best_params["n_estimators"],
        max_depth=best_params["max_depth"],
        learning_rate=best_params["learning_rate"],
        subsample=best_params["subsample"],
        colsample_bytree=best_params["colsample_bytree"],
        random_state=42,
        eval_metric="logloss",
    )
    model.fit(train_df[feature_cols], train_df[TARGET])

    probs_a = model.predict_proba(test_df[feature_cols])[:, 1]
    y_true = test_df[TARGET].astype(int)
    y_pred = (probs_a >= 0.5).astype(int)

    accuracy = float(accuracy_score(y_true, y_pred))
    loss = float(log_loss(y_true, probs_a, labels=[0, 1]))

    pred_df = _build_backtest_like_predictions(test_df, probs_a)
    clv_mean, clv_matched = _compute_clv_mean(
        pred_df, valid_odds_df, invalid_odds_df
    )

    return {
        "experiment": name,
        "num_features": len(feature_cols),
        "removed_features": ",".join(remove_features) if remove_features else "",
        "accuracy": accuracy,
        "log_loss": loss,
        "clv_mean": clv_mean,
        "clv_matched_rows": clv_matched,
    }


def main() -> None:
    root = project_root()
    n_trials = int(os.getenv("CHAL_ABLATION_TRIALS", "30"))

    features_df = pd.read_csv(path_challenger_processed_features_csv(root))
    features_df["date"] = pd.to_datetime(features_df["date"], errors="coerce")
    features_df = features_df.dropna(subset=["date"]).copy()

    cutoff = pd.Timestamp(CHALLENGER_MODEL_EVAL_CUTOFF_DATE)
    train_df = features_df[features_df["date"] < cutoff].copy()
    test_full_df = features_df[features_df["date"] >= cutoff].copy()
    if train_df.empty or test_full_df.empty:
        raise ValueError("Train/test split empty for challenger ablation.")

    test_df = _deduplicate_test_rows(test_full_df)
    valid_odds_df, invalid_odds_df, note = prepare_sofascore_odds(
        root / SOFASCORE_ODDS_RELATIVE_PATH
    )

    print(f"ablation trials per experiment: {n_trials}", flush=True)
    print(f"train rows: {len(train_df)}", flush=True)
    print(f"test rows (deduplicated): {len(test_df)}", flush=True)
    print(f"odds rows: {len(valid_odds_df)}", flush=True)
    print(f"odds filter: {note}", flush=True)

    results: list[dict] = []
    for experiment_name, remove_features in EXPERIMENTS:
        print(f"\nRunning {experiment_name}...", flush=True)
        result = run_experiment(
            name=experiment_name,
            remove_features=remove_features,
            train_df=train_df,
            test_df=test_df,
            valid_odds_df=valid_odds_df,
            invalid_odds_df=invalid_odds_df,
            n_trials=n_trials,
        )
        results.append(result)
        print(
            f"{experiment_name}: accuracy={result['accuracy']:.6f}, "
            f"log_loss={result['log_loss']:.6f}, "
            f"clv_mean={result['clv_mean']:.6f}, "
            f"clv_matched_rows={result['clv_matched_rows']}",
            flush=True,
        )

    result_df = pd.DataFrame(results)
    result_df = result_df[
        [
            "experiment",
            "num_features",
            "accuracy",
            "log_loss",
            "clv_mean",
            "clv_matched_rows",
            "removed_features",
        ]
    ]

    print("\nAblation summary:")
    print(result_df.to_string(index=False, float_format=lambda v: f"{v:.6f}"), flush=True)


if __name__ == "__main__":
    main()
