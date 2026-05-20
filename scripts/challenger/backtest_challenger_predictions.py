from __future__ import annotations

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

from challenger_config import (
    CHALLENGER_MODEL_EVAL_CUTOFF_DATE,
    assert_challenger_processed_output_path,
    path_challenger_backtest_predictions_csv,
    path_challenger_model_pkl,
    path_challenger_processed_features_csv,
    project_root,
)
from tennis_pipeline import FEATURES


def _validate_output_path(root) -> None:
    output_path = path_challenger_backtest_predictions_csv(root).resolve()
    expected = (root / "data" / "processed" / "challenger" / "backtest_predictions.csv").resolve()
    forbidden = (root / "data" / "processed" / "atp" / "backtest_predictions.csv").resolve()

    if output_path != expected:
        raise ValueError(
            "Safety check failed: challenger backtest output path is unexpected "
            f"({output_path})."
        )
    if output_path == forbidden:
        raise ValueError(
            "Safety check failed: refusing to overwrite ATP backtest output path."
        )
    assert_challenger_processed_output_path(output_path, root)


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


def main() -> None:
    root = project_root()
    _validate_output_path(root)

    features_path = path_challenger_processed_features_csv(root)
    model_path = path_challenger_model_pkl(root)
    output_path = path_challenger_backtest_predictions_csv(root)

    df = pd.read_csv(features_path)
    df["date"] = pd.to_datetime(df["date"])
    print(f"input rows: {len(df)}", flush=True)

    cutoff = pd.Timestamp(CHALLENGER_MODEL_EVAL_CUTOFF_DATE)
    test_df = df[df["date"] >= cutoff].copy()
    print(f"2026 rows: {len(test_df)}", flush=True)
    if test_df.empty:
        raise ValueError("No rows found with date >= 2026-01-01.")

    test_df = dedupe_match_rows(test_df)
    print(f"deduplicated match rows: {len(test_df)}", flush=True)

    model = joblib.load(model_path)
    probs_a = model.predict_proba(test_df[FEATURES])[:, 1]
    probs_b = 1.0 - probs_a
    y_true = test_df["result"].astype(int)
    y_pred = (probs_a >= 0.5).astype(int)

    accuracy = accuracy_score(y_true, y_pred)
    loss = log_loss(y_true, probs_a, labels=[0, 1])
    print(f"accuracy: {accuracy:.6f}", flush=True)
    print(f"log loss: {loss:.6f}", flush=True)

    output_df = pd.DataFrame(
        {
            "date": test_df["date"].dt.strftime("%Y-%m-%d"),
            "tourney_id": test_df.get("tourney_id", ""),
            "tourney_name": test_df.get("tourney_name", ""),
            "tourney_date": pd.to_datetime(
                test_df.get("tourney_date", test_df["date"])
            ).dt.strftime("%Y-%m-%d"),
            "round": test_df.get("round", ""),
            "player_a": test_df["player_a_name"],
            "player_b": test_df["player_b_name"],
            "surface": test_df["surface"],
            "predicted_prob_a": probs_a,
            "predicted_prob_b": probs_b,
            "actual_winner": y_true,
        }
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)
    print(f"output path: {output_path}", flush=True)
    print("new/modified files:", flush=True)
    print(f"- {output_path}", flush=True)


if __name__ == "__main__":
    main()
