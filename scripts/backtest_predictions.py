from __future__ import annotations

import joblib
import pandas as pd

from tennis_pipeline import (
    FEATURES,
    path_processed_backtest_predictions_csv,
    path_processed_features_csv,
    path_trained_model_pkl,
    project_root,
)


def main() -> None:
    root = project_root()

    df = pd.read_csv(path_processed_features_csv(root))
    df["date"] = pd.to_datetime(df["date"])

    # Historical test window only: matches from 2026-01-01 onward.
    test_df = df[df["date"] >= pd.Timestamp("2026-01-01")].copy()

    # Keep one row per real match. features.csv stores each match twice, once per
    # player perspective, so collapse the mirrored pair into a stable canonical
    # orientation by sorting the player ids.
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
    test_df = test_df.drop_duplicates(subset=["match_key"], keep="first")

    model = joblib.load(path_trained_model_pkl(root))

    X_backtest = test_df[FEATURES]
    predicted_prob_a = model.predict_proba(X_backtest)[:, 1]

    output_df = pd.DataFrame(
        {
            "date": test_df["date"].dt.strftime("%Y-%m-%d"),
            "player_a": test_df["player_a_name"],
            "player_b": test_df["player_b_name"],
            "surface": test_df["surface"],
            "predicted_prob_a": predicted_prob_a,
            "actual_winner": test_df["result"].astype(int),
        }
    )

    output_df.to_csv(path_processed_backtest_predictions_csv(root), index=False)


if __name__ == "__main__":
    main()
