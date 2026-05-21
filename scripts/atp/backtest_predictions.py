from __future__ import annotations

import joblib
import pandas as pd

from clv_calculator import (
    add_benchmark_probabilities,
    load_atp_backtest_odds,
    match_single_prediction,
    name_match_key,
    surface_key,
)
from tennis_pipeline import (
    FEATURES,
    MODEL_EVAL_CUTOFF_DATE,
    path_processed_backtest_predictions_csv,
    path_processed_features_csv,
    path_trained_model_pkl,
    project_root,
)


def _attach_market_columns(output_df: pd.DataFrame, root) -> pd.DataFrame:
    predictions_df = output_df.copy()
    predictions_df["date"] = pd.to_datetime(predictions_df["date"], errors="coerce").dt.normalize()
    predictions_df["surface_key"] = predictions_df["surface"].map(surface_key)
    predictions_df["player_a_key"] = predictions_df["player_a"].map(name_match_key)
    predictions_df["player_b_key"] = predictions_df["player_b"].map(name_match_key)
    predictions_df["prediction_row_id"] = range(len(predictions_df))

    odds_df = load_atp_backtest_odds(root)
    odds_df["date"] = pd.to_datetime(odds_df["Date"], dayfirst=True, errors="coerce").dt.normalize()
    odds_df["odds_date"] = odds_df["date"]
    odds_df["odds_surface_key"] = odds_df["Surface"].map(surface_key)
    odds_df["winner_key"] = odds_df["Winner"].map(name_match_key)
    odds_df["loser_key"] = odds_df["Loser"].map(name_match_key)
    odds_df["odds_row_id"] = range(len(odds_df))

    for benchmark_name, odds_prefix in (
        ("b365", "B365"),
        ("bf", "BFE"),
        ("pinnacle", "PS"),
        ("max", "Max"),
    ):
        if f"{odds_prefix}W" in odds_df.columns and f"{odds_prefix}L" in odds_df.columns:
            odds_df = add_benchmark_probabilities(odds_df, benchmark_name, odds_prefix)

    matched_records = []
    for _, prediction_row in predictions_df.iterrows():
        matched_record, _, _ = match_single_prediction(prediction_row, odds_df)
        if matched_record is not None:
            matched_records.append(matched_record)

    matched_df = pd.DataFrame(matched_records)
    if matched_df.empty:
        return output_df.iloc[0:0].assign(
            market_prob_b365=pd.Series(dtype=float),
            clv_b365=pd.Series(dtype=float),
            market_prob_bf=pd.Series(dtype=float),
            clv_bf=pd.Series(dtype=float),
        )

    matched_df = matched_df.sort_values(
        ["prediction_row_id", "date_diff_days", "surface_match"],
        ascending=[True, True, False],
        kind="mergesort",
    ).drop_duplicates(subset=["prediction_row_id"], keep="first")
    matched_df["market_prob_b365"] = matched_df.apply(
        lambda row: row["b365_true_prob_winner"]
        if row["match_orientation"] == "player_a_is_winner"
        else row["b365_true_prob_loser"],
        axis=1,
    )
    matched_df["market_prob_bf"] = matched_df.apply(
        lambda row: row["bf_true_prob_winner"]
        if row["match_orientation"] == "player_a_is_winner"
        else row["bf_true_prob_loser"],
        axis=1,
    )
    matched_df["clv_b365"] = matched_df["predicted_prob_a"] - matched_df["market_prob_b365"]
    matched_df["clv_bf"] = matched_df["predicted_prob_a"] - matched_df["market_prob_bf"]

    market_cols = matched_df[
        [
            "prediction_row_id",
            "market_prob_b365",
            "clv_b365",
            "market_prob_bf",
            "clv_bf",
            "odds_a_b365",
            "odds_b_b365",
            "odds_a_bf",
            "odds_b_bf",
        ]
    ]
    return predictions_df.merge(market_cols, on="prediction_row_id", how="inner").drop(
        columns=["surface_key", "player_a_key", "player_b_key", "prediction_row_id"]
    )


def main() -> None:
    root = project_root()

    df = pd.read_csv(path_processed_features_csv(root))
    df["date"] = pd.to_datetime(df["date"])

    # Historical test window only: matches from the model evaluation cutoff onward.
    test_df = df[df["date"] >= pd.Timestamp(MODEL_EVAL_CUTOFF_DATE)].copy()

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
            "year": test_df["date"].dt.year.astype(int),
            "player_a": test_df["player_a_name"],
            "player_b": test_df["player_b_name"],
            "surface": test_df["surface"],
            "predicted_prob_a": predicted_prob_a,
            "actual_winner": test_df["result"].astype(int),
        }
    )

    output_df = _attach_market_columns(output_df, root)
    output_df.to_csv(path_processed_backtest_predictions_csv(root), index=False)


if __name__ == "__main__":
    main()
