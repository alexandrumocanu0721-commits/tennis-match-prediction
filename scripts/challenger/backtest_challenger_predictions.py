from __future__ import annotations

from pathlib import Path

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
from clv_challenger_calculator import (
    prepare_predictions,
    prepare_sofascore_odds,
    resolve_candidate,
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


def _sofascore_challenger_dir(root: Path) -> Path:
    return root / "data" / "sofascore" / "challenger"


def _full_year_odds_paths(root: Path) -> list[Path]:
    odds_dir = _sofascore_challenger_dir(root)
    return [
        path
        for path in [
            odds_dir / "challenger_2025_odds.csv",
            odds_dir / "challenger_2026_odds.csv",
        ]
        if path.exists()
    ]


def _load_covered_prediction_ids(predictions_df: pd.DataFrame, root: Path) -> set[int]:
    odds_paths = _full_year_odds_paths(root)
    if not odds_paths:
        raise FileNotFoundError(
            f"No full-year Challenger odds files found in {_sofascore_challenger_dir(root)}"
        )

    odds_frames = []
    invalid_frames = []
    for odds_path in odds_paths:
        odds_df, invalid_df, singles_filter_note = prepare_sofascore_odds(odds_path)
        print(f"{odds_path.name} SofaScore odds filter: {singles_filter_note}", flush=True)
        odds_df["odds_source_file"] = odds_path.name
        invalid_df["odds_source_file"] = odds_path.name
        odds_frames.append(odds_df)
        invalid_frames.append(invalid_df)

    odds_df = pd.concat(odds_frames, ignore_index=True)
    invalid_odds_df = pd.concat(invalid_frames, ignore_index=True)

    # The 2026 full-year replacement overlaps older partial exports. Deduplicate on
    # match identity before applying the unchanged reconciliation routine.
    odds_df["match_identity"] = odds_df.apply(
        lambda row: (
            row["match_start_date_utc"],
            row["tournament_key"],
            tuple(sorted((row["home_player_key"], row["away_player_key"]))),
        ),
        axis=1,
    )
    before = len(odds_df)
    odds_df = odds_df.sort_values(
        ["match_start_date_utc", "tournament_key", "home_player_key", "away_player_key"],
        kind="mergesort",
    ).drop_duplicates(subset=["match_identity"], keep="last")
    print(
        f"loaded Challenger full-year odds rows: {before}; after match-identity dedupe: {len(odds_df)}",
        flush=True,
    )

    covered_ids: set[int] = set()
    for _, prediction_row in predictions_df.iterrows():
        matched_row, _, _, _ = resolve_candidate(
            prediction_row, odds_df, invalid_odds_df
        )
        if matched_row is not None:
            covered_ids.add(int(prediction_row["prediction_row_id"]))
    return covered_ids


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
    print(f"evaluation rows: {len(test_df)}", flush=True)
    if test_df.empty:
        raise ValueError(
            f"No rows found with date >= {CHALLENGER_MODEL_EVAL_CUTOFF_DATE}."
        )

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
            "year": test_df["date"].dt.year.astype(int),
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

    prepared_predictions = prepare_predictions(output_df)
    covered_ids = _load_covered_prediction_ids(prepared_predictions, root)
    output_df = output_df.iloc[sorted(covered_ids)].copy()
    print(f"odds-covered backtest rows: {len(output_df)}", flush=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)
    print(f"output path: {output_path}", flush=True)
    print("new/modified files:", flush=True)
    print(f"- {output_path}", flush=True)


if __name__ == "__main__":
    main()
