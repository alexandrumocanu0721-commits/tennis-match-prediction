from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from near_dog_specialist_common import (
    BACKTEST_BY_TOUR_PATH,
    BACKTEST_CALIBRATION_PATH,
    BACKTEST_PREDICTIONS_PATH,
    BACKTEST_SUMMARY_PATH,
    DATASET_PATH,
    FINAL_RECOMMENDATION_PATH,
    MODEL_PATH,
    backtest_summary,
    backtest_by_tour,
    calibration_table,
    final_recommendation,
    load_model_artifact,
    score_with_artifact,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backtest the saved near-underdog specialist model on train/tune/final periods."
    )
    parser.add_argument(
        "--dataset",
        default=str(DATASET_PATH),
        help="Near-dog dataset CSV built by build_near_dog_dataset.py.",
    )
    parser.add_argument(
        "--model",
        default=str(MODEL_PATH),
        help="Saved near-dog specialist joblib artifact.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = pd.read_csv(args.dataset)
    artifact = load_model_artifact(Path(args.model))

    scored = score_with_artifact(dataset, artifact)
    summary = backtest_summary(scored)
    by_tour = backtest_by_tour(scored)
    calibration = calibration_table(scored)
    recommendation = final_recommendation(scored, artifact)

    BACKTEST_PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    scored.round(6).to_csv(BACKTEST_PREDICTIONS_PATH, index=False)
    summary.round(6).to_csv(BACKTEST_SUMMARY_PATH, index=False)
    by_tour.round(6).to_csv(BACKTEST_BY_TOUR_PATH, index=False)
    calibration.round(6).to_csv(BACKTEST_CALIBRATION_PATH, index=False)
    FINAL_RECOMMENDATION_PATH.write_text(recommendation, encoding="utf-8")

    print(recommendation)
    print(f"wrote predictions: {BACKTEST_PREDICTIONS_PATH}")
    print(f"wrote summary: {BACKTEST_SUMMARY_PATH}")
    print(f"wrote by-tour summary: {BACKTEST_BY_TOUR_PATH}")
    print(f"wrote calibration: {BACKTEST_CALIBRATION_PATH}")


if __name__ == "__main__":
    main()
