from __future__ import annotations

import argparse

import pandas as pd

from near_dog_specialist_common import (
    DATASET_PATH,
    TRAIN_RESULTS_PATH,
    build_near_dog_dataset,
    save_model_artifact,
    train_specialist,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the near-underdog specialist model and select its edge gate on tune data."
    )
    parser.add_argument(
        "--rebuild-dataset",
        action="store_true",
        help="Rebuild the near-dog dataset before training.",
    )
    parser.add_argument(
        "--tour",
        choices=["all", "atp", "challenger"],
        default="all",
        help="Tour artifacts to include when --rebuild-dataset is used.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.rebuild_dataset or not DATASET_PATH.exists():
        dataset, _ = build_near_dog_dataset(args.tour)
        DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
        dataset.to_csv(DATASET_PATH, index=False)
    else:
        dataset = pd.read_csv(DATASET_PATH)

    artifact, tune_results = train_specialist(dataset)
    save_model_artifact(artifact)
    TRAIN_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    tune_results.round(6).to_csv(TRAIN_RESULTS_PATH, index=False)

    print(f"selected candidate: {artifact['candidate_name']}")
    print(f"selection status: {artifact['selection_status']}")
    print(f"shrinkage factor: {artifact['shrinkage_factor']:.2f}")
    print(f"edge threshold: {artifact['edge_threshold']:.4f}")
    print("selected tune metrics:")
    for key, value in artifact["selected_tune_metrics"].items():
        print(f"- {key}: {value}")
    print(f"wrote model: {artifact['candidate_name']} -> models/underdog_specialist/near_dog_specialist.pkl")
    print(f"wrote train results: {TRAIN_RESULTS_PATH}")


if __name__ == "__main__":
    main()
