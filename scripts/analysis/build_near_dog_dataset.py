from __future__ import annotations

import argparse

from near_dog_specialist_common import (
    DATASET_PATH,
    DATASET_SUMMARY_PATH,
    OUTPUT_DIR,
    build_near_dog_dataset,
    dataset_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build two-sided 2.20-3.00 near-underdog specialist dataset."
    )
    parser.add_argument(
        "--tour",
        choices=["all", "atp", "challenger"],
        default="all",
        help="Tour artifacts to include in the specialist dataset.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset, merge_summary = build_near_dog_dataset(args.tour)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dataset.to_csv(DATASET_PATH, index=False)
    summary = dataset_summary(dataset, merge_summary)
    summary.round(6).to_csv(DATASET_SUMMARY_PATH, index=False)

    print(f"near-dog rows: {len(dataset)}")
    print(f"date range: {dataset['date'].min()} -> {dataset['date'].max()}")
    print("rows by period:")
    print(dataset["wf_period"].value_counts(dropna=False).to_string())
    print("rows by tour:")
    print(dataset["tour"].value_counts(dropna=False).to_string())
    print(f"wrote dataset: {DATASET_PATH}")
    print(f"wrote summary: {DATASET_SUMMARY_PATH}")


if __name__ == "__main__":
    main()
