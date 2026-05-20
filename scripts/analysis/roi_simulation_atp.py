from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_CSV = PROJECT_ROOT / "data" / "processed" / "atp" / "clv_results.csv"
OUTPUT_SUMMARY = PROJECT_ROOT / "data" / "processed" / "atp" / "roi_summary_atp.csv"
OUTPUT_BY_SURFACE = PROJECT_ROOT / "data" / "processed" / "atp" / "roi_by_surface_atp.csv"
OUTPUT_BY_BUCKET = PROJECT_ROOT / "data" / "processed" / "atp" / "roi_by_bucket_atp.csv"

EV_THRESHOLDS = [0.00, 0.01, 0.02, 0.03, 0.05]
SURFACES = ["Hard", "Clay", "Grass"]
ODDS_BUCKETS = ["favorite", "balanced", "underdog"]


def _print_labeled_df(label: str, df: pd.DataFrame) -> None:
    print(label)
    if df.empty:
        print("(no rows)")
    else:
        print(df.to_string(index=False))
    print()


def _distribution_bins(series: pd.Series, width: float) -> pd.Series:
    clean = series.dropna().astype(float)
    if clean.empty:
        return pd.Series(dtype="int64")

    left = np.floor(clean.min() / width) * width
    right = np.ceil(clean.max() / width) * width
    if np.isclose(left, right):
        right = left + width

    bins = np.arange(left, right + width, width)
    binned = pd.cut(clean, bins=bins, right=False, include_lowest=True)
    counts = binned.value_counts(sort=False)
    return counts


def _format_float(val: float, decimals: int) -> float | str:
    if pd.isna(val):
        return "NaN"
    return round(float(val), decimals)


def _summarize(df: pd.DataFrame, threshold: float) -> dict:
    filtered = df.loc[
        df["clv_betfair"].ge(threshold)
        & df["actual_winner"].notna()
        & df["betfair_true_prob_a"].notna()
        & df["odds_a"].notna()
    ].copy()

    bet_count = int(len(filtered))
    if bet_count == 0:
        return {
            "ev_threshold": threshold,
            "bet_count": 0,
            "hit_rate": np.nan,
            "avg_odds": np.nan,
            "mean_clv": np.nan,
            "roi": np.nan,
        }

    stake = 1.0
    wins = filtered["actual_winner"].astype(float).eq(1.0)
    profits = np.where(wins, (filtered["odds_a"] - 1.0) * stake, -stake)

    hit_rate = float(wins.mean())
    avg_odds = float(filtered["odds_a"].mean())
    mean_clv = float(filtered["clv_betfair"].mean())
    roi = float(np.sum(profits) / bet_count)

    return {
        "ev_threshold": threshold,
        "bet_count": bet_count,
        "hit_rate": hit_rate,
        "avg_odds": avg_odds,
        "mean_clv": mean_clv,
        "roi": roi,
    }


def _format_table(df: pd.DataFrame, include_flag: bool = False) -> pd.DataFrame:
    out = df.copy()
    out["ev_threshold"] = out["ev_threshold"].map(lambda x: _format_float(x, 2))
    out["hit_rate"] = out["hit_rate"].map(lambda x: _format_float(x, 3))
    out["avg_odds"] = out["avg_odds"].map(lambda x: _format_float(x, 3))
    out["mean_clv"] = out["mean_clv"].map(lambda x: _format_float(x, 4))
    out["roi"] = out["roi"].map(lambda x: _format_float(x, 4))
    if include_flag and "sample_flag" in out.columns:
        out["sample_flag"] = out["sample_flag"].fillna("")
    return out


def _round_for_csv(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ev_threshold"] = out["ev_threshold"].round(2)
    out["hit_rate"] = out["hit_rate"].round(3)
    out["avg_odds"] = out["avg_odds"].round(3)
    out["mean_clv"] = out["mean_clv"].round(4)
    out["roi"] = out["roi"].round(4)
    return out


def main() -> None:
    start = time.perf_counter()

    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)

    print(f"Shape: {df.shape}")
    print()
    print("Head(3):")
    print(df.head(3).to_string(index=False))
    print()
    print("Null counts (key columns):")
    print(df[["clv_betfair", "betfair_true_prob_a", "actual_winner"]].isna().sum().to_string())
    print()
    print("Surface counts:")
    print(df["surface"].value_counts(dropna=False).to_string())
    print()

    print("CLV Betfair distribution (0.05 bins):")
    clv_dist = _distribution_bins(df["clv_betfair"], 0.05)
    if clv_dist.empty:
        print("(no non-null clv_betfair values)")
    else:
        print(clv_dist.to_string())
    print()

    print("Predicted Prob A distribution (0.10 bins):")
    prob_dist = _distribution_bins(df["predicted_prob_a"], 0.10)
    if prob_dist.empty:
        print("(no non-null predicted_prob_a values)")
    else:
        print(prob_dist.to_string())
    print()

    if "actual_winner" not in df.columns or df["actual_winner"].isna().all():
        print("actual_winner is missing or all null. Stopping before simulation.")
        elapsed = time.perf_counter() - start
        print(f"Total runtime: {elapsed:.2f} seconds")
        return

    df = df.copy()
    prob = df["betfair_true_prob_a"].astype(float)
    df["odds_a"] = np.where(prob > 0, 1.0 / prob, np.nan)

    df["odds_bucket"] = pd.cut(
        df["odds_a"],
        bins=[-np.inf, 1.80, 2.20, np.inf],
        labels=ODDS_BUCKETS,
        right=True,
        include_lowest=True,
    ).astype("object")

    print("Odds bucket row counts:")
    print(df["odds_bucket"].value_counts(dropna=False).to_string())
    print()

    summary_rows: list[dict] = []
    by_surface_rows: list[dict] = []
    by_bucket_rows: list[dict] = []

    for threshold in EV_THRESHOLDS:
        base_row = _summarize(df, threshold)
        summary_rows.append(base_row)

        for surface in SURFACES:
            row = _summarize(df.loc[df["surface"].eq(surface)], threshold)
            row["surface"] = surface
            row["sample_flag"] = "LOW_SAMPLE" if row["bet_count"] < 20 else ""
            by_surface_rows.append(row)

        for bucket in ODDS_BUCKETS:
            row = _summarize(df.loc[df["odds_bucket"].eq(bucket)], threshold)
            row["odds_bucket"] = bucket
            row["sample_flag"] = "LOW_SAMPLE" if row["bet_count"] < 20 else ""
            by_bucket_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)[
        ["ev_threshold", "bet_count", "hit_rate", "avg_odds", "mean_clv", "roi"]
    ]
    by_surface_df = pd.DataFrame(by_surface_rows)[
        ["ev_threshold", "surface", "bet_count", "hit_rate", "avg_odds", "mean_clv", "roi", "sample_flag"]
    ]
    by_bucket_df = pd.DataFrame(by_bucket_rows)[
        ["ev_threshold", "odds_bucket", "bet_count", "hit_rate", "avg_odds", "mean_clv", "roi", "sample_flag"]
    ]

    assert (summary_df["bet_count"] > 0).any(), "No bets found for any threshold."

    for name, table in [
        ("summary", summary_df),
        ("surface", by_surface_df),
        ("bucket", by_bucket_df),
    ]:
        for idx, row in table.iterrows():
            roi = row["roi"]
            hit_rate = row["hit_rate"]

            if not pd.isna(roi) and not (-1.0 <= float(roi) <= 5.0):
                print(
                    f"SUSPICIOUS ROI in {name} row {idx}: "
                    f"ev_threshold={row['ev_threshold']:.2f}, roi={roi:.4f}"
                )
            if not pd.isna(hit_rate):
                assert 0.0 <= float(hit_rate) <= 1.0, (
                    f"Invalid hit_rate in {name} row {idx}: {hit_rate}"
                )

    _round_for_csv(summary_df).to_csv(OUTPUT_SUMMARY, index=False)
    _round_for_csv(by_surface_df).to_csv(OUTPUT_BY_SURFACE, index=False)
    _round_for_csv(by_bucket_df).to_csv(OUTPUT_BY_BUCKET, index=False)

    _print_labeled_df("=== ATP ROI SUMMARY ===", _format_table(summary_df))
    _print_labeled_df("=== ATP ROI BY SURFACE ===", _format_table(by_surface_df, include_flag=True))
    _print_labeled_df("=== ATP ROI BY ODDS BUCKET ===", _format_table(by_bucket_df, include_flag=True))

    elapsed = time.perf_counter() - start
    print(f"Total runtime: {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()
