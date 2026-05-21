from __future__ import annotations

import time
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_CSV = PROJECT_ROOT / "data" / "processed" / "challenger" / "clv_results.csv"
BACKTEST_CSV = PROJECT_ROOT / "data" / "processed" / "challenger" / "backtest_predictions.csv"
OUTPUT_SUMMARY = PROJECT_ROOT / "data" / "processed" / "challenger" / "roi_summary.csv"
OUTPUT_BY_SURFACE = PROJECT_ROOT / "data" / "processed" / "challenger" / "roi_by_surface.csv"
OUTPUT_BY_BUCKET = PROJECT_ROOT / "data" / "processed" / "challenger" / "roi_by_bucket.csv"

EV_THRESHOLDS = [0.00, 0.01, 0.02, 0.03, 0.05]
SURFACES = ["Hard", "Clay", "Grass"]
ODDS_BUCKETS = ["favorite", "balanced", "underdog"]


def _print_df(label: str, df: pd.DataFrame) -> None:
    print(label)
    if df.empty:
        print("(no rows)")
    else:
        print(df.to_string(index=False))
    print()


def _format_optional_float(value: float | None, decimals: int) -> float | str:
    if pd.isna(value):
        return "NaN"
    return round(float(value), decimals)


def _summarize_bets(
    df: pd.DataFrame,
    ev_threshold: float,
    year: int,
    group_label: str | None = None,
    group_value: str | None = None,
) -> dict:
    working = df.loc[
        df["clv_bet365"].ge(ev_threshold)
        & df["actual_result"].notna()
        & df["odds_a"].notna()
    ].copy()

    if group_label is not None:
        working = working.loc[working[group_label].eq(group_value)]

    bet_count = int(len(working))
    if bet_count == 0:
        result = {
            "year": year,
            "ev_threshold": ev_threshold,
            "bet_count": 0,
            "hit_rate": float("nan"),
            "avg_odds": float("nan"),
            "mean_clv": float("nan"),
            "roi": float("nan"),
            "sample_flag": "LOW_SAMPLE",
        }
        if group_label is not None:
            result[group_label] = group_value
        return result

    wins = int(working["actual_result"].sum())
    hit_rate = wins / bet_count
    profits = working["odds_a"].where(working["actual_result"].eq(1), other=0.0) - 1.0
    profits = profits.where(working["actual_result"].eq(1), other=-1.0)
    total_profit = float(profits.sum())
    roi = total_profit / bet_count

    result = {
        "year": year,
        "ev_threshold": ev_threshold,
        "bet_count": bet_count,
        "hit_rate": hit_rate,
        "avg_odds": float(working["odds_a"].mean()),
        "mean_clv": float(working["clv_bet365"].mean()),
        "roi": roi,
        "sample_flag": "LOW_SAMPLE" if bet_count < 20 else "",
    }
    if group_label is not None:
        result[group_label] = group_value
    return result


def _format_table(df: pd.DataFrame, float_columns: dict[str, int]) -> pd.DataFrame:
    formatted = df.copy()
    for col, decimals in float_columns.items():
        if col in formatted.columns:
            formatted[col] = formatted[col].map(lambda x, d=decimals: _format_optional_float(x, d))
    return formatted


def _round_table(df: pd.DataFrame, float_columns: dict[str, int]) -> pd.DataFrame:
    rounded = df.copy()
    for col, decimals in float_columns.items():
        if col in rounded.columns:
            rounded[col] = rounded[col].round(decimals)
    return rounded


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
    print("Null counts:")
    print(df[["clv_bet365", "odds_a", "actual_result"]].isna().sum().to_string())
    print()
    print("Surface counts:")
    print(df["surface"].value_counts(dropna=False).to_string())
    print()

    if "actual_result" not in df.columns or df["actual_result"].isna().all():
        print("actual_result is missing or all null. Stopping before simulation.")
        return

    df = df.copy()
    if "year" not in df.columns:
        df["year"] = pd.to_datetime(df["date"], errors="coerce").dt.year.astype("Int64")
    df["odds_a"] = pd.to_numeric(df["odds_a"], errors="coerce")
    df["odds_bucket"] = pd.cut(
        df["odds_a"],
        bins=[float("-inf"), 1.80, 2.20, float("inf")],
        labels=ODDS_BUCKETS,
        right=True,
        include_lowest=True,
    )

    summary_rows = []
    surface_rows = []
    bucket_rows = []

    for year in sorted(df["year"].dropna().astype(int).unique()):
        year_df = df.loc[df["year"].eq(year)].copy()
        for threshold in EV_THRESHOLDS:
            summary_rows.append(_summarize_bets(year_df, threshold, year))
            for surface in SURFACES:
                row = _summarize_bets(year_df, threshold, year, "surface", surface)
                surface_rows.append(row)
            for bucket in ODDS_BUCKETS:
                row = _summarize_bets(year_df, threshold, year, "odds_bucket", bucket)
                bucket_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    by_surface = pd.DataFrame(surface_rows)
    by_bucket = pd.DataFrame(bucket_rows)

    assert (summary["bet_count"] > 0).any(), "No bets found for any threshold."

    for name, result_df in [
        ("summary", summary),
        ("surface", by_surface),
        ("bucket", by_bucket),
    ]:
        for idx, row in result_df.iterrows():
            if not pd.isna(row["roi"]) and not (-1.0 <= float(row["roi"]) <= 5.0):
                print(
                    f"SUSPICIOUS ROI in {name} at row {idx}: "
                    f"ev_threshold={row['ev_threshold']}, roi={row['roi']}"
                )
            if not pd.isna(row["hit_rate"]):
                assert 0.0 <= float(row["hit_rate"]) <= 1.0, (
                    f"Invalid hit_rate in {name} at row {idx}: {row['hit_rate']}"
                )

    summary_out = _format_table(summary, {"ev_threshold": 2, "hit_rate": 3, "avg_odds": 3, "mean_clv": 4, "roi": 4})
    surface_out = _format_table(by_surface, {"ev_threshold": 2, "hit_rate": 3, "avg_odds": 3, "mean_clv": 4, "roi": 4})
    bucket_out = _format_table(by_bucket, {"ev_threshold": 2, "hit_rate": 3, "avg_odds": 3, "mean_clv": 4, "roi": 4})

    summary_csv = _round_table(summary, {"ev_threshold": 2, "hit_rate": 3, "avg_odds": 3, "mean_clv": 4, "roi": 4})
    surface_csv = _round_table(by_surface, {"ev_threshold": 2, "hit_rate": 3, "avg_odds": 3, "mean_clv": 4, "roi": 4})
    bucket_csv = _round_table(by_bucket, {"ev_threshold": 2, "hit_rate": 3, "avg_odds": 3, "mean_clv": 4, "roi": 4})

    summary_csv.to_csv(OUTPUT_SUMMARY, index=False)
    surface_csv.to_csv(OUTPUT_BY_SURFACE, index=False)
    bucket_csv.to_csv(OUTPUT_BY_BUCKET, index=False)

    _print_df("=== CHALLENGER ROI SUMMARY ===", summary_out)
    _print_df("=== CHALLENGER ROI BY SURFACE ===", surface_out)
    _print_df("=== CHALLENGER ROI BY ODDS BUCKET ===", bucket_out)

    elapsed = time.perf_counter() - start
    print(f"Total runtime: {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()
