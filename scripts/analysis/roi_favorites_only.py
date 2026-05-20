from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_CSV = PROJECT_ROOT / "data" / "processed" / "challenger" / "clv_results_calibrated.csv"
OUTPUT_SUMMARY = PROJECT_ROOT / "data" / "processed" / "challenger" / "roi_favorites_summary.csv"
OUTPUT_BY_SURFACE = PROJECT_ROOT / "data" / "processed" / "challenger" / "roi_favorites_by_surface.csv"
OUTPUT_BY_TIER = PROJECT_ROOT / "data" / "processed" / "challenger" / "roi_favorites_by_tier.csv"
OUTPUT_BY_MONTH = PROJECT_ROOT / "data" / "processed" / "challenger" / "roi_favorites_by_month.csv"

EV_THRESHOLDS = [0.00, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10]
SURFACES = ["Hard", "Clay", "Grass"]
KEY_COLS = ["calibrated_clv", "calibrated_prob_a", "odds_a", "actual_result"]


def _print_df(label: str, df: pd.DataFrame) -> None:
    print(label)
    if df.empty:
        print("(no rows)")
    else:
        print(df.to_string(index=False))
    print()


def _distribution_bins(series: pd.Series, width: float) -> pd.DataFrame:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return pd.DataFrame(columns=["clv_bin", "count"])

    left = np.floor(clean.min() / width) * width
    right = np.ceil(clean.max() / width) * width
    if np.isclose(left, right):
        right = left + width

    bins = np.arange(left, right + width, width)
    binned = pd.cut(clean, bins=bins, right=False, include_lowest=True)
    counts = binned.value_counts(sort=False)

    out = counts.reset_index()
    out.columns = ["clv_bin", "count"]
    out["clv_bin"] = out["clv_bin"].astype(str)
    out["count"] = out["count"].astype(int)
    return out


def _summary_from_filtered(filtered: pd.DataFrame) -> tuple[int, float, float, float, float]:
    bet_count = int(len(filtered))
    if bet_count == 0:
        return 0, float("nan"), float("nan"), float("nan"), float("nan")

    wins = filtered["actual_result"].astype(float).eq(1.0)
    profits = np.where(wins, filtered["odds_a"].astype(float) - 1.0, -1.0)

    hit_rate = float(wins.mean())
    avg_odds = float(pd.to_numeric(filtered["odds_a"], errors="coerce").mean())
    mean_clv = float(pd.to_numeric(filtered["calibrated_clv"], errors="coerce").mean())
    roi = float(np.sum(profits) / bet_count)
    return bet_count, hit_rate, avg_odds, mean_clv, roi


def _make_row(filtered: pd.DataFrame) -> dict:
    bet_count, hit_rate, avg_odds, mean_clv, roi = _summary_from_filtered(filtered)
    row = {
        "bet_count": bet_count,
        "hit_rate": hit_rate,
        "avg_odds": avg_odds,
        "mean_calibrated_clv": mean_clv,
        "roi": roi,
    }
    if not pd.isna(roi) and not (-1.0 <= float(roi) <= 5.0):
        row["flag"] = "SUSPICIOUS"
    return row


def _add_low_sample_flag(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "flag" not in out.columns:
        out["flag"] = ""

    def _combine_flags(row: pd.Series) -> str:
        flags = []
        if int(row["bet_count"]) < 15:
            flags.append("LOW_SAMPLE")
        existing = str(row.get("flag", "")).strip()
        if existing:
            flags.append(existing)
        return "|".join(flags)

    out["flag"] = out.apply(_combine_flags, axis=1)
    return out


def _round_for_csv(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "ev_threshold" in out.columns:
        out["ev_threshold"] = out["ev_threshold"].round(2)
    for col, decimals in [
        ("hit_rate", 3),
        ("avg_odds", 3),
        ("mean_calibrated_clv", 4),
        ("roi", 4),
    ]:
        if col in out.columns:
            out[col] = out[col].round(decimals)
    return out


def _display_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    def _fmt(series: pd.Series, decimals: int) -> pd.Series:
        return series.map(lambda x: "NaN" if pd.isna(x) else round(float(x), decimals))

    if "ev_threshold" in out.columns:
        out["ev_threshold"] = _fmt(out["ev_threshold"], 2)
    for col, decimals in [
        ("hit_rate", 3),
        ("avg_odds", 3),
        ("mean_calibrated_clv", 4),
        ("roi", 4),
    ]:
        if col in out.columns:
            out[col] = _fmt(out[col], decimals)
    if "flag" in out.columns:
        out["flag"] = out["flag"].fillna("")
    return out


def main() -> None:
    start = time.perf_counter()

    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)

    print(f"Shape: {df.shape}")
    print()
    print("Null counts (key columns):")
    print(df[KEY_COLS].isna().sum().to_string())
    print()

    favorites = df.loc[pd.to_numeric(df["odds_a"], errors="coerce") < 1.80].copy()

    print(f"Favorites row count (odds_a < 1.80): {len(favorites)}")
    print()
    print("Favorites calibrated_clv distribution (0.05 bins):")
    dist = _distribution_bins(favorites["calibrated_clv"], 0.05)
    if dist.empty:
        print("(no non-null calibrated_clv values for favorites)")
    else:
        print(dist.to_string(index=False))
    print()

    if len(favorites) < 100:
        print("WARNING: fewer than 100 favorite rows available.")
        print()

    print(f"Row count after favorites-only filter: {len(favorites)}")
    print()

    summary_rows = []
    surface_rows = []

    for threshold in EV_THRESHOLDS:
        filtered = favorites.loc[
            pd.to_numeric(favorites["calibrated_clv"], errors="coerce").ge(threshold)
            & favorites["actual_result"].notna()
            & favorites["odds_a"].notna()
        ].copy()

        row = {"ev_threshold": threshold}
        row.update(_make_row(filtered))
        summary_rows.append(row)

        for surface in SURFACES:
            sf = filtered.loc[filtered["surface"].eq(surface)]
            srow = {"ev_threshold": threshold, "surface": surface}
            srow.update(_make_row(sf))
            surface_rows.append(srow)

    summary_df = pd.DataFrame(summary_rows)[
        ["ev_threshold", "bet_count", "hit_rate", "avg_odds", "mean_calibrated_clv", "roi", "flag"]
        if "flag" in pd.DataFrame(summary_rows).columns
        else ["ev_threshold", "bet_count", "hit_rate", "avg_odds", "mean_calibrated_clv", "roi"]
    ]

    if "flag" not in summary_df.columns:
        summary_df["flag"] = ""

    assert int(summary_df.loc[summary_df["ev_threshold"].eq(0.00), "bet_count"].iloc[0]) > 0, (
        "No bets found for EV threshold 0.00 in favorites-only data."
    )

    surface_df = pd.DataFrame(surface_rows)
    if "flag" not in surface_df.columns:
        surface_df["flag"] = ""
    surface_df = surface_df[
        ["ev_threshold", "surface", "bet_count", "hit_rate", "avg_odds", "mean_calibrated_clv", "roi", "flag"]
    ]
    surface_df = _add_low_sample_flag(surface_df)

    tier_df_source = favorites.loc[
        favorites["actual_result"].notna() & favorites["odds_a"].notna() & favorites["calibrated_clv"].notna()
    ].copy()

    tier_rows = []
    tiers = [
        ("low_edge", tier_df_source["calibrated_clv"].ge(0.00) & tier_df_source["calibrated_clv"].lt(0.05)),
        ("medium_edge", tier_df_source["calibrated_clv"].ge(0.05) & tier_df_source["calibrated_clv"].lt(0.10)),
        ("high_edge", tier_df_source["calibrated_clv"].ge(0.10)),
    ]

    for tier_name, mask in tiers:
        tier_filtered = tier_df_source.loc[mask]
        row = {"tier": tier_name}
        row.update(_make_row(tier_filtered))
        tier_rows.append(row)

    tier_df = pd.DataFrame(tier_rows)
    if "flag" not in tier_df.columns:
        tier_df["flag"] = ""
    tier_df = tier_df[["tier", "bet_count", "hit_rate", "avg_odds", "mean_calibrated_clv", "roi", "flag"]]
    tier_df = _add_low_sample_flag(tier_df)

    month_source = favorites.loc[
        favorites["actual_result"].notna() & favorites["odds_a"].notna() & favorites["calibrated_clv"].notna()
    ].copy()
    month_source["date_parsed"] = pd.to_datetime(month_source["date"], errors="coerce")
    month_source = month_source.loc[month_source["date_parsed"].notna()].copy()
    month_source["month"] = month_source["date_parsed"].dt.to_period("M").astype(str)

    month_rows = []
    for month, group in month_source.groupby("month", sort=True):
        row = {"month": month}
        row.update(_make_row(group))
        month_rows.append(row)

    month_df = pd.DataFrame(month_rows)
    if month_df.empty:
        month_df = pd.DataFrame(
            columns=["month", "bet_count", "hit_rate", "avg_odds", "mean_calibrated_clv", "roi", "flag"]
        )
    else:
        if "flag" not in month_df.columns:
            month_df["flag"] = ""
        month_df = month_df[["month", "bet_count", "hit_rate", "avg_odds", "mean_calibrated_clv", "roi", "flag"]]

    for name, table in [
        ("summary", summary_df),
        ("surface", surface_df),
        ("tier", tier_df),
        ("month", month_df),
    ]:
        for idx, row in table.iterrows():
            roi = row.get("roi")
            if not pd.isna(roi) and not (-1.0 <= float(roi) <= 5.0):
                print(
                    f"SUSPICIOUS ROI in {name} row {idx}: "
                    f"roi={float(roi):.4f}"
                )

    _round_for_csv(summary_df).to_csv(OUTPUT_SUMMARY, index=False)
    _round_for_csv(surface_df).to_csv(OUTPUT_BY_SURFACE, index=False)
    _round_for_csv(tier_df).to_csv(OUTPUT_BY_TIER, index=False)
    _round_for_csv(month_df).to_csv(OUTPUT_BY_MONTH, index=False)

    _print_df("=== FAVORITES ROI SUMMARY ===", _display_table(summary_df))
    _print_df("=== FAVORITES ROI BY SURFACE ===", _display_table(surface_df))
    _print_df("=== FAVORITES ROI BY CLV TIER ===", _display_table(tier_df))
    _print_df("=== FAVORITES ROI BY MONTH ===", _display_table(month_df))

    elapsed = time.perf_counter() - start
    print(f"Total runtime: {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()
