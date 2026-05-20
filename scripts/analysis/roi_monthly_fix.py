from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_CSV = PROJECT_ROOT / "data" / "processed" / "challenger" / "clv_results_calibrated.csv"


def _print_df(label: str, df: pd.DataFrame) -> None:
    print(label)
    if df.empty:
        print("(no rows)")
    else:
        print(df.to_string(index=False))
    print()


def _monthly_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[
                "month",
                "bet_count",
                "hit_rate",
                "avg_odds",
                "mean_calibrated_clv",
                "roi",
                "flag",
            ]
        )

    working = df.copy()
    wins = working["actual_result"].astype(float).eq(1.0)
    working["profit"] = np.where(wins, working["odds_a"].astype(float) - 1.0, -1.0)

    grouped = (
        working.groupby("month", sort=True)
        .agg(
            bet_count=("actual_result", "size"),
            hit_rate=("actual_result", "mean"),
            avg_odds=("odds_a", "mean"),
            mean_calibrated_clv=("calibrated_clv", "mean"),
            roi=("profit", "mean"),
        )
        .reset_index()
    )
    grouped["flag"] = np.where(grouped["bet_count"] < 10, "LOW_SAMPLE", "")
    return grouped


def _round_for_display(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col, dec in [
        ("hit_rate", 3),
        ("avg_odds", 3),
        ("mean_calibrated_clv", 4),
        ("roi", 4),
    ]:
        if col in out.columns:
            out[col] = out[col].map(lambda x, d=dec: "NaN" if pd.isna(x) else round(float(x), d))
    return out


def _prepare_base(df: pd.DataFrame) -> pd.DataFrame:
    working = df.copy()
    working["date_parsed"] = pd.to_datetime(working["date"], errors="coerce")

    base = working.loc[
        pd.to_numeric(working["odds_a"], errors="coerce").lt(1.80)
        & working["actual_result"].notna()
        & working["odds_a"].notna()
    ].copy()

    base["month"] = base["date_parsed"].dt.to_period("M").astype(str)
    return base


def _cumulative_hard_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["date", "bet_number", "profit", "cumulative_profit", "cumulative_roi"])

    working = df.sort_values(["date_parsed", "event_id", "prediction_row_id"], kind="mergesort").copy()
    wins = working["actual_result"].astype(float).eq(1.0)
    working["profit"] = np.where(wins, working["odds_a"].astype(float) - 1.0, -1.0)
    working["bet_number"] = np.arange(1, len(working) + 1)
    working["cumulative_profit"] = working["profit"].cumsum()
    working["cumulative_roi"] = working["cumulative_profit"] / working["bet_number"]

    out = working[["date", "bet_number", "profit", "cumulative_profit", "cumulative_roi"]].reset_index(drop=True)

    n = len(out)
    selected = {0, n - 1}
    selected.update(i for i in range(9, n, 10))
    out = out.iloc[sorted(selected)].copy()
    return out


def main() -> None:
    start = time.perf_counter()

    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_CSV}")

    raw = pd.read_csv(INPUT_CSV)
    base = _prepare_base(raw)

    print(f"Total rows after base filters: {len(base)}")
    print()

    surface_counts = (
        base["surface"].fillna("NaN").value_counts(dropna=False).rename_axis("surface").reset_index(name="row_count")
    )
    _print_df("=== ROW COUNTS BY SURFACE (BASE FILTERED) ===", surface_counts)

    month_counts = (
        base.loc[base["date_parsed"].notna(), "month"]
        .value_counts(sort=False)
        .rename_axis("month")
        .reset_index(name="row_count")
        .sort_values("month")
        .reset_index(drop=True)
    )
    _print_df("=== ROW COUNTS BY MONTH (BASE FILTERED) ===", month_counts)

    step1 = base.loc[pd.to_numeric(base["calibrated_clv"], errors="coerce").ge(0.00)].copy()
    table1 = _monthly_summary(step1)
    _print_df("=== MONTHLY ROI ALL SURFACES (EV >= 0.00) ===", _round_for_display(table1))

    hard_ev0 = base.loc[
        base["surface"].eq("Hard")
        & pd.to_numeric(base["calibrated_clv"], errors="coerce").ge(0.00)
    ].copy()
    table2 = _monthly_summary(hard_ev0)
    _print_df("=== MONTHLY ROI HARD ONLY (EV >= 0.00) ===", _round_for_display(table2))

    hard_ev5 = base.loc[
        base["surface"].eq("Hard")
        & pd.to_numeric(base["calibrated_clv"], errors="coerce").ge(0.05)
    ].copy()
    table3 = _monthly_summary(hard_ev5)
    _print_df("=== MONTHLY ROI HARD ONLY (EV >= 0.05) ===", _round_for_display(table3))

    clay_ev0 = base.loc[
        base["surface"].eq("Clay")
        & pd.to_numeric(base["calibrated_clv"], errors="coerce").ge(0.00)
    ].copy()
    table4 = _monthly_summary(clay_ev0)
    _print_df("=== MONTHLY ROI CLAY ONLY (EV >= 0.00) ===", _round_for_display(table4))

    cum_table = _cumulative_hard_table(hard_ev0)
    _print_df(
        "=== HARD CUMULATIVE ROI OVER TIME (EV >= 0.00, EVERY 10TH + FIRST/LAST) ===",
        _round_for_display(cum_table),
    )

    elapsed = time.perf_counter() - start
    print(f"Total runtime: {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()
