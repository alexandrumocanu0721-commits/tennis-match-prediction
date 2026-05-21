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
OUTPUT_DIAGNOSTIC = PROJECT_ROOT / "data" / "processed" / "challenger" / "favorite_sleeve_diagnostic.txt"
CHALLENGER_CLV_RAW = PROJECT_ROOT / "data" / "processed" / "challenger" / "clv_results.csv"
CHALLENGER_BACKTEST = PROJECT_ROOT / "data" / "processed" / "challenger" / "backtest_predictions.csv"
ATP_BACKTEST = PROJECT_ROOT / "data" / "processed" / "atp" / "backtest_predictions.csv"

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


def _safe_read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def _missing_columns(df: pd.DataFrame | None, required: list[str]) -> list[str]:
    if df is None:
        return required
    return [col for col in required if col not in df.columns]


def _assign_odds_bucket_from_odds(odds: pd.Series) -> pd.Series:
    numeric_odds = pd.to_numeric(odds, errors="coerce")
    out = pd.Series(index=numeric_odds.index, dtype="object")
    out.loc[numeric_odds < 1.80] = "favorite"
    out.loc[(numeric_odds >= 1.80) & (numeric_odds <= 2.20)] = "balanced"
    out.loc[numeric_odds > 2.20] = "underdog"
    return out


def _ensure_year(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    out = df.copy()
    if "year" not in out.columns and date_col in out.columns:
        out["year"] = pd.to_datetime(out[date_col], errors="coerce").dt.year.astype("Int64")
    return out


def _ensure_month(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    out = df.copy()
    out["date_parsed"] = pd.to_datetime(out[date_col], errors="coerce")
    out["month"] = out["date_parsed"].dt.to_period("M").astype(str)
    out.loc[out["date_parsed"].isna(), "month"] = np.nan
    return out


def _flat_stake_roi(working: pd.DataFrame) -> float:
    bet_count = int(len(working))
    if bet_count == 0:
        return float("nan")
    wins = pd.to_numeric(working["actual_result"], errors="coerce").eq(1.0)
    returns = np.where(wins, pd.to_numeric(working["odds_a"], errors="coerce"), 0.0)
    return float((np.sum(returns) - bet_count) / bet_count)


def _favorite_breakdown_row(group: pd.DataFrame) -> dict:
    bet_count = int(len(group))
    if bet_count == 0:
        return {
            "bet_count": 0,
            "hit_rate": float("nan"),
            "mean_odds_a": float("nan"),
            "roi": float("nan"),
            "flag": "LOW_SAMPLE",
        }
    return {
        "bet_count": bet_count,
        "hit_rate": float(pd.to_numeric(group["actual_result"], errors="coerce").mean()),
        "mean_odds_a": float(pd.to_numeric(group["odds_a"], errors="coerce").mean()),
        "roi": _flat_stake_roi(group),
        "flag": "LOW_SAMPLE" if bet_count < 30 else "",
    }


def _format_metric_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col, decimals in [("hit_rate", 3), ("mean_odds_a", 3), ("roi", 4)]:
        if col in out.columns:
            out[col] = out[col].map(lambda x, d=decimals: "NaN" if pd.isna(x) else round(float(x), d))
    if "flag" in out.columns:
        out["flag"] = out["flag"].fillna("")
    return out


def _table_to_lines(df: pd.DataFrame) -> list[str]:
    if df.empty:
        return ["(no rows)"]
    return [df.to_string(index=False)]


def _error_table_lines(path: Path, missing_cols: list[str]) -> list[str]:
    if not path.exists():
        return [f"ERROR: missing file {path}"]
    return [f"ERROR: missing column(s) in {path}: {', '.join(missing_cols)}"]


def _build_group_breakdown(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows = []
    for group_value, group in df.groupby(group_col, sort=True, dropna=False):
        row = {group_col: group_value}
        row.update(_favorite_breakdown_row(group))
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=[group_col, "bet_count", "hit_rate", "mean_odds_a", "roi", "flag"])
    out = pd.DataFrame(rows)
    return out[[group_col, "bet_count", "hit_rate", "mean_odds_a", "roi", "flag"]]


def _month_surface_odds_sections(base_df: pd.DataFrame, year: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    working = _ensure_month(_ensure_year(base_df))
    working["odds_bucket"] = _assign_odds_bucket_from_odds(working["odds_a"])
    filtered = working.loc[
        working["year"].eq(year)
        & working["odds_bucket"].eq("favorite")
        & working["actual_result"].notna()
        & working["odds_a"].notna()
    ].copy()

    month_df = _build_group_breakdown(filtered.loc[filtered["month"].notna()].copy(), "month")

    surface_df = _build_group_breakdown(filtered.loc[filtered["surface"].notna()].copy(), "surface")
    if not surface_df.empty:
        surface_df["surface"] = pd.Categorical(surface_df["surface"], categories=SURFACES, ordered=True)
        surface_df = surface_df.sort_values("surface").reset_index(drop=True)
        surface_df["surface"] = surface_df["surface"].astype(str)

    odds_df = filtered.copy()
    odds_df["odds_sub_bucket"] = pd.Series(pd.NA, index=odds_df.index, dtype="object")
    odds_df.loc[pd.to_numeric(odds_df["odds_a"], errors="coerce") < 1.30, "odds_sub_bucket"] = "tight"
    odds_df.loc[
        pd.to_numeric(odds_df["odds_a"], errors="coerce").ge(1.30)
        & pd.to_numeric(odds_df["odds_a"], errors="coerce").lt(1.50),
        "odds_sub_bucket",
    ] = "mid"
    odds_df.loc[
        pd.to_numeric(odds_df["odds_a"], errors="coerce").ge(1.50)
        & pd.to_numeric(odds_df["odds_a"], errors="coerce").lt(1.70),
        "odds_sub_bucket",
    ] = "wide"
    odds_df = _build_group_breakdown(odds_df.loc[odds_df["odds_sub_bucket"].notna()].copy(), "odds_sub_bucket")
    if not odds_df.empty:
        odds_df["odds_sub_bucket"] = pd.Categorical(
            odds_df["odds_sub_bucket"],
            categories=["tight", "mid", "wide"],
            ordered=True,
        )
        odds_df = odds_df.sort_values("odds_sub_bucket").reset_index(drop=True)
        odds_df["odds_sub_bucket"] = odds_df["odds_sub_bucket"].astype(str)

    return month_df, surface_df, odds_df


def _describe_bucket_definition(label: str, source_col: str) -> list[str]:
    return [
        f"{label}: favorite if odds < 1.80; balanced if 1.80 <= odds <= 2.20; underdog if odds > 2.20.",
        f"{label}: bucket assignment source column = {source_col}.",
    ]


def _favorite_distribution_table(
    df: pd.DataFrame,
    market_prob_col: str,
    odds_col: str,
    year_col: str = "year",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    working = _ensure_year(df)
    working[odds_col] = pd.to_numeric(working[odds_col], errors="coerce")
    working[market_prob_col] = pd.to_numeric(working[market_prob_col], errors="coerce")
    working["odds_bucket"] = _assign_odds_bucket_from_odds(working[odds_col])
    favorites = working.loc[
        working["odds_bucket"].eq("favorite")
        & working[market_prob_col].notna()
        & working[odds_col].notna()
    ].copy()

    if favorites.empty:
        dist = pd.DataFrame(
            [{"mean": float("nan"), "median": float("nan"), "p10": float("nan"), "p90": float("nan")}]
        )
        by_year = pd.DataFrame(columns=["year", "mean_odds_a"])
        return dist, by_year

    market_prob = favorites[market_prob_col]
    dist = pd.DataFrame(
        [
            {
                "mean": float(market_prob.mean()),
                "median": float(market_prob.median()),
                "p10": float(market_prob.quantile(0.10)),
                "p90": float(market_prob.quantile(0.90)),
            }
        ]
    )
    by_year = (
        favorites.groupby(year_col, dropna=True, sort=True)
        .agg(mean_odds_a=(odds_col, "mean"))
        .reset_index()
    )
    return dist, by_year


def _format_distribution_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col, decimals in [("mean", 4), ("median", 4), ("p10", 4), ("p90", 4), ("mean_odds_a", 3)]:
        if col in out.columns:
            out[col] = out[col].map(lambda x, d=decimals: "NaN" if pd.isna(x) else round(float(x), d))
    return out


def _rank_diff_bucket(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").abs()
    bins = [-np.inf, 50, 100, 200, 500, np.inf]
    labels = ["0-50", "50-100", "100-200", "200-500", "500+"]
    return pd.cut(numeric, bins=bins, labels=labels, right=False)


def _confidence_bucket(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    bins = [0.65, 0.70, 0.75, 0.80, np.inf]
    labels = ["0.65-0.70", "0.70-0.75", "0.75-0.80", "0.80+"]
    return pd.cut(numeric, bins=bins, labels=labels, right=False)


def _build_task4_tables(base_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    working = _ensure_year(base_df)
    working["odds_bucket"] = _assign_odds_bucket_from_odds(working["odds_a"])
    filtered = working.loc[
        working["year"].eq(2026)
        & working["odds_bucket"].eq("favorite")
        & working["actual_result"].notna()
        & working["odds_a"].notna()
    ].copy()

    rank_df = pd.DataFrame(columns=["rank_diff_bucket", "bet_count", "hit_rate", "mean_odds_a", "roi", "flag"])
    if "rank_diff" in filtered.columns:
        rank_working = filtered.copy()
        rank_working["rank_diff_bucket"] = _rank_diff_bucket(rank_working["rank_diff"])
        rank_df = _build_group_breakdown(rank_working.loc[rank_working["rank_diff_bucket"].notna()].copy(), "rank_diff_bucket")
        if not rank_df.empty:
            rank_df["rank_diff_bucket"] = pd.Categorical(
                rank_df["rank_diff_bucket"],
                categories=["0-50", "50-100", "100-200", "200-500", "500+"],
                ordered=True,
            )
            rank_df = rank_df.sort_values("rank_diff_bucket").reset_index(drop=True)
            rank_df["rank_diff_bucket"] = rank_df["rank_diff_bucket"].astype(str)

    confidence_df = pd.DataFrame(
        columns=["confidence_bucket", "bet_count", "hit_rate", "mean_odds_a", "roi", "flag"]
    )
    if "predicted_prob_a" in filtered.columns:
        conf_working = filtered.copy()
        conf_working["confidence_bucket"] = _confidence_bucket(conf_working["predicted_prob_a"])
        confidence_df = _build_group_breakdown(
            conf_working.loc[conf_working["confidence_bucket"].notna()].copy(),
            "confidence_bucket",
        )
        if not confidence_df.empty:
            confidence_df["confidence_bucket"] = pd.Categorical(
                confidence_df["confidence_bucket"],
                categories=["0.65-0.70", "0.70-0.75", "0.75-0.80", "0.80+"],
                ordered=True,
            )
            confidence_df = confidence_df.sort_values("confidence_bucket").reset_index(drop=True)
            confidence_df["confidence_bucket"] = confidence_df["confidence_bucket"].astype(str)

    return rank_df, confidence_df


def _summary_assessment(
    task1_month: pd.DataFrame,
    task1_surface: pd.DataFrame,
    task2_month: pd.DataFrame,
    task2_surface: pd.DataFrame,
    task2_odds: pd.DataFrame,
    challenger_dist: pd.DataFrame | None,
    atp_dist: pd.DataFrame | None,
    challenger_odds_year: pd.DataFrame | None,
    atp_odds_year: pd.DataFrame | None,
    rank_df: pd.DataFrame,
    confidence_df: pd.DataFrame,
) -> str:
    lines: list[str] = []

    positive_months = 0
    if not task1_month.empty:
        positive_months = int(((pd.to_numeric(task1_month["roi"], errors="coerce") > 0) & (task1_month["bet_count"] >= 30)).sum())
    hard_ok = False
    clay_ok = False
    if not task1_surface.empty:
        hard_row = task1_surface.loc[task1_surface["surface"].eq("Hard")]
        clay_row = task1_surface.loc[task1_surface["surface"].eq("Clay")]
        hard_ok = not hard_row.empty and float(hard_row["roi"].iloc[0]) > 0 and int(hard_row["bet_count"].iloc[0]) >= 30
        clay_ok = not clay_row.empty and float(clay_row["roi"].iloc[0]) > 0 and int(clay_row["bet_count"].iloc[0]) >= 30

    if positive_months >= 4 and hard_ok and clay_ok:
        lines.append("2026 Challenger favorite ROI looks distributed rather than concentrated: positive ROI appears across multiple 30+ bet months and both Hard and Clay are positive.")
    elif positive_months <= 2:
        lines.append("2026 Challenger favorite ROI looks concentrated in a small number of months, which weakens the case that this is a stable edge.")
    else:
        lines.append("2026 Challenger favorite ROI is mixed: some spread exists, but the month and surface tables should be treated as only partial support for a stable edge.")

    if not task2_odds.empty:
        worst_odds = task2_odds.sort_values("roi").iloc[0]
        lines.append(
            f"2025 Challenger favorite losses are most visible in the {worst_odds['odds_sub_bucket']} odds sleeve"
            f" (ROI {float(worst_odds['roi']):.4f}, {int(worst_odds['bet_count'])} bets)."
        )
    elif not task2_month.empty or not task2_surface.empty:
        lines.append("2025 Challenger favorite losses can be inspected in the month and surface tables; no odds sub-bucket conclusion was available.")
    else:
        lines.append("2025 Challenger favorite loss attribution could not be established from the required columns.")

    if challenger_dist is not None and atp_dist is not None and not challenger_dist.empty and not atp_dist.empty:
        chall_mean = float(pd.to_numeric(challenger_dist["mean"], errors="coerce").iloc[0])
        atp_mean = float(pd.to_numeric(atp_dist["mean"], errors="coerce").iloc[0])
        odds_gap_msg = "suggests the hit-rate gap is at least partly definitional"
        if challenger_odds_year is not None and atp_odds_year is not None and not challenger_odds_year.empty and not atp_odds_year.empty:
            chall_2026 = challenger_odds_year.loc[challenger_odds_year["year"].astype(int).eq(2026), "mean_odds_a"]
            atp_2026 = atp_odds_year.loc[atp_odds_year["year"].astype(int).eq(2026), "mean_odds_a"]
            if not chall_2026.empty and not atp_2026.empty:
                if abs(float(chall_2026.iloc[0]) - float(atp_2026.iloc[0])) < 0.10 and abs(chall_mean - atp_mean) < 0.05:
                    odds_gap_msg = "suggests the hit-rate gap is more likely genuine than purely definitional"
        lines.append(
            f"ATP vs Challenger favorite bucket comparison: Challenger favorite mean market_prob {chall_mean:.4f} vs ATP {atp_mean:.4f}; this {odds_gap_msg}."
        )
    else:
        lines.append("ATP vs Challenger favorite bucket comparison is incomplete because one of the required favorite-distribution inputs was missing.")

    if not rank_df.empty and not confidence_df.empty:
        best_rank = rank_df.sort_values("roi", ascending=False).iloc[0]
        best_conf = confidence_df.sort_values("roi", ascending=False).iloc[0]
        lines.append(
            f"Within 2026 Challenger favorites, the strongest slices are rank_diff {best_rank['rank_diff_bucket']}"
            f" and confidence {best_conf['confidence_bucket']}; if these also carry decent sample sizes, the edge is more intuitive and defensible."
        )
    elif not rank_df.empty or not confidence_df.empty:
        lines.append("Only one of the rank-diff or confidence interaction tables was available, so stability assessment on feature interaction is partial.")
    else:
        lines.append("Rank-diff and confidence interaction checks were unavailable.")

    return " ".join(lines)


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

    diagnostic_lines: list[str] = []

    def add_section(title: str, lines: list[str]) -> None:
        diagnostic_lines.append(title)
        diagnostic_lines.extend(lines if lines else ["(no rows)"])
        diagnostic_lines.append("")

    task1_missing = _missing_columns(df, ["year", "date", "surface", "odds_a", "actual_result"])
    task1_month_df = pd.DataFrame()
    task1_surface_df = pd.DataFrame()
    if task1_missing:
        error_lines = _error_table_lines(INPUT_CSV, task1_missing)
        add_section("=== Task 1: 2026 Challenger Favorites — Monthly Breakdown ===", error_lines)
        add_section("=== Task 1: 2026 Challenger Favorites — Surface Breakdown ===", error_lines)
    else:
        task1_month_df, task1_surface_df, _ = _month_surface_odds_sections(df, 2026)
        add_section(
            "=== Task 1: 2026 Challenger Favorites — Monthly Breakdown ===",
            _table_to_lines(_format_metric_table(task1_month_df)),
        )
        add_section(
            "=== Task 1: 2026 Challenger Favorites — Surface Breakdown ===",
            _table_to_lines(_format_metric_table(task1_surface_df)),
        )

    task2_missing = _missing_columns(df, ["year", "date", "surface", "odds_a", "actual_result"])
    task2_month_df = pd.DataFrame()
    task2_surface_df = pd.DataFrame()
    task2_odds_df = pd.DataFrame()
    if task2_missing:
        error_lines = _error_table_lines(INPUT_CSV, task2_missing)
        add_section("=== Task 2: 2025 Challenger Favorites — Monthly Breakdown ===", error_lines)
        add_section("=== Task 2: 2025 Challenger Favorites — Surface Breakdown ===", error_lines)
        add_section("=== Task 2: 2025 Challenger Favorites — Odds Sub-bucket Breakdown ===", error_lines)
    else:
        task2_month_df, task2_surface_df, task2_odds_df = _month_surface_odds_sections(df, 2025)
        add_section(
            "=== Task 2: 2025 Challenger Favorites — Monthly Breakdown ===",
            _table_to_lines(_format_metric_table(task2_month_df)),
        )
        add_section(
            "=== Task 2: 2025 Challenger Favorites — Surface Breakdown ===",
            _table_to_lines(_format_metric_table(task2_surface_df)),
        )
        add_section(
            "=== Task 2: 2025 Challenger Favorites — Odds Sub-bucket Breakdown ===",
            _table_to_lines(_format_metric_table(task2_odds_df)),
        )

    task3_lines: list[str] = []
    task3_lines.extend(_describe_bucket_definition("Challenger pipeline", "odds_a from clv_results/clv_results_calibrated"))
    task3_lines.extend(_describe_bucket_definition("ATP pipeline", "odds_a_b365-derived odds / odds_a in pipeline CLV output"))
    task3_lines.append(
        "Challenger backtest_predictions.csv lacks market_prob/odds/actual_result, so favorite-bucket distribution stats use the existing Challenger CLV file already consumed by this script."
    )
    task3_lines.append("")

    challenger_source = df.copy()
    challenger_dist_df: pd.DataFrame | None = None
    challenger_odds_year_df: pd.DataFrame | None = None
    challenger_missing = _missing_columns(challenger_source, ["market_prob_a", "odds_a", "year"])
    if challenger_missing:
        raw_challenger = _safe_read_csv(CHALLENGER_CLV_RAW)
        raw_missing = _missing_columns(raw_challenger, ["market_prob_a", "odds_a", "year"])
        if raw_missing:
            task3_lines.extend(_error_table_lines(CHALLENGER_CLV_RAW, raw_missing))
        else:
            challenger_dist_df, challenger_odds_year_df = _favorite_distribution_table(raw_challenger, "market_prob_a", "odds_a")
    else:
        challenger_dist_df, challenger_odds_year_df = _favorite_distribution_table(challenger_source, "market_prob_a", "odds_a")

    atp_backtest_df = _safe_read_csv(ATP_BACKTEST)
    atp_dist_df: pd.DataFrame | None = None
    atp_odds_year_df: pd.DataFrame | None = None
    atp_missing = _missing_columns(atp_backtest_df, ["market_prob_b365", "odds_a_b365", "year"])
    if atp_missing:
        task3_lines.extend(_error_table_lines(ATP_BACKTEST, atp_missing))
    else:
        atp_dist_df, atp_odds_year_df = _favorite_distribution_table(atp_backtest_df, "market_prob_b365", "odds_a_b365")

    if challenger_dist_df is not None:
        task3_lines.append("Challenger favorite market_prob distribution:")
        task3_lines.extend(_table_to_lines(_format_distribution_table(challenger_dist_df)))
        task3_lines.append("")
    if challenger_odds_year_df is not None:
        task3_lines.append("Challenger favorite mean odds_a by year:")
        task3_lines.extend(_table_to_lines(_format_distribution_table(challenger_odds_year_df)))
        task3_lines.append("")
    if atp_dist_df is not None:
        task3_lines.append("ATP favorite market_prob distribution:")
        task3_lines.extend(_table_to_lines(_format_distribution_table(atp_dist_df)))
        task3_lines.append("")
    if atp_odds_year_df is not None:
        task3_lines.append("ATP favorite mean odds_a by year:")
        task3_lines.extend(_table_to_lines(_format_distribution_table(atp_odds_year_df)))
        task3_lines.append("")
    add_section("=== Task 3: Odds Bucket Definition Comparison ===", task3_lines)

    rank_df = pd.DataFrame()
    confidence_df = pd.DataFrame()
    task4_missing = _missing_columns(df, ["year", "odds_a", "actual_result"])
    if task4_missing:
        error_lines = _error_table_lines(INPUT_CSV, task4_missing)
        add_section("=== Task 4: 2026 Challenger Favorites — Rank Diff Breakdown ===", error_lines)
        add_section("=== Task 4: 2026 Challenger Favorites — Confidence Breakdown ===", error_lines)
    else:
        rank_df, confidence_df = _build_task4_tables(df)
        if "rank_diff" not in df.columns:
            add_section(
                "=== Task 4: 2026 Challenger Favorites — Rank Diff Breakdown ===",
                _error_table_lines(INPUT_CSV, ["rank_diff"]),
            )
        else:
            add_section(
                "=== Task 4: 2026 Challenger Favorites — Rank Diff Breakdown ===",
                _table_to_lines(_format_metric_table(rank_df)),
            )
        if "predicted_prob_a" not in df.columns:
            add_section(
                "=== Task 4: 2026 Challenger Favorites — Confidence Breakdown ===",
                _error_table_lines(INPUT_CSV, ["predicted_prob_a"]),
            )
        else:
            add_section(
                "=== Task 4: 2026 Challenger Favorites — Confidence Breakdown ===",
                _table_to_lines(_format_metric_table(confidence_df)),
            )

    add_section(
        "=== Summary ===",
        [
            _summary_assessment(
                task1_month_df,
                task1_surface_df,
                task2_month_df,
                task2_surface_df,
                task2_odds_df,
                challenger_dist_df,
                atp_dist_df,
                challenger_odds_year_df,
                atp_odds_year_df,
                rank_df,
                confidence_df,
            )
        ],
    )

    OUTPUT_DIAGNOSTIC.write_text("\n".join(diagnostic_lines), encoding="utf-8")
    print(f"Diagnostic report written to: {OUTPUT_DIAGNOSTIC}")

    elapsed = time.perf_counter() - start
    print(f"Total runtime: {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()
