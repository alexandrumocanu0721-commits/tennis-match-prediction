from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PREFERRED = PROJECT_ROOT / "models" / "challenger" / "challenger_xgboost_model.pkl"
MODELS_DIR = PROJECT_ROOT / "models" / "challenger"
CLV_RESULTS = PROJECT_ROOT / "data" / "processed" / "challenger" / "clv_results.csv"
BACKTEST_PREDICTIONS = PROJECT_ROOT / "data" / "processed" / "challenger" / "backtest_predictions.csv"
OUTPUT_CALIBRATED = PROJECT_ROOT / "data" / "processed" / "challenger" / "clv_results_calibrated.csv"
OUTPUT_RELIABILITY_BY_YEAR = PROJECT_ROOT / "data" / "processed" / "challenger" / "calibration_reliability_by_year.csv"
OUTPUT_BUCKET_BY_YEAR = PROJECT_ROOT / "data" / "processed" / "challenger" / "calibration_bucket_by_year.csv"

EV_THRESHOLDS = [0.00, 0.01, 0.02, 0.03, 0.05]
MIN_CALIBRATION_ROWS = 50


def _format_value(value: float | int | str | None, decimals: int | None = None) -> str:
    if isinstance(value, str):
        return value
    if value is None or pd.isna(value):
        return "NaN"
    if decimals is None:
        return str(value)
    return f"{float(value):.{decimals}f}"


def _print_table(title: str, df: pd.DataFrame, round_map: dict[str, int]) -> None:
    print(title)
    if df.empty:
        print("(no rows)")
        print()
        return
    rendered = df.copy()
    for col, digits in round_map.items():
        if col in rendered.columns:
            rendered[col] = rendered[col].map(lambda x, d=digits: _format_value(x, d))
    print(rendered.to_string(index=False))
    print()


def _find_model_path() -> Path:
    if MODEL_PREFERRED.exists():
        return MODEL_PREFERRED

    model_files = sorted(MODELS_DIR.rglob("*.pkl"))
    if not model_files:
        raise FileNotFoundError(f"No .pkl model found under {MODELS_DIR}")

    challenger_candidates = [p for p in model_files if "challenger" in p.as_posix().lower()]
    if challenger_candidates:
        return challenger_candidates[0]
    return model_files[0]


def _load_calibration_source() -> tuple[pd.DataFrame, str]:
    if not CLV_RESULTS.exists():
        raise FileNotFoundError(f"Missing input file: {CLV_RESULTS}")
    if not BACKTEST_PREDICTIONS.exists():
        raise FileNotFoundError(f"Missing fallback file: {BACKTEST_PREDICTIONS}")

    clv = pd.read_csv(CLV_RESULTS)
    labeled_clv = clv["actual_result"].notna().sum() if "actual_result" in clv.columns else 0
    if labeled_clv >= MIN_CALIBRATION_ROWS:
        return clv, "clv_results.csv"

    backtest = pd.read_csv(BACKTEST_PREDICTIONS)
    labeled_backtest = backtest["actual_result"].notna().sum() if "actual_result" in backtest.columns else 0
    if labeled_backtest > labeled_clv:
        print(
            f"Calibration rows in clv_results.csv ({labeled_clv}) are below minimum "
            f"({MIN_CALIBRATION_ROWS}). Using backtest_predictions.csv for fitting."
        )
        return backtest, "backtest_predictions.csv"
    return clv, "clv_results.csv"


def _assign_odds_bucket(odds: pd.Series) -> pd.Series:
    out = pd.Series(index=odds.index, dtype="object")
    out.loc[odds < 1.80] = "favorite"
    out.loc[(odds >= 1.80) & (odds <= 2.20)] = "balanced"
    out.loc[odds > 2.20] = "underdog"
    return out


def _roi_summary(df: pd.DataFrame, threshold: float, bucket: str | None = None) -> dict[str, float | int | str]:
    working = df.loc[
        df["calibrated_clv"].ge(threshold)
        & df["actual_result"].notna()
        & df["odds_a"].notna()
    ].copy()
    if bucket is not None:
        working = working.loc[working["odds_bucket"].eq(bucket)]

    bet_count = int(len(working))
    if bet_count == 0:
        result: dict[str, float | int | str] = {
            "ev_threshold": threshold,
            "bet_count": 0,
            "hit_rate": np.nan,
            "avg_odds": np.nan,
            "mean_calibrated_clv": np.nan,
            "roi": np.nan,
            "sample_flag": "LOW_SAMPLE",
        }
        if bucket is not None:
            result["odds_bucket"] = bucket
        return result

    wins = int(working["actual_result"].sum())
    hit_rate = wins / bet_count
    profits = np.where(working["actual_result"].eq(1), working["odds_a"] - 1.0, -1.0)
    roi = float(np.sum(profits) / bet_count)

    result = {
        "ev_threshold": threshold,
        "bet_count": bet_count,
        "hit_rate": hit_rate,
        "avg_odds": float(working["odds_a"].mean()),
        "mean_calibrated_clv": float(working["calibrated_clv"].mean()),
        "roi": roi,
        "sample_flag": "LOW_SAMPLE" if bet_count < 20 else "",
    }
    if bucket is not None:
        result["odds_bucket"] = bucket
    return result


def _reliability_table(df: pd.DataFrame, prob_col: str, label: str) -> pd.DataFrame:
    bins = np.linspace(0.0, 1.0, 11)
    working = df.loc[df[prob_col].notna() & df["actual_result"].notna(), [prob_col, "actual_result"]].copy()
    working["bin"] = pd.cut(working[prob_col], bins=bins, include_lowest=True)
    grouped = (
        working.groupby("bin", observed=False)
        .agg(
            n=("actual_result", "size"),
            mean_pred=(prob_col, "mean"),
            actual_rate=("actual_result", "mean"),
        )
        .reset_index()
    )
    grouped["table"] = label
    return grouped[["table", "bin", "n", "mean_pred", "actual_rate"]]


def _reliability_table_by_year(df: pd.DataFrame, prob_col: str, label: str) -> pd.DataFrame:
    rows = []
    for year in sorted(df["year"].dropna().astype(int).unique()):
        table = _reliability_table(df.loc[df["year"].eq(year)].copy(), prob_col, label)
        table.insert(0, "year", int(year))
        rows.append(table)
    if not rows:
        return pd.DataFrame(columns=["year", "table", "bin", "n", "mean_pred", "actual_rate"])
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    start = time.perf_counter()

    model_path = _find_model_path()
    print(f"Model file selected: {model_path}")

    df, source_name = _load_calibration_source()
    print(f"Calibration data source: {source_name}")
    print()
    if "year" not in df.columns:
        df["year"] = pd.to_datetime(df["date"], errors="coerce").dt.year.astype("Int64")

    required_columns = ["predicted_prob_a", "actual_result", "odds_a", "surface", "market_prob_a"]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    print("STEP 0 — INSPECT")
    print(f"Shape: {df.shape}")
    null_counts = df[["predicted_prob_a", "actual_result", "odds_a"]].isna().sum()
    print("Null counts:")
    print(null_counts.to_string())
    print()

    bins = np.linspace(0.0, 1.0, 11)
    pred_bin_counts = pd.cut(df["predicted_prob_a"], bins=bins, include_lowest=True).value_counts(sort=False, dropna=False)
    print("predicted_prob_a distribution (0.1 bins):")
    print(pred_bin_counts.to_string())
    print()

    underdog_count = int((df["odds_a"] > 2.20).sum())
    print(f"Underdog bucket count (odds_a > 2.20): {underdog_count}")
    print()

    if df["actual_result"].isna().all():
        print("actual_result is all null. Stopping.")
        elapsed = time.perf_counter() - start
        print(f"Total runtime: {elapsed:.2f} seconds")
        return

    cal_df = df.loc[df["actual_result"].notna() & df["predicted_prob_a"].notna()].copy()
    if len(cal_df) < MIN_CALIBRATION_ROWS:
        raise ValueError(
            f"Not enough labeled rows with predicted_prob_a for calibration: {len(cal_df)} "
            f"(minimum {MIN_CALIBRATION_ROWS})"
        )

    print("STEP 1 — PREPARE CALIBRATION DATA")
    X_cal = cal_df["predicted_prob_a"].to_numpy(dtype=float).reshape(-1, 1)
    y_cal = cal_df["actual_result"].to_numpy(dtype=float)
    print(f"Calibration sample size: {len(cal_df)}")
    print()

    print("STEP 2 — FIT ISOTONIC CALIBRATION")
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(X_cal.ravel(), y_cal)
    calibrated_probs_cal = ir.predict(X_cal.ravel())

    mean_raw = float(np.mean(X_cal))
    mean_calibrated = float(np.mean(calibrated_probs_cal))
    mean_actual = float(np.mean(y_cal))
    print(f"Mean raw predicted_prob_a: {_format_value(mean_raw, 4)}")
    print(f"Mean calibrated_prob_a: {_format_value(mean_calibrated, 4)}")
    print(f"Mean actual_result: {_format_value(mean_actual, 4)}")
    print()

    df = df.copy()
    df["calibrated_prob_a"] = np.nan
    pred_notna = df["predicted_prob_a"].notna()
    df.loc[pred_notna, "calibrated_prob_a"] = ir.predict(df.loc[pred_notna, "predicted_prob_a"].to_numpy(dtype=float))

    print("STEP 3 — EVALUATE CALIBRATION QUALITY")
    eval_df = df.loc[
        df["actual_result"].notna()
        & df["odds_a"].notna()
        & df["predicted_prob_a"].notna()
        & df["calibrated_prob_a"].notna()
    ].copy()
    if "year" not in eval_df.columns:
        eval_df["year"] = pd.to_datetime(eval_df["date"], errors="coerce").dt.year.astype("Int64")
    eval_df["odds_bucket"] = _assign_odds_bucket(eval_df["odds_a"])

    bucket_rows = []
    for year in sorted(eval_df["year"].dropna().astype(int).unique()):
        year_eval_df = eval_df.loc[eval_df["year"].eq(year)].copy()
        for bucket in ["favorite", "balanced", "underdog"]:
            bucket_df = year_eval_df.loc[year_eval_df["odds_bucket"].eq(bucket)]
            n = int(len(bucket_df))
            mean_raw_prob = float(bucket_df["predicted_prob_a"].mean()) if n else np.nan
            mean_cal_prob = float(bucket_df["calibrated_prob_a"].mean()) if n else np.nan
            mean_actual_bucket = float(bucket_df["actual_result"].mean()) if n else np.nan
            raw_error = abs(mean_raw_prob - mean_actual_bucket) if n else np.nan
            cal_error = abs(mean_cal_prob - mean_actual_bucket) if n else np.nan
            bucket_rows.append(
                {
                    "year": int(year),
                    "bucket": bucket,
                    "n": n,
                    "mean_raw_prob": mean_raw_prob,
                    "mean_calibrated_prob": mean_cal_prob,
                    "mean_actual_result": mean_actual_bucket,
                    "raw_error": raw_error,
                    "calibrated_error": cal_error,
                }
            )

    bucket_summary = pd.DataFrame(bucket_rows)
    _print_table(
        "Bucket calibration diagnostics:",
        bucket_summary,
        {
            "mean_raw_prob": 4,
            "mean_calibrated_prob": 4,
            "mean_actual_result": 4,
            "raw_error": 4,
            "calibrated_error": 4,
        },
    )

    underdog_rows = bucket_summary.loc[bucket_summary["bucket"].eq("underdog")]
    for _, underdog_row in underdog_rows.iterrows():
        if pd.notna(underdog_row["calibrated_error"]) and pd.notna(underdog_row["raw_error"]) and float(underdog_row["calibrated_error"]) > float(underdog_row["raw_error"]):
            print(
                f"WARNING: {int(underdog_row['year'])} underdog calibrated_error is worse than raw_error. "
                "Global isotonic fit can prioritize other probability regions when the underdog slice is sparse/noisy."
            )
            print()

    raw_reliability = _reliability_table_by_year(df, "predicted_prob_a", "raw")
    calibrated_reliability = _reliability_table_by_year(df, "calibrated_prob_a", "calibrated")
    reliability = pd.concat([raw_reliability, calibrated_reliability], ignore_index=True)
    _print_table(
        "Reliability table by year (10 equal-width bins):",
        reliability,
        {"mean_pred": 4, "actual_rate": 4},
    )

    print("STEP 4 — SAVE CALIBRATED PREDICTIONS")
    df["calibrated_clv"] = df["calibrated_prob_a"] - df["market_prob_a"]
    df.to_csv(OUTPUT_CALIBRATED, index=False)
    reliability.to_csv(OUTPUT_RELIABILITY_BY_YEAR, index=False)
    bucket_summary.to_csv(OUTPUT_BUCKET_BY_YEAR, index=False)
    print(f"Saved: {OUTPUT_CALIBRATED}")
    print(f"Saved: {OUTPUT_RELIABILITY_BY_YEAR}")
    print(f"Saved: {OUTPUT_BUCKET_BY_YEAR}")
    print()

    print("STEP 5 — QUICK ROI PREVIEW ON CALIBRATED CLV")
    roi_df = df.copy()
    roi_df["odds_bucket"] = _assign_odds_bucket(roi_df["odds_a"])

    overall_rows = []
    bucket_rows = []
    for threshold in EV_THRESHOLDS:
        overall_rows.append(_roi_summary(roi_df, threshold))
        for bucket in ["favorite", "balanced", "underdog"]:
            bucket_rows.append(_roi_summary(roi_df, threshold, bucket=bucket))

    overall = pd.DataFrame(overall_rows)
    by_bucket = pd.DataFrame(bucket_rows)

    _print_table(
        "ROI SUMMARY (overall):",
        overall[
            ["ev_threshold", "bet_count", "hit_rate", "avg_odds", "mean_calibrated_clv", "roi", "sample_flag"]
        ],
        {
            "ev_threshold": 2,
            "hit_rate": 3,
            "avg_odds": 3,
            "mean_calibrated_clv": 4,
            "roi": 4,
        },
    )
    _print_table(
        "ROI BY ODDS BUCKET:",
        by_bucket[
            ["ev_threshold", "odds_bucket", "bet_count", "hit_rate", "avg_odds", "mean_calibrated_clv", "roi", "sample_flag"]
        ],
        {
            "ev_threshold": 2,
            "hit_rate": 3,
            "avg_odds": 3,
            "mean_calibrated_clv": 4,
            "roi": 4,
        },
    )

    elapsed = time.perf_counter() - start
    print(f"Total runtime: {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()
