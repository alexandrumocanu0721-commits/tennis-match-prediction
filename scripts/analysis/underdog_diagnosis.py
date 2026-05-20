#!/usr/bin/env python3
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


START_TIME = time.time()


def fmt(x):
    if pd.isna(x):
        return "nan"
    return f"{float(x):.4f}"


def safe_spearman(x: pd.Series, y: pd.Series) -> float:
    pair = pd.concat([x, y], axis=1).dropna()
    if pair.shape[0] < 3:
        return np.nan
    if pair.iloc[:, 0].nunique(dropna=True) < 2 or pair.iloc[:, 1].nunique(dropna=True) < 2:
        return np.nan
    return pair.iloc[:, 0].corr(pair.iloc[:, 1], method="spearman")


def print_table(headers, rows):
    print(" | ".join(headers))
    for r in rows:
        print(" | ".join(str(v) for v in r))


def choose_model_path(models_dir: Path) -> Path:
    preferred = models_dir / "challenger" / "challenger_xgboost_model.pkl"
    if preferred.exists():
        return preferred
    pkl_files = sorted(models_dir.rglob("*.pkl"))
    if not pkl_files:
        raise FileNotFoundError(f"No .pkl model found under {models_dir}")
    return pkl_files[0]


def find_join_key(df_features: pd.DataFrame, df_clv: pd.DataFrame):
    candidates = [
        ["match_id"],
        ["prediction_row_id"],
        ["player_a_name", "tourney_date", "round"],
    ]

    print("\nSTEP 0 — INSPECT")
    print(f"features.csv shape: {df_features.shape}")
    print("features.csv columns:")
    print(df_features.columns.tolist())
    print(f"clv_results_calibrated.csv shape: {df_clv.shape}")

    null_cols = ["actual_result", "odds_a", "calibrated_prob_a"]
    print("Null counts in clv_results_calibrated.csv:")
    for c in null_cols:
        if c in df_clv.columns:
            print(f"- {c}: {int(df_clv[c].isna().sum())}")
        else:
            print(f"- {c}: MISSING COLUMN")

    print("\nJoin key checks:")
    viable = []
    for cols in candidates:
        missing_features = [c for c in cols if c not in df_features.columns]
        missing_clv = [c for c in cols if c not in df_clv.columns]
        if missing_features or missing_clv:
            print(
                f"- {cols}: unavailable (missing in features={missing_features}, missing in clv={missing_clv})"
            )
            continue

        left_unique = df_features[cols].drop_duplicates()
        right_unique = df_clv[cols].drop_duplicates()
        overlap_unique = left_unique.merge(right_unique, on=cols, how="inner").shape[0]

        joined = df_features.merge(df_clv, on=cols, how="inner", suffixes=("", "_clv"))
        joined_rows = joined.shape[0]
        print(
            f"- {cols}: available, overlapping unique keys={overlap_unique}, joined rows={joined_rows}"
        )

        if joined_rows > 0:
            viable.append((cols, joined_rows))

    if not viable:
        return None

    viable.sort(key=lambda x: x[1], reverse=True)
    chosen = viable[0][0]
    print(f"Chosen join key: {chosen}")
    return chosen


def main():
    base_dir = Path(__file__).resolve().parents[2]
    features_path = base_dir / "data" / "processed" / "challenger" / "features.csv"
    clv_path = base_dir / "data" / "processed" / "challenger" / "clv_results_calibrated.csv"
    model_path = choose_model_path(base_dir / "models")

    if not features_path.exists() or not clv_path.exists():
        raise FileNotFoundError("Missing required input CSV files.")

    df_features = pd.read_csv(features_path)
    df_clv = pd.read_csv(clv_path)

    join_cols = find_join_key(df_features, df_clv)
    if join_cols is None:
        print("STOP: No join key could be found with the requested key strategies.")
        runtime = time.time() - START_TIME
        print(f"\nTotal runtime: {runtime:.4f} seconds")
        return

    print("\nSTEP 1 — MERGE FEATURES WITH CLV RESULTS")
    before_features = df_features.shape[0]
    df_merged = df_features.merge(df_clv, on=join_cols, how="inner", suffixes=("", "_clv"))
    after_join = df_merged.shape[0]
    rows_lost = before_features - after_join
    loss_pct = (rows_lost / before_features * 100.0) if before_features else 0.0

    print(f"Rows before join (features): {before_features}")
    print(f"Rows after join: {after_join}")
    print(f"Rows lost in join: {rows_lost}")
    print(f"Join loss percentage: {loss_pct:.4f}%")
    if loss_pct > 20.0:
        print(
            "WARNING: More than 20% of rows were lost in the join. "
            "Likely cause: missing CLV matches or key mismatch between training features and CLV rows."
        )

    for col in ["odds_a", "actual_result", "calibrated_prob_a"]:
        if col not in df_merged.columns:
            raise KeyError(f"Required column missing after merge: {col}")
        df_merged[col] = pd.to_numeric(df_merged[col], errors="coerce")

    df_merged["odds_bucket"] = np.where(
        df_merged["odds_a"] < 1.80,
        "favorite",
        np.where(df_merged["odds_a"] <= 2.20, "balanced", "underdog"),
    )

    bucket_counts = df_merged["odds_bucket"].value_counts(dropna=False)
    print("Odds bucket counts:")
    print(f"- favorite: {int(bucket_counts.get('favorite', 0))}")
    print(f"- balanced: {int(bucket_counts.get('balanced', 0))}")
    print(f"- underdog: {int(bucket_counts.get('underdog', 0))}")

    low_sample = int(bucket_counts.get("underdog", 0)) < 200
    if low_sample:
        print("WARNING: Fewer than 200 underdog rows after join. All diagnostic tables are LOW_SAMPLE.")

    print("\nSTEP 2 — GLOBAL FEATURE IMPORTANCE")
    model = joblib.load(model_path)
    booster = model.get_booster()

    gain_imp = booster.get_score(importance_type="gain")
    weight_imp = booster.get_score(importance_type="weight")
    cover_imp = booster.get_score(importance_type="cover")

    model_features = list(getattr(model, "feature_names_in_", []))
    if not model_features:
        model_features = list(booster.feature_names or [])

    all_importance_features = sorted(set(gain_imp) | set(weight_imp) | set(cover_imp) | set(model_features))

    records = []
    for feat in all_importance_features:
        records.append(
            {
                "feature": feat,
                "gain": float(gain_imp.get(feat, 0.0)),
                "weight": float(weight_imp.get(feat, 0.0)),
                "cover": float(cover_imp.get(feat, 0.0)),
            }
        )

    imp_df = pd.DataFrame(records).sort_values("gain", ascending=False).reset_index(drop=True)

    headers = ["rank", "feature_name", "importance_gain", "importance_weight", "importance_cover"]
    rows = []
    for i, row in imp_df.head(20).iterrows():
        rows.append(
            [
                i + 1,
                row["feature"],
                fmt(row["gain"]),
                fmt(row["weight"]),
                fmt(row["cover"]),
            ]
        )
    print_table(headers, rows)

    blocked_prefixes = ("shuffled_", "random_market_centered_")

    # Candidate feature columns from features.csv only, numeric only, excluding target-like columns.
    feature_candidates = []
    for c in df_features.columns:
        if c in {"result"}:
            continue
        if c.startswith(blocked_prefixes):
            continue
        if c in df_merged.columns and pd.api.types.is_numeric_dtype(df_merged[c]):
            feature_candidates.append(c)

    # Prefer model feature list when available; fallback to numeric candidates.
    usable_features = [
        c
        for c in model_features
        if c in df_merged.columns and c not in {"result"} and not c.startswith(blocked_prefixes)
    ]
    if not usable_features:
        usable_features = feature_candidates

    print("\nSTEP 3 — FEATURE IMPORTANCE ON UNDERDOG SUBSET")
    underdog = df_merged[df_merged["odds_bucket"] == "underdog"].copy()

    diagnostics = []
    for f in usable_features:
        if not pd.api.types.is_numeric_dtype(underdog[f]):
            continue
        corr_actual = safe_spearman(underdog[f], underdog["actual_result"])
        corr_calib = safe_spearman(underdog[f], underdog["calibrated_prob_a"])
        gap = abs(corr_actual) - abs(corr_calib) if not (pd.isna(corr_actual) or pd.isna(corr_calib)) else np.nan
        diagnostics.append(
            {
                "feature": f,
                "corr_actual": corr_actual,
                "corr_calib": corr_calib,
                "gap": gap,
            }
        )

    diag_df = pd.DataFrame(diagnostics).dropna(subset=["corr_actual", "corr_calib", "gap"]) 

    prefix = "[LOW_SAMPLE] " if low_sample else ""

    # TABLE A: top by abs(corr_actual)
    table_a = diag_df.assign(abs_corr_actual=diag_df["corr_actual"].abs()).sort_values(
        "abs_corr_actual", ascending=False
    )
    print(f"{prefix}TABLE A — features most correlated with actual_result in underdog matches")
    headers = ["rank", "feature", "corr_with_actual", "corr_with_calibrated_prob", "gap"]
    rows = []
    for i, row in table_a.head(20).reset_index(drop=True).iterrows():
        rows.append(
            [
                i + 1,
                row["feature"],
                fmt(row["corr_actual"]),
                fmt(row["corr_calib"]),
                fmt(row["gap"]),
            ]
        )
    print_table(headers, rows)

    # TABLE B: sorted by gap descending
    table_b = diag_df.sort_values("gap", ascending=False)
    print(f"\n{prefix}TABLE B — features where the model diverges most from reality in underdogs")
    rows = []
    for i, row in table_b.head(20).reset_index(drop=True).iterrows():
        rows.append(
            [
                i + 1,
                row["feature"],
                fmt(row["corr_actual"]),
                fmt(row["corr_calib"]),
                fmt(row["gap"]),
            ]
        )
    print_table(headers, rows)

    print("\nSTEP 4 — DISTRIBUTION COMPARISON: UNDERDOGS VS FAVORITES")
    top10_gain_features = imp_df["feature"].head(10).tolist()
    favorite = df_merged[df_merged["odds_bucket"] == "favorite"].copy()

    headers = [
        "feature",
        "mean_favorites",
        "mean_underdogs",
        "mean_diff",
        "corr_actual_favorites",
        "corr_actual_underdogs",
    ]
    rows = []
    for f in top10_gain_features:
        if f not in df_merged.columns or not pd.api.types.is_numeric_dtype(df_merged[f]):
            rows.append([f, "nan", "nan", "nan", "nan", "nan"])
            continue

        mean_fav = favorite[f].mean()
        mean_dog = underdog[f].mean()
        mean_diff = mean_dog - mean_fav
        corr_fav = safe_spearman(favorite[f], favorite["actual_result"])
        corr_dog = safe_spearman(underdog[f], underdog["actual_result"])

        rows.append([f, fmt(mean_fav), fmt(mean_dog), fmt(mean_diff), fmt(corr_fav), fmt(corr_dog)])
    if low_sample:
        print("[LOW_SAMPLE]")
    print_table(headers, rows)

    print("\nSTEP 5 — PROBABILITY OVERSHOOT ANALYSIS")
    underdog = underdog.copy()
    underdog["overshoot"] = underdog["calibrated_prob_a"] - underdog["actual_result"]

    overshoot_records = []
    for f in usable_features:
        if f not in underdog.columns or not pd.api.types.is_numeric_dtype(underdog[f]):
            continue
        corr_over = safe_spearman(underdog[f], underdog["overshoot"])
        if pd.isna(corr_over):
            continue
        if corr_over > 0.10:
            interp = "model overshoots MORE when this feature is high"
        elif corr_over < -0.10:
            interp = "model overshoots LESS (or undershoots) when this feature is high"
        else:
            interp = "weak signal"

        overshoot_records.append({"feature": f, "corr": corr_over, "abs_corr": abs(corr_over), "interp": interp})

    over_df = pd.DataFrame(overshoot_records)
    if not over_df.empty:
        over_df = over_df.sort_values("abs_corr", ascending=False).reset_index(drop=True)

    if low_sample:
        print("[LOW_SAMPLE]")

    headers = ["rank", "feature", "corr_with_overshoot", "interpretation"]
    rows = []
    for i, row in over_df.head(15).iterrows():
        rows.append([i + 1, row["feature"], fmt(row["corr"]), row["interp"]])
    print_table(headers, rows)

    runtime = time.time() - START_TIME
    print(f"\nTotal runtime: {runtime:.4f} seconds")


if __name__ == "__main__":
    main()
