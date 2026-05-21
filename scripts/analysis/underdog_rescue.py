from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]

ODDS_BUCKETS = ["favorite", "balanced", "underdog"]
UNDERDOG_ODDS_BINS = [2.20, 3.00, 4.00, 6.00, 10.00, float("inf")]
UNDERDOG_PROB_BINS = np.arange(0.0, 1.0001, 0.05)
EV_THRESHOLD = 0.0

MIN_CALIBRATION_ROWS = 50
MIN_MODEL_ROWS = 80
MIN_TUNE_BETS = 20
MIN_FINAL_BETS = 30
MAX_CALIBRATION_ERROR = 0.08
MAX_FINAL_CLUSTER_SHARE = 0.60

RANK_FEATURES = ["rank_diff", "points_diff", "rank_momentum_diff"]
ELO_FEATURES = ["elo_diff", "surface_elo_diff"]
SERVE_RETURN_FEATURES = [
    "serve_rating_diff",
    "return_rating_diff",
    "bp_save_rate_diff",
    "bp_convert_rate_diff",
    "serve_rating_surface_diff",
    "return_rating_surface_diff",
    "bp_save_rate_surface_diff",
    "bp_convert_rate_surface_diff",
]
RECENT_FORM_FEATURES = [
    "recent_form_diff",
    "recent_surface_form_diff",
    "win_pct_diff",
    "matches_played_diff",
    "deciding_set_win_rate_diff",
]
H2H_FEATURES = ["h2h_win_rate_diff", "h2h_surface_win_rate_diff"]
FEATURE_FAMILIES = {
    "rank": RANK_FEATURES,
    "elo": ELO_FEATURES,
    "serve_return": SERVE_RETURN_FEATURES,
    "recent_form": RECENT_FORM_FEATURES,
    "h2h": H2H_FEATURES,
}


@dataclass(frozen=True)
class TourSpec:
    name: str
    clv_path: Path
    features_path: Path
    output_dir: Path
    actual_col: str
    market_prob_col: str
    odds_col: str | None
    raw_prob_col: str
    artifact_prob_col: str | None
    artifact_clv_col: str | None


@dataclass(frozen=True)
class CandidateResult:
    name: str
    description: str
    selectable: bool
    status: str
    adjusted_prob: pd.Series
    removed_family: str = ""
    feature_count: int = 0


def tour_specs(root: Path = PROJECT_ROOT) -> dict[str, TourSpec]:
    return {
        "atp": TourSpec(
            name="atp",
            clv_path=root / "data" / "processed" / "atp" / "clv_results.csv",
            features_path=root / "data" / "processed" / "atp" / "features.csv",
            output_dir=root / "data" / "processed" / "atp" / "underdog_rescue",
            actual_col="actual_winner",
            market_prob_col="betfair_true_prob_a",
            odds_col=None,
            raw_prob_col="predicted_prob_a",
            artifact_prob_col=None,
            artifact_clv_col="clv_betfair",
        ),
        "challenger": TourSpec(
            name="challenger",
            clv_path=root
            / "data"
            / "processed"
            / "challenger"
            / "clv_results_calibrated.csv",
            features_path=root / "data" / "processed" / "challenger" / "features.csv",
            output_dir=root
            / "data"
            / "processed"
            / "challenger"
            / "underdog_rescue",
            actual_col="actual_result",
            market_prob_col="market_prob_a",
            odds_col="odds_a",
            raw_prob_col="predicted_prob_a",
            artifact_prob_col="calibrated_prob_a",
            artifact_clv_col="calibrated_clv",
        ),
    }


def clip_prob(values: pd.Series | np.ndarray) -> pd.Series:
    series = pd.Series(values)
    return series.clip(1e-6, 1.0 - 1e-6)


def assign_odds_bucket(odds: pd.Series) -> pd.Series:
    numeric_odds = pd.to_numeric(odds, errors="coerce")
    out = pd.Series("unknown", index=numeric_odds.index, dtype="object")
    out.loc[numeric_odds < 1.80] = "favorite"
    out.loc[numeric_odds.ge(1.80) & numeric_odds.le(2.20)] = "balanced"
    out.loc[numeric_odds > 2.20] = "underdog"
    return out


def safe_log_loss(y_true: pd.Series, prob: pd.Series) -> float:
    pair = pd.concat([y_true, prob], axis=1).dropna()
    if pair.empty:
        return float("nan")
    y = pair.iloc[:, 0].astype(float)
    p = clip_prob(pair.iloc[:, 1]).astype(float)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def flat_stake_profit(df: pd.DataFrame) -> pd.Series:
    wins = pd.to_numeric(df["actual_result"], errors="coerce").eq(1.0)
    odds = pd.to_numeric(df["odds_a"], errors="coerce")
    return pd.Series(np.where(wins, odds - 1.0, -1.0), index=df.index)


def summarize_rows(
    df: pd.DataFrame,
    prob_col: str,
    edge_threshold: float | None = None,
) -> dict[str, float | int]:
    working = df.copy()
    if edge_threshold is not None:
        working = working.loc[
            pd.to_numeric(working[prob_col], errors="coerce")
            .sub(pd.to_numeric(working["market_prob_a"], errors="coerce"))
            .ge(edge_threshold)
        ].copy()

    working = working.loc[
        working["actual_result"].notna()
        & working["odds_a"].notna()
        & working["market_prob_a"].notna()
        & working[prob_col].notna()
    ].copy()

    bet_count = int(len(working))
    if bet_count == 0:
        return {
            "bet_count": 0,
            "hit_rate": np.nan,
            "avg_odds": np.nan,
            "actual_win_rate": np.nan,
            "mean_model_prob": np.nan,
            "mean_market_prob": np.nan,
            "mean_clv": np.nan,
            "calibration_error": np.nan,
            "brier": np.nan,
            "log_loss": np.nan,
            "roi": np.nan,
        }

    actual = pd.to_numeric(working["actual_result"], errors="coerce")
    model_prob = pd.to_numeric(working[prob_col], errors="coerce")
    market_prob = pd.to_numeric(working["market_prob_a"], errors="coerce")
    profits = flat_stake_profit(working)

    actual_win_rate = float(actual.mean())
    mean_model_prob = float(model_prob.mean())
    return {
        "bet_count": bet_count,
        "hit_rate": actual_win_rate,
        "avg_odds": float(pd.to_numeric(working["odds_a"], errors="coerce").mean()),
        "actual_win_rate": actual_win_rate,
        "mean_model_prob": mean_model_prob,
        "mean_market_prob": float(market_prob.mean()),
        "mean_clv": float((model_prob - market_prob).mean()),
        "calibration_error": abs(mean_model_prob - actual_win_rate),
        "brier": float(np.mean((model_prob - actual) ** 2)),
        "log_loss": safe_log_loss(actual, model_prob),
        "roi": float(profits.mean()),
    }


def output_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    return [column for column in columns if column in df.columns]


def load_tour_dataset(spec: TourSpec) -> pd.DataFrame:
    if not spec.clv_path.exists():
        raise FileNotFoundError(f"Missing CLV input: {spec.clv_path}")

    raw = pd.read_csv(spec.clv_path)
    df = raw.copy()
    df["tour"] = spec.name
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()

    if spec.name == "atp":
        df["player_a_name"] = df["player_a"].astype(str)
        df["player_b_name"] = df["player_b"].astype(str)
    else:
        df["player_a_name"] = df["player_a_name"].astype(str)
        df["player_b_name"] = df["player_b_name"].astype(str)

    df["actual_result"] = pd.to_numeric(df[spec.actual_col], errors="coerce")
    df["market_prob_a"] = pd.to_numeric(df[spec.market_prob_col], errors="coerce")
    df["odds_a"] = (
        pd.to_numeric(df[spec.odds_col], errors="coerce")
        if spec.odds_col is not None
        else np.where(df["market_prob_a"] > 0, 1.0 / df["market_prob_a"], np.nan)
    )
    df["raw_prob_a"] = pd.to_numeric(df[spec.raw_prob_col], errors="coerce")

    if spec.artifact_prob_col and spec.artifact_prob_col in df.columns:
        df["artifact_prob_a"] = pd.to_numeric(df[spec.artifact_prob_col], errors="coerce")
    else:
        df["artifact_prob_a"] = df["raw_prob_a"]

    df["artifact_clv"] = (
        pd.to_numeric(df[spec.artifact_clv_col], errors="coerce")
        if spec.artifact_clv_col and spec.artifact_clv_col in df.columns
        else df["artifact_prob_a"] - df["market_prob_a"]
    )
    df["raw_clv"] = df["raw_prob_a"] - df["market_prob_a"]
    df["odds_bucket"] = assign_odds_bucket(df["odds_a"])
    df["month"] = df["date"].dt.to_period("M").astype(str)
    df["date_key"] = df["date"].dt.strftime("%Y-%m-%d")
    return df


def merge_features(df: pd.DataFrame, spec: TourSpec) -> tuple[pd.DataFrame, list[str], str]:
    if not spec.features_path.exists():
        return df.copy(), [], f"features file missing: {spec.features_path}"

    feature_cols = sorted({feature for family in FEATURE_FAMILIES.values() for feature in family})
    base_cols = ["date", "player_a_name", "player_b_name", "surface", *feature_cols]
    optional_cols = ["tourney_id", "round"]

    header = pd.read_csv(spec.features_path, nrows=0)
    usecols = output_columns(header, [*base_cols, *optional_cols])
    features = pd.read_csv(spec.features_path, usecols=usecols)
    features["date"] = pd.to_datetime(features["date"], errors="coerce").dt.normalize()
    features["date_key"] = features["date"].dt.strftime("%Y-%m-%d")
    features["player_a_name"] = features["player_a_name"].astype(str)
    features["player_b_name"] = features["player_b_name"].astype(str)

    merge_cols = ["date_key", "player_a_name", "player_b_name", "surface"]
    if spec.name == "challenger":
        for col in ["tourney_id", "round"]:
            if col in df.columns and col in features.columns:
                df[col] = df[col].astype(str)
                features[col] = features[col].astype(str)
                merge_cols.append(col)

    available_features = [column for column in feature_cols if column in features.columns]
    features = features.drop_duplicates(subset=merge_cols, keep="first")
    merged = df.merge(
        features[merge_cols + available_features],
        on=merge_cols,
        how="left",
        suffixes=("", "_feature"),
    )
    matched_feature_rows = int(merged[available_features].notna().any(axis=1).sum()) if available_features else 0
    note = (
        f"merged {matched_feature_rows}/{len(df)} CLV rows with feature values "
        f"using keys {merge_cols}"
    )
    return merged, available_features, note


def assign_walk_forward_periods(
    df: pd.DataFrame,
    train_fraction: float = 0.60,
    tune_fraction: float = 0.20,
) -> pd.DataFrame:
    if df.empty:
        out = df.copy()
        out["wf_period"] = pd.Series(dtype="object")
        return out

    dates = sorted(pd.to_datetime(df["date"], errors="coerce").dropna().unique())
    if len(dates) < 3:
        raise ValueError("Need at least three unique dates for train/tune/final split.")

    train_end_idx = max(1, int(len(dates) * train_fraction))
    tune_end_idx = max(train_end_idx + 1, int(len(dates) * (train_fraction + tune_fraction)))
    tune_end_idx = min(tune_end_idx, len(dates) - 1)

    train_end_date = pd.Timestamp(dates[train_end_idx])
    final_start_date = pd.Timestamp(dates[tune_end_idx])

    out = df.copy()
    out["wf_period"] = "final"
    out.loc[out["date"] < train_end_date, "wf_period"] = "train"
    out.loc[
        out["date"].ge(train_end_date) & out["date"].lt(final_start_date),
        "wf_period",
    ] = "tune"
    return out


def build_regime_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for prob_col, label in [
        ("raw_prob_a", "raw_model"),
        ("artifact_prob_a", "artifact_model"),
    ]:
        for period in ["all", "train", "tune", "final"]:
            period_df = df if period == "all" else df.loc[df["wf_period"].eq(period)]
            for bucket in ODDS_BUCKETS:
                row = {
                    "prob_source": label,
                    "wf_period": period,
                    "odds_bucket": bucket,
                }
                row.update(summarize_rows(period_df.loc[period_df["odds_bucket"].eq(bucket)], prob_col))
                rows.append(row)
    return pd.DataFrame(rows)


def build_underdog_calibration(df: pd.DataFrame, prob_col: str, prob_source: str) -> pd.DataFrame:
    rows: list[dict] = []
    dogs = df.loc[df["odds_bucket"].eq("underdog")].copy()
    for period in ["all", "train", "tune", "final"]:
        period_df = dogs if period == "all" else dogs.loc[dogs["wf_period"].eq(period)]
        working = period_df.loc[
            period_df[prob_col].notna() & period_df["actual_result"].notna()
        ].copy()
        if working.empty:
            continue
        working["prob_bin"] = pd.cut(
            pd.to_numeric(working[prob_col], errors="coerce"),
            bins=UNDERDOG_PROB_BINS,
            include_lowest=True,
            right=False,
        )
        grouped = (
            working.groupby("prob_bin", observed=False)
            .agg(
                bet_count=("actual_result", "size"),
                actual_win_rate=("actual_result", "mean"),
                mean_model_prob=(prob_col, "mean"),
                mean_market_prob=("market_prob_a", "mean"),
                avg_odds=("odds_a", "mean"),
            )
            .reset_index()
        )
        grouped = grouped.loc[grouped["bet_count"].gt(0)].copy()
        grouped["prob_source"] = prob_source
        grouped["wf_period"] = period
        grouped["prob_bin"] = grouped["prob_bin"].astype(str)
        grouped["calibration_error"] = (
            grouped["mean_model_prob"] - grouped["actual_win_rate"]
        ).abs()
        rows.extend(grouped.to_dict("records"))
    return pd.DataFrame(rows)


def build_underdog_odds_bins(df: pd.DataFrame) -> pd.DataFrame:
    dogs = df.loc[df["odds_bucket"].eq("underdog")].copy()
    rows: list[pd.DataFrame] = []
    for prob_col, source in [("raw_prob_a", "raw_model"), ("artifact_prob_a", "artifact_model")]:
        for period in ["all", "train", "tune", "final"]:
            period_df = dogs if period == "all" else dogs.loc[dogs["wf_period"].eq(period)]
            working = period_df.loc[period_df["odds_a"].notna()].copy()
            if working.empty:
                continue
            working["odds_bin"] = pd.cut(
                working["odds_a"],
                bins=UNDERDOG_ODDS_BINS,
                include_lowest=True,
                right=False,
            )
            grouped = (
                working.groupby("odds_bin", observed=False)
                .agg(
                    bet_count=("actual_result", "size"),
                    actual_win_rate=("actual_result", "mean"),
                    mean_model_prob=(prob_col, "mean"),
                    mean_market_prob=("market_prob_a", "mean"),
                    avg_odds=("odds_a", "mean"),
                )
                .reset_index()
            )
            grouped = grouped.loc[grouped["bet_count"].gt(0)].copy()
            grouped["prob_source"] = source
            grouped["wf_period"] = period
            grouped["odds_bin"] = grouped["odds_bin"].astype(str)
            rows.append(grouped)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def build_concentration_summary(df: pd.DataFrame, prob_col: str) -> pd.DataFrame:
    dogs = df.loc[
        df["wf_period"].eq("final")
        & df["odds_bucket"].eq("underdog")
        & pd.to_numeric(df[prob_col], errors="coerce").sub(df["market_prob_a"]).ge(EV_THRESHOLD)
    ].copy()
    rows: list[dict] = []
    for group_col in ["month", "surface", "tourney_name"]:
        if group_col not in dogs.columns or dogs.empty:
            continue
        counts = dogs[group_col].fillna("UNKNOWN").value_counts(dropna=False)
        for value, count in counts.items():
            rows.append(
                {
                    "candidate": prob_col,
                    "group": group_col,
                    "value": value,
                    "bet_count": int(count),
                    "share": float(count / len(dogs)),
                }
            )
    return pd.DataFrame(rows)


def fit_isotonic(train_df: pd.DataFrame, prob_col: str) -> Callable[[pd.Series], pd.Series] | None:
    working = train_df.loc[
        train_df[prob_col].notna() & train_df["actual_result"].notna()
    ].copy()
    if len(working) < MIN_CALIBRATION_ROWS or working["actual_result"].nunique() < 2:
        return None

    model = IsotonicRegression(out_of_bounds="clip")
    model.fit(
        pd.to_numeric(working[prob_col], errors="coerce").to_numpy(dtype=float),
        pd.to_numeric(working["actual_result"], errors="coerce").to_numpy(dtype=float),
    )

    def predict(values: pd.Series) -> pd.Series:
        out = pd.Series(np.nan, index=values.index, dtype=float)
        mask = values.notna()
        if mask.any():
            out.loc[mask] = model.predict(values.loc[mask].to_numpy(dtype=float))
        return out

    return predict


def bucket_isotonic_candidate(df: pd.DataFrame, train_df: pd.DataFrame) -> CandidateResult:
    adjusted = pd.Series(np.nan, index=df.index, dtype=float)
    fitted_buckets = []
    for bucket in ODDS_BUCKETS:
        calibrator = fit_isotonic(train_df.loc[train_df["odds_bucket"].eq(bucket)], "raw_prob_a")
        if calibrator is None:
            continue
        mask = df["odds_bucket"].eq(bucket)
        adjusted.loc[mask] = calibrator(df.loc[mask, "raw_prob_a"])
        fitted_buckets.append(bucket)

    if adjusted.notna().sum() == 0:
        return CandidateResult(
            name="bucket_isotonic",
            description="post-model isotonic calibration fit separately by odds bucket",
            selectable=True,
            status="SKIPPED_INSUFFICIENT_BUCKET_ROWS",
            adjusted_prob=df["raw_prob_a"],
        )

    adjusted = adjusted.fillna(df["raw_prob_a"])
    return CandidateResult(
        name="bucket_isotonic",
        description=f"post-model isotonic calibration fit for buckets: {','.join(fitted_buckets)}",
        selectable=True,
        status="OK",
        adjusted_prob=adjusted,
    )


def dog_isotonic_candidate(df: pd.DataFrame, train_df: pd.DataFrame) -> CandidateResult:
    calibrator = fit_isotonic(train_df.loc[train_df["odds_bucket"].eq("underdog")], "raw_prob_a")
    if calibrator is None:
        return CandidateResult(
            name="dog_isotonic",
            description="underdog-only isotonic calibration",
            selectable=True,
            status="SKIPPED_INSUFFICIENT_DOG_ROWS",
            adjusted_prob=df["raw_prob_a"],
        )
    adjusted = df["raw_prob_a"].copy()
    mask = df["odds_bucket"].eq("underdog")
    adjusted.loc[mask] = calibrator(df.loc[mask, "raw_prob_a"])
    return CandidateResult(
        name="dog_isotonic",
        description="underdog-only isotonic calibration fit on train period",
        selectable=True,
        status="OK",
        adjusted_prob=adjusted,
    )


def cap_candidate(df: pd.DataFrame, cap: float) -> CandidateResult:
    adjusted = df["raw_prob_a"].copy()
    mask = df["odds_bucket"].eq("underdog")
    adjusted.loc[mask] = adjusted.loc[mask].clip(upper=cap)
    return CandidateResult(
        name=f"dog_cap_{str(cap).replace('.', '')}",
        description=f"lower-cap underdog model probability at {cap:.2f}",
        selectable=True,
        status="OK",
        adjusted_prob=adjusted,
    )


def market_delta_shrink_candidate(df: pd.DataFrame, shrink: float) -> CandidateResult:
    adjusted = df["raw_prob_a"].copy()
    mask = df["odds_bucket"].eq("underdog")
    adjusted.loc[mask] = df.loc[mask, "market_prob_a"] + shrink * (
        df.loc[mask, "raw_prob_a"] - df.loc[mask, "market_prob_a"]
    )
    return CandidateResult(
        name=f"dog_market_delta_shrink_{int(shrink * 100)}",
        description=(
            "market-implied probability anchor: "
            f"market_prob + {shrink:.2f} * model_edge for underdogs"
        ),
        selectable=True,
        status="OK",
        adjusted_prob=adjusted,
    )


def logistic_candidate(
    df: pd.DataFrame,
    train_df: pd.DataFrame,
    feature_cols: list[str],
    name: str,
    description: str,
    removed_family: str = "",
) -> CandidateResult:
    base_features = ["raw_prob_a", "market_prob_a"]
    all_features = base_features + feature_cols
    train_dogs = train_df.loc[train_df["odds_bucket"].eq("underdog")].copy()
    train_dogs = train_dogs.dropna(subset=all_features + ["actual_result"])
    if len(train_dogs) < MIN_MODEL_ROWS or train_dogs["actual_result"].nunique() < 2:
        return CandidateResult(
            name=name,
            description=description,
            selectable=True,
            status="SKIPPED_INSUFFICIENT_DOG_ROWS",
            adjusted_prob=df["raw_prob_a"],
            removed_family=removed_family,
            feature_count=len(all_features),
        )

    model = Pipeline(
        steps=[
            ("scale", StandardScaler()),
            ("logit", LogisticRegression(max_iter=2000)),
        ]
    )
    model.fit(train_dogs[all_features], train_dogs["actual_result"].astype(int))

    adjusted = df["raw_prob_a"].copy()
    dog_mask = df["odds_bucket"].eq("underdog")
    pred_source = df.loc[dog_mask, all_features].copy()
    valid_mask = pred_source.notna().all(axis=1)
    if valid_mask.any():
        adjusted.loc[pred_source.loc[valid_mask].index] = model.predict_proba(
            pred_source.loc[valid_mask]
        )[:, 1]

    return CandidateResult(
        name=name,
        description=description,
        selectable=True,
        status="OK",
        adjusted_prob=adjusted,
        removed_family=removed_family,
        feature_count=len(all_features),
    )


def build_candidates(
    df: pd.DataFrame,
    train_df: pd.DataFrame,
    available_features: list[str],
) -> list[CandidateResult]:
    candidates = [
        CandidateResult(
            name="raw_model",
            description="uncalibrated model probability baseline",
            selectable=False,
            status="BASELINE",
            adjusted_prob=df["raw_prob_a"],
        ),
        CandidateResult(
            name="artifact_current",
            description=(
                "current persisted artifact probability; not selectable because it may "
                "include global calibration fitted outside the walk-forward train period"
            ),
            selectable=False,
            status="DIAGNOSTIC_ONLY",
            adjusted_prob=df["artifact_prob_a"],
        ),
        bucket_isotonic_candidate(df, train_df),
        dog_isotonic_candidate(df, train_df),
    ]

    for cap in [0.30, 0.35, 0.40]:
        candidates.append(cap_candidate(df, cap))

    for shrink in [0.25, 0.50, 0.75]:
        candidates.append(market_delta_shrink_candidate(df, shrink))

    usable_features = [feature for feature in available_features if feature in df.columns]
    if usable_features:
        candidates.append(
            logistic_candidate(
                df=df,
                train_df=train_df,
                feature_cols=usable_features,
                name="dog_logit_all",
                description="separate underdog logistic model using market, raw model, and all merged features",
            )
        )
        for family_name, family_features in FEATURE_FAMILIES.items():
            kept = [feature for feature in usable_features if feature not in set(family_features)]
            candidates.append(
                logistic_candidate(
                    df=df,
                    train_df=train_df,
                    feature_cols=kept,
                    name=f"dog_logit_no_{family_name}",
                    description=f"underdog logistic ablation removing {family_name} feature family",
                    removed_family=family_name,
                )
            )

    return candidates


def evaluate_candidate(
    df: pd.DataFrame,
    candidate: CandidateResult,
    edge_threshold: float = EV_THRESHOLD,
) -> list[dict]:
    prob_col = f"candidate_prob__{candidate.name}"
    working = df.copy()
    working[prob_col] = candidate.adjusted_prob
    rows = []
    for period in ["tune", "final"]:
        period_dogs = working.loc[
            working["wf_period"].eq(period) & working["odds_bucket"].eq("underdog")
        ].copy()
        row = {
            "candidate": candidate.name,
            "wf_period": period,
            "selectable": candidate.selectable,
            "status": candidate.status,
            "description": candidate.description,
            "removed_family": candidate.removed_family,
            "feature_count": candidate.feature_count,
            "edge_threshold": edge_threshold,
        }
        row.update(summarize_rows(period_dogs, prob_col, edge_threshold=edge_threshold))
        rows.append(row)
    return rows


def select_candidate_on_tune(experiments: pd.DataFrame) -> tuple[str | None, str]:
    tune = experiments.loc[
        experiments["wf_period"].eq("tune")
        & experiments["selectable"].eq(True)
        & experiments["status"].eq("OK")
        & experiments["bet_count"].ge(MIN_TUNE_BETS)
    ].copy()
    if tune.empty:
        return None, "NO_TUNE_CANDIDATE_WITH_MIN_BETS"

    tune["tune_pass"] = (
        tune["roi"].ge(0.0)
        & tune["mean_clv"].ge(0.0)
        & tune["calibration_error"].le(MAX_CALIBRATION_ERROR)
    )
    passing = tune.loc[tune["tune_pass"]].copy()
    if passing.empty:
        tune = tune.sort_values(
            ["roi", "calibration_error", "bet_count"],
            ascending=[False, True, False],
            kind="mergesort",
        )
        return str(tune.iloc[0]["candidate"]), "BEST_TUNE_ONLY_NO_PASS"

    passing = passing.sort_values(
        ["roi", "calibration_error", "bet_count"],
        ascending=[False, True, False],
        kind="mergesort",
    )
    return str(passing.iloc[0]["candidate"]), "TUNE_PASS"


def final_cluster_share(df: pd.DataFrame, prob_col: str) -> float:
    final_bets = df.loc[
        df["wf_period"].eq("final")
        & df["odds_bucket"].eq("underdog")
        & pd.to_numeric(df[prob_col], errors="coerce").sub(df["market_prob_a"]).ge(EV_THRESHOLD)
    ].copy()
    if final_bets.empty:
        return float("nan")

    max_share = 0.0
    for group_col in ["month", "surface", "tourney_name"]:
        if group_col not in final_bets.columns:
            continue
        share = float(final_bets[group_col].fillna("UNKNOWN").value_counts(normalize=True).max())
        max_share = max(max_share, share)
    return max_share


def build_recommendation(
    df: pd.DataFrame,
    candidates: list[CandidateResult],
    experiments: pd.DataFrame,
    selected_candidate: str | None,
    selection_status: str,
    feature_merge_note: str,
) -> str:
    lines = [
        "Underdog Rescue Recommendation",
        "==============================",
        "",
        f"feature merge: {feature_merge_note}",
        f"selection status: {selection_status}",
    ]

    if selected_candidate is None:
        lines.extend(
            [
                "recommendation: BLOCK_UNDERDOGS",
                "reason: no selectable rescue candidate produced enough tune-period bets.",
            ]
        )
        return "\n".join(lines) + "\n"

    selected = next(candidate for candidate in candidates if candidate.name == selected_candidate)
    prob_col = f"candidate_prob__{selected.name}"
    scored = df.copy()
    scored[prob_col] = selected.adjusted_prob
    cluster_share = final_cluster_share(scored, prob_col)

    final_row = experiments.loc[
        experiments["candidate"].eq(selected_candidate)
        & experiments["wf_period"].eq("final")
    ].iloc[0]

    final_pass = (
        int(final_row["bet_count"]) >= MIN_FINAL_BETS
        and float(final_row["roi"]) >= 0.0
        and float(final_row["mean_clv"]) >= 0.0
        and float(final_row["calibration_error"]) <= MAX_CALIBRATION_ERROR
        and (pd.isna(cluster_share) or cluster_share <= MAX_FINAL_CLUSTER_SHARE)
    )
    recommendation = "ALLOW_UNDERDOG_RESEARCH_ONLY" if final_pass else "BLOCK_UNDERDOGS"

    lines.extend(
        [
            f"selected candidate: {selected_candidate}",
            f"selected description: {selected.description}",
            f"final bet count: {int(final_row['bet_count'])}",
            f"final roi: {float(final_row['roi']):.4f}",
            f"final mean_clv: {float(final_row['mean_clv']):.4f}",
            f"final calibration_error: {float(final_row['calibration_error']):.4f}",
            f"final max month/surface/tournament share: {cluster_share:.4f}"
            if pd.notna(cluster_share)
            else "final max month/surface/tournament share: n/a",
            f"recommendation: {recommendation}",
        ]
    )

    if not final_pass:
        lines.append(
            "reason: selected tune-period rescue did not satisfy all untouched final-period pass criteria."
        )
    return "\n".join(lines) + "\n"


def write_outputs(
    spec: TourSpec,
    df: pd.DataFrame,
    available_features: list[str],
    feature_merge_note: str,
) -> None:
    spec.output_dir.mkdir(parents=True, exist_ok=True)
    df = assign_walk_forward_periods(df)

    train_df = df.loc[df["wf_period"].eq("train")].copy()
    candidates = build_candidates(df, train_df, available_features)

    scored_df = df.copy()
    experiment_rows: list[dict] = []
    feature_rows: list[dict] = []
    for candidate in candidates:
        scored_df[f"candidate_prob__{candidate.name}"] = candidate.adjusted_prob
        experiment_rows.extend(evaluate_candidate(df, candidate))
        if candidate.name.startswith("dog_logit"):
            feature_rows.append(
                {
                    "candidate": candidate.name,
                    "status": candidate.status,
                    "removed_family": candidate.removed_family,
                    "feature_count": candidate.feature_count,
                    "description": candidate.description,
                }
            )

    regime_summary = build_regime_summary(df)
    calibration = pd.concat(
        [
            build_underdog_calibration(df, "raw_prob_a", "raw_model"),
            build_underdog_calibration(df, "artifact_prob_a", "artifact_model"),
        ],
        ignore_index=True,
    )
    odds_bins = build_underdog_odds_bins(df)
    experiments = pd.DataFrame(experiment_rows)
    feature_ablation = pd.DataFrame(feature_rows)

    selected_candidate, selection_status = select_candidate_on_tune(experiments)
    recommendation = build_recommendation(
        scored_df,
        candidates,
        experiments,
        selected_candidate,
        selection_status,
        feature_merge_note,
    )

    concentration = pd.DataFrame()
    if selected_candidate is not None:
        concentration = build_concentration_summary(
            scored_df,
            f"candidate_prob__{selected_candidate}",
        )

    regime_summary.round(6).to_csv(spec.output_dir / "regime_summary.csv", index=False)
    calibration.round(6).to_csv(spec.output_dir / "underdog_calibration.csv", index=False)
    odds_bins.round(6).to_csv(spec.output_dir / "underdog_odds_bins.csv", index=False)
    experiments.round(6).to_csv(spec.output_dir / "rescue_experiments.csv", index=False)
    feature_ablation.to_csv(spec.output_dir / "feature_family_ablation.csv", index=False)
    concentration.round(6).to_csv(spec.output_dir / "final_concentration.csv", index=False)
    (spec.output_dir / "final_recommendation.txt").write_text(
        recommendation,
        encoding="utf-8",
    )

    print(f"=== {spec.name.upper()} UNDERDOG RESCUE ===")
    print(feature_merge_note)
    print(recommendation)
    print(f"Wrote outputs under {spec.output_dir}")
    print()


def run_for_tour(tour: str, root: Path = PROJECT_ROOT) -> None:
    specs = tour_specs(root)
    spec = specs[tour]
    df = load_tour_dataset(spec)
    df, available_features, merge_note = merge_features(df, spec)
    write_outputs(spec, df, available_features, merge_note)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run odds-regime diagnostics and walk-forward underdog rescue experiments."
    )
    parser.add_argument(
        "--tour",
        choices=["all", "atp", "challenger"],
        default="all",
        help="Tour artifacts to analyze.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tours = ["atp", "challenger"] if args.tour == "all" else [args.tour]
    for tour in tours:
        run_for_tour(tour)


if __name__ == "__main__":
    main()
