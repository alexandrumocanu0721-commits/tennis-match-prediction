from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from underdog_rescue import (
    ELO_FEATURES,
    FEATURE_FAMILIES,
    H2H_FEATURES,
    PROJECT_ROOT,
    RANK_FEATURES,
    RECENT_FORM_FEATURES,
    SERVE_RETURN_FEATURES,
    assign_walk_forward_periods,
    flat_stake_profit,
    load_tour_dataset,
    merge_features,
    tour_specs,
)


OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "underdog_specialist"
MODEL_DIR = PROJECT_ROOT / "models" / "underdog_specialist"
DATASET_PATH = OUTPUT_DIR / "near_dog_dataset.csv"
DATASET_SUMMARY_PATH = OUTPUT_DIR / "near_dog_dataset_summary.csv"
MODEL_PATH = MODEL_DIR / "near_dog_specialist.pkl"
MODEL_METADATA_PATH = MODEL_DIR / "near_dog_specialist_metadata.json"
TRAIN_RESULTS_PATH = OUTPUT_DIR / "near_dog_train_results.csv"
BACKTEST_PREDICTIONS_PATH = OUTPUT_DIR / "near_dog_backtest_predictions.csv"
BACKTEST_SUMMARY_PATH = OUTPUT_DIR / "near_dog_backtest_summary.csv"
BACKTEST_BY_TOUR_PATH = OUTPUT_DIR / "near_dog_backtest_by_tour.csv"
BACKTEST_CALIBRATION_PATH = OUTPUT_DIR / "near_dog_backtest_calibration.csv"
FINAL_RECOMMENDATION_PATH = OUTPUT_DIR / "near_dog_final_recommendation.txt"

NEAR_DOG_MIN_ODDS = 2.20
NEAR_DOG_MAX_ODDS = 3.00
SHRINKAGE_FACTORS = [0.25, 0.50, 0.75, 1.00]
EDGE_THRESHOLDS = [0.00, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10]
MIN_TUNE_BETS = 20
MIN_FINAL_BETS = 40
MAX_FINAL_CALIBRATION_ERROR = 0.08
MAX_FINAL_CLUSTER_SHARE = 0.60

DIFF_FEATURES = sorted(
    {
        feature
        for family in [
            RANK_FEATURES,
            ELO_FEATURES,
            SERVE_RETURN_FEATURES,
            RECENT_FORM_FEATURES,
            H2H_FEATURES,
        ]
        for feature in family
    }
)
BASE_NUMERIC_FEATURES = [
    "raw_prob",
    "artifact_prob",
    "market_prob",
    "raw_edge",
    "artifact_edge",
    "odds",
    "log_odds",
]
CATEGORICAL_FEATURES = ["tour", "surface"]


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    numeric_features: list[str]
    categorical_features: list[str]
    removed_family: str = ""


def _round_float(value: float | int | str | None, digits: int = 6) -> float | str:
    if value is None or pd.isna(value):
        return "NaN"
    if isinstance(value, str):
        return value
    return round(float(value), digits)


def available_diff_features(df: pd.DataFrame) -> list[str]:
    return [feature for feature in DIFF_FEATURES if feature in df.columns]


def _base_side_columns(row: pd.Series, side: str) -> dict[str, Any]:
    side_is_a = side == "A"
    player_col = "player_a_name" if side_is_a else "player_b_name"
    opponent_col = "player_b_name" if side_is_a else "player_a_name"

    raw_prob_a = float(row["raw_prob_a"])
    artifact_prob_a = float(row["artifact_prob_a"])
    market_prob_a = float(row["market_prob_a"])
    actual_a = float(row["actual_result"])

    if side_is_a:
        raw_prob = raw_prob_a
        artifact_prob = artifact_prob_a
        market_prob = market_prob_a
        odds = float(row["odds_a"])
        actual = actual_a
    else:
        raw_prob = 1.0 - raw_prob_a
        artifact_prob = 1.0 - artifact_prob_a
        market_prob = 1.0 - market_prob_a
        odds = (
            float(row["odds_b"])
            if "odds_b" in row.index and pd.notna(row["odds_b"])
            else 1.0 / market_prob
            if market_prob > 0
            else np.nan
        )
        actual = 1.0 - actual_a

    return {
        "candidate_id": f"{row['tour']}:{row['date_key']}:{row.name}:{side}",
        "tour": row["tour"],
        "date": row["date"],
        "date_key": row["date_key"],
        "month": row.get("month", ""),
        "side": side,
        "player_name": row[player_col],
        "opponent_name": row[opponent_col],
        "surface": row.get("surface", ""),
        "tourney_name": row.get("tourney_name", ""),
        "tourney_id": row.get("tourney_id", ""),
        "round": row.get("round", ""),
        "raw_prob": raw_prob,
        "artifact_prob": artifact_prob,
        "market_prob": market_prob,
        "odds": odds,
        "actual_result": actual,
        "source_bookmaker": row.get("source_bookmaker", ""),
        "odds_source_confidence": row.get("odds_source_confidence", ""),
    }


def build_two_sided_candidates(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    required = ["raw_prob_a", "artifact_prob_a", "market_prob_a", "odds_a", "actual_result"]
    valid = df.dropna(subset=required).copy()
    for _, row in valid.iterrows():
        for side in ["A", "B"]:
            record = _base_side_columns(row, side)
            sign = 1.0 if side == "A" else -1.0
            for feature in feature_cols:
                value = pd.to_numeric(row.get(feature), errors="coerce")
                record[feature] = sign * float(value) if pd.notna(value) else np.nan
            rows.append(record)

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["raw_edge"] = out["raw_prob"] - out["market_prob"]
    out["artifact_edge"] = out["artifact_prob"] - out["market_prob"]
    out["log_odds"] = np.log(pd.to_numeric(out["odds"], errors="coerce"))
    out["is_near_dog"] = pd.to_numeric(out["odds"], errors="coerce").ge(
        NEAR_DOG_MIN_ODDS
    ) & pd.to_numeric(out["odds"], errors="coerce").lt(NEAR_DOG_MAX_ODDS)
    return out


def build_near_dog_dataset(tour: str = "all") -> tuple[pd.DataFrame, pd.DataFrame]:
    specs = tour_specs(PROJECT_ROOT)
    tours = ["atp", "challenger"] if tour == "all" else [tour]
    candidate_frames: list[pd.DataFrame] = []
    merge_notes: list[dict[str, Any]] = []

    for tour_name in tours:
        spec = specs[tour_name]
        tour_df = load_tour_dataset(spec)
        merged, feature_cols, merge_note = merge_features(tour_df, spec)
        candidates = build_two_sided_candidates(merged, feature_cols)
        candidate_frames.append(candidates)
        merge_notes.append(
            {
                "tour": tour_name,
                "input_rows": int(len(tour_df)),
                "candidate_rows": int(len(candidates)),
                "near_dog_rows": int(candidates["is_near_dog"].sum()) if not candidates.empty else 0,
                "feature_count": int(len(feature_cols)),
                "merge_note": merge_note,
            }
        )

    if not candidate_frames:
        raise ValueError("No candidate frames were built.")

    all_candidates = pd.concat(candidate_frames, ignore_index=True)
    near_dogs = all_candidates.loc[all_candidates["is_near_dog"]].copy()
    near_dogs = near_dogs.dropna(subset=["actual_result", "market_prob", "odds", "raw_prob"])
    near_dogs = near_dogs.sort_values(["date", "tour", "player_name"], kind="mergesort")
    near_dogs = assign_walk_forward_periods(near_dogs).reset_index(drop=True)
    near_dogs["target"] = near_dogs["actual_result"].astype(int)

    summary = pd.DataFrame(merge_notes)
    return near_dogs, summary


def summarize_signal(
    df: pd.DataFrame,
    prob_col: str,
    edge_threshold: float | None = None,
) -> dict[str, float | int]:
    working = df.copy()
    if edge_threshold is not None:
        working = working.loc[
            pd.to_numeric(working[prob_col], errors="coerce")
            .sub(pd.to_numeric(working["market_prob"], errors="coerce"))
            .ge(edge_threshold)
        ].copy()

    working = working.dropna(subset=[prob_col, "market_prob", "odds", "actual_result"])
    if working.empty:
        return {
            "bet_count": 0,
            "hit_rate": np.nan,
            "avg_odds": np.nan,
            "mean_prob": np.nan,
            "mean_market_prob": np.nan,
            "mean_edge": np.nan,
            "roi": np.nan,
            "calibration_error": np.nan,
            "log_loss": np.nan,
        }

    actual = pd.to_numeric(working["actual_result"], errors="coerce")
    prob = pd.to_numeric(working[prob_col], errors="coerce")
    market = pd.to_numeric(working["market_prob"], errors="coerce")
    profits = flat_stake_profit(
        working.rename(columns={"odds": "odds_a"})[["actual_result", "odds_a"]]
    )
    clipped = prob.clip(1e-6, 1.0 - 1e-6)
    return {
        "bet_count": int(len(working)),
        "hit_rate": float(actual.mean()),
        "avg_odds": float(pd.to_numeric(working["odds"], errors="coerce").mean()),
        "mean_prob": float(prob.mean()),
        "mean_market_prob": float(market.mean()),
        "mean_edge": float((prob - market).mean()),
        "roi": float(profits.mean()),
        "calibration_error": float(abs(prob.mean() - actual.mean())),
        "log_loss": float(log_loss(actual, clipped, labels=[0, 1])),
    }


def dataset_summary(df: pd.DataFrame, merge_summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for period in ["all", "train", "tune", "final"]:
        period_df = df if period == "all" else df.loc[df["wf_period"].eq(period)]
        row = {"section": "period", "name": period}
        row.update(summarize_signal(period_df, "raw_prob"))
        rows.append(row)

    for tour, group in df.groupby("tour", dropna=False):
        row = {"section": "tour", "name": tour}
        row.update(summarize_signal(group, "raw_prob"))
        rows.append(row)

    for _, merge_row in merge_summary.iterrows():
        rows.append(
            {
                "section": "merge",
                "name": merge_row["tour"],
                "bet_count": merge_row["near_dog_rows"],
                "hit_rate": np.nan,
                "avg_odds": np.nan,
                "mean_prob": np.nan,
                "mean_market_prob": np.nan,
                "mean_edge": np.nan,
                "roi": np.nan,
                "calibration_error": np.nan,
                "log_loss": np.nan,
                "note": merge_row["merge_note"],
            }
        )

    return pd.DataFrame(rows)


def candidate_specs(columns: pd.Index) -> list[CandidateSpec]:
    available_base = [column for column in BASE_NUMERIC_FEATURES if column in columns]
    available_diffs = [feature for feature in DIFF_FEATURES if feature in columns]
    available_cats = [column for column in CATEGORICAL_FEATURES if column in columns]
    specs = [
        CandidateSpec(
            name="near_dog_market_only",
            numeric_features=available_base,
            categorical_features=available_cats,
        ),
        CandidateSpec(
            name="near_dog_logit_all",
            numeric_features=available_base + available_diffs,
            categorical_features=available_cats,
        ),
    ]

    for family_name, family_features in FEATURE_FAMILIES.items():
        kept_diffs = [
            feature
            for feature in available_diffs
            if feature not in set(family_features)
        ]
        specs.append(
            CandidateSpec(
                name=f"near_dog_logit_no_{family_name}",
                numeric_features=available_base + kept_diffs,
                categorical_features=available_cats,
                removed_family=family_name,
            )
        )
    return specs


def build_model(spec: CandidateSpec) -> Pipeline:
    transformers = []
    if spec.numeric_features:
        transformers.append(("num", StandardScaler(), spec.numeric_features))
    if spec.categorical_features:
        transformers.append(
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                spec.categorical_features,
            )
        )
    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", LogisticRegression(max_iter=2000, C=0.5)),
        ]
    )


def score_candidate(model: Pipeline, df: pd.DataFrame, spec: CandidateSpec) -> pd.Series:
    features = spec.numeric_features + spec.categorical_features
    scored = pd.Series(np.nan, index=df.index, dtype=float)
    valid = df[features].notna().all(axis=1)
    if valid.any():
        scored.loc[valid] = model.predict_proba(df.loc[valid, features])[:, 1]
    return scored


def apply_market_shrinkage(
    model_prob: pd.Series,
    market_prob: pd.Series,
    shrinkage_factor: float,
) -> pd.Series:
    adjusted = market_prob + shrinkage_factor * (model_prob - market_prob)
    return adjusted.clip(1e-6, 1.0 - 1e-6)


def evaluate_thresholds(
    df: pd.DataFrame,
    prob_col: str,
    period: str,
    candidate_name: str,
    shrinkage_factor: float,
    edge_thresholds: list[float] = EDGE_THRESHOLDS,
) -> pd.DataFrame:
    period_df = df.loc[df["wf_period"].eq(period)].copy()
    rows = []
    for threshold in edge_thresholds:
        row = {
            "candidate": candidate_name,
            "wf_period": period,
            "shrinkage_factor": shrinkage_factor,
            "edge_threshold": threshold,
        }
        row.update(summarize_signal(period_df, prob_col, edge_threshold=threshold))
        rows.append(row)
    return pd.DataFrame(rows)


def select_best_tune_row(results: pd.DataFrame) -> tuple[pd.Series, str]:
    candidates = results.loc[
        results["wf_period"].eq("tune") & results["bet_count"].ge(MIN_TUNE_BETS)
    ].copy()
    if candidates.empty:
        candidates = results.loc[results["wf_period"].eq("tune")].copy()
        candidates = candidates.sort_values(
            ["bet_count", "roi", "calibration_error"],
            ascending=[False, False, True],
            kind="mergesort",
        )
        return candidates.iloc[0], "NO_TUNE_ROW_WITH_MIN_BETS"

    candidates["tune_pass"] = (
        candidates["roi"].ge(0.0)
        & candidates["mean_edge"].ge(0.0)
        & candidates["calibration_error"].le(MAX_FINAL_CALIBRATION_ERROR)
    )
    passing = candidates.loc[candidates["tune_pass"]].copy()
    if passing.empty:
        candidates = candidates.sort_values(
            ["roi", "calibration_error", "bet_count"],
            ascending=[False, True, False],
            kind="mergesort",
        )
        return candidates.iloc[0], "BEST_TUNE_ONLY_NO_PASS"

    passing = passing.sort_values(
        ["roi", "calibration_error", "bet_count"],
        ascending=[False, True, False],
        kind="mergesort",
    )
    return passing.iloc[0], "TUNE_PASS"


def train_specialist(dataset: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    train_df = dataset.loc[dataset["wf_period"].eq("train")].copy()
    tune_frames: list[pd.DataFrame] = []
    fitted: dict[str, tuple[Pipeline, CandidateSpec]] = {}

    for spec in candidate_specs(dataset.columns):
        features = spec.numeric_features + spec.categorical_features
        fit_df = train_df.dropna(subset=features + ["target"]).copy()
        if fit_df.empty or fit_df["target"].nunique() < 2:
            continue
        model = build_model(spec)
        model.fit(fit_df[features], fit_df["target"].astype(int))
        fitted[spec.name] = (model, spec)

        scored = dataset.copy()
        base_prob_col = f"base_prob__{spec.name}"
        scored[base_prob_col] = score_candidate(model, scored, spec)
        for shrinkage_factor in SHRINKAGE_FACTORS:
            prob_col = f"prob__{spec.name}__shrink_{int(shrinkage_factor * 100)}"
            scored[prob_col] = apply_market_shrinkage(
                scored[base_prob_col],
                scored["market_prob"],
                shrinkage_factor,
            )
            tune_frames.append(
                evaluate_thresholds(
                    scored,
                    prob_col,
                    "tune",
                    spec.name,
                    shrinkage_factor,
                )
            )

    if not tune_frames:
        raise ValueError("No specialist candidate models could be trained.")

    tune_results = pd.concat(tune_frames, ignore_index=True)
    best_row, selection_status = select_best_tune_row(tune_results)
    selected_name = str(best_row["candidate"])
    selected_model, selected_spec = fitted[selected_name]

    artifact = {
        "model": selected_model,
        "candidate_name": selected_name,
        "selection_status": selection_status,
        "shrinkage_factor": float(best_row["shrinkage_factor"]),
        "edge_threshold": float(best_row["edge_threshold"]),
        "numeric_features": selected_spec.numeric_features,
        "categorical_features": selected_spec.categorical_features,
        "removed_family": selected_spec.removed_family,
        "near_dog_min_odds": NEAR_DOG_MIN_ODDS,
        "near_dog_max_odds": NEAR_DOG_MAX_ODDS,
        "min_tune_bets": MIN_TUNE_BETS,
        "selected_tune_metrics": {
            key: _round_float(best_row[key])
            for key in [
                "bet_count",
                "hit_rate",
                "avg_odds",
                "mean_prob",
                "mean_market_prob",
                "mean_edge",
                "roi",
                "calibration_error",
                "log_loss",
            ]
            if key in best_row.index
        },
    }
    return artifact, tune_results


def save_model_artifact(artifact: dict[str, Any]) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, MODEL_PATH)
    metadata = {key: value for key, value in artifact.items() if key != "model"}
    MODEL_METADATA_PATH.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def load_model_artifact(path: Path = MODEL_PATH) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing specialist model artifact: {path}")
    return joblib.load(path)


def score_with_artifact(dataset: pd.DataFrame, artifact: dict[str, Any]) -> pd.DataFrame:
    spec = CandidateSpec(
        name=str(artifact["candidate_name"]),
        numeric_features=list(artifact["numeric_features"]),
        categorical_features=list(artifact["categorical_features"]),
        removed_family=str(artifact.get("removed_family", "")),
    )
    out = dataset.copy()
    out["specialist_model_prob"] = score_candidate(artifact["model"], out, spec)
    out["specialist_shrinkage_factor"] = float(artifact.get("shrinkage_factor", 1.0))
    out["specialist_prob"] = apply_market_shrinkage(
        out["specialist_model_prob"],
        out["market_prob"],
        float(artifact.get("shrinkage_factor", 1.0)),
    )
    out["specialist_edge"] = out["specialist_prob"] - out["market_prob"]
    out["specialist_edge_threshold"] = float(artifact["edge_threshold"])
    out["specialist_bet"] = out["specialist_edge"].ge(float(artifact["edge_threshold"]))
    return out


def max_cluster_share(df: pd.DataFrame) -> float:
    bets = df.loc[df["specialist_bet"]].copy()
    if bets.empty:
        return float("nan")
    max_share = 0.0
    for col in ["month", "surface", "tourney_name"]:
        if col not in bets.columns:
            continue
        max_share = max(
            max_share,
            float(bets[col].fillna("UNKNOWN").value_counts(normalize=True).max()),
        )
    return max_share


def backtest_summary(scored: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for period in ["train", "tune", "final", "all"]:
        period_df = scored if period == "all" else scored.loc[scored["wf_period"].eq(period)]
        row = {"section": "period", "name": period}
        row.update(summarize_signal(period_df, "specialist_prob", edge_threshold=float(scored["specialist_edge_threshold"].iloc[0])))
        rows.append(row)

    for tour, group in scored.groupby("tour", dropna=False):
        row = {"section": "tour", "name": tour}
        row.update(summarize_signal(group, "specialist_prob", edge_threshold=float(scored["specialist_edge_threshold"].iloc[0])))
        rows.append(row)
    return pd.DataFrame(rows)


def backtest_by_tour(scored: pd.DataFrame) -> pd.DataFrame:
    rows = []
    threshold = float(scored["specialist_edge_threshold"].iloc[0])
    for tour, tour_df in scored.groupby("tour", dropna=False):
        for period in ["train", "tune", "final", "all"]:
            period_df = tour_df if period == "all" else tour_df.loc[tour_df["wf_period"].eq(period)]
            row = {
                "tour": tour,
                "wf_period": period,
            }
            row.update(summarize_signal(period_df, "specialist_prob", edge_threshold=threshold))
            row["max_cluster_share"] = max_cluster_share(period_df)
            row["recommendation"] = recommendation_from_metrics(row)
            rows.append(row)
    return pd.DataFrame(rows)


def calibration_table(scored: pd.DataFrame) -> pd.DataFrame:
    rows = []
    bins = np.arange(0.0, 1.0001, 0.05)
    for period in ["tune", "final", "all"]:
        period_df = scored if period == "all" else scored.loc[scored["wf_period"].eq(period)]
        working = period_df.dropna(subset=["specialist_prob", "actual_result"]).copy()
        if working.empty:
            continue
        working["prob_bin"] = pd.cut(
            working["specialist_prob"],
            bins=bins,
            include_lowest=True,
            right=False,
        )
        grouped = (
            working.groupby("prob_bin", observed=False)
            .agg(
                row_count=("actual_result", "size"),
                actual_win_rate=("actual_result", "mean"),
                mean_specialist_prob=("specialist_prob", "mean"),
                mean_market_prob=("market_prob", "mean"),
                avg_odds=("odds", "mean"),
            )
            .reset_index()
        )
        grouped = grouped.loc[grouped["row_count"].gt(0)].copy()
        grouped["wf_period"] = period
        grouped["prob_bin"] = grouped["prob_bin"].astype(str)
        rows.append(grouped)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def recommendation_from_metrics(metrics: dict[str, Any] | pd.Series) -> str:
    cluster_share = metrics.get("max_cluster_share", np.nan)
    passes = (
        int(metrics["bet_count"]) >= MIN_FINAL_BETS
        and float(metrics["roi"]) >= 0.0
        and float(metrics["mean_edge"]) >= 0.0
        and float(metrics["calibration_error"]) <= MAX_FINAL_CALIBRATION_ERROR
        and (pd.isna(cluster_share) or float(cluster_share) <= MAX_FINAL_CLUSTER_SHARE)
    )
    return "ALLOW_NEAR_DOG_RESEARCH_ONLY" if passes else "BLOCK_NEAR_DOGS"


def final_recommendation(scored: pd.DataFrame, artifact: dict[str, Any]) -> str:
    final_df = scored.loc[scored["wf_period"].eq("final")].copy()
    metrics = summarize_signal(
        final_df,
        "specialist_prob",
        edge_threshold=float(artifact["edge_threshold"]),
    )
    cluster_share = max_cluster_share(final_df)
    recommendation = recommendation_from_metrics(
        {
            **metrics,
            "max_cluster_share": cluster_share,
        }
    )
    lines = [
        "Near-Dog Specialist Recommendation",
        "==================================",
        "",
        f"candidate: {artifact['candidate_name']}",
        f"selection_status: {artifact['selection_status']}",
        f"shrinkage_factor: {float(artifact.get('shrinkage_factor', 1.0)):.2f}",
        f"edge_threshold: {float(artifact['edge_threshold']):.4f}",
        f"final_bet_count: {int(metrics['bet_count'])}",
        f"final_roi: {float(metrics['roi']):.4f}",
        f"final_mean_edge: {float(metrics['mean_edge']):.4f}",
        f"final_calibration_error: {float(metrics['calibration_error']):.4f}",
        f"final_max_cluster_share: {cluster_share:.4f}" if pd.notna(cluster_share) else "final_max_cluster_share: n/a",
        f"recommendation: {recommendation}",
    ]
    if recommendation == "BLOCK_NEAR_DOGS":
        lines.append(
            "reason: near-dog specialist did not satisfy untouched final-period pass criteria."
        )
    return "\n".join(lines) + "\n"
