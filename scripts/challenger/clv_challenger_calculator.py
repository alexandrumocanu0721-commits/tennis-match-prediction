from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from challenger_config import (
    assert_challenger_processed_output_path,
    path_challenger_backtest_predictions_csv,
    path_challenger_processed_features_csv,
    path_challenger_clv_results_csv,
    path_challenger_clv_unmatched_csv,
    project_root,
)

SOFASCORE_ODDS_RELATIVE_PATH = Path("data") / "sofascore" / "challenger" / "march-april_sofascore_odds.csv"
CLV_AMBIGUOUS_FILENAME = "clv_ambiguous.csv"
CLV_PLACEBO_FILENAME = "clv_placebo.csv"
MAX_DATE_DISTANCE_DAYS = 7
PLACEBO_RANDOM_SEED = 42
PLACEBO_BETA_CONCENTRATION = 20.0
RANK_DIFF_BUCKET_LABELS = ["0-50", "50-100", "100-200", "200-500", "500+"]


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKD", str(value))
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.replace(".", " ").replace(",", " ").replace("-", " ").replace("/", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def name_key(value: object) -> str:
    return normalize_text(value).upper()


def tournament_tokens(value: object) -> set[str]:
    stopwords = {
        "CH",
        "CHALLENGER",
        "ATP",
        "WTA",
        "OPEN",
        "SINGLES",
        "DOUBLES",
        "MEN",
        "WOMEN",
        "QUALIFYING",
        "QUALIFICATION",
        "QUAL",
    }
    return {
        token
        for token in name_key(value).split()
        if len(token) > 1 and token not in stopwords and not token.isdigit()
    }


def surface_key(value: object) -> str:
    text = normalize_text(value).lower()
    if not text:
        return ""
    if "hard" in text:
        return "HARD"
    if "clay" in text:
        return "CLAY"
    if "grass" in text:
        return "GRASS"
    if "carpet" in text:
        return "CARPET"
    return text.upper()


def _first_existing_column(df: pd.DataFrame, options: list[str]) -> str:
    for column in options:
        if column in df.columns:
            return column
    raise KeyError(f"None of these columns exist: {options}")


def _optional_existing_column(df: pd.DataFrame, options: list[str]) -> str | None:
    for column in options:
        if column in df.columns:
            return column
    return None


def _validate_output_paths(root: Path) -> None:
    expected_results = (root / "data" / "processed" / "challenger" / "clv_results.csv").resolve()
    expected_unmatched = (root / "data" / "processed" / "challenger" / "clv_unmatched.csv").resolve()
    forbidden_results = (root / "data" / "processed" / "atp" / "clv_results.csv").resolve()
    forbidden_unmatched = (root / "data" / "processed" / "atp" / "clv_unmatched.csv").resolve()

    actual_results = path_challenger_clv_results_csv(root).resolve()
    actual_unmatched = path_challenger_clv_unmatched_csv(root).resolve()

    if actual_results != expected_results or actual_unmatched != expected_unmatched:
        raise ValueError("Safety check failed: challenger CLV output paths are unexpected.")
    if actual_results == forbidden_results or actual_unmatched == forbidden_unmatched:
        raise ValueError("Safety check failed: refusing to write ATP CLV outputs.")
    assert_challenger_processed_output_path(actual_results, root)
    assert_challenger_processed_output_path(actual_unmatched, root)


def prepare_predictions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    player_a_col = _first_existing_column(df, ["player_a_name", "player_a"])
    player_b_col = _first_existing_column(df, ["player_b_name", "player_b"])
    actual_col = _first_existing_column(df, ["actual_result", "actual_winner"])
    tournament_col = _optional_existing_column(
        df, ["tournament_name", "unique_tournament_name", "tourney_name", "tournament"]
    )
    tourney_id_col = _optional_existing_column(df, ["tourney_id", "tournament_id"])
    tourney_date_col = _optional_existing_column(df, ["tourney_date", "tournament_date"])
    round_col = _optional_existing_column(df, ["round", "round_name"])

    out = pd.DataFrame(
        {
            "date": pd.to_datetime(df["date"], errors="coerce").dt.normalize(),
            "tourney_id": df[tourney_id_col].astype(str) if tourney_id_col else "",
            "tourney_name": df[tournament_col].astype(str) if tournament_col else "",
            "tourney_date": (
                pd.to_datetime(df[tourney_date_col], errors="coerce").dt.normalize()
                if tourney_date_col
                else pd.to_datetime(df["date"], errors="coerce").dt.normalize()
            ),
            "round": df[round_col].astype(str) if round_col else "",
            "player_a_name": df[player_a_col].astype(str),
            "player_b_name": df[player_b_col].astype(str),
            "surface": df["surface"].astype(str),
            "predicted_prob_a": pd.to_numeric(df["predicted_prob_a"], errors="coerce"),
            "actual_result": pd.to_numeric(df[actual_col], errors="coerce"),
        }
    )
    out["player_a_key"] = out["player_a_name"].map(name_key)
    out["player_b_key"] = out["player_b_name"].map(name_key)
    out["surface_key"] = out["surface"].map(surface_key)
    out["tournament_key"] = out["tourney_name"].map(name_key)
    out = out.dropna(subset=["date"])
    out = out.reset_index(drop=True)
    out["prediction_row_id"] = range(len(out))
    return out


def enrich_predictions_with_rank_diff(predictions_df: pd.DataFrame, root: Path) -> pd.DataFrame:
    features_path = path_challenger_processed_features_csv(root)
    if not features_path.exists():
        out = predictions_df.copy()
        out["rank_diff"] = float("nan")
        return out

    feature_header = pd.read_csv(features_path, nrows=0)
    feature_cols = ["date", "surface", "player_a_name", "player_b_name", "rank_diff"]
    optional_merge_cols = ["tourney_id", "round"]
    feature_cols.extend(
        column for column in optional_merge_cols if column in feature_header.columns
    )
    features_df = pd.read_csv(features_path, usecols=feature_cols)
    features_df["date"] = pd.to_datetime(features_df["date"], errors="coerce").dt.normalize()
    features_df["rank_diff"] = pd.to_numeric(features_df["rank_diff"], errors="coerce")
    features_df = features_df.dropna(subset=["date"])
    merge_cols = ["date", "surface", "player_a_name", "player_b_name"]
    merge_cols.extend(
        column
        for column in optional_merge_cols
        if column in features_df.columns and column in predictions_df.columns
    )
    features_df = features_df.drop_duplicates(
        subset=merge_cols, keep="first"
    )

    return predictions_df.merge(
        features_df,
        how="left",
        on=merge_cols,
    )


def _apply_singles_filter(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    out = df.copy()
    applied = False

    if "is_doubles" in out.columns:
        out = out[~out["is_doubles"].astype(str).str.lower().isin(["1", "true", "yes"])]
        applied = True

    for column in ["match_type", "event_type", "competition_type", "category"]:
        if column not in out.columns:
            continue
        values = out[column].astype(str).str.lower()
        out = out[~values.str.contains("double", na=False)]
        if values.str.contains("single", na=False).any():
            out = out[values.str.contains("single", na=False)]
        applied = True

    for column in ["tournament_name", "unique_tournament_name"]:
        if column in out.columns:
            out = out[
                ~out[column].astype(str).str.lower().str.contains("doubles", na=False)
            ]
            applied = True

    note = (
        "explicit/event-name singles filter applied"
        if applied
        else "no singles filter columns found; relying on player-name matching"
    )
    return out.copy(), note


def prepare_sofascore_odds(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    df = pd.read_csv(path)
    out = df.copy()
    out["match_start_date_utc"] = pd.to_datetime(
        out["match_start_date_utc"], errors="coerce"
    ).dt.normalize()
    out["home_latest_no_vig"] = pd.to_numeric(out["home_latest_no_vig"], errors="coerce")
    out["away_latest_no_vig"] = pd.to_numeric(out["away_latest_no_vig"], errors="coerce")
    out["home_latest_decimal"] = pd.to_numeric(out["home_latest_decimal"], errors="coerce")
    out["away_latest_decimal"] = pd.to_numeric(out["away_latest_decimal"], errors="coerce")

    out["home_player_key"] = out["home_player"].map(name_key)
    out["away_player_key"] = out["away_player"].map(name_key)
    out["player_pair_key"] = out.apply(
        lambda row: tuple(sorted((row["home_player_key"], row["away_player_key"]))), axis=1
    )
    out["surface_key"] = out["surface"].map(surface_key)
    out["tournament_key"] = out["tournament_name"].map(name_key)
    out["unique_tournament_key"] = out["unique_tournament_name"].map(name_key)
    out["tournament_tokens"] = out.apply(
        lambda row: tournament_tokens(row.get("tournament_name"))
        | tournament_tokens(row.get("unique_tournament_name")),
        axis=1,
    )

    valid_mask = out["match_start_date_utc"].notna()
    if "source_bookmaker" in out.columns:
        valid_mask &= out["source_bookmaker"].astype(str) == "bet365_via_sofascore"
    valid_mask &= out["home_latest_no_vig"].between(0, 1, inclusive="both")
    valid_mask &= out["away_latest_no_vig"].between(0, 1, inclusive="both")
    valid_mask &= (out["home_latest_no_vig"] + out["away_latest_no_vig"]).sub(1).abs() <= 0.01
    valid_mask &= out["home_latest_decimal"] > 1
    valid_mask &= out["away_latest_decimal"] > 1
    valid_mask &= out["surface"].notna()
    valid_mask &= out["home_player"].notna()
    valid_mask &= out["away_player"].notna()

    valid_out = out[valid_mask].copy()
    invalid_out = out[~valid_mask].copy()
    valid_out, singles_note = _apply_singles_filter(valid_out)
    return valid_out.copy(), invalid_out.copy(), singles_note


def resolve_candidate(
    prediction_row: pd.Series, odds_df: pd.DataFrame, invalid_odds_df: pd.DataFrame
) -> tuple[pd.Series | None, str | None, pd.DataFrame, pd.DataFrame]:
    player_pair = tuple(
        sorted((prediction_row["player_a_key"], prediction_row["player_b_key"]))
    )
    player_candidates = odds_df[odds_df["player_pair_key"] == player_pair].copy()
    if player_candidates.empty:
        invalid_candidates = invalid_odds_df[
            invalid_odds_df["player_pair_key"] == player_pair
        ].copy()
        if not invalid_candidates.empty:
            return None, "invalid_odds", invalid_candidates, invalid_candidates
        return None, "no_candidate", player_candidates, player_candidates

    player_candidates["date_diff_days"] = (
        player_candidates["match_start_date_utc"] - prediction_row["date"]
    ).dt.days.abs()
    within_window = player_candidates[
        player_candidates["date_diff_days"] <= MAX_DATE_DISTANCE_DAYS
    ].copy()
    if within_window.empty:
        return None, "date_conflict", player_candidates, within_window

    tournament_key = prediction_row.get("tournament_key", "")
    tournament_filtered = within_window
    if isinstance(tournament_key, str) and tournament_key:
        prediction_tournament = prediction_row.get("tourney_name", "")
        prediction_tokens = tournament_tokens(prediction_tournament)
        tournament_matches = within_window[
            (within_window["tournament_key"] == tournament_key)
            | (within_window["unique_tournament_key"] == tournament_key)
            | within_window["tournament_tokens"].map(
                lambda tokens: bool(prediction_tokens and tokens & prediction_tokens)
            )
        ].copy()
        if tournament_matches.empty:
            return None, "tournament_conflict", player_candidates, within_window
        tournament_filtered = tournament_matches

    min_distance = tournament_filtered["date_diff_days"].min()
    closest_date = tournament_filtered[
        tournament_filtered["date_diff_days"] == min_distance
    ].copy()
    if len(closest_date) == 1:
        return closest_date.iloc[0], None, player_candidates, closest_date

    surface_filtered = closest_date[
        closest_date["surface_key"] == prediction_row["surface_key"]
    ].copy()
    if len(surface_filtered) == 1:
        return surface_filtered.iloc[0], None, player_candidates, surface_filtered
    if not surface_filtered.empty:
        return None, "multiple_candidates", player_candidates, surface_filtered

    return None, "multiple_candidates", player_candidates, closest_date


def map_row_to_player_a_market(
    prediction_row: pd.Series, odds_row: pd.Series
) -> tuple[float, float, float, float]:
    if prediction_row["player_a_key"] == odds_row["home_player_key"]:
        market_prob_a = odds_row["home_latest_no_vig"]
        odds_a = odds_row["home_latest_decimal"]
        odds_b = odds_row["away_latest_decimal"]
    else:
        market_prob_a = odds_row["away_latest_no_vig"]
        odds_a = odds_row["away_latest_decimal"]
        odds_b = odds_row["home_latest_decimal"]

    if pd.isna(market_prob_a) or pd.isna(odds_a) or pd.isna(odds_b):
        raise ValueError("Invalid matched odds row contains missing market values.")
    if not 0 <= float(market_prob_a) <= 1 or float(odds_a) <= 1 or float(odds_b) <= 1:
        raise ValueError("Invalid matched odds row contains out-of-range market values.")

    clv_bet365 = prediction_row["predicted_prob_a"] - market_prob_a
    return float(market_prob_a), float(odds_a), float(odds_b), float(clv_bet365)


def build_diagnostics(
    results_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    monthly_df = (
        results_df.assign(month=pd.to_datetime(results_df["date"]).dt.to_period("M").astype(str))
        .groupby("month", dropna=False)
        .agg(
            count=("clv_bet365", "size"),
            mean_clv=("clv_bet365", "mean"),
            median_clv=("clv_bet365", "median"),
        )
        .reset_index()
        .sort_values("month", kind="mergesort")
    )

    surface_mapped = results_df["surface"].map(surface_key)
    surface_name = (
        surface_mapped.map({"HARD": "Hard", "CLAY": "Clay", "GRASS": "Grass"}).fillna("Other")
    )
    surface_df = (
        results_df.assign(surface_group=surface_name)
        .groupby("surface_group", dropna=False)
        .agg(
            count=("clv_bet365", "size"),
            mean_clv=("clv_bet365", "mean"),
            median_clv=("clv_bet365", "median"),
        )
    )
    surface_df = surface_df.reindex(["Hard", "Clay", "Grass"])
    surface_df["count"] = surface_df["count"].fillna(0).astype(int)
    surface_df = surface_df.reset_index().rename(columns={"surface_group": "surface"})

    odds_bucket = pd.Series("balanced", index=results_df.index)
    odds_bucket = odds_bucket.mask(results_df["market_prob_a"] > 0.65, "favorite")
    odds_bucket = odds_bucket.mask(results_df["market_prob_a"] < 0.35, "underdog")
    odds_bucket_df = (
        results_df.assign(odds_bucket=odds_bucket)
        .groupby("odds_bucket", dropna=False)
        .agg(count=("clv_bet365", "size"), mean_clv=("clv_bet365", "mean"))
    )
    odds_bucket_df = odds_bucket_df.reindex(["favorite", "balanced", "underdog"])
    odds_bucket_df["count"] = odds_bucket_df["count"].fillna(0).astype(int)
    odds_bucket_df = odds_bucket_df.reset_index()

    confidence_bins = [0.50, 0.55, 0.60, 0.65, 0.70, float("inf")]
    confidence_labels = ["0.50-0.55", "0.55-0.60", "0.60-0.65", "0.65-0.70", "0.70+"]
    confidence_bucket = pd.cut(
        results_df["predicted_prob_a"],
        bins=confidence_bins,
        labels=confidence_labels,
        right=False,
        include_lowest=True,
    )
    confidence_bucket_df = (
        results_df.assign(confidence_bucket=confidence_bucket)
        .groupby("confidence_bucket", dropna=False)
        .agg(count=("clv_bet365", "size"), mean_clv=("clv_bet365", "mean"))
    )
    confidence_bucket_df = confidence_bucket_df.reindex(confidence_labels)
    confidence_bucket_df["count"] = confidence_bucket_df["count"].fillna(0).astype(int)
    confidence_bucket_df = confidence_bucket_df.reset_index()

    rank_diff_abs = results_df["rank_diff"].abs()
    rank_diff_bucket = pd.cut(
        rank_diff_abs,
        bins=[0, 50, 100, 200, 500, float("inf")],
        labels=RANK_DIFF_BUCKET_LABELS,
        right=False,
        include_lowest=True,
    )
    rank_diff_bucket_df = (
        results_df.assign(rank_diff_bucket=rank_diff_bucket)
        .groupby("rank_diff_bucket", dropna=False)
        .agg(
            count=("clv_bet365", "size"),
            mean_clv=("clv_bet365", "mean"),
            median_clv=("clv_bet365", "median"),
        )
    )
    rank_diff_bucket_df = rank_diff_bucket_df.reindex(RANK_DIFF_BUCKET_LABELS)
    rank_diff_bucket_df["count"] = rank_diff_bucket_df["count"].fillna(0).astype(int)
    rank_diff_bucket_df = rank_diff_bucket_df.reset_index()

    return monthly_df, surface_df, odds_bucket_df, confidence_bucket_df, rank_diff_bucket_df


def add_placebo_probability_columns(
    results_df: pd.DataFrame, predictions_df: pd.DataFrame
) -> pd.DataFrame:
    out = results_df.copy()
    if results_df.empty:
        out["shuffled_predicted_prob_a"] = pd.Series(dtype=float)
        out["shuffled_clv_bet365"] = pd.Series(dtype=float)
        out["random_market_centered_prob_a"] = pd.Series(dtype=float)
        out["random_market_centered_clv_bet365"] = pd.Series(dtype=float)
        return out

    rng = np.random.default_rng(PLACEBO_RANDOM_SEED)
    market_prob = out["market_prob_a"].astype(float).to_numpy()

    if "prediction_row_id" in results_df.columns:
        shuffled_all = predictions_df["predicted_prob_a"].astype(float).to_numpy().copy()
        rng.shuffle(shuffled_all)
        row_ids = results_df["prediction_row_id"].astype(int).to_numpy()
        shuffled_prob = shuffled_all[row_ids]
    else:
        shuffled_prob = out["predicted_prob_a"].astype(float).to_numpy().copy()
        rng.shuffle(shuffled_prob)

    centered_market = np.clip(market_prob, 1e-6, 1 - 1e-6)
    alpha = centered_market * PLACEBO_BETA_CONCENTRATION
    beta = (1 - centered_market) * PLACEBO_BETA_CONCENTRATION
    random_prob = rng.beta(alpha, beta)

    out["shuffled_predicted_prob_a"] = shuffled_prob
    out["shuffled_clv_bet365"] = shuffled_prob - market_prob
    out["random_market_centered_prob_a"] = random_prob
    out["random_market_centered_clv_bet365"] = random_prob - market_prob
    return out


def build_placebo_clv_summary(results_df: pd.DataFrame) -> pd.DataFrame:
    columns = ["experiment", "mean_clv", "median_clv", "matched_rows"]
    if results_df.empty:
        return pd.DataFrame(columns=columns)

    experiments = {
        "actual_predictions": results_df["clv_bet365"].astype(float).to_numpy(),
        "shuffled_predictions": results_df["shuffled_clv_bet365"].astype(float).to_numpy(),
        "random_market_centered": results_df[
            "random_market_centered_clv_bet365"
        ].astype(float).to_numpy(),
    }

    rows = []
    for name, clv_values in experiments.items():
        rows.append(
            {
                "experiment": name,
                "mean_clv": float(np.mean(clv_values)),
                "median_clv": float(np.median(clv_values)),
                "matched_rows": int(len(clv_values)),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def build_summary_text(
    backtest_rows: int,
    odds_rows: int,
    results_df: pd.DataFrame,
    unmatched_df: pd.DataFrame,
    monthly_df: pd.DataFrame,
    surface_df: pd.DataFrame,
    odds_bucket_df: pd.DataFrame,
    confidence_bucket_df: pd.DataFrame,
    rank_diff_bucket_df: pd.DataFrame,
    placebo_df: pd.DataFrame,
) -> str:
    lines: list[str] = []
    matched_rows = len(results_df)
    unmatched_rows = len(unmatched_df)
    match_rate = (matched_rows / backtest_rows * 100.0) if backtest_rows else 0.0

    lines.append(f"backtest rows: {backtest_rows}")
    lines.append(f"odds rows: {odds_rows}")
    lines.append(f"matched rows: {matched_rows}")
    lines.append(f"unmatched rows: {unmatched_rows}")
    lines.append(f"match rate: {match_rate:.2f}%")

    if results_df.empty:
        lines.append("overall CLV: n/a")
        lines.append("monthly CLV: n/a")
        lines.append("surface CLV: n/a")
        lines.append("odds bucket CLV: n/a")
        lines.append("confidence bucket CLV: n/a")
        lines.append("rank_diff bucket CLV: n/a")
        return "\n".join(lines)

    lines.append("overall CLV:")
    lines.append(f"count: {matched_rows}")
    lines.append(f"mean CLV: {results_df['clv_bet365'].mean(skipna=True):.6f}")
    lines.append(f"median CLV: {results_df['clv_bet365'].median(skipna=True):.6f}")

    distance_distribution = (
        results_df["matched_date_distance"]
        .value_counts(dropna=False)
        .reindex(range(0, MAX_DATE_DISTANCE_DAYS + 1), fill_value=0)
    )
    lines.append("distance distribution:")
    for days, count in distance_distribution.items():
        label = "day" if days == 1 else "days"
        lines.append(f"{days} {label}: {int(count)}")

    lines.append("monthly CLV:")
    lines.append(monthly_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    lines.append("surface CLV:")
    lines.append(surface_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    lines.append("odds bucket CLV:")
    lines.append(odds_bucket_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    lines.append("confidence bucket CLV:")
    lines.append(confidence_bucket_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    rank_diff_available = int(results_df["rank_diff"].notna().sum())
    lines.append(f"rank_diff coverage: {rank_diff_available}/{matched_rows}")
    lines.append("rank_diff bucket CLV:")
    lines.append(rank_diff_bucket_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    lines.append("placebo CLV:")
    lines.append(placebo_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    return "\n".join(lines)


def main() -> None:
    root = project_root()
    _validate_output_paths(root)

    prediction_path = path_challenger_backtest_predictions_csv(root)
    sofascore_odds_path = root / SOFASCORE_ODDS_RELATIVE_PATH
    output_results_path = path_challenger_clv_results_csv(root)
    output_unmatched_path = path_challenger_clv_unmatched_csv(root)
    output_ambiguous_path = output_results_path.parent / CLV_AMBIGUOUS_FILENAME
    output_placebo_path = output_results_path.parent / CLV_PLACEBO_FILENAME
    assert_challenger_processed_output_path(output_ambiguous_path, root)
    assert_challenger_processed_output_path(output_placebo_path, root)
    legacy_diagnostic_paths = [
        output_results_path.parent / "clv_summary.txt",
        output_results_path.parent / "clv_surface.csv",
        output_results_path.parent / "clv_monthly.csv",
        output_results_path.parent / "clv_odds_bucket.csv",
        output_results_path.parent / "clv_confidence_bucket.csv",
    ]

    predictions_df = prepare_predictions(prediction_path)
    predictions_df = enrich_predictions_with_rank_diff(predictions_df, root)
    odds_df, invalid_odds_df, singles_filter_note = prepare_sofascore_odds(
        sofascore_odds_path
    )
    print(f"SofaScore odds filter: {singles_filter_note}")
    print(f"valid odds rows: {len(odds_df)}")
    print(f"invalid odds rows filtered out: {len(invalid_odds_df)}")

    matched_records: list[dict] = []
    unmatched_records: list[dict] = []
    ambiguous_records: list[dict] = []

    for _, prediction_row in predictions_df.iterrows():
        matched_row, reason, player_candidates, narrowed_candidates = resolve_candidate(
            prediction_row, odds_df, invalid_odds_df
        )

        if matched_row is not None:
            market_prob_a, odds_a, odds_b, clv_bet365 = map_row_to_player_a_market(
                prediction_row, matched_row
            )
            matched_records.append(
                {
                    "date": prediction_row["date"].strftime("%Y-%m-%d"),
                    "prediction_row_id": prediction_row["prediction_row_id"],
                    "tourney_id": prediction_row["tourney_id"],
                    "tourney_name": prediction_row["tourney_name"],
                    "tourney_date": prediction_row["tourney_date"].strftime("%Y-%m-%d"),
                    "round": prediction_row["round"],
                    "event_id": matched_row["event_id"],
                    "surface": prediction_row["surface"],
                    "player_a_name": prediction_row["player_a_name"],
                    "player_b_name": prediction_row["player_b_name"],
                    "predicted_prob_a": prediction_row["predicted_prob_a"],
                    "rank_diff": prediction_row.get("rank_diff"),
                    "market_prob_a": market_prob_a,
                    "clv_bet365": clv_bet365,
                    "date_distance_days": int(matched_row["date_diff_days"]),
                    "matched_date_distance": int(matched_row["date_diff_days"]),
                    "odds_a": odds_a,
                    "odds_b": odds_b,
                    "actual_result": prediction_row["actual_result"],
                    "source_bookmaker": matched_row.get("source_bookmaker"),
                    "odds_source_confidence": matched_row.get("odds_source_confidence"),
                }
            )
            continue

        unmatched_records.append(
            {
                "date": prediction_row["date"].strftime("%Y-%m-%d"),
                "tourney_id": prediction_row["tourney_id"],
                "tourney_name": prediction_row["tourney_name"],
                "tourney_date": prediction_row["tourney_date"].strftime("%Y-%m-%d"),
                "round": prediction_row["round"],
                "player_a_name": prediction_row["player_a_name"],
                "player_b_name": prediction_row["player_b_name"],
                "surface": prediction_row["surface"],
                "reason": reason,
                "name_candidate_count": len(player_candidates),
                "narrowed_candidate_count": len(narrowed_candidates),
            }
        )

        if reason == "multiple_candidates":
            for _, candidate in narrowed_candidates.iterrows():
                ambiguous_records.append(
                    {
                        "date": prediction_row["date"].strftime("%Y-%m-%d"),
                        "tourney_id": prediction_row["tourney_id"],
                        "tourney_name": prediction_row["tourney_name"],
                        "tourney_date": prediction_row["tourney_date"].strftime(
                            "%Y-%m-%d"
                        ),
                        "round": prediction_row["round"],
                        "player_a_name": prediction_row["player_a_name"],
                        "player_b_name": prediction_row["player_b_name"],
                        "surface": prediction_row["surface"],
                        "predicted_prob_a": prediction_row["predicted_prob_a"],
                        "event_id": candidate.get("event_id"),
                        "match_start_date_utc": candidate["match_start_date_utc"].strftime(
                            "%Y-%m-%d"
                        ),
                        "candidate_surface": candidate.get("surface"),
                        "home_player": candidate.get("home_player"),
                        "away_player": candidate.get("away_player"),
                        "date_diff_days": candidate.get("date_diff_days"),
                        "source_bookmaker": candidate.get("source_bookmaker"),
                        "odds_source_confidence": candidate.get("odds_source_confidence"),
                    }
                )

    results_df = pd.DataFrame(
        matched_records,
        columns=[
            "date",
            "prediction_row_id",
            "tourney_id",
            "tourney_name",
            "tourney_date",
            "round",
            "event_id",
            "surface",
            "player_a_name",
            "player_b_name",
            "predicted_prob_a",
            "rank_diff",
            "market_prob_a",
            "clv_bet365",
            "date_distance_days",
            "matched_date_distance",
            "odds_a",
            "odds_b",
            "actual_result",
            "source_bookmaker",
            "odds_source_confidence",
        ],
    )
    unmatched_df = pd.DataFrame(unmatched_records)
    ambiguous_df = pd.DataFrame(ambiguous_records)
    results_df = add_placebo_probability_columns(results_df, predictions_df)

    if results_df.empty:
        monthly_df = pd.DataFrame(columns=["month", "count", "mean_clv", "median_clv"])
        surface_df = pd.DataFrame(columns=["surface", "count", "mean_clv", "median_clv"])
        odds_bucket_df = pd.DataFrame(columns=["odds_bucket", "count", "mean_clv"])
        confidence_bucket_df = pd.DataFrame(
            columns=["confidence_bucket", "count", "mean_clv"]
        )
        rank_diff_bucket_df = pd.DataFrame(
            columns=["rank_diff_bucket", "count", "mean_clv", "median_clv"]
        )
    else:
        (
            monthly_df,
            surface_df,
            odds_bucket_df,
            confidence_bucket_df,
            rank_diff_bucket_df,
        ) = build_diagnostics(results_df)
    placebo_df = build_placebo_clv_summary(results_df)

    output_results_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_results_path, index=False)
    unmatched_df.to_csv(output_unmatched_path, index=False)
    ambiguous_df.to_csv(output_ambiguous_path, index=False)
    placebo_df.to_csv(output_placebo_path, index=False)

    for legacy_path in legacy_diagnostic_paths:
        # Disabled: do not delete files automatically. Remove manually if needed.
        # if legacy_path.exists():
        #     legacy_path.unlink()
        pass

    summary_text = build_summary_text(
        backtest_rows=len(predictions_df),
        odds_rows=len(odds_df),
        results_df=results_df,
        unmatched_df=unmatched_df,
        monthly_df=monthly_df,
        surface_df=surface_df,
        odds_bucket_df=odds_bucket_df,
        confidence_bucket_df=confidence_bucket_df,
        rank_diff_bucket_df=rank_diff_bucket_df,
        placebo_df=placebo_df,
    )
    print(summary_text)
    print(f"output path: {output_results_path}")
    print(f"unmatched path: {output_unmatched_path}")
    print(f"ambiguous path: {output_ambiguous_path}")
    print(f"placebo path: {output_placebo_path}")


if __name__ == "__main__":
    main()
