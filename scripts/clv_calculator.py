from __future__ import annotations

import re
import unicodedata

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from tennis_pipeline import (
    path_backtest_real_2026_odds_csv,
    path_processed_backtest_predictions_csv,
    path_processed_clv_results_csv,
    project_root,
)


def normalize_text(value: object) -> str:
    """ASCII-fold and strip punctuation so name and surface keys compare cleanly."""
    if pd.isna(value):
        return ""

    text = unicodedata.normalize("NFKD", str(value))
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.replace(".", " ").replace(",", " ").replace("-", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def name_match_key(name: object) -> str:
    """Reduce a player name to ``last-name|first-initial`` for fuzzy matching.

    Examples:
        ``Jannik Sinner`` -> ``SINNER|J``
        ``Sinner J.`` -> ``SINNER|J``
        ``Pablo Carreno Busta`` -> ``CARRENO BUSTA|P``
    """
    text = normalize_text(name)
    if not text:
        return ""

    tokens = text.split()
    if len(tokens) == 1:
        surname = tokens[0]
        first_initial = ""
    elif len(tokens[-1]) == 1:
        surname = " ".join(tokens[:-1])
        first_initial = tokens[-1][0]
    else:
        surname = " ".join(tokens[1:])
        first_initial = tokens[0][0]

    return f"{surname.upper()}|{first_initial.upper()}"


def surface_key(surface: object) -> str:
    """Normalize surface labels for a soft sanity check only."""
    return normalize_text(surface).upper()


def true_prob_from_pair(winner_odd: object, loser_odd: object) -> tuple[float, float]:
    """Remove the bookmaker margin from a two-way market."""
    winner = pd.to_numeric(str(winner_odd).replace(",", "."), errors="coerce")
    loser = pd.to_numeric(str(loser_odd).replace(",", "."), errors="coerce")

    if pd.isna(winner) or pd.isna(loser) or winner <= 0 or loser <= 0:
        return float("nan"), float("nan")

    raw_prob_winner = 1.0 / float(winner)
    raw_prob_loser = 1.0 / float(loser)
    overround = raw_prob_winner + raw_prob_loser

    if overround <= 0:
        return float("nan"), float("nan")

    return raw_prob_winner / overround, raw_prob_loser / overround


def add_benchmark_probabilities(
    df: pd.DataFrame, benchmark_name: str, odds_prefix: str
) -> pd.DataFrame:
    """Attach winner/loser true probabilities for one odds source."""
    winner_col = f"{odds_prefix}W"
    loser_col = f"{odds_prefix}L"

    winner_probs = []
    loser_probs = []
    for winner_odd, loser_odd in zip(df[winner_col], df[loser_col]):
        winner_true, loser_true = true_prob_from_pair(winner_odd, loser_odd)
        winner_probs.append(winner_true)
        loser_probs.append(loser_true)

    df[f"{benchmark_name}_true_prob_winner"] = winner_probs
    df[f"{benchmark_name}_true_prob_loser"] = loser_probs
    return df


def match_single_prediction(
    prediction_row: pd.Series, odds_df: pd.DataFrame
) -> tuple[dict | None, pd.DataFrame, pd.DataFrame]:
    """Find exactly one odds row for a single prediction, or skip it.

    The search is:
    1. Collect every odds row within +/- 3 days of the prediction date.
    2. Among those, keep rows whose player-name keys match exactly, ignoring
       winner/loser order.
    3. Use the row only if exactly one candidate matches.
    """
    date_diff_days = (odds_df["odds_date"] - prediction_row["date"]).dt.days.abs()
    candidate_mask = date_diff_days <= 3
    candidate_rows = odds_df[candidate_mask].copy()
    candidate_rows["date_diff_days"] = date_diff_days[candidate_mask].astype(int)

    if candidate_rows.empty:
        return None, candidate_rows, pd.DataFrame()

    name_match_mask = (
        (candidate_rows["winner_key"] == prediction_row["player_a_key"])
        & (candidate_rows["loser_key"] == prediction_row["player_b_key"])
    ) | (
        (candidate_rows["winner_key"] == prediction_row["player_b_key"])
        & (candidate_rows["loser_key"] == prediction_row["player_a_key"])
    )
    name_matches = candidate_rows[name_match_mask].copy()

    if len(name_matches) != 1:
        return None, candidate_rows, name_matches

    odds_row = name_matches.iloc[0]
    orientation = (
        "player_a_is_winner"
        if odds_row["winner_key"] == prediction_row["player_a_key"]
        else "player_a_is_loser"
    )

    record = {
        "date": prediction_row["date"],
        "player_a": prediction_row["player_a"],
        "player_b": prediction_row["player_b"],
        "surface": prediction_row["surface"],
        "predicted_prob_a": prediction_row["predicted_prob_a"],
        "actual_winner": prediction_row["actual_winner"],
        "prediction_row_id": prediction_row["prediction_row_id"],
        "surface_key": prediction_row["surface_key"],
        "player_a_key": prediction_row["player_a_key"],
        "player_b_key": prediction_row["player_b_key"],
        "odds_row_id": odds_row["odds_row_id"],
        "odds_date": odds_row["odds_date"],
        "odds_surface_key": odds_row["odds_surface_key"],
        "Winner": odds_row["Winner"],
        "Loser": odds_row["Loser"],
        "Surface": odds_row["Surface"],
        "winner_key": odds_row["winner_key"],
        "loser_key": odds_row["loser_key"],
        "match_orientation": orientation,
        "date_diff_days": int(abs((prediction_row["date"] - odds_row["odds_date"]).days)),
        "surface_match": prediction_row["surface_key"] == odds_row["odds_surface_key"],
        "betfair_true_prob_winner": odds_row["betfair_true_prob_winner"],
        "betfair_true_prob_loser": odds_row["betfair_true_prob_loser"],
        "pinnacle_true_prob_winner": odds_row["pinnacle_true_prob_winner"],
        "pinnacle_true_prob_loser": odds_row["pinnacle_true_prob_loser"],
        "max_true_prob_winner": odds_row["max_true_prob_winner"],
        "max_true_prob_loser": odds_row["max_true_prob_loser"],
    }
    return record, candidate_rows, name_matches


def main() -> None:
    root = project_root()

    # Load the two historical views: model backtest output and the odds snapshot.
    predictions_df = pd.read_csv(path_processed_backtest_predictions_csv(root))
    odds_df = pd.read_csv(path_backtest_real_2026_odds_csv(root))

    predictions_df["date"] = pd.to_datetime(predictions_df["date"], errors="coerce").dt.normalize()
    predictions_df["surface_key"] = predictions_df["surface"].map(surface_key)
    predictions_df["player_a_key"] = predictions_df["player_a"].map(name_match_key)
    predictions_df["player_b_key"] = predictions_df["player_b"].map(name_match_key)
    predictions_df["prediction_row_id"] = range(len(predictions_df))

    odds_df["date"] = pd.to_datetime(odds_df["Date"], dayfirst=True, errors="coerce").dt.normalize()
    odds_df["odds_date"] = odds_df["date"]
    odds_df["odds_surface_key"] = odds_df["Surface"].map(surface_key)
    odds_df["winner_key"] = odds_df["Winner"].map(name_match_key)
    odds_df["loser_key"] = odds_df["Loser"].map(name_match_key)
    odds_df["odds_row_id"] = range(len(odds_df))

    for benchmark_name, odds_prefix in (
        ("betfair", "BFE"),
        ("pinnacle", "PS"),
        ("max", "Max"),
    ):
        odds_df = add_benchmark_probabilities(odds_df, benchmark_name, odds_prefix)

    matched_records: list[dict] = []
    name_mismatch_examples: list[dict] = []

    for _, prediction_row in predictions_df.iterrows():
        matched_record, candidate_rows, name_matches = match_single_prediction(
            prediction_row, odds_df
        )

        if matched_record is not None:
            matched_records.append(matched_record)
            continue

    matched_df = pd.DataFrame(matched_records)

    if matched_df.empty:
        output_columns = [
            "date",
            "player_a",
            "player_b",
            "surface",
            "predicted_prob_a",
            "betfair_true_prob_a",
            "clv_betfair",
            "pinnacle_true_prob_a",
            "clv_pinnacle",
            "max_true_prob_a",
            "clv_max",
            "actual_winner",
        ]
        empty_df = pd.DataFrame(columns=output_columns)
        empty_df.to_csv(path_processed_clv_results_csv(root), index=False)
        print("Matched matches: 0")
        print("Average CLV Betfair: n/a")
        print("Average CLV Pinnacle: n/a")
        print("Average CLV Max: n/a")
        return

    # Prefer rows whose surface label matches after normalization, then keep one row
    # per original backtest prediction.
    matched_df = matched_df.sort_values(
        ["prediction_row_id", "date_diff_days", "surface_match"],
        ascending=[True, True, False],
        kind="mergesort",
    )
    matched_df = matched_df.drop_duplicates(subset=["prediction_row_id"], keep="first")

    # Map the odds-side benchmark probabilities to player_a's side.
    matched_df["betfair_true_prob_a"] = matched_df.apply(
        lambda row: row["betfair_true_prob_winner"]
        if row["match_orientation"] == "player_a_is_winner"
        else row["betfair_true_prob_loser"],
        axis=1,
    )
    matched_df["pinnacle_true_prob_a"] = matched_df.apply(
        lambda row: row["pinnacle_true_prob_winner"]
        if row["match_orientation"] == "player_a_is_winner"
        else row["pinnacle_true_prob_loser"],
        axis=1,
    )
    matched_df["max_true_prob_a"] = matched_df.apply(
        lambda row: row["max_true_prob_winner"]
        if row["match_orientation"] == "player_a_is_winner"
        else row["max_true_prob_loser"],
        axis=1,
    )

    matched_df["clv_betfair"] = matched_df["predicted_prob_a"] - matched_df["betfair_true_prob_a"]
    matched_df["clv_pinnacle"] = matched_df["predicted_prob_a"] - matched_df["pinnacle_true_prob_a"]
    matched_df["clv_max"] = matched_df["predicted_prob_a"] - matched_df["max_true_prob_a"]

    output_df = matched_df[
        [
            "date",
            "player_a",
            "player_b",
            "surface",
            "predicted_prob_a",
            "betfair_true_prob_a",
            "clv_betfair",
            "pinnacle_true_prob_a",
            "clv_pinnacle",
            "max_true_prob_a",
            "clv_max",
            "actual_winner",
        ]
    ].copy()

    output_df["date"] = output_df["date"].dt.strftime("%Y-%m-%d")
    output_df = output_df.sort_values(["date", "player_a", "player_b"], kind="mergesort")
    output_path = path_processed_clv_results_csv(root)
    output_df.to_csv(output_path, index=False)

    print(f"Matched matches: {len(output_df)}")
    print(f"Average CLV Betfair: {output_df['clv_betfair'].mean(skipna=True):.6f}")
    print(f"Average CLV Pinnacle: {output_df['clv_pinnacle'].mean(skipna=True):.6f}")
    print(f"Average CLV Max: {output_df['clv_max'].mean(skipna=True):.6f}")
    print()
    print("CLV by surface:")
    surface_summary = output_df.groupby("surface", dropna=False)[
        ["clv_betfair", "clv_pinnacle", "clv_max"]
    ].mean(numeric_only=True)
    print(surface_summary.to_string(float_format=lambda value: f"{value:.6f}"))

    # Additional analyses operate on the persisted CLV output so they reflect
    # the exact artifact written to disk.
    clv_df = pd.read_csv(output_path, parse_dates=["date"])

    print()
    print("CLV over time by month:")
    monthly_summary = (
        clv_df.assign(month=clv_df["date"].dt.to_period("M").astype(str))
        .groupby("month")
        .agg(match_count=("clv_betfair", "size"), avg_clv_betfair=("clv_betfair", "mean"))
    )
    print(monthly_summary.to_string(float_format=lambda value: f"{value:.6f}"))

    print()
    print("CLV by probability bucket:")
    bucket_bins = [0.0, 0.4, 0.5, 0.6, 0.7, 1.0000001]
    bucket_labels = ["0.0-0.4", "0.4-0.5", "0.5-0.6", "0.6-0.7", "0.7+"]
    clv_df["probability_bucket"] = pd.cut(
        clv_df["predicted_prob_a"],
        bins=bucket_bins,
        labels=bucket_labels,
        right=False,
        include_lowest=True,
    )
    bucket_summary = (
        clv_df.groupby("probability_bucket", dropna=False)
        .agg(match_count=("clv_betfair", "size"), avg_clv_betfair=("clv_betfair", "mean"))
    )
    print(bucket_summary.to_string(float_format=lambda value: f"{value:.6f}"))

    # Cumulative average CLV over time, saved for quick visual inspection.
    cumulative_df = clv_df.sort_values("date", kind="mergesort").copy()
    cumulative_df["cumulative_avg_clv_betfair"] = cumulative_df["clv_betfair"].expanding().mean()
    plt.figure(figsize=(10, 5))
    plt.plot(cumulative_df["date"], cumulative_df["cumulative_avg_clv_betfair"])
    plt.title("Cumulative Average CLV Betfair")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Avg CLV")
    plt.tight_layout()
    plt.savefig(root / "data" / "processed" / "clv_cumulative.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
