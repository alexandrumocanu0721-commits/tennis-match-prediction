# =============================================================================
# tennis_pipeline.py — Shared constants, I/O paths, and match-state simulation
# =============================================================================
# Used by build_features.py (build features.csv) and predict_match.py (replay
# history, then score). Keeps FEATURE order, Elo math, and ranking logic aligned
# so training and inference see the same definitions.
#
# Temporal split for train_model / evaluate_model (match ``date`` from features.csv):
#   • Train: date < MODEL_EVAL_CUTOFF_DATE (everything strictly before start of 2026).
#   • Test:  date >= MODEL_EVAL_CUTOFF_DATE (from 2026-01-01 onward, any later year).
# features.csv rows are chronological by construction (build_features replay order);
# the split only needs parsed dates, not row order.
# =============================================================================

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

# --- Feature / label columns (order must match XGBoost training matrix) ---
FEATURES: list[str] = [
    "elo_diff",
    "surface_elo_diff",
    "is_grand_slam",
    "is_masters",
    "is_atp_open",
    "rank_diff",
    "points_diff",
    "rank_momentum_diff",
    "recent_form_diff",
    "recent_surface_form_diff",
    "win_pct_diff",
    "matches_played_diff",
    "deciding_set_win_rate_diff",
    "serve_rating_diff",
    "return_rating_diff",
    "bp_save_rate_diff",
    "bp_convert_rate_diff",
    "serve_rating_surface_diff",
    "return_rating_surface_diff",
    "bp_save_rate_surface_diff",
    "bp_convert_rate_surface_diff",
    "h2h_win_rate_diff",
    "h2h_surface_win_rate_diff",
]
TARGET = "result"

# --- Elo update strength (standard scale) ---
ELO_K = 32

# --- Rank momentum lookback window (weeks) ---
RANK_MOMENTUM_WEEKS = 12

# --- Temporal split: single calendar boundary (ISO date, midnight) ---
# Train / Optuna never see rows at or after this instant; evaluation uses this instant onward.
MODEL_EVAL_CUTOFF_DATE = "2026-01-01"

# Portion of pre-cutoff training dates reserved for Optuna temporal validation.
MODEL_TUNING_VALIDATION_FRACTION = 0.20


def project_root() -> Path:
    """Return repo root (directory above ``scripts/``)."""
    return Path(__file__).resolve().parent.parent


def path_raw_dir(root: Path | None = None) -> Path:
    """Directory containing raw ATP CSVs (matches + rankings)."""
    return (root or project_root()) / "data" / "raw"


def path_processed_features_csv(root: Path | None = None) -> Path:
    """Processed feature matrix produced by ``build_features.py``."""
    return (root or project_root()) / "data" / "processed" / "features.csv"


def path_predict_today_csv(root: Path | None = None) -> Path:
    """Input: upcoming matches (player_a, player_b, surface)."""
    return (root or project_root()) / "data" / "predict" / "today_matches.csv"


def path_predict_output_csv(root: Path | None = None) -> Path:
    """Output: batch predictions from ``predict_match.py``."""
    return (root or project_root()) / "data" / "predict" / "predictions.csv"


def path_processed_backtest_predictions_csv(root: Path | None = None) -> Path:
    """Output: retroactive predictions from ``backtest_predictions.py``."""
    return (root or project_root()) / "data" / "processed" / "backtest_predictions.csv"


def path_processed_clv_results_csv(root: Path | None = None) -> Path:
    """Output: CLV report built from backtest predictions and historical odds."""
    return (root or project_root()) / "data" / "processed" / "clv_results.csv"


def path_backtest_real_2026_odds_csv(root: Path | None = None) -> Path:
    """Historical odds snapshot used to benchmark backtest predictions."""
    return (root or project_root()) / "data" / "backtest" / "real_2026_odds.csv"


def path_trained_model_pkl(root: Path | None = None) -> Path:
    """Serialized XGBoost from ``train_model.py``."""
    return (root or project_root()) / "models" / "xgboost_model.pkl"


def load_match_history_csvs(root: Path | None = None) -> pd.DataFrame:
    """Load and concatenate all ``atp_matches_*.csv`` files (not yet sorted)."""
    raw = path_raw_dir(root)
    csv_files = sorted(raw.glob("atp_matches_*.csv"))
    dfs = [pd.read_csv(file) for file in csv_files]
    return pd.concat(dfs, ignore_index=True)


def load_rankings_csv(root: Path | None = None) -> pd.DataFrame:
    """Load ATP rankings table used for rank/points as-of match date."""
    return pd.read_csv(path_raw_dir(root) / "atp_rankings_20s.csv")


def prepare_matches_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Parse ``tourney_date`` and sort matches chronologically (required for no leakage)."""
    out = df.copy()
    out["tourney_date"] = pd.to_datetime(out["tourney_date"], format="%Y%m%d")
    return out.sort_values("tourney_date")


def prepare_rankings_dataframe(rankings_df: pd.DataFrame) -> pd.DataFrame:
    """Parse ``ranking_date`` and sort globally by date (stable for tie-breaks)."""
    out = rankings_df.copy()
    out["ranking_date"] = pd.to_datetime(out["ranking_date"], format="%Y%m%d")
    return out.sort_values("ranking_date")


def build_rankings_by_player(rankings_df: pd.DataFrame) -> dict[Any, pd.DataFrame]:
    """Split rankings into one time-ordered frame per ``player`` (faster lookups).

    ``get_player_ranking`` only scans that player's rows instead of the full table.
    """
    by_player: dict[Any, pd.DataFrame] = {}
    for player_id, group in rankings_df.groupby("player", sort=False):
        by_player[player_id] = group.sort_values("ranking_date").reset_index(
            drop=True
        )
    return by_player


def get_player_ranking(
    player_id: Any,
    match_date: pd.Timestamp,
    rankings_by_player: dict[Any, pd.DataFrame],
) -> tuple[Any, Any]:
    """Latest ATP rank and points on or before ``match_date``; defaults if missing."""
    history = rankings_by_player.get(player_id)
    if history is None or history.empty:
        return 2000, 0

    player_history = history[history["ranking_date"] <= match_date]
    if player_history.empty:
        return 2000, 0

    latest_entry = player_history.iloc[-1]
    return latest_entry["rank"], latest_entry["points"]


def _get_player_rank_points_asof(
    player_id: Any,
    match_date: pd.Timestamp,
    rankings_by_player: dict[Any, pd.DataFrame],
) -> tuple[int, int] | None:
    """Return the most recent rank/points on or before ``match_date``."""
    history = rankings_by_player.get(player_id)
    if history is None or history.empty:
        return None

    player_history = history[history["ranking_date"] <= match_date]
    if player_history.empty:
        return None

    latest_entry = player_history.iloc[-1]
    return int(latest_entry["rank"]), int(latest_entry["points"])


def compute_rank_momentum(
    player_id: Any,
    match_date: pd.Timestamp,
    rankings_by_player: dict[Any, pd.DataFrame],
    weeks_back: int = RANK_MOMENTUM_WEEKS,
) -> float:
    """Percentage change in ranking points over the lookback window."""
    current_snapshot = _get_player_rank_points_asof(
        player_id, match_date, rankings_by_player
    )
    if current_snapshot is None:
        return 0.0

    past_date = match_date - pd.Timedelta(weeks=weeks_back)
    past_snapshot = _get_player_rank_points_asof(
        player_id, past_date, rankings_by_player
    )
    if past_snapshot is None:
        return 0.0

    current_points = current_snapshot[1]
    past_points = past_snapshot[1]
    return (current_points - past_points) / (past_points + 1)


def resolve_rank_points_for_player(
    player_id: Any,
    match_date: pd.Timestamp,
    rankings_by_player: dict[Any, pd.DataFrame],
) -> tuple[int, int]:
    """ATP rank/points for this player at ``match_date``.

    Uses the latest weekly row in ``atp_rankings_20s`` on or before ``match_date``.
    This keeps rank/points logic consistent between training feature generation and
    live inference.
    """
    rank, points = get_player_ranking(player_id, match_date, rankings_by_player)
    return int(rank), int(points)


def refresh_two_players_rankings(
    players: dict,
    rankings_by_player: dict[Any, pd.DataFrame],
    left_id: Any,
    right_id: Any,
    match_date: pd.Timestamp,
) -> None:
    """Overwrite stored ``rank``/``points`` before building rank/points feature diffs."""
    lr, lp = resolve_rank_points_for_player(left_id, match_date, rankings_by_player)
    rr, rp = resolve_rank_points_for_player(right_id, match_date, rankings_by_player)
    players[left_id]["rank"] = lr
    players[left_id]["points"] = lp
    players[right_id]["rank"] = rr
    players[right_id]["points"] = rp


def reference_date_for_prediction_row(
    row: Any,
    predict_df_columns: pd.Index,
    default_as_of: pd.Timestamp,
) -> pd.Timestamp:
    """Optional per-row event date from ``today_matches``.

    Prefers ``date`` (recommended), then ``match_date`` or ``tourney_date`` for
    backward compatibility. Falls back to ``default_as_of`` when all are absent
    or empty.
    """
    for col in ("date", "match_date", "tourney_date"):
        if col not in predict_df_columns:
            continue
        raw = getattr(row, col, None)
        if raw is None or pd.isna(raw):
            continue
        parsed = pd.to_datetime(raw, errors="coerce")
        if pd.notna(parsed):
            return pd.Timestamp(parsed)
    return default_as_of


def _exponential_decay_mean(values: list, alpha: float) -> float:
    """Return an exponential-decay weighted mean over an already-sliced window."""
    if not values:
        return 0.5

    weighted_total = 0.0
    weight_sum = 0.0
    n = len(values)
    for idx, value in enumerate(values):
        weight = alpha ** (n - 1 - idx)
        weighted_total += float(value) * weight
        weight_sum += weight
    return weighted_total / weight_sum


def recent_win_rate(
    results: list, window: int = 10, alpha: float = 0.85
) -> float:
    """Decay-weighted mean of the last ``window`` binary results (1/0); 0.5 if no history yet."""
    recent = results[-window:]
    return _exponential_decay_mean(recent, alpha)


def rolling_serve_stat(
    history: list, window: int = 20, alpha: float = 0.85
) -> float:
    """Decay-weighted mean of the last ``window`` values; 0.5 if no history yet."""
    recent = history[-window:]
    return _exponential_decay_mean(recent, alpha)


def count_completed_sets(score: str) -> int:
    """Count completed sets in a score string, returning 0 on malformed input."""
    try:
        if score is None or pd.isna(score):
            return 0
        score_str = str(score).strip()
        if not score_str:
            return 0

        set_pattern = re.compile(r"^\d+-\d+(?:\(\d+\))?$")
        completed_sets = 0
        for token in score_str.split():
            if set_pattern.match(token):
                completed_sets += 1
        return completed_sets
    except Exception:
        return 0


def is_deciding_set_match(score: str, tourney_level: str) -> bool:
    """Return whether a match score implies a deciding set was played."""
    try:
        set_count = count_completed_sets(score)
        if set_count == 3:
            return str(tourney_level).strip().upper() != "G"
        if set_count == 5:
            return str(tourney_level).strip().upper() == "G"
        return False
    except Exception:
        return False


def surface_or_global_stat(
    surface_history: list,
    global_history: list,
    min_matches: int = 5,
    window: int = 20,
    alpha: float = 0.85,
) -> float:
    """Use the surface history once it is long enough, otherwise fall back to global history."""
    if len(surface_history) >= min_matches:
        return rolling_serve_stat(surface_history, window, alpha)
    return rolling_serve_stat(global_history, window, alpha)


def encode_round(round_str: str) -> int:
    """Encode ATP round labels into a compact ordinal scale."""
    if round_str is None or pd.isna(round_str):
        return 3

    round_map = {
        "R128": 1,
        "R64": 2,
        "R32": 3,
        "R16": 4,
        "QF": 5,
        "SF": 6,
        "F": 7,
        "RR": 3,
    }
    return round_map.get(str(round_str), 3)


def encode_tourney_level(tourney_level: str) -> dict[str, int]:
    """Encode tournament level as three one-hot flags with finals and unknowns as baseline."""
    if tourney_level is None or pd.isna(tourney_level):
        return {
            "is_grand_slam": 0,
            "is_masters": 0,
            "is_atp_open": 0,
        }

    level = str(tourney_level).strip().upper()
    return {
        "is_grand_slam": int(level == "G"),
        "is_masters": int(level == "M"),
        "is_atp_open": int(level == "A"),
    }


def initialize_player(
    players: dict,
    rankings_by_player: dict[Any, pd.DataFrame],
    player_id: Any,
    player_name: str,
    match_date: pd.Timestamp,
) -> None:
    """Create default state the first time we see ``player_id`` (rank/points set from file)."""
    if player_id in players:
        return

    rank, points = get_player_ranking(
        player_id,
        match_date,
        rankings_by_player,
    )

    players[player_id] = {
        "name": player_name,
        "elo": 1500,
        "rank": int(rank),
        "points": int(points),
        "surface_elo": {
            "Hard": 1500,
            "Clay": 1500,
            "Grass": 1500,
        },
        "recent_results": [],
        "surface_recent_results": {
            "Hard": [],
            "Clay": [],
            "Grass": [],
        },
        "serve_rating_history": [],
        "return_rating_history": [],
        "bp_save_rate_history": [],
        "bp_convert_rate_history": [],
        "deciding_set_results": [],
        "surface_serve_rating_history": {
            "Hard": [],
            "Clay": [],
            "Grass": [],
        },
        "surface_return_rating_history": {
            "Hard": [],
            "Clay": [],
            "Grass": [],
        },
        "surface_bp_save_rate_history": {
            "Hard": [],
            "Clay": [],
            "Grass": [],
        },
        "surface_bp_convert_rate_history": {
            "Hard": [],
            "Clay": [],
            "Grass": [],
        },
        "matches_played": 0,
        "wins": 0,
        "last_match_date": None,
        "recent_minutes": [],
        "tournament_round": 0,
    }


def initialize_h2h_records() -> dict:
    """Create empty head-to-head state."""
    return {}


def compute_serve_return_stats(row: Any) -> dict[str, dict[str, float]]:
    """Compute match serve/return stats from raw ATP match columns."""
    row_dict = row._asdict()

    def safe_value(key: str) -> float | None:
        value = row_dict.get(key)
        if value is None or pd.isna(value):
            return None
        return value

    def safe_ratio_value(numerator: float | None, denominator: float | None) -> float:
        if numerator is None or denominator is None or denominator == 0:
            return 0.5
        return numerator / denominator

    def serve_rating(prefix: str) -> float:
        first_in = safe_ratio_value(
            safe_value(f"{prefix}_1stIn"), safe_value(f"{prefix}_svpt")
        )
        first_won = safe_ratio_value(
            safe_value(f"{prefix}_1stWon"), safe_value(f"{prefix}_1stIn")
        )
        second_denominator = safe_value(f"{prefix}_svpt")
        first_in_raw = safe_value(f"{prefix}_1stIn")
        second_won = safe_ratio_value(
            safe_value(f"{prefix}_2ndWon"),
            None
            if second_denominator is None or first_in_raw is None
            else second_denominator - first_in_raw,
        )
        return (first_in * 0.4) + (first_won * 0.35) + (second_won * 0.25)

    def bp_save_rate(prefix: str) -> float:
        numerator = safe_value(f"{prefix}_bpSaved")
        denominator = safe_value(f"{prefix}_bpFaced")
        if numerator is None or denominator is None or denominator == 0:
            return 0.5
        return numerator / denominator

    winner_serve_rating = serve_rating("w")
    loser_serve_rating = serve_rating("l")
    winner_bp_save_rate = bp_save_rate("w")
    loser_bp_save_rate = bp_save_rate("l")
    winner_svpt = safe_value("w_svpt")
    serve_data_valid = winner_svpt is not None and winner_svpt > 0

    return {
        "valid": serve_data_valid,
        "winner": {
            "serve_rating": winner_serve_rating,
            "return_rating": 1 - loser_serve_rating,
            "bp_save_rate": winner_bp_save_rate,
            "bp_convert_rate": 1 - loser_bp_save_rate,
        },
        "loser": {
            "serve_rating": loser_serve_rating,
            "return_rating": 1 - winner_serve_rating,
            "bp_save_rate": loser_bp_save_rate,
            "bp_convert_rate": 1 - winner_bp_save_rate,
        },
    }


def compute_fatigue_stats(row: Any, players: dict, winner: Any, loser: Any) -> dict[str, float]:
    """Compute pre-match fatigue deltas from current row and player state."""
    row_dict = row._asdict()
    match_date = row_dict.get("tourney_date")
    if match_date is None or pd.isna(match_date):
        match_date = None

    def safe_days_rest(player_id: Any) -> float:
        last_match_date = players[player_id]["last_match_date"]
        if last_match_date is None or match_date is None:
            return 30.0
        try:
            days_rest = (match_date - last_match_date).days
        except Exception:
            return 30.0
        if days_rest is None or pd.isna(days_rest):
            return 30.0
        return float(min(days_rest, 30))

    def safe_fatigue_minutes(player_id: Any) -> float:
        history = players[player_id]["recent_minutes"]
        if len(history) == 0:
            return 90.0
        return float(rolling_serve_stat(history, window=5))

    days_rest_winner = safe_days_rest(winner)
    days_rest_loser = safe_days_rest(loser)
    fatigue_minutes_winner = safe_fatigue_minutes(winner)
    fatigue_minutes_loser = safe_fatigue_minutes(loser)

    return {
        "days_rest_diff": days_rest_winner - days_rest_loser,
        "fatigue_minutes_diff": fatigue_minutes_winner - fatigue_minutes_loser,
    }


def fill_player_names_from_matches(df: pd.DataFrame) -> dict[Any, str]:
    """Build ``player_id`` → official name map from every historical match row."""
    player_names: dict[Any, str] = {}
    for row in df.itertuples(index=False):
        player_names[row.winner_id] = row.winner_name
        player_names[row.loser_id] = row.loser_name
    return player_names


def compute_winner_perspective_diffs(
    winner: Any,
    loser: Any,
    match_date: pd.Timestamp,
    surface: str,
    players: dict,
    rankings_by_player: dict[Any, pd.DataFrame],
    h2h_diffs: dict | None = None,
    fatigue_stats: dict | None = None,
) -> dict[str, float]:
    """All feature deltas from the real winner’s perspective vs loser (pre-match snapshot)."""
    winner_recent_form = recent_win_rate(players[winner]["recent_results"])
    loser_recent_form = recent_win_rate(players[loser]["recent_results"])
    recent_form_diff = winner_recent_form - loser_recent_form

    winner_surface_form = recent_win_rate(
        players[winner]["surface_recent_results"][surface]
    )
    loser_surface_form = recent_win_rate(
        players[loser]["surface_recent_results"][surface]
    )
    recent_surface_form_diff = winner_surface_form - loser_surface_form

    winner_matches = max(players[winner]["matches_played"], 1)
    loser_matches = max(players[loser]["matches_played"], 1)
    winner_win_pct = players[winner]["wins"] / winner_matches
    loser_win_pct = players[loser]["wins"] / loser_matches
    win_pct_diff = winner_win_pct - loser_win_pct
    matches_played_diff = (
        players[winner]["matches_played"] - players[loser]["matches_played"]
    )
    deciding_set_win_rate_diff = deciding_set_win_rate(
        players[winner]["deciding_set_results"]
    ) - deciding_set_win_rate(players[loser]["deciding_set_results"])

    elo_diff = players[winner]["elo"] - players[loser]["elo"]
    surface_elo_diff = (
        players[winner]["surface_elo"][surface]
        - players[loser]["surface_elo"][surface]
    )
    rank_diff = players[loser]["rank"] - players[winner]["rank"]
    points_diff = players[winner]["points"] - players[loser]["points"]
    winner_rank_momentum = compute_rank_momentum(
        winner, match_date, rankings_by_player
    )
    loser_rank_momentum = compute_rank_momentum(
        loser, match_date, rankings_by_player
    )
    rank_momentum_diff = winner_rank_momentum - loser_rank_momentum
    serve_rating_diff = rolling_serve_stat(
        players[winner]["serve_rating_history"]
    ) - rolling_serve_stat(players[loser]["serve_rating_history"])
    return_rating_diff = rolling_serve_stat(
        players[winner]["return_rating_history"]
    ) - rolling_serve_stat(players[loser]["return_rating_history"])
    bp_save_rate_diff = rolling_serve_stat(
        players[winner]["bp_save_rate_history"]
    ) - rolling_serve_stat(players[loser]["bp_save_rate_history"])
    bp_convert_rate_diff = rolling_serve_stat(
        players[winner]["bp_convert_rate_history"]
    ) - rolling_serve_stat(players[loser]["bp_convert_rate_history"])
    serve_rating_surface_diff = surface_or_global_stat(
        players[winner]["surface_serve_rating_history"][surface],
        players[winner]["serve_rating_history"],
    ) - surface_or_global_stat(
        players[loser]["surface_serve_rating_history"][surface],
        players[loser]["serve_rating_history"],
    )
    return_rating_surface_diff = surface_or_global_stat(
        players[winner]["surface_return_rating_history"][surface],
        players[winner]["return_rating_history"],
    ) - surface_or_global_stat(
        players[loser]["surface_return_rating_history"][surface],
        players[loser]["return_rating_history"],
    )
    bp_save_rate_surface_diff = surface_or_global_stat(
        players[winner]["surface_bp_save_rate_history"][surface],
        players[winner]["bp_save_rate_history"],
    ) - surface_or_global_stat(
        players[loser]["surface_bp_save_rate_history"][surface],
        players[loser]["bp_save_rate_history"],
    )
    bp_convert_rate_surface_diff = surface_or_global_stat(
        players[winner]["surface_bp_convert_rate_history"][surface],
        players[winner]["bp_convert_rate_history"],
    ) - surface_or_global_stat(
        players[loser]["surface_bp_convert_rate_history"][surface],
        players[loser]["bp_convert_rate_history"],
    )
    h2h_win_rate_diff = 0.0
    h2h_surface_win_rate_diff = 0.0
    if h2h_diffs is not None:
        h2h_win_rate_diff = h2h_diffs.get("h2h_win_rate_diff", 0.0)
        h2h_surface_win_rate_diff = h2h_diffs.get(
            "h2h_surface_win_rate_diff", 0.0
        )

    return {
        "elo_diff": elo_diff,
        "surface_elo_diff": surface_elo_diff,
        "rank_diff": rank_diff,
        "points_diff": points_diff,
        "rank_momentum_diff": rank_momentum_diff,
        "recent_form_diff": recent_form_diff,
        "recent_surface_form_diff": recent_surface_form_diff,
        "win_pct_diff": win_pct_diff,
        "matches_played_diff": matches_played_diff,
        "deciding_set_win_rate_diff": deciding_set_win_rate_diff,
        "serve_rating_diff": serve_rating_diff,
        "return_rating_diff": return_rating_diff,
        "bp_save_rate_diff": bp_save_rate_diff,
        "bp_convert_rate_diff": bp_convert_rate_diff,
        "serve_rating_surface_diff": serve_rating_surface_diff,
        "return_rating_surface_diff": return_rating_surface_diff,
        "bp_save_rate_surface_diff": bp_save_rate_surface_diff,
        "bp_convert_rate_surface_diff": bp_convert_rate_surface_diff,
        "h2h_win_rate_diff": h2h_win_rate_diff,
        "h2h_surface_win_rate_diff": h2h_surface_win_rate_diff,
    }


def build_symmetric_training_rows(
    match_date: pd.Timestamp,
    winner: Any,
    loser: Any,
    surface: str,
    player_names: dict[Any, str],
    diffs: dict[str, float],
    tourney_level_encoded: dict[str, int],
) -> tuple[dict, dict]:
    """Return two labeled rows: player A = winner (y=1) and player A = loser (y=0), mirrored features."""
    winner_row = {
        "date": match_date,
        "player_a_id": winner,
        "player_b_id": loser,
        "player_a_name": player_names[winner],
        "player_b_name": player_names[loser],
        "surface": surface,
        "elo_diff": diffs["elo_diff"],
        "surface_elo_diff": diffs["surface_elo_diff"],
        "is_grand_slam": tourney_level_encoded["is_grand_slam"],
        "is_masters": tourney_level_encoded["is_masters"],
        "is_atp_open": tourney_level_encoded["is_atp_open"],
        "rank_diff": diffs["rank_diff"],
        "points_diff": diffs["points_diff"],
        "rank_momentum_diff": diffs["rank_momentum_diff"],
        "recent_form_diff": diffs["recent_form_diff"],
        "recent_surface_form_diff": diffs["recent_surface_form_diff"],
        "win_pct_diff": diffs["win_pct_diff"],
        "matches_played_diff": diffs["matches_played_diff"],
        "deciding_set_win_rate_diff": diffs["deciding_set_win_rate_diff"],
        "serve_rating_diff": diffs["serve_rating_diff"],
        "return_rating_diff": diffs["return_rating_diff"],
        "bp_save_rate_diff": diffs["bp_save_rate_diff"],
        "bp_convert_rate_diff": diffs["bp_convert_rate_diff"],
        "serve_rating_surface_diff": diffs["serve_rating_surface_diff"],
        "return_rating_surface_diff": diffs["return_rating_surface_diff"],
        "bp_save_rate_surface_diff": diffs["bp_save_rate_surface_diff"],
        "bp_convert_rate_surface_diff": diffs["bp_convert_rate_surface_diff"],
        "h2h_win_rate_diff": diffs["h2h_win_rate_diff"],
        "h2h_surface_win_rate_diff": diffs["h2h_surface_win_rate_diff"],
        "result": 1,
    }
    loser_row = {
        "date": match_date,
        "player_a_id": loser,
        "player_b_id": winner,
        "player_a_name": player_names[loser],
        "player_b_name": player_names[winner],
        "surface": surface,
        "elo_diff": (-1) * diffs["elo_diff"],
        "surface_elo_diff": (-1) * diffs["surface_elo_diff"],
        "is_grand_slam": tourney_level_encoded["is_grand_slam"],
        "is_masters": tourney_level_encoded["is_masters"],
        "is_atp_open": tourney_level_encoded["is_atp_open"],
        "rank_diff": (-1) * diffs["rank_diff"],
        "points_diff": (-1) * diffs["points_diff"],
        "rank_momentum_diff": (-1) * diffs["rank_momentum_diff"],
        "recent_form_diff": -diffs["recent_form_diff"],
        "recent_surface_form_diff": -diffs["recent_surface_form_diff"],
        "win_pct_diff": -diffs["win_pct_diff"],
        "matches_played_diff": -diffs["matches_played_diff"],
        "deciding_set_win_rate_diff": -diffs["deciding_set_win_rate_diff"],
        "serve_rating_diff": -diffs["serve_rating_diff"],
        "return_rating_diff": -diffs["return_rating_diff"],
        "bp_save_rate_diff": -diffs["bp_save_rate_diff"],
        "bp_convert_rate_diff": -diffs["bp_convert_rate_diff"],
        "serve_rating_surface_diff": -diffs["serve_rating_surface_diff"],
        "return_rating_surface_diff": -diffs["return_rating_surface_diff"],
        "bp_save_rate_surface_diff": -diffs["bp_save_rate_surface_diff"],
        "bp_convert_rate_surface_diff": -diffs["bp_convert_rate_surface_diff"],
        "h2h_win_rate_diff": -diffs["h2h_win_rate_diff"],
        "h2h_surface_win_rate_diff": -diffs["h2h_surface_win_rate_diff"],
        "result": 0,
    }
    return winner_row, loser_row


def append_recent_result_lists(
    players: dict,
    winner: Any,
    loser: Any,
    surface: str,
    serve_stats: dict[str, dict[str, float]] | None = None,
    serve_data_valid: bool = True,
    is_deciding_set: bool = False,
) -> None:
    """Push this match’s 1/0 outcomes onto overall and surface-specific form deques (lists)."""
    players[winner]["recent_results"].append(1)
    players[loser]["recent_results"].append(0)
    players[winner]["surface_recent_results"][surface].append(1)
    players[loser]["surface_recent_results"][surface].append(0)
    if is_deciding_set:
        players[winner]["deciding_set_results"].append(1)
        players[loser]["deciding_set_results"].append(0)
    if serve_stats is None or not serve_data_valid:
        return
    players[winner]["serve_rating_history"].append(
        serve_stats["winner"]["serve_rating"]
    )
    players[loser]["serve_rating_history"].append(
        serve_stats["loser"]["serve_rating"]
    )
    players[winner]["return_rating_history"].append(
        serve_stats["winner"]["return_rating"]
    )
    players[loser]["return_rating_history"].append(
        serve_stats["loser"]["return_rating"]
    )
    players[winner]["bp_save_rate_history"].append(
        serve_stats["winner"]["bp_save_rate"]
    )
    players[loser]["bp_save_rate_history"].append(
        serve_stats["loser"]["bp_save_rate"]
    )
    players[winner]["bp_convert_rate_history"].append(
        serve_stats["winner"]["bp_convert_rate"]
    )
    players[loser]["bp_convert_rate_history"].append(
        serve_stats["loser"]["bp_convert_rate"]
    )
    players[winner]["surface_serve_rating_history"][surface].append(
        serve_stats["winner"]["serve_rating"]
    )
    players[loser]["surface_serve_rating_history"][surface].append(
        serve_stats["loser"]["serve_rating"]
    )
    players[winner]["surface_return_rating_history"][surface].append(
        serve_stats["winner"]["return_rating"]
    )
    players[loser]["surface_return_rating_history"][surface].append(
        serve_stats["loser"]["return_rating"]
    )
    players[winner]["surface_bp_save_rate_history"][surface].append(
        serve_stats["winner"]["bp_save_rate"]
    )
    players[loser]["surface_bp_save_rate_history"][surface].append(
        serve_stats["loser"]["bp_save_rate"]
    )
    players[winner]["surface_bp_convert_rate_history"][surface].append(
        serve_stats["winner"]["bp_convert_rate"]
    )
    players[loser]["surface_bp_convert_rate_history"][surface].append(
        serve_stats["loser"]["bp_convert_rate"]
    )


def update_h2h_records(
    h2h_records: dict,
    winner: Any,
    loser: Any,
    surface: str,
    match_date: pd.Timestamp,
) -> None:
    """Append this match to the symmetric H2H history bucket."""
    key = frozenset({winner, loser})
    if key not in h2h_records:
        h2h_records[key] = []
    h2h_records[key].append((match_date, surface, winner))


def update_fatigue_state(
    players: dict,
    winner: Any,
    loser: Any,
    match_date: pd.Timestamp,
    row: Any,
) -> None:
    """Update fatigue-related player state after the match is saved."""
    row_dict = row._asdict()
    raw_minutes = row_dict.get("minutes")
    minutes = 90.0
    if raw_minutes is not None and not pd.isna(raw_minutes):
        try:
            minutes = float(raw_minutes)
        except (TypeError, ValueError):
            minutes = 90.0

    round_value = encode_round(row_dict.get("round"))
    players[winner]["last_match_date"] = match_date
    players[loser]["last_match_date"] = match_date
    players[winner]["recent_minutes"].append(minutes)
    players[loser]["recent_minutes"].append(minutes)
    players[winner]["tournament_round"] = round_value
    players[loser]["tournament_round"] = round_value


def update_elo_and_match_counts(
    players: dict, winner: Any, loser: Any, surface: str, k: int = ELO_K
) -> None:
    """Apply Elo + surface-Elo increments and update wins / matches_played after the result."""
    winner_elo = players[winner]["elo"]
    loser_elo = players[loser]["elo"]
    expected_win = 1 / (1 + 10 ** ((loser_elo - winner_elo) / 400))
    players[winner]["elo"] += k * (1 - expected_win)
    players[loser]["elo"] += k * (0 - (1 - expected_win))

    winner_surface_elo = players[winner]["surface_elo"][surface]
    loser_surface_elo = players[loser]["surface_elo"][surface]
    expected_surface_win = 1 / (
        1 + 10 ** ((loser_surface_elo - winner_surface_elo) / 400)
    )
    players[winner]["surface_elo"][surface] += k * (1 - expected_surface_win)
    players[loser]["surface_elo"][surface] += k * (0 - (1 - expected_surface_win))

    players[winner]["wins"] += 1
    players[winner]["matches_played"] += 1
    players[loser]["matches_played"] += 1


def compute_h2h_win_rate(
    h2h_records: dict,
    player_a: Any,
    player_b: Any,
    match_date: pd.Timestamp,
    surface_filter: str | None = None,
) -> float:
    """Time-decayed win rate for ``player_a`` against ``player_b``."""
    key = frozenset({player_a, player_b})
    records = h2h_records.get(key)
    if not records:
        return 0.5

    weighted_wins = 0.0
    weighted_total = 0.0
    for entry_date, entry_surface, winner_id in records:
        if entry_date >= match_date:
            continue
        if surface_filter is not None and entry_surface != surface_filter:
            continue
        years_ago = (match_date - entry_date).days / 365.25
        if years_ago <= 1:
            weight = 1.0
        elif years_ago <= 3:
            weight = 0.75
        elif years_ago <= 5:
            weight = 0.5
        else:
            weight = 0.25
        weighted_total += weight
        if winner_id == player_a:
            weighted_wins += weight

    if weighted_total < 2.0:
        return 0.5
    return weighted_wins / weighted_total


def compute_h2h_diffs(
    h2h_records: dict,
    winner: Any,
    loser: Any,
    surface: str,
    match_date: pd.Timestamp,
) -> dict[str, float]:
    """Head-to-head feature deltas from the winner's perspective."""
    return {
        "h2h_win_rate_diff": compute_h2h_win_rate(
            h2h_records, winner, loser, match_date
        )
        - compute_h2h_win_rate(h2h_records, loser, winner, match_date),
        "h2h_surface_win_rate_diff": compute_h2h_win_rate(
            h2h_records, winner, loser, match_date, surface_filter=surface
        )
        - compute_h2h_win_rate(
            h2h_records, loser, winner, match_date, surface_filter=surface
        ),
    }


def compute_player_a_perspective_features(
    player_a_id: Any,
    player_b_id: Any,
    match_date: pd.Timestamp,
    surface: str,
    players: dict,
    rankings_by_player: dict[Any, pd.DataFrame],
    tourney_level_encoded: dict[str, int],
    h2h_diffs: dict | None = None,
    fatigue_stats: dict | None = None,
) -> dict[str, float]:
    """Same features as training, with player A fixed as the “positive” side."""
    elo_diff = players[player_a_id]["elo"] - players[player_b_id]["elo"]
    surface_elo_diff = (
        players[player_a_id]["surface_elo"][surface]
        - players[player_b_id]["surface_elo"][surface]
    )
    rank_diff = (
        players[player_b_id]["rank"] - players[player_a_id]["rank"]
    )
    points_diff = (
        players[player_a_id]["points"] - players[player_b_id]["points"]
    )
    player_a_rank_momentum = compute_rank_momentum(
        player_a_id, match_date, rankings_by_player
    )
    player_b_rank_momentum = compute_rank_momentum(
        player_b_id, match_date, rankings_by_player
    )
    rank_momentum_diff = player_a_rank_momentum - player_b_rank_momentum
    recent_form_diff = recent_win_rate(
        players[player_a_id]["recent_results"]
    ) - recent_win_rate(players[player_b_id]["recent_results"])
    recent_surface_form_diff = recent_win_rate(
        players[player_a_id]["surface_recent_results"][surface]
    ) - recent_win_rate(
        players[player_b_id]["surface_recent_results"][surface]
    )
    player_a_matches = max(players[player_a_id]["matches_played"], 1)
    player_b_matches = max(players[player_b_id]["matches_played"], 1)
    win_pct_diff = (
        players[player_a_id]["wins"] / player_a_matches
        - players[player_b_id]["wins"] / player_b_matches
    )
    matches_played_diff = (
        players[player_a_id]["matches_played"]
        - players[player_b_id]["matches_played"]
    )
    deciding_set_win_rate_diff = deciding_set_win_rate(
        players[player_a_id]["deciding_set_results"]
    ) - deciding_set_win_rate(players[player_b_id]["deciding_set_results"])
    serve_rating_diff = rolling_serve_stat(
        players[player_a_id]["serve_rating_history"]
    ) - rolling_serve_stat(players[player_b_id]["serve_rating_history"])
    return_rating_diff = rolling_serve_stat(
        players[player_a_id]["return_rating_history"]
    ) - rolling_serve_stat(players[player_b_id]["return_rating_history"])
    bp_save_rate_diff = rolling_serve_stat(
        players[player_a_id]["bp_save_rate_history"]
    ) - rolling_serve_stat(players[player_b_id]["bp_save_rate_history"])
    bp_convert_rate_diff = rolling_serve_stat(
        players[player_a_id]["bp_convert_rate_history"]
    ) - rolling_serve_stat(players[player_b_id]["bp_convert_rate_history"])
    serve_rating_surface_diff = surface_or_global_stat(
        players[player_a_id]["surface_serve_rating_history"][surface],
        players[player_a_id]["serve_rating_history"],
    ) - surface_or_global_stat(
        players[player_b_id]["surface_serve_rating_history"][surface],
        players[player_b_id]["serve_rating_history"],
    )
    return_rating_surface_diff = surface_or_global_stat(
        players[player_a_id]["surface_return_rating_history"][surface],
        players[player_a_id]["return_rating_history"],
    ) - surface_or_global_stat(
        players[player_b_id]["surface_return_rating_history"][surface],
        players[player_b_id]["return_rating_history"],
    )
    bp_save_rate_surface_diff = surface_or_global_stat(
        players[player_a_id]["surface_bp_save_rate_history"][surface],
        players[player_a_id]["bp_save_rate_history"],
    ) - surface_or_global_stat(
        players[player_b_id]["surface_bp_save_rate_history"][surface],
        players[player_b_id]["bp_save_rate_history"],
    )
    bp_convert_rate_surface_diff = surface_or_global_stat(
        players[player_a_id]["surface_bp_convert_rate_history"][surface],
        players[player_a_id]["bp_convert_rate_history"],
    ) - surface_or_global_stat(
        players[player_b_id]["surface_bp_convert_rate_history"][surface],
        players[player_b_id]["bp_convert_rate_history"],
    )
    h2h_win_rate_diff = 0.0
    h2h_surface_win_rate_diff = 0.0
    if h2h_diffs is not None:
        h2h_win_rate_diff = h2h_diffs.get("h2h_win_rate_diff", 0.0)
        h2h_surface_win_rate_diff = h2h_diffs.get(
            "h2h_surface_win_rate_diff", 0.0
        )
    return {
        "elo_diff": elo_diff,
        "surface_elo_diff": surface_elo_diff,
        "is_grand_slam": tourney_level_encoded["is_grand_slam"],
        "is_masters": tourney_level_encoded["is_masters"],
        "is_atp_open": tourney_level_encoded["is_atp_open"],
        "rank_diff": rank_diff,
        "points_diff": points_diff,
        "rank_momentum_diff": rank_momentum_diff,
        "recent_form_diff": recent_form_diff,
        "recent_surface_form_diff": recent_surface_form_diff,
        "win_pct_diff": win_pct_diff,
        "matches_played_diff": matches_played_diff,
        "deciding_set_win_rate_diff": deciding_set_win_rate_diff,
        "serve_rating_diff": serve_rating_diff,
        "return_rating_diff": return_rating_diff,
        "bp_save_rate_diff": bp_save_rate_diff,
        "bp_convert_rate_diff": bp_convert_rate_diff,
        "serve_rating_surface_diff": serve_rating_surface_diff,
        "return_rating_surface_diff": return_rating_surface_diff,
        "bp_save_rate_surface_diff": bp_save_rate_surface_diff,
        "bp_convert_rate_surface_diff": bp_convert_rate_surface_diff,
        "h2h_win_rate_diff": h2h_win_rate_diff,
        "h2h_surface_win_rate_diff": h2h_surface_win_rate_diff,
    }


def deciding_set_win_rate(
    results: list, window: int = 20, min_matches: int = 3, alpha: float = 0.85
) -> float:
    """Decay-weighted deciding-set win rate with a small-sample fallback."""
    if len(results) < min_matches:
        return 0.5
    return rolling_serve_stat(results[-window:], window, alpha)


def temporal_train_test_split_for_modeling(
    df: pd.DataFrame, date_column: str = "date"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train on ``date < MODEL_EVAL_CUTOFF_DATE``; test on ``date >=`` that cutoff.

    Matches “all history until the start of 2026” vs “from 2026 onward”. Later
    seasons (2027+, if present in ``features.csv``) stay in the evaluation set.
    """
    dates = pd.to_datetime(df[date_column])
    cutoff = pd.Timestamp(MODEL_EVAL_CUTOFF_DATE)
    train_mask = dates < cutoff
    test_mask = dates >= cutoff
    return df[train_mask], df[test_mask]


def temporal_train_validation_split(
    train_df: pd.DataFrame,
    date_column: str = "date",
    validation_fraction: float = MODEL_TUNING_VALIDATION_FRACTION,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split pre-cutoff training rows into earlier-train and later-validation windows."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1.")

    dates = pd.to_datetime(train_df[date_column])
    unique_dates = sorted(dates.dropna().unique())
    if len(unique_dates) < 2:
        raise ValueError("Need at least two unique dates to create temporal validation split.")

    split_index = int(len(unique_dates) * (1 - validation_fraction))
    split_index = max(1, min(split_index, len(unique_dates) - 1))
    validation_start = pd.Timestamp(unique_dates[split_index])

    fit_mask = dates < validation_start
    val_mask = dates >= validation_start

    fit_df = train_df[fit_mask]
    val_df = train_df[val_mask]
    if fit_df.empty or val_df.empty:
        raise ValueError("Temporal validation split produced an empty partition.")

    return fit_df, val_df
