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

from pathlib import Path
from typing import Any

import pandas as pd

# --- Feature / label columns (order must match XGBoost training matrix) ---
FEATURES: list[str] = [
    "elo_diff",
    "surface_elo_diff",
    "rank_diff",
    "points_diff",
    "recent_form_diff",
    "recent_surface_form_diff",
    "win_pct_diff",
    "matches_played_diff",
]
TARGET = "result"

# --- Elo update strength (standard scale) ---
ELO_K = 32

# --- Temporal split: single calendar boundary (ISO date, midnight) ---
# Train / Optuna never see rows at or after this instant; evaluation uses this instant onward.
MODEL_EVAL_CUTOFF_DATE = "2026-01-01"


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


def recent_win_rate(results: list, window: int = 10) -> float:
    """Mean of the last ``window`` binary results (1/0); 0.5 if no history yet."""
    if len(results) == 0:
        return 0.5
    recent = results[-window:]
    return sum(recent) / len(recent)


def initialize_player(
    players: dict,
    rankings_by_player: dict[Any, pd.DataFrame],
    player_id: Any,
    player_name: str,
    match_date: pd.Timestamp,
) -> None:
    """Create default state + rank/points from rankings the first time we see ``player_id``."""
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
        "matches_played": 0,
        "wins": 0,
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
    surface: str,
    players: dict,
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

    elo_diff = players[winner]["elo"] - players[loser]["elo"]
    surface_elo_diff = (
        players[winner]["surface_elo"][surface]
        - players[loser]["surface_elo"][surface]
    )
    rank_diff = players[loser]["rank"] - players[winner]["rank"]
    points_diff = players[winner]["points"] - players[loser]["points"]

    return {
        "elo_diff": elo_diff,
        "surface_elo_diff": surface_elo_diff,
        "rank_diff": rank_diff,
        "points_diff": points_diff,
        "recent_form_diff": recent_form_diff,
        "recent_surface_form_diff": recent_surface_form_diff,
        "win_pct_diff": win_pct_diff,
        "matches_played_diff": matches_played_diff,
    }


def build_symmetric_training_rows(
    match_date: pd.Timestamp,
    winner: Any,
    loser: Any,
    surface: str,
    player_names: dict[Any, str],
    diffs: dict[str, float],
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
        "rank_diff": diffs["rank_diff"],
        "points_diff": diffs["points_diff"],
        "recent_form_diff": diffs["recent_form_diff"],
        "recent_surface_form_diff": diffs["recent_surface_form_diff"],
        "win_pct_diff": diffs["win_pct_diff"],
        "matches_played_diff": diffs["matches_played_diff"],
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
        "rank_diff": (-1) * diffs["rank_diff"],
        "points_diff": (-1) * diffs["points_diff"],
        "recent_form_diff": -diffs["recent_form_diff"],
        "recent_surface_form_diff": -diffs["recent_surface_form_diff"],
        "win_pct_diff": -diffs["win_pct_diff"],
        "matches_played_diff": -diffs["matches_played_diff"],
        "result": 0,
    }
    return winner_row, loser_row


def append_recent_result_lists(
    players: dict, winner: Any, loser: Any, surface: str
) -> None:
    """Push this match’s 1/0 outcomes onto overall and surface-specific form deques (lists)."""
    players[winner]["recent_results"].append(1)
    players[loser]["recent_results"].append(0)
    players[winner]["surface_recent_results"][surface].append(1)
    players[loser]["surface_recent_results"][surface].append(0)


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


def compute_player_a_perspective_features(
    player_a_id: Any,
    player_b_id: Any,
    surface: str,
    players: dict,
) -> dict[str, float]:
    """Same eight features as training, with player A fixed as the “positive” side."""
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
    return {
        "elo_diff": elo_diff,
        "surface_elo_diff": surface_elo_diff,
        "rank_diff": rank_diff,
        "points_diff": points_diff,
        "recent_form_diff": recent_form_diff,
        "recent_surface_form_diff": recent_surface_form_diff,
        "win_pct_diff": win_pct_diff,
        "matches_played_diff": matches_played_diff,
    }


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
