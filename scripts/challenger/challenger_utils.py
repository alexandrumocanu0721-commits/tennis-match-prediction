from __future__ import annotations

from pathlib import Path
import re
from typing import Any

import pandas as pd

from challenger_config import (
    CHALLENGER_ALLOWED_TOURNEY_LEVEL,
    CHALLENGER_DEFAULT_PLAYER_POINTS,
    CHALLENGER_DEFAULT_PLAYER_RANK,
    CHALLENGER_INVALID_OUTCOME_REGEX,
    CHALLENGER_MATCH_GLOB,
    CHALLENGER_MATCH_HISTORY_START_YEAR,
    path_challenger_raw_dir,
    path_external_atp_rankings_csv,
)
from tennis_pipeline import (
    build_rankings_by_player,
    fill_player_names_from_matches,
    initialize_player,
    prepare_matches_dataframe,
    prepare_rankings_dataframe,
    refresh_two_players_rankings,
)


def _assert_challenger_match_path(path: Path, raw_dir: Path) -> None:
    resolved_path = path.resolve()
    resolved_raw_dir = raw_dir.resolve()
    if resolved_raw_dir not in resolved_path.parents:
        raise ValueError(f"Unsafe match path outside Challenger raw dir: {path}")
    if not path.name.startswith("atp_matches_qual_chall_"):
        raise ValueError(f"Unsafe Challenger filename: {path.name}")


def _year_from_challenger_match_filename(path: Path) -> int | None:
    match = re.fullmatch(r"atp_matches_qual_chall_(\d{4})\.csv", path.name)
    return int(match.group(1)) if match else None


def load_challenger_match_history_csvs(root: Path | None = None) -> pd.DataFrame:
    raw_dir = path_challenger_raw_dir(root)
    csv_files = [
        file
        for file in sorted(raw_dir.glob(CHALLENGER_MATCH_GLOB))
        if (_year_from_challenger_match_filename(file) or 0)
        >= CHALLENGER_MATCH_HISTORY_START_YEAR
    ]
    if not csv_files:
        raise FileNotFoundError(
            "No Challenger match files found in "
            f"{raw_dir} with glob {CHALLENGER_MATCH_GLOB} "
            f"from {CHALLENGER_MATCH_HISTORY_START_YEAR} onward"
        )
    frames: list[pd.DataFrame] = []
    for csv_path in csv_files:
        _assert_challenger_match_path(csv_path, raw_dir)
        frames.append(pd.read_csv(csv_path))
    return pd.concat(frames, ignore_index=True)


def prepare_challenger_matches_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    return prepare_matches_dataframe(df)


def filter_challenger_matches(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out[
        out["tourney_level"].astype(str).str.strip().str.upper()
        == CHALLENGER_ALLOWED_TOURNEY_LEVEL
    ]
    out = out[
        ~out["score"].astype(str).str.contains(
            CHALLENGER_INVALID_OUTCOME_REGEX,
            case=False,
            na=False,
        )
    ]
    out = out.dropna(subset=["winner_id", "loser_id", "surface", "tourney_date"])
    # Keep player ids on an integer keyspace so ranking lookups are stable.
    out["winner_id"] = pd.to_numeric(out["winner_id"], errors="coerce").astype("Int64")
    out["loser_id"] = pd.to_numeric(out["loser_id"], errors="coerce").astype("Int64")
    out = out.dropna(subset=["winner_id", "loser_id"])
    out["winner_id"] = out["winner_id"].astype("int64")
    out["loser_id"] = out["loser_id"].astype("int64")
    return out


def build_challenger_player_name_map(df: pd.DataFrame) -> dict[Any, str]:
    return fill_player_names_from_matches(df)


def challenger_tourney_level_encoding(_tourney_level: str) -> dict[str, int]:
    return {
        "is_grand_slam": 0,
        "is_masters": 0,
        "is_atp_open": 0,
    }


def neutral_rankings_context() -> dict[Any, pd.DataFrame]:
    return {}


def load_external_atp_rankings_context(
    root: Path | None = None,
) -> dict[Any, pd.DataFrame]:
    rankings_df = load_external_atp_rankings_dataframe(root)
    return build_external_atp_rankings_context(rankings_df)


def load_external_atp_rankings_dataframe(root: Path | None = None) -> pd.DataFrame:
    rankings_path = path_external_atp_rankings_csv(root)
    rankings_df = pd.read_csv(rankings_path)
    rankings_df["player"] = (
        pd.to_numeric(rankings_df["player"], errors="coerce").astype("Int64")
    )
    rankings_df = rankings_df.dropna(subset=["player"])
    rankings_df["player"] = rankings_df["player"].astype("int64")
    return prepare_rankings_dataframe(rankings_df)


def build_external_atp_rankings_context(
    rankings_df: pd.DataFrame,
) -> dict[Any, pd.DataFrame]:
    return build_rankings_by_player(rankings_df)


def initialize_challenger_player(
    players: dict,
    player_id: Any,
    player_name: str,
    match_date: pd.Timestamp,
    rankings_by_player: dict[Any, pd.DataFrame] | None = None,
    use_external_rankings: bool = False,
) -> None:
    rankings = rankings_by_player if rankings_by_player is not None else {}
    initialize_player(players, rankings, player_id, player_name, match_date)
    if not use_external_rankings:
        players[player_id]["rank"] = CHALLENGER_DEFAULT_PLAYER_RANK
        players[player_id]["points"] = CHALLENGER_DEFAULT_PLAYER_POINTS


def refresh_challenger_player_rankings(
    players: dict,
    left_id: Any,
    right_id: Any,
    match_date: pd.Timestamp,
    rankings_by_player: dict[Any, pd.DataFrame] | None = None,
    use_external_rankings: bool = False,
) -> None:
    rankings = rankings_by_player if rankings_by_player is not None else {}
    refresh_two_players_rankings(
        players=players,
        rankings_by_player=rankings,
        left_id=left_id,
        right_id=right_id,
        match_date=match_date,
    )
    if not use_external_rankings:
        players[left_id]["rank"] = CHALLENGER_DEFAULT_PLAYER_RANK
        players[left_id]["points"] = CHALLENGER_DEFAULT_PLAYER_POINTS
        players[right_id]["rank"] = CHALLENGER_DEFAULT_PLAYER_RANK
        players[right_id]["points"] = CHALLENGER_DEFAULT_PLAYER_POINTS
