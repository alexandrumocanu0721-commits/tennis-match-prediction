# =============================================================================
# build_features.py — Chronological ATP feature generation for model training
# =============================================================================
# Replays every match in date order, updates Elo / form, snapshots ATP rank/points
# per match date, then writes two symmetric labeled rows per match (A=winner vs A=loser).
# Output feeds train_model.py. All simulation rules live in tennis_pipeline.py.
# =============================================================================

import pandas as pd

from tennis_pipeline import (
    ELO_K,
    append_recent_result_lists,
    build_rankings_by_player,
    build_symmetric_training_rows,
    compute_winner_perspective_diffs,
    fill_player_names_from_matches,
    initialize_player,
    load_match_history_csvs,
    load_rankings_csv,
    path_processed_features_csv,
    prepare_matches_dataframe,
    prepare_rankings_dataframe,
    project_root,
    refresh_two_players_rankings,
    update_elo_and_match_counts,
)

# --- Paths: always relative to repo root (not the shell’s cwd) ---
root = project_root()

# --- Raw ATP matches + rankings, chronologically ordered ---
df = prepare_matches_dataframe(load_match_history_csvs(root))
rankings_df = prepare_rankings_dataframe(load_rankings_csv(root))
rankings_by_player = build_rankings_by_player(rankings_df)

# --- Mutable state for the chronological replay ---
players: dict = {}
player_names = fill_player_names_from_matches(df)
feature_rows: list[dict] = []

# --- One match at a time: snapshot → append form lists → save rows → update Elo ---
for row in df.itertuples(index=False):
    winner = row.winner_id
    loser = row.loser_id
    match_date = row.tourney_date
    surface = row.surface
    if pd.isna(surface):
        continue

    initialize_player(
        players, rankings_by_player, winner, player_names[winner], match_date
    )
    initialize_player(
        players, rankings_by_player, loser, player_names[loser], match_date
    )

    wr = getattr(row, "winner_rank", None)
    wp = getattr(row, "winner_rank_points", None)
    lr = getattr(row, "loser_rank", None)
    lp = getattr(row, "loser_rank_points", None)
    refresh_two_players_rankings(
        players,
        rankings_by_player,
        winner,
        loser,
        match_date,
        left_rank=wr,
        left_points=wp,
        right_rank=lr,
        right_points=lp,
    )

    diffs = compute_winner_perspective_diffs(winner, loser, surface, players)
    winner_row, loser_row = build_symmetric_training_rows(
        match_date, winner, loser, surface, player_names, diffs
    )

    append_recent_result_lists(players, winner, loser, surface)
    feature_rows.append(winner_row)
    feature_rows.append(loser_row)
    update_elo_and_match_counts(players, winner, loser, surface, k=ELO_K)

features_df = pd.DataFrame(feature_rows)
features_df.to_csv(path_processed_features_csv(root), index=False)
