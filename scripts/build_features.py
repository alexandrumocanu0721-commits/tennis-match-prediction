# =============================================================================
# build_features.py — Chronological ATP feature generation for model training
# =============================================================================
# Replays every match in date order, updates Elo / form, snapshots ATP rank/points
# from rankings history as-of each match date, then writes two symmetric labeled rows
# per match (A=winner vs A=loser).
# Output feeds train_model.py. All simulation rules live in tennis_pipeline.py.
# =============================================================================

import pandas as pd

from tennis_pipeline import (
    ELO_K,
    append_recent_result_lists,
    build_rankings_by_player,
    build_symmetric_training_rows,
    compute_h2h_diffs,
    compute_fatigue_stats,
    compute_winner_perspective_diffs,
    compute_serve_return_stats,
    is_deciding_set_match,
    encode_tourney_level,
    fill_player_names_from_matches,
    initialize_h2h_records,
    initialize_player,
    load_match_history_csvs,
    load_rankings_csv,
    path_processed_features_csv,
    prepare_matches_dataframe,
    prepare_rankings_dataframe,
    project_root,
    refresh_two_players_rankings,
    update_elo_and_match_counts,
    update_h2h_records,
    update_fatigue_state,
)

# --- Paths: always relative to repo root (not the shell’s cwd) ---
root = project_root()

# --- Raw ATP matches + rankings, chronologically ordered ---
df = prepare_matches_dataframe(load_match_history_csvs(root))
original_len = len(df)
df = df[
    ~df["score"].str.contains(
        "RET|W/O|Walkover|DEF", case=False, na=False
    )
    & ~df["tourney_level"].str.strip().str.upper().isin(["D", "O"])
]
print(f"Removed {original_len - len(df)} retirement/walkover/Davis Cup/Olympics rows")
rankings_df = prepare_rankings_dataframe(load_rankings_csv(root))
rankings_by_player = build_rankings_by_player(rankings_df)

# --- Mutable state for the chronological replay ---
players: dict = {}
player_names = fill_player_names_from_matches(df)
feature_rows: list[dict] = []
h2h_records = initialize_h2h_records()

# --- One match at a time: snapshot → build rows → save rows → update state ---
for row in df.itertuples(index=False):
    winner = row.winner_id
    loser = row.loser_id
    match_date = row.tourney_date
    surface = row.surface
    if pd.isna(surface):
        continue
    tourney_level_encoded = encode_tourney_level(row.tourney_level)

    initialize_player(
        players, rankings_by_player, winner, player_names[winner], match_date
    )
    initialize_player(
        players, rankings_by_player, loser, player_names[loser], match_date
    )

    refresh_two_players_rankings(
        players,
        rankings_by_player,
        winner,
        loser,
        match_date,
    )

    h2h_diffs = compute_h2h_diffs(h2h_records, winner, loser, surface, match_date)
    fatigue_stats = compute_fatigue_stats(row, players, winner, loser)
    diffs = compute_winner_perspective_diffs(
        winner,
        loser,
        match_date,
        surface,
        players,
        rankings_by_player,
        h2h_diffs=h2h_diffs,
        fatigue_stats=fatigue_stats,
    )
    winner_row, loser_row = build_symmetric_training_rows(
        match_date,
        winner,
        loser,
        surface,
        player_names,
        diffs,
        tourney_level_encoded,
    )

    serve_stats = compute_serve_return_stats(row)
    deciding_set = is_deciding_set_match(row.score, row.tourney_level)
    append_recent_result_lists(
        players,
        winner,
        loser,
        surface,
        serve_stats,
        serve_data_valid=serve_stats["valid"],
        is_deciding_set=deciding_set,
    )
    feature_rows.append(winner_row)
    feature_rows.append(loser_row)
    update_elo_and_match_counts(players, winner, loser, surface, k=ELO_K)
    update_h2h_records(h2h_records, winner, loser, surface, match_date)
    update_fatigue_state(players, winner, loser, match_date, row)

features_df = pd.DataFrame(feature_rows)
features_df.to_csv(path_processed_features_csv(root), index=False)
