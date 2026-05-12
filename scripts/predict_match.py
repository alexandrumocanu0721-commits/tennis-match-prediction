# =============================================================================
# predict_match.py — Replay history, then score rows from today_matches.csv
# =============================================================================
# Rebuilds the same player state as build_features (through latest raw match),
# then reads upcoming pairings, stacks features, runs one predict_proba batch,
# and writes predictions.csv. Model + paths come from tennis_pipeline.
# =============================================================================

import joblib
import pandas as pd

from tennis_pipeline import (
    ELO_K,
    append_recent_result_lists,
    build_rankings_by_player,
    compute_player_a_perspective_features,
    FEATURES,
    fill_player_names_from_matches,
    initialize_player,
    load_match_history_csvs,
    load_rankings_csv,
    path_predict_output_csv,
    path_predict_today_csv,
    path_trained_model_pkl,
    prepare_matches_dataframe,
    prepare_rankings_dataframe,
    project_root,
    update_elo_and_match_counts,
)

root = project_root()
model = joblib.load(path_trained_model_pkl(root))

# --- Converge state on full ATP history (no training rows written here) ---
df = prepare_matches_dataframe(load_match_history_csvs(root))
rankings_df = prepare_rankings_dataframe(load_rankings_csv(root))
rankings_by_player = build_rankings_by_player(rankings_df)

players: dict = {}
player_names = fill_player_names_from_matches(df)

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

    append_recent_result_lists(players, winner, loser, surface)
    update_elo_and_match_counts(players, winner, loser, surface, k=ELO_K)

# --- Resolve CSV names to internal ids (last id wins on duplicate spellings) ---
name_to_id: dict[str, object] = {}
for player_id, info in players.items():
    name_to_id[info["name"]] = player_id

predict_df = pd.read_csv(path_predict_today_csv(root))

# --- Batch inference: one matrix → one predict_proba (column order = FEATURES) ---
feature_dicts: list[dict] = []
meta_rows: list[tuple[str, str]] = []

for row in predict_df.itertuples(index=False):
    player_a_name = row.player_a
    player_b_name = row.player_b
    surface = row.surface

    player_a_id = name_to_id[player_a_name]
    player_b_id = name_to_id[player_b_name]

    feature_dicts.append(
        compute_player_a_perspective_features(
            player_a_id, player_b_id, surface, players
        )
    )
    meta_rows.append((player_a_name, player_b_name))

X_pred = pd.DataFrame(feature_dicts, columns=FEATURES)
probs = model.predict_proba(X_pred)[:, 1]

results = []
for i, (player_a_name, player_b_name) in enumerate(meta_rows):
    prob = float(probs[i])
    results.append(
        {
            "match": f"{player_a_name} vs {player_b_name}",
            "predicted_winner": (
                player_a_name if prob >= 0.5 else player_b_name
            ),
            "winner_probability": round(max(prob, 1 - prob) * 100, 2),
        }
    )

results_df = pd.DataFrame(results)
results_df.to_csv(path_predict_output_csv(root), index=False)
