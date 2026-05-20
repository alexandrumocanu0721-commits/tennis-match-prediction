import copy

import pandas as pd

from challenger_config import (
    USE_EXTERNAL_ATP_RANKINGS_FOR_CHALLENGER,
    assert_challenger_processed_output_path,
    path_challenger_processed_features_csv,
    project_root,
)
from challenger_utils import (
    build_external_atp_rankings_context,
    build_challenger_player_name_map,
    challenger_tourney_level_encoding,
    filter_challenger_matches,
    initialize_challenger_player,
    load_challenger_match_history_csvs,
    load_external_atp_rankings_dataframe,
    neutral_rankings_context,
    prepare_challenger_matches_dataframe,
    refresh_challenger_player_rankings,
)
from tennis_pipeline import (
    ELO_K,
    FEATURES,
    append_recent_result_lists,
    build_symmetric_training_rows,
    compute_fatigue_stats,
    compute_h2h_diffs,
    compute_serve_return_stats,
    compute_winner_perspective_diffs,
    initialize_h2h_records,
    is_deciding_set_match,
    update_elo_and_match_counts,
    update_fatigue_state,
    update_h2h_records,
)

root = project_root()

ROUND_UPDATE_ORDER = {
    "Q1": 1,
    "Q2": 2,
    "Q3": 3,
    "R128": 4,
    "R64": 5,
    "R32": 6,
    "R16": 7,
    "QF": 8,
    "SF": 9,
    "F": 10,
    "RR": 6,
}


def round_update_order(round_value: object) -> int:
    if pd.isna(round_value):
        return 50
    return ROUND_UPDATE_ORDER.get(str(round_value).strip().upper(), 50)


def match_metadata(row, match_date) -> dict:
    return {
        "tourney_id": getattr(row, "tourney_id", ""),
        "tourney_name": getattr(row, "tourney_name", ""),
        "tourney_date": match_date,
        "round": getattr(row, "round", ""),
    }


df = load_challenger_match_history_csvs(root)
df = prepare_challenger_matches_dataframe(df)
original_len = len(df)
df = filter_challenger_matches(df)
print(f"Removed {original_len - len(df)} non-Challenger/invalid-outcome rows")

if USE_EXTERNAL_ATP_RANKINGS_FOR_CHALLENGER:
    rankings_df = load_external_atp_rankings_dataframe(root)
    rankings_by_player = build_external_atp_rankings_context(rankings_df)
else:
    rankings_df = pd.DataFrame(columns=["player"])
    rankings_by_player = neutral_rankings_context()
players: dict = {}
player_names = build_challenger_player_name_map(df)
feature_rows: list[dict] = []
h2h_records = initialize_h2h_records()

df = df.sort_values(["tourney_date", "tourney_id"], kind="mergesort")
for match_date, group in df.groupby("tourney_date", sort=False, dropna=False):
    pending_updates: list[dict] = []
    day_players = set()

    for row in group.itertuples(index=False):
        winner = row.winner_id
        loser = row.loser_id
        surface = row.surface
        if pd.isna(surface):
            continue

        initialize_challenger_player(
            players,
            winner,
            player_names[winner],
            match_date,
            rankings_by_player,
            use_external_rankings=USE_EXTERNAL_ATP_RANKINGS_FOR_CHALLENGER,
        )
        initialize_challenger_player(
            players,
            loser,
            player_names[loser],
            match_date,
            rankings_by_player,
            use_external_rankings=USE_EXTERNAL_ATP_RANKINGS_FOR_CHALLENGER,
        )

        refresh_challenger_player_rankings(
            players,
            winner,
            loser,
            match_date,
            rankings_by_player=rankings_by_player,
            use_external_rankings=USE_EXTERNAL_ATP_RANKINGS_FOR_CHALLENGER,
        )
        day_players.update([winner, loser])

    player_snapshot = {
        player_id: copy.deepcopy(players[player_id]) for player_id in day_players
    }

    for row in group.itertuples(index=False):
        winner = row.winner_id
        loser = row.loser_id
        surface = row.surface
        if pd.isna(surface):
            continue

        h2h_diffs = compute_h2h_diffs(h2h_records, winner, loser, surface, match_date)
        fatigue_stats = compute_fatigue_stats(row, player_snapshot, winner, loser)
        diffs = compute_winner_perspective_diffs(
            winner,
            loser,
            match_date,
            surface,
            player_snapshot,
            rankings_by_player,
            h2h_diffs=h2h_diffs,
            fatigue_stats=fatigue_stats,
        )
        winner_row, loser_row = build_symmetric_training_rows(
            match_date=match_date,
            winner=winner,
            loser=loser,
            surface=surface,
            player_names=player_names,
            diffs=diffs,
            tourney_level_encoded=challenger_tourney_level_encoding(row.tourney_level),
        )
        metadata = match_metadata(row, match_date)
        winner_row.update(metadata)
        loser_row.update(metadata)

        serve_stats = compute_serve_return_stats(row)
        deciding_set = is_deciding_set_match(row.score, row.tourney_level)

        feature_rows.append(winner_row)
        feature_rows.append(loser_row)
        pending_updates.append(
            {
                "row": row,
                "winner": winner,
                "loser": loser,
                "surface": surface,
                "serve_stats": serve_stats,
                "deciding_set": deciding_set,
                "match_date": match_date,
                "round_order": round_update_order(getattr(row, "round", "")),
                "match_num": getattr(row, "match_num", 0),
            }
        )

    pending_updates.sort(key=lambda item: (item["round_order"], item["match_num"]))
    for update in pending_updates:
        append_recent_result_lists(
            players,
            update["winner"],
            update["loser"],
            update["surface"],
            update["serve_stats"],
            serve_data_valid=update["serve_stats"]["valid"],
            is_deciding_set=update["deciding_set"],
        )
        update_elo_and_match_counts(
            players,
            update["winner"],
            update["loser"],
            update["surface"],
            k=ELO_K,
        )
        update_h2h_records(
            h2h_records,
            update["winner"],
            update["loser"],
            update["surface"],
            update["match_date"],
        )
        update_fatigue_state(
            players,
            update["winner"],
            update["loser"],
            update["match_date"],
            update["row"],
        )

feature_columns = [
    "date",
    "tourney_id",
    "tourney_name",
    "tourney_date",
    "round",
    "player_a_id",
    "player_b_id",
    "player_a_name",
    "player_b_name",
    "surface",
    *FEATURES,
    "result",
]
features_df = pd.DataFrame(feature_rows, columns=feature_columns)

output_path = path_challenger_processed_features_csv(root)
assert_challenger_processed_output_path(output_path, root)
output_path.parent.mkdir(parents=True, exist_ok=True)
features_df.to_csv(output_path, index=False)
print(f"Wrote {len(features_df)} rows to {output_path}")

challenger_players = set(
    pd.concat([df["winner_id"], df["loser_id"]], ignore_index=True).tolist()
)
ranking_players = set(rankings_df["player"].tolist()) if len(rankings_df) else set()
challenger_players_with_rankings = len(challenger_players & ranking_players)
non_neutral_rank_pct = (
    (features_df["rank_diff"] != 0).mean() * 100 if len(features_df) else 0.0
)
non_neutral_points_pct = (
    (features_df["points_diff"] != 0).mean() * 100 if len(features_df) else 0.0
)

print("Ranking debug summary:")
print("- ranking refresh strategy: most recent snapshot where ranking_date <= match_date")
print(f"- rankings rows loaded: {len(rankings_df)}")
print(f"- unique ranking players loaded: {len(ranking_players)}")
print(f"- Challenger players with at least one ranking match: {challenger_players_with_rankings}")
print(f"- percent of feature rows with non-neutral rank_diff: {non_neutral_rank_pct:.2f}%")
print(f"- percent of feature rows with non-neutral points_diff: {non_neutral_points_pct:.2f}%")
