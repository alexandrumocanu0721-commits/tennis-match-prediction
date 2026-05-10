import pandas as pd
from pathlib import Path

k = 32

raw_path = Path(__file__).resolve().parent.parent / "data" / "raw"
csv_files = raw_path.glob("atp_matches_*.csv")

dfs = [pd.read_csv(file) for file in csv_files]
df = pd.concat(dfs, ignore_index=True)

df = df.sort_values("tourney_date")

rankings_path = Path(__file__).resolve().parent.parent / "data" / "raw"
rankings_df = pd.read_csv(rankings_path / "atp_rankings_20s.csv")

rankings_df["ranking_date"] = pd.to_datetime(
    rankings_df["ranking_date"],
    format="%Y%m%d"
)

df["tourney_date"] = pd.to_datetime(
    df["tourney_date"],
    format="%Y%m%d"
)

rankings_df = rankings_df.sort_values("ranking_date")

players = {}
player_names = {} 
feature_rows = []
for _, row in df.iterrows(): 
    winner_id = row["winner_id"] 
    winner_name = row["winner_name"] 
    
    loser_id = row["loser_id"] 
    loser_name = row["loser_name"] 
    
    player_names[winner_id] = winner_name 
    player_names[loser_id] = loser_name

def get_player_ranking(player_id, match_date):
    player_history = rankings_df[
        ((rankings_df["player"] == player_id))
        &
        (rankings_df["ranking_date"] <= match_date)
    ]
    
    if player_history.empty:
        return 2000, 0

    latest_entry = player_history.iloc[-1]

    rank = latest_entry["rank"]
    points = latest_entry["points"]

    return rank, points

def initialize_player(player_id, player_name, match_date):

    if player_id in players:
        return

    rank, points = get_player_ranking(
        player_id,
        match_date
    )

    players[player_id] = {

        "name": player_name,

        "elo": 1500,

        "rank": int(rank),
        "points": int(points),

        "surface_elo": {
            "Hard": 1500,
            "Clay": 1500,
            "Grass": 1500
        },

        "recent_results": [], 
        
        "surface_recent_results": { 
            "Hard": [], 
            "Clay": [], 
            "Grass": [] 
        },

        "matches_played": 0,
        "wins": 0
    }

def recent_win_rate(results, window=10): 
    if len(results) == 0: 
        return 0.5 
        
    recent = results[-window:] 
    return sum(recent) / len(recent)

for _, row in df.iterrows():

    winner = row["winner_id"]
    loser = row["loser_id"]
    match_date = row["tourney_date"]
    surface = row["surface"]
    if pd.isna(surface): 
        continue

    initialize_player(
        winner,
        player_names[winner],
        match_date
    )

    initialize_player(
        loser,
        player_names[loser],
        match_date
   )

    winner_recent_form = recent_win_rate(
        players[winner]["recent_results"]
    )
    loser_recent_form = recent_win_rate(
        players[loser]["recent_results"]
    )

    recent_form_diff = (
        winner_recent_form
        -
        loser_recent_form
    )

    winner_surface_form = recent_win_rate(
        players[winner]["surface_recent_results"][surface]
    )
    loser_surface_form = recent_win_rate(
        players[loser]["surface_recent_results"][surface]
    )

    recent_surface_form_diff = (
        winner_surface_form
        -
        loser_surface_form
    )

    winner_matches = max(
        players[winner]["matches_played"],
        1
    )
    loser_matches = max(
        players[loser]["matches_played"],
        1
    )

    winner_win_pct = (
        players[winner]["wins"]
        / winner_matches
    )
    loser_win_pct = (
        players[loser]["wins"]
        / loser_matches
    )

    win_pct_diff = (
        winner_win_pct
        -
        loser_win_pct
    )

    matches_played_diff = (
        players[winner]["matches_played"]
        -
        players[loser]["matches_played"]
    )

    elo_diff = (
        players[winner]["elo"]
        -
        players[loser]["elo"]
    )
    surface_elo_diff = (
        players[winner]["surface_elo"][surface]
        -
        players[loser]["surface_elo"][surface]
    )
    rank_diff = (
        players[loser]["rank"]
        -
        players[winner]["rank"]
    )
    points_diff = (
        players[winner]["points"]
        -
        players[loser]["points"]
    )

    winner_row = {

        "date": match_date,

        "player_a_id": winner,
        "player_b_id": loser,

        "player_a_name": player_names[winner],
        "player_b_name": player_names[loser],

        "surface": surface,

        "elo_diff": elo_diff,
        "surface_elo_diff": surface_elo_diff,
        "rank_diff": rank_diff,
        "points_diff": points_diff,

        "recent_form_diff": recent_form_diff,
        "recent_surface_form_diff": recent_surface_form_diff,
        "win_pct_diff": win_pct_diff,
        "matches_played_diff": matches_played_diff,

        "result": 1
    }
    loser_row = {

        "date": match_date,

        "player_a_id": loser,
        "player_b_id": winner,

        "player_a_name": player_names[loser],
        "player_b_name": player_names[winner],

        "surface": surface,

        "elo_diff": (-1) * elo_diff,
        "surface_elo_diff": (-1) * surface_elo_diff,
        "rank_diff": (-1) * rank_diff,
        "points_diff": (-1) * points_diff,

        "recent_form_diff": -recent_form_diff,
        "recent_surface_form_diff": -recent_surface_form_diff,
        "win_pct_diff": -win_pct_diff,
        "matches_played_diff": -matches_played_diff,

        "result": 0
    }
    players[winner]["recent_results"].append(1)
    players[loser]["recent_results"].append(0)
    players[winner]["surface_recent_results"][surface].append(1)
    players[loser]["surface_recent_results"][surface].append(0)

    feature_rows.append(winner_row) 
    feature_rows.append(loser_row)

    winner_elo = players[winner]["elo"]
    loser_elo = players[loser]["elo"]
    expected_win = 1 / (
        1 + 10 ** ((loser_elo - winner_elo) / 400)
    )

    players[winner]["elo"] += k * (1 - expected_win)
    players[loser]["elo"] += k * (0 - (1 - expected_win))

    winner_surface_elo = (
        players[winner]["surface_elo"][surface]
    )
    loser_surface_elo = (
        players[loser]["surface_elo"][surface]
    )
    expected_surface_win = 1 / (
        1 + 10 ** (
            (loser_surface_elo - winner_surface_elo) / 400
        )
    )
    players[winner]["surface_elo"][surface] += (
        k * (1 - expected_surface_win)
    )
    players[loser]["surface_elo"][surface] += (
        k * (0 - (1 - expected_surface_win))
    )

    players[winner]["wins"] += 1
    players[winner]["matches_played"] += 1
    players[loser]["matches_played"] += 1


features_df = pd.DataFrame(feature_rows)

features_df.to_csv(
    "./data/processed/features.csv",
    index=False
)




