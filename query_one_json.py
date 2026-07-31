import json 
import json
import re
import sys
import pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
from sentence_transformers import SentenceTransformer
from nflreadpy import load_player_stats
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os 
import json

H_JSON = {
  "team_name": "Team H",
  "season": 2025,
  "week": 7,
  "roster": {
    "starters": [
      {
        "slot_id": 1,
        "slot": "QB",
        "first_name": "Patrick",
        "last_name": "Mahomes"
      },
      {
        "slot_id": 2,
        "slot": "RB",
        "first_name": "Christian",
        "last_name": "McCaffrey"
      },
      {
        "slot_id": 3,
        "slot": "RB",
        "first_name": "Javonte",
        "last_name": "Williams"
      },
      {
        "slot_id": 4,
        "slot": "WR",
        "first_name": "Davante",
        "last_name": "Adams"
      },
      {
        "slot_id": 5,
        "slot": "WR",
        "first_name": "DK",
        "last_name": "Metcalf"
      },
      {
        "slot_id": 6,
        "slot": "TE",
        "first_name": "Evan",
        "last_name": "Engram"
      },
      {
        "slot_id": 7,
        "slot": "FLEX",
        "first_name": "David",
        "last_name": "Montgomery"
      },
      {
        "slot_id": 8,
        "slot": "D/ST",
        "first_name": "Steelers",
        "last_name": "Defense"
      },
      {
        "slot_id": 9,
        "slot": "K",
        "first_name": "Joshua",
        "last_name": "Karty"
      }
    ],
    "bench": [
      {
        "slot_id": 1,
        "slot": "BN",
        "first_name": "Brock",
        "last_name": "Bowers"
      },
      {
        "slot_id": 2,
        "slot": "BN",
        "first_name": "Jakobi",
        "last_name": "Meyers"
      },
      {
        "slot_id": 3,
        "slot": "BN",
        "first_name": "Michael",
        "last_name": "Pittman Jr."
      },
      {
        "slot_id": 4,
        "slot": "BN",
        "first_name": "Nick",
        "last_name": "Chubb"
      },
      {
        "slot_id": 5,
        "slot": "BN",
        "first_name": "Aaron",
        "last_name": "Rodgers"
      },
      {
        "slot_id": 6,
        "slot": "BN",
        "first_name": "Elic",
        "last_name": "Ayomanor"
      },
      {
        "slot_id": 7,
        "slot": "BN",
        "first_name": "Noah",
        "last_name": "Fant"
      }
    ]
  }
}

#load json takes a json string and separates the starting lineup and bench into two lists
def load_json(json_string):
    data =json_string
    starting_lineup = []
    bench = []
    starting_lineup_json = data["roster"]["starters"]
    bench_json = data["roster"]["bench"]
    for elem in starting_lineup_json:
        if elem["last_name"] != "Defense":
            starting_lineup.append(elem["first_name"] + " " + elem["last_name"] + " week: " + str(data["week"]))
    for elem in bench_json:
        if elem["last_name"] != "Defense":
            bench.append(elem["first_name"] + " " + elem["last_name"] + " week: " + str(data["week"]))
    return starting_lineup, bench

#normalize name converts the name to lowercase and removes all non alphabetic characters
def normalize_name(name):
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())

#query takes a player string and returns a matched row based on name match and similarity score
def query(player_string, week):
    cache_path = f"embeddings_try_2_{week}.npy"
    if not os.path.exists(cache_path):
        chunks = []
        filtered_rows = []
        stats = load_player_stats([2025])
        rows = stats.to_dicts()
        for row in rows: 
            if row["week"] == week:
                chunk = row_to_semantic_text(row) 
                chunks.append(chunk)
                filtered_rows.append(row)
        embeddings = embed(chunks)
        np.save(cache_path, embeddings)
        json.dump(chunks, open(f"chunks.json_2_{week}", "w")) 
        json.dump(filtered_rows, open(f"rows.json_2_{week}", "w"))
    else:
        embeddings = np.load(cache_path)
        chunks = json.load(open(f"chunks.json_2_{week}"))
        filtered_rows = json.load(open(f"rows.json_2_{week}"))
    query = player_string
    print(query)
    top_k = 7
    query_vec = embed([query])
    sims = cosine_similarity(query_vec, embeddings)[0]  
    top_indices = np.argsort(sims)[::-1][:top_k]
    norm_query_name = normalize_name(player_string.split(" week:")[0])
    matches = [(i, sims[i]) for i in top_indices 
           if normalize_name(filtered_rows[i].get("player_display_name", "")) == norm_query_name]

    if matches:
        best_idx = max(matches, key=lambda x: x[1])[0]
    else:
        best_idx = top_indices[0]  # fallback to top by score, or return no_match

    row = filtered_rows[best_idx]
    similarity = sims[best_idx]
    return similarity, row
   
#embed takes a list of chunks and returns a list of embedded chunks
def embed(chunks: list[str]):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embedded_chunks= model.encode(chunks)
    return embedded_chunks

#row_to_semantic_text takes a row and returns a string of the row with relevant embedding information
def row_to_semantic_text(row: dict) -> str:
    parts = []
    SEARCHABLE_COLUMNS = ['player_display_name', 'week']
    for col in SEARCHABLE_COLUMNS:
        if col in row and row[col] is not None:
            parts.append(f"{col}: {row[col]}")
    return "\n".join(parts)

#get relevant stats takes a row and returns a dictionary of relevant stats for the player
def get_relevant_stats(row: dict) -> dict:
    relevant_stats = {}
    if row["position"] == "QB":
        relevant_stats = {
        "player_display_name": row["player_display_name"],
        "week": row["week"],
        "completions": row["completions"],
        "passing_yards": row["passing_yards"],
        "passing_touchdowns": row["passing_tds"],
        "passing_interceptions": row["passing_interceptions"],
        "rushing_yards": row["rushing_yards"],
        "rushing_touchdowns": row["rushing_tds"],
        "rushing_attempts": row["carries"],
        "sacks_suffered": row["sacks_suffered"],
        "sack_yards_lost": row["sack_yards_lost"],
        "fumbles_lost": row["sack_fumbles_lost"],
        "fantasy_points": row["fantasy_points"],
        "fantasy_points_ppr": row["fantasy_points_ppr"],
    }
    if row["position"] == "RB":
        relevant_stats = {
        "player_display_name": row["player_display_name"],
        "week": row["week"],
        "rushing_attempts": row["carries"],
        "rushing_yards": row["rushing_yards"],
        "rushing_touchdowns": row["rushing_tds"],
        "rushing_fumbles": row["rushing_fumbles"],
        "receptions": row["receptions"],
        "receiving_yards": row["receiving_yards"],
        "receiving_touchdowns": row["receiving_tds"],
        "receiving_fumbles": row["receiving_fumbles"],
        "fantasy_points": row["fantasy_points"],
        "fantasy_points_ppr": row["fantasy_points_ppr"],
    }
    if row["position"] == "WR" or row["position"] == "TE":
        relevant_stats = {
        "player_display_name": row["player_display_name"],
        "week": row["week"],
        "receptions": row["receptions"],
        "receiving_yards": row["receiving_yards"],
        "receiving_touchdowns": row["receiving_tds"],
        "receiving_fumbles": row["receiving_fumbles"],
        "fantasy_points": row["fantasy_points"],
        "fantasy_points_ppr": row["fantasy_points_ppr"],
        }
    if row["position_group"] == "DL" or row["position_group"] == "LB" or row["position_group"] == "DB":
        relevant_stats = {
        "player_display_name": row["player_display_name"],
        "week": row["week"],
        "def_tackles_solo": row["def_tackles_solo"],
        "def_tackles_with_assist": row["def_tackles_with_assist"],
        "def_sacks": row["def_sacks"],
        "def_interceptions": row["def_interceptions"],
        "def_interception_return_yards": row["def_interception_yards"],
        "def_safeties": row["def_safeties"],
        "def_touchdowns": row["def_tds"],
        "defense_sacks": row["def_sacks"],
        "defense_interceptions": row["def_interceptions"],
        "defensive_interception_yards": row["def_interception_yards"],
        "defense_passes_defended": row["def_pass_defended"],
        "def_fumbles_forced": row["def_fumbles_forced"],
        "def_qb_hits": row["def_qb_hits"],
        "def_fumble_recovery_own": row["fumble_recovery_own"],
        "fantasy_points": row["fantasy_points"],
        "fantasy_points_ppr": row["fantasy_points_ppr"],
    }
    if row["position"] == "K":
        relevant_stats = {
        "player_display_name": row["player_display_name"],
        "week": row["week"],
        "field_goals_attempted": row["fg_att"],
        "field_goals_made": row["fg_made"],
        "field_goals_yards": row["fg_made_list"],
        "extra_points_made": row["pat_made"],
        "extra_points_attempted": row["pat_att"],
        # "fantasy_points": row["fantasy_points"],
        # "fantasy_points_ppr": row["fantasy_points_ppr"],
    }
    return relevant_stats

#process FF teams takes in a json from langchain_doc_retrieval and returns a list of
#starting lineup and bench stats for the team
def process_FF_teams(JSON, week):
    starting_lineup, bench = load_json(JSON)
    starting_lineup_stats = []
    bench_stats = []
    for elem in starting_lineup:
        similarity, row = query(elem, week)
        if similarity >= 0.6:
            row = get_relevant_stats(row)
            starting_lineup_stats.append(row)
    for elem in bench:
        similarity, row = query(elem, week)
        row = get_relevant_stats(row)
        if similarity >=0.6:
            bench_stats.append(row)
    return starting_lineup_stats, bench_stats


if __name__ == "__main__":
    starting_lineup_stats, bench_stats = process_FF_teams(H_JSON, week=7)
    print(starting_lineup_stats)
    print(bench_stats)

