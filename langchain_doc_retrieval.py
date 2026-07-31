import json 
import os 
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from query_one_json import process_FF_teams


JSON = {
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

#query FF teams takes a string query and returns a list of starting lineup and bench stats for the team
def query_FF_teams(string_query):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    text_chunks = []
    metadatas = []
    file_path = "FF Teams/"
    json_paths = [
    os.path.join(file_path, f)
    for f in os.listdir(file_path)
]
    for path in json_paths:
        with open(path,"r") as file:
            team = json.load(file)
            team_name = team["team_name"]
            team_week = team["week"]
            metadatas.append({"path": path})
            text_chunks.append(f"team_name: {team_name}, week: {team_week}")
    vectorstore = FAISS.from_texts(text_chunks, embeddings,  metadatas=metadatas)
    results = vectorstore.similarity_search(string_query, k=2)
    return results

# get_FF_team_stats (main) function takes a string query and returns a list of team stats (one entry per retrieved team)
def get_FF_team_stats(query):
    if query is None:
        query = input("Enter your query: ")
    else:
        query = query
    results = query_FF_teams(query)
    teams = []
    for result in results:
        print(result)
        with open(result.metadata["path"], 'r') as f:
            content = f.read()
            data = json.loads(content)
            starting_lineup_stats, bench_stats = process_FF_teams(data, data["week"])
            teams.append({
                "team_name": data.get("team_name", "Unknown"),
                "week": data.get("week"),
                "starting_lineup_stats": starting_lineup_stats,
                "bench_stats": bench_stats,
            })
    return teams

if __name__ == "__main__":
    teams = get_FF_team_stats(None)
    for t in teams:
        print(t["team_name"], t["week"])
        print("  starters:", t["starting_lineup_stats"])
        print("  bench:", t["bench_stats"])
    
