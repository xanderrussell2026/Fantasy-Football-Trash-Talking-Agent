from google.cloud import aiplatform
import os
from langchain_doc_retrieval import get_FF_team_stats, query_FF_teams
from google import genai
from datetime import datetime

def _load_dotenv(path: str = ".env") -> None:
    """Load KEY=VALUE pairs from a local .env file into the environment."""
    if not os.path.isfile(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))

_load_dotenv()
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")

#Takes team stats and builds a prompt for the model
def build_prompt_from_teams(teams: list, instruction: str) -> str:
    prompt = instruction + "\n\n"
    for team in teams:
        prompt += f"Team: {team['team_name']}, Week: {team['week']}\n"
        prompt += f"Starters: {team['starting_lineup_stats']}\n"
        prompt += f"Bench: {team['bench_stats']}\n\n"
    return prompt

#Takes teams without stats and builds a prompt for the statless model
def build_statless_prompt(statless_teams: list) -> str:
        instruction = (
          "You are a trash-talking fantasy football assistant. "
          "Given the teams below, roast this team in a fun, sports bar tone."
          "If 2 teams are provided, roast both teams in a fun, sports bar tone."
          "DO NOT MENTION EMPTY SLOTS"
          "Get the stats for the teams from the prompt and use them to roast the teams."
          "The teams are: {statless_teams}"
           )
        return instruction.format(statless_teams=statless_teams)

#Takes the response from the model and the team stats and builds a prompt for the verification model
def build_verification_prompt(response: str, stats_prompt: str) -> str:
    template = (
        "You are going to fact check my fantasy football commentator. "
        "The prompt with statistics are: {prompt}\n"
        "The response is: {response}\n"
        "The response should be fact checked against the prompt and statistics. "
        "Say when there are any errors in the response that have to do with factual errors or hallucinations. "
        "At the end, give an ACCURACY SCORE out of 100 (0 = many factual errors/hallucinations, 100 = fully accurate to the data). "
        "Format the score on its own line as: ACCURACY SCORE: X/100"
    )
    return template.format(prompt=stats_prompt, response=response)

#Calls the model with the given prompt and key
def call_model_with_key(prompt: str, *, key: str | None = None, model: str | None = None) -> str:
    """Call Gemini with API key. Key from argument, GOOGLE_API_KEY global, or env."""
    api_key = key or GOOGLE_API_KEY
    if not api_key:
        raise ValueError("Set GOOGLE_API_KEY in .env (see .env.example) or pass key=")
    model_id = model or "gemini-2.5-flash"
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(model=model_id, contents=prompt)
    return response.text

def main():
      user_query = input("Enter your FF team query (e.g. 'Roast the Sharks and the Robots week 7'): ")
      sections = [f"QUERY: {user_query}\n"]
      teams = get_FF_team_stats(user_query)
      instruction = (
          "You are a trash-talking fantasy football assistant. "
          "Given the team stats below, roast this team in a fun, sports bar tone."
          "If 2 teams are provided, roast both teams in a fun, sports bar tone."
          "Rules: Do not hallucinate any stats only use the stats provided."
          "DO NOT MENTION EMPTY SLOTS"
          "Do not make inferences about empty slots, but feel free to mention when players did well or poorly, and were benched"
          "Do not comment on what stats are missing, only make judgements on the stats provided."
      )
      prompt = build_prompt_from_teams(teams, instruction)
      print("PROMPT SENT TO GEMINI:")
      print(prompt)
      sections.append("PROMPT (WITH STATS):" + prompt + "\n")

      print("MODEL RESPONSE:")

      #To change the model, get the response from model here: 
      response = call_model_with_key(prompt)
      print(response)
      sections.append("COMMENTATOR RESPONSE (WITH STATS):" + response + "\n")

      verification_prompt = build_verification_prompt(response, prompt)
      print("VERIFICATION PROMPT: ")
      print(verification_prompt)
      print("VERIFICATION RESPONSE: ")
      stats_verification_response = call_model_with_key(verification_prompt)
      print(stats_verification_response)
      sections.append("VERIFICATION RESPONSE (WITH STATS): " + stats_verification_response + "\n")

      statless_teams = query_FF_teams(user_query)
      statless_query = build_statless_prompt(statless_teams)
      #To change the model, get the response from model here: 
      statless_response = call_model_with_key(statless_query)
      print("STATLESS RESPONSE:")
      print(statless_response)
      
      sections.append("PROMPT (STATLESS): " + statless_query + "\n")
      sections.append("COMMENTATOR RESPONSE (STATLESS): " + statless_response + "\n")
      print("STATLESS VERIFICATION RESPONSE: ")

      statless_verification_prompt = build_verification_prompt(statless_response, statless_query)
      print(statless_verification_prompt)
      statless_verification_response = call_model_with_key(statless_verification_prompt)
      print(statless_verification_response)
      sections.append("VERIFICATION RESPONSE (STATLESS): " + statless_verification_response + "\n")
      accuracy_score = statless_verification_response.split("ACCURACY SCORE: ")[1].split("/100")[0]
      print("ACCURACY SCORE (STATLESS): ")
      print(accuracy_score)
      sections.append("ACCURACY SCORE (STATLESS): " + accuracy_score + "\n")
      output_path = f"ff_run_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
      with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(sections))
      print(f"ALL OUTPUT WRITTEN TO {output_path}")

if __name__ == "__main__":
    main()

     

# to operate without generating statless model
# def main():
#       user_query = input("Enter your FF team query (e.g. 'Team H week 7'): ")
#       teams = get_FF_team_stats(user_query)
#       instruction = (
#           "You are a trash-talking fantasy football assistant. "
#           "Given the team stats below, roast this team in a fun, sports bar tone."
#           "If 2 teams are provided, roast both teams in a fun, sports bar tone."
#           "Rules: Do not hallucinate any stats only use the stats provided."
#           "DO NOT MENTION EMPTY SLOTS"
#           "Do not make inferences about empty slots, but feel free to mention when players did well or poorly, and were benched"
#           "Do not comment on what stats are missing, only make judgements on the stats provided."
#       )
#       prompt = build_prompt_from_teams(teams, instruction)
#       print("\n===== PROMPT SENT TO GEMINI =====\n")
#       print(prompt)
#       print("\n===== MODEL RESPONSE =====\n")
#       response = call_model_with_key(prompt)
#       print(response)
#       print("\n===== SAVE RESPONSE TO FILE =====\n")
#       with open("response.txt", "w") as f:
#         f.write(response)
#       verification_prompt = build_verification_prompt(response,prompt)
#       print("\n===== VERIFICATION PROMPT =====\n")
#       print(verification_prompt)
#       print("\n===== VERIFICATION RESPONSE =====\n")
#       verification_response = call_model_with_key(verification_prompt)
#       print(verification_response)
#       print("\n===== SAVE VERIFICATION RESPONSE TO FILE =====\n")
#       with open("verification_response.txt", "w") as f:
#         f.write(verification_response)


# def build_verification_prompt(response: str, stats_prompt: str) -> str:
#     template = (
#         "You are going to fact check my fantasy football commentator. "
#         "The prompt with statistics are: {prompt}\n"
#         "The response is: {response}\n"
#         "The response should be fact checked against the prompt and statistics. "
#         "Say when there are any errors in the response that have to do with factual errors or hallucinations."
#     )
#     return template.format(prompt=stats_prompt, response=response)