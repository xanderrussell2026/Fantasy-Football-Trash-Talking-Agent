# RAG FF Agent

Fantasy football RAG agent that pulls team/player stats, generates trash-talk commentary with Gemini, and fact-checks the response.

## Setup (after cloning)

1. **Create a virtual environment** (recommended):

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

2. **Install dependencies:**

   ```powershell
   pip install -r requirements.txt
   ```

3. **Add your Gemini API key:**

   ```powershell
   copy .env.example .env
   ```

   Open `.env` and set your key from [Google AI Studio](https://aistudio.google.com/apikey):

   ```
   GOOGLE_API_KEY=your_api_key_here
   ```

   Do not commit `.env` — it is gitignored.

4. **Run the agent:**

   ```powershell
   python ff_vertex_client.py
   ```

   Example query: `Roast the Sharks and the Robots week 7`

## Project layout

| Path | Purpose |
|------|---------|
| `ff_vertex_client.py` | Main script: prompt Gemini + verification |
| `langchain_doc_retrieval.py` | Team lookup / stats retrieval |
| `query_one_json.py` | Player stats matching helpers |
| `FF Teams/` | Sample team JSON inputs |
| `.env.example` | Template for API key setup |
| `requirements.txt` | Python dependencies |
