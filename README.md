# 🌾 Kisaan AI — Crop Doctor & Yield Optimizer

An AI-powered multimodal agricultural assistant that analyzes crop images, retrieves organic farming knowledge, and generates hyper-local farming recommendations using Vision-Language Models, RAG, weather intelligence, and soil analysis.

Built with Groq LLMs, LangChain, LangGraph, FAISS, Hugging Face embeddings, an MCP tool server, FastAPI, and Streamlit.

**Deployed Link** https://kisaan-ai.streamlit.app/

---

# 🚀 Features

- 📷 Crop disease detection from one or more photos using a vision LLM
- 🌱 Organic, chemical-free treatment and care recommendations
- 🔍 Retrieval-Augmented Generation (FAISS search + cross-encoder reranking)
- 🌦️ Live weather from OpenWeatherMap, served through an MCP tool
- 🌍 Location-based soil profile lookup for Indian states and cities
- 🛡️ Graceful degradation — if weather, soil or retrieval fails, you still get advice, with a warning saying what was missing
- 📊 RAGAS evaluation (faithfulness + context precision) against a 30-question golden dataset

---

# 🧠 Tech Stack

| Layer | Tools |
|---|---|
| LLMs (via Groq) | `qwen/qwen3.6-27b` (vision), `openai/gpt-oss-120b` (reasoning) |
| Orchestration | LangChain (LCEL), LangGraph |
| Retrieval | FAISS, `all-MiniLM-L6-v2` embeddings, `cross-encoder/ms-marco-MiniLM-L-6-v2` reranker |
| Tools | MCP server built with FastMCP, OpenWeatherMap API |
| Backend | FastAPI + Uvicorn |
| Frontend | Streamlit |
| Evaluation | RAGAS |

---

# 🏗️ System Architecture

The Streamlit frontend sends the request to a FastAPI backend, which runs a LangGraph pipeline:

```text
Streamlit UI  ──POST /diagnose──▶  FastAPI backend (LangGraph)
                                      │
                                      ▼
                              1. vision_node
                                 Vision LLM analyzes the photo(s)
                                 (skipped for text-only requests)
                                      │
                                      ▼
                              2. weather_soil_node
                                 Weather + soil fetched in parallel
                                 over one long-lived MCP session
                                 (mcp_server.py → OpenWeatherMap)
                                      │
                                      ▼
                              3. retrieval_node
                                 FAISS top-10 → cross-encoder rerank → top-5
                                      │
                                      ▼
                              4. synthesis_node
                                 gpt-oss-120b writes the organic
                                 recommendations from all collected context
                                      │
                                      ▼
                              Markdown answer + status flags
```

Vision and synthesis are critical steps: if they fail after retries, the API returns an `error`. Weather, soil and retrieval are non-critical: if they fail, the pipeline continues and sets `weather_soil_failed` or `retrieval_failed` so the UI can warn the user.

---

# 📂 Project Structure

```text
Kisaan-AI/
├── backend/
│   ├── app.py              # FastAPI app + LangGraph pipeline
│   ├── mcp_server.py       # MCP server: get_weather, get_soil_condition tools
│   ├── setup_rag.py        # Builds the FAISS index from the KisanVaani dataset
│   └── requirements.txt
├── frontend/
│   ├── app.py              # Streamlit UI
│   └── requirements.txt
├── evaluation/
│   ├── eval.py             # Runs the golden set through the API and scores it with RAGAS
│   ├── golden_dataset.json # 30 questions across 10 crop/disease categories
│   ├── images/             # Sample crop images for the golden set
│   └── requirements.txt
├── faiss_index/            # Prebuilt vector index (index.faiss + index.pkl)
├── logs/                   # Backend logs (created at runtime, git-ignored)
└── .env                    # API keys (you create this; git-ignored)
```

---

# ⚙️ Installation

Requires **Python 3.11**.

## 1️⃣ Clone the repository

```bash
git clone https://github.com/<your-github-username>/<repo-name>.git
cd <repo-name>
```

## 2️⃣ Create a virtual environment

### Windows
```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / Mac
```bash
python3 -m venv venv
source venv/bin/activate
```

## 3️⃣ Install dependencies

Each component has its own requirements file. Install the ones you need:

```bash
pip install -r backend/requirements.txt
pip install -r frontend/requirements.txt
pip install -r evaluation/requirements.txt   # only needed to run the evaluation
```

---

# 🔑 Environment Variables

Create a `.env` file in the **repository root**:

```env
GROQ_API_KEY=your_groq_api_key
OPENWEATHER_API_KEY=your_openweather_api_key
```

Put `OPENWEATHER_API_KEY` in `.env` rather than only exporting it in your shell. The MCP server runs as a subprocess with a minimal environment and reads the key from `.env` itself.

Without an OpenWeatherMap key the app still works, but weather is reported as unavailable and the UI shows a warning.

### Optional settings

| Variable | Default | Purpose |
|---|---|---|
| `BACKEND_URL` | `http://localhost:8000` | Where the frontend and evaluation script send requests |
| `FAISS_INDEX_DIR` | `<repo>/faiss_index` | Location of the vector index |
| `RAG_CANDIDATE_K` | `10` | Documents fetched from FAISS before reranking |
| `RAG_RERANK_TOP_N` | `5` | Documents kept after reranking |
| `GROQ_RETRY_MAX_ATTEMPTS` | `3` | Attempts for each LLM call |
| `GROQ_RETRY_BACKOFF_BASE` | `2` | Exponential backoff base, in seconds |
| `MCP_TOOL_TIMEOUT_SECONDS` | `20` | Timeout for each weather/soil tool call |
| `RAGAS_JUDGE_MODEL` | `openai/gpt-oss-120b` | Judge LLM used by the evaluation |

---

# 📚 RAG Database

A prebuilt index is included in `faiss_index/`, so this step is optional. If the index is missing, the backend builds it automatically on first start.

To rebuild it manually:

```bash
python backend/setup_rag.py
```

This script:
- Downloads the first 2,000 entries of the KisanVaani agriculture QA dataset
- Splits them into ~800-character chunks
- Embeds them with `all-MiniLM-L6-v2` (runs locally, no API key needed)
- Saves the FAISS index to `faiss_index/`

---

# ▶️ Run the Application

Start the backend and the frontend in two terminals.

**Terminal 1 — backend**
```bash
cd backend
uvicorn app:app --port 8000
```

The first start downloads the embedding and reranker models (~180 MB in total). Check it's up with `curl http://localhost:8000/health`.

**Terminal 2 — frontend**
```bash
streamlit run frontend/app.py
```

Then open http://localhost:8501.

---

# 🔌 API

### `POST /diagnose`

Request:

```json
{
  "images_base64": ["<base64 JPEG>", "..."],
  "user_query": "How do I treat this disease organically?",
  "location": "Lucknow, India"
}
```

`images_base64` is optional. Without it, the pipeline skips image analysis and answers from the question and the knowledge base. The Streamlit UI always requires at least one image.

Response:

| Field | Description |
|---|---|
| `final_answer` | Markdown recommendations (`null` if `error` is set) |
| `error` | Set if image analysis or answer generation failed |
| `weather_soil_failed` | `true` if weather or soil data was unavailable |
| `retrieval_failed` | `true` if the knowledge-base lookup failed |
| `rag_context` | Retrieved documents joined into one string, as given to the LLM |
| `rag_contexts` | The same documents as a list, in rerank order |

### `GET /health`

Returns `{"status": "ok"}`.

---

# 📊 Evaluation

With the backend running:

```bash
python evaluation/eval.py
```

The script sends every question in `golden_dataset.json` to `/diagnose`, saves the raw results to `evaluation/eval_results.json`, and scores the successful ones with RAGAS:

- **Faithfulness** — is the answer supported by the retrieved context?
- **Context precision** — are the relevant documents ranked near the top?

The golden set has 30 questions: 3 for each of 10 crop/disease categories (apple scab, corn common rust, grape black rot, bell-pepper bacterial spot, potato early and late blight, strawberry leaf scorch, tomato early and late blight, and healthy tomato). Evaluation currently runs in text-only mode.

---

# 📸 How It Works

1. Upload one or more crop photos
2. Enter your location and question
3. The vision model analyzes crop health
4. Weather and soil conditions are fetched for your location
5. Relevant agricultural knowledge is retrieved and reranked
6. The LLM writes personalized organic recommendations

---

# 📊 Dataset

- **KisanVaani/agriculture-qa-english-only** (Hugging Face Datasets)
- Agricultural questions and answers covering crop diseases, pests and farming practices

---

# ⚠️ Known Limitations

- Soil data comes from a small built-in table of Indian states and cities. Other locations get a generic "Standard Loamy Soil" default.
- If the MCP server crashes while the backend is running, weather and soil stay unavailable until the backend restarts.
- Dependency versions are not pinned.

---

# 🔥 Future Improvements

- Multi-language farmer support
- Voice-based interaction
- Live satellite crop monitoring
- Mobile application deployment
- Fine-tuned agriculture-specific LLM

---

# 👨‍💻 Author

Kushal Punem

---

# ⭐ Acknowledgements

- Groq
- Hugging Face
- LangChain
- Streamlit
- OpenWeatherMap
- KisanVaani Dataset
