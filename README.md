# 🕒 Chronos-Nexus – Temporal Contradiction Engine

**Chronos-Nexus** is an AI-powered "Temporal Contradiction Engine" designed to answer queries by analyzing how facts, policies, and narratives evolve over time (specifically across **2024 → 2025 → 2026**).

Unlike standard RAG chatbots that assume truth is static, Chronos-Nexus actively detects policy reversals, scientific disagreements, and market shifts, providing a structured Evolution Brief for any open-domain topic (Finance, Politics, Technology, Healthcare, etc.).

---

## 🎯 Key Features

| Feature | Description |
| :--- | :--- |
| 🔍 **Temporal RAG** | Indexes data from 2024, 2025, and 2026 into a Chroma vector store. Semantic retrieval ensures the most relevant historical context is fetched. |
| 🛠️ **Intelligent Tool Calling** | The LangGraph agent autonomously calls web search (Tavily), fact-checkers, confidence/volatility scorers, and a report exporter to enrich the analysis. |
| 🧠 **Dual-Layer Memory** | Short-term (`MemorySaver`) maintains conversation context. Long-term (`InMemoryStore`) alerts the user if new 2026 findings contradict their previous queries from earlier sessions. |
| 🔄 **Cyclic LangGraph Workflow** | A state-machine loop: Search → RAG → Contradiction Detection → (if contradiction found → Call Tools → Loop Back to RAG) → Memory Update → Report. |
| 📊 **Streamlit UI** | Beautiful, tabbed interface with markdown rendering, raw data inspection, memory visualization, and a download button for the report. |

---

## 🛠️ Tech Stack & Architecture

Chronos-Nexus is built as a cyclic state machine using **LangGraph**:

- **Orchestration**: LangChain + LangGraph (Cyclic State Machine)
- **LLM**: **xAI Grok** (`grok-2-1212`) or **Groq Cloud** (`llama-3.3-70b-versatile`) with fallback to **Google Gemini 2.0 Flash**
- **Embeddings**: Google Generative AI Embeddings (`models/embedding-001`)
- **Vector DB**: Chroma (In-Memory)
- **Web Search**: Tavily Search API (falls back to Smart Mock Data if API key is missing)
- **Frontend**: Streamlit

---

## 📦 Installation & Setup

### 1. Prerequisites
Ensure you have **Python 3.11+** installed.

### 2. Clone and Setup
```bash
git clone https://github.com/PrakharM9/Chronos-Nexus.git
cd Chronos-Nexus
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Environment Variables
Create a `.env` file in the root directory and add your API keys:
```env
# Required for LLM (Grok or Groq)
GROK_API_KEY="your_xai_grok_key_here"
GROQ_API_KEY="your_groq_api_key_here"

# Required for Vector Embeddings
GEMINI_API_KEY="your_gemini_api_key_here"

# Required for Live Web Search
TAVILY_API_KEY="your_tavily_api_key_here"
```

---

## 🚀 How to Run

### Option A: Command-Line Chatbot
```bash
python code.py
```

### Option B: Streamlit Web UI (Recommended)
```bash
streamlit run streamlit_app.py
```
Open `http://localhost:8501` in your browser.

---

## 📂 Project Structure

```text
chronos-nexus/
├── code.py                  # Core LangGraph workflow (RAG, Tools, Memory, Agents)
├── streamlit_app.py         # Streamlit UI interface
├── requirements.txt         # Python dependencies
├── .env                     # API keys config
└── README.md                # Project documentation
```
