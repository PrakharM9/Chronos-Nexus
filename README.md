# 🕒 Chronos-Nexus – Temporal Contradiction Engine

**Chronos-Nexus** is an AI-powered "Temporal Contradiction Engine" designed to answer queries by analyzing how facts, policies, and narratives evolve over time (specifically across **2024 → 2025 → 2026**). 

Unlike standard chatbots that provide a static, single answer, Chronos-Nexus explicitly highlights reversals, shifts, and contradictions across different years.

---

## 🎯 Features

- **Temporal Search**: Searches the live web using Tavily, partitioning results by target years (2024, 2025, 2026).
- **RAG-based Context Retrieval**: Embeds yearly reports and retrieves context relevant to the user query using Gemini Embeddings and a Chroma vector store.
- **Contradiction Detection**: Employs an LLM to identify changes and reversals across the timeline.
- **Long-Term Memory Alerts**: Remembers past user queries. If a new query yields findings that contradict a past query, it triggers a memory alert to notify you of the shift.
- **Auto-Export**: Automatically exports the generated temporal report to a Markdown file.
- **Streamlit Interface**: Clean, reactive web interface for interacting with the engine and viewing raw data, analysis, and memory logs.

---

## 🛠️ Architecture & Tech Stack

Chronos-Nexus is built as a state graph agent using **LangGraph**:

1. **Orchestrator/Graph State**: Tracks the conversation history, yearly data, RAG context, and contradiction logs.
2. **LLM Engine**: Supports **xAI Grok** or **Groq Cloud (Llama 3.3)** with automatic fallback to **Google Gemini** depending on available API keys.
3. **Embeddings**: Uses Google Generative AI embeddings.
4. **Search Tool**: Live search using Tavily.
5. **Memory Store**: Custom `InMemoryStore` for long-term historical query tracking.

---

## 🚀 Setup & Installation

### 1. Prerequisites
Ensure you have **Python 3.11+** installed.

### 2. Install Dependencies
Clone the repository and install the required packages:
```bash
pip install -r requirements.txt
```

### 3. Environment Configuration
Create a `.env` file in the root directory (based on the provided template) and add your API keys:
```env
# Required for LLM (Grok/Groq) and Embeddings (Gemini)
GROQ_API_KEY="your-groq-key-here"
GROK_API_KEY="your-xai-grok-key-here"
GEMINI_API_KEY="your-gemini-key-here"

# Required for Web Searching
TAVILY_API_KEY="your-tavily-key-here"
```

---

## 💻 How to Run

### Command Line Interface
To run the CLI chatbot interface:
```bash
python code.py
```

### Streamlit Web App
To run the web interface:
```bash
streamlit run streamlit_app.py
```
Open `http://localhost:8501` in your browser.
