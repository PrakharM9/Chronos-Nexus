🕒 Chronos-Nexus: The Temporal RAG Engine


Chronos-Nexus is an AI-powered "Temporal Contradiction Engine" that doesn't just answer your question—it shows you how the answer has evolved over the last 3 years (2024 → 2025 → 2026).

Unlike standard RAG chatbots that assume truth is static, Chronos-Nexus actively detects policy reversals, scientific disagreements, and market shifts, providing a structured Evolution Brief for any open-domain topic (Finance, Politics, Technology, Healthcare, etc.).

✨ Key Features
Feature	Description
🔍 Temporal RAG	Indexes data from 2024, 2025, and 2026 into a Chroma vector store. Semantic retrieval ensures the most relevant historical context is fetched.
🛠️ Intelligent Tool Calling	The LangGraph agent autonomously calls web search (Tavily), fact-checkers, confidence/volatility scorers, and a report exporter to enrich the analysis.
🧠 Dual-Layer Memory	Short-term (MemorySaver) maintains conversation context. Long-term (InMemoryStore) alerts the user if new 2026 findings contradict their previous queries from earlier sessions.
🔄 Cyclic LangGraph Workflow	A state-machine loop: Search → RAG → Contradiction Detection → (if contradiction found → Call Tools → Loop Back to RAG) → Memory Update → Report.
📊 Streamlit UI	Beautiful, tabbed interface with markdown rendering, raw data inspection, memory visualization, and a download button for the report.
🏗️ System Architecture










🛠️ Tech Stack
Orchestration: LangChain + LangGraph (Cyclic State Machine)

LLM: Google Gemini 2.0 Flash (Free Tier) / Fallback Mock LLM

Embeddings: Google Generative AI Embeddings (models/embedding-001)

Vector DB: Chroma (In-Memory)

Web Search: Tavily API (Optional, falls back to Smart Mock Data)

Frontend: Streamlit

Memory: InMemoryStore (Long-term) + MemorySaver (Short-term)

📦 Installation & Setup
1. Clone the Repository
bash
git clone https://github.com/yourusername/chronos-nexus.git
cd chronos-nexus
2. Create a Virtual Environment
bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
3. Install Dependencies
bash
pip install -r requirements.txt
requirements.txt:

text
langgraph
langchain
langchain-core
langchain-google-genai
langchain-chroma
langchain-community
tavily-python
streamlit
python-dotenv
4. Environment Variables (Optional)
Create a .env file in the root directory:

env
GOOGLE_API_KEY=your_gemini_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
Note: The system runs perfectly without API keys using the Smart Mock Data Generator.

🚀 How to Run
Option A: Command-Line Chatbot
bash
python code.py
Option B: Streamlit Web UI (Recommended)
bash
streamlit run streamlit_app.py
📂 Project Structure
text
chronos-nexus/
├── code.py                # Core LangGraph workflow (RAG, Tools, Memory, Agents)
├── streamlit_app.py       # Streamlit UI interface
├── chronos_latest_report.md # Exported report (generated on query)
├── requirements.txt       # Python dependencies
├── .env                   # API keys (optional)
└── README.md              # This file
