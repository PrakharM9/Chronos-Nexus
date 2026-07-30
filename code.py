import os
import json
from typing import Annotated, List, Dict, Literal, Any, TypedDict
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# ==================== IMPORTS ====================
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI

# Gemini Imports (with Safety Settings)
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_google_genai import HarmCategory, HarmBlockThreshold

# Tavily (try modern, fallback to legacy)
try:
    from langchain_tavily import TavilySearch
    tavily_available = True
    TAVILY_CLASS = TavilySearch
except ImportError:
    try:
        from langchain_community.tools.tavily_search import TavilySearchResults
        tavily_available = True
        TAVILY_CLASS = TavilySearchResults
    except ImportError:
        tavily_available = False

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.prebuilt import ToolNode

# ==================== 1. CONFIGURATION ====================
CURRENT_YEAR = datetime.now().year
YEARS_TO_CHECK = [CURRENT_YEAR - 2, CURRENT_YEAR - 1, CURRENT_YEAR]

print(f"🕒 System initialized. Current year: {CURRENT_YEAR}")
print(f"📅 Analyzing temporal data across: {YEARS_TO_CHECK}")

# ---------- Setup LLM and Embeddings ----------
GROK_API_KEY = os.getenv("GROK_API_KEY") or os.getenv("XAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

# 1. Initialize LLM
if GROK_API_KEY:
    LLM = ChatOpenAI(
        model="grok-2-1212",
        api_key=GROK_API_KEY,
        base_url="https://api.xai.com/v1",
        temperature=0.7
    )
    print("✅ Using xAI Grok API for LLM.")
elif GROQ_API_KEY:
    LLM = ChatOpenAI(
        model="llama-3.3-70b-versatile",
        api_key=GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1",
        temperature=0.7
    )
    print("✅ Using Groq API (Llama 3.3) for LLM.")
elif GOOGLE_API_KEY:
    safety_settings = {
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    }
    LLM = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.7,
        google_api_key=GOOGLE_API_KEY,
        safety_settings=safety_settings
    )
    print("✅ Using Google Gemini for LLM.")
else:
    from langchain_community.llms import FakeListLLM
    LLM = FakeListLLM(responses=["Fallback: No LLM key found."])
    print("⚠️ No LLM key found (GROK_API_KEY, GROQ_API_KEY, or GOOGLE_API_KEY). Running with Fake LLM.")

# 2. Initialize Embeddings (independent of LLM selection)
if GOOGLE_API_KEY:
    EMBEDDINGS = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GOOGLE_API_KEY
    )
    print("✅ Using Google Gemini for Vector Embeddings.")
else:
    from langchain_community.embeddings import FakeEmbeddings
    EMBEDDINGS = FakeEmbeddings(size=384)
    print("⚠️ GOOGLE_API_KEY missing. Using Fake Embeddings.")

# ---------- Setup Tavily ----------
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
if TAVILY_API_KEY and tavily_available:
    try:
        tavily_tool = TAVILY_CLASS(api_key=TAVILY_API_KEY, max_results=4)
        print("✅ Tavily Search enabled (Live web data).")
    except Exception as e:
        tavily_tool = None
        print(f"⚠️ Tavily init failed: {e}. Using mock.")
else:
    tavily_tool = None
    print("⚠️ TAVILY_API_KEY missing. Running with MOCK search data.")

# ==================== 2. MEMORY SETUP ====================
long_term_store = InMemoryStore()
checkpointer = MemorySaver()

# ==================== 3. SMART MOCK DATA GENERATOR ====================
def generate_smart_mock(query: str, year: int) -> str:
    topics = {
        "rupee": f"In {year}, the Indian Rupee faced significant pressure against the USD. Forex reserves fluctuated, and RBI intervened to stabilize the currency.",
        "janta": f"In {year}, the 'Janta' movements gained significant traction.",
        "anti-national": f"In {year}, the term 'anti-national' was heavily debated.",
        "ai": f"In {year}, AI regulation became a global priority.",
        "stock": f"In {year}, the stock market saw a 15% correction.",
        "climate": f"In {year}, global temperatures reached record highs.",
    }
    for key in topics.keys():
        if key in query.lower():
            return f"{topics[key]} This marked a turning point."
    
    return f"In {year}, significant developments occurred regarding '{query}'. Stakeholders held summits and policy changes were announced."

# ==================== 4. TOOL CALLING DEFINITIONS ====================
@tool
def web_search_temporal(query: str, year: int) -> str:
    """Searches the web for a given query filtered for a specific year."""
    if tavily_tool:
        try:
            # ✅ FIX 2: Changed from .search() to .invoke()
            results = tavily_tool.invoke({"query": f"{query} {year} news"})
            formatted = f"--- Results for {year} ---\n"
            # Handle if results is a list of dicts or just a string
            if isinstance(results, list):
                for res in results[:3]:
                    formatted += f"Title: {res.get('title', '')}\nContent: {res.get('content', '')}\nURL: {res.get('url', '')}\n\n"
            else:
                formatted += str(results)
            return formatted
        except Exception as e:
            return f"Error searching {year}: {str(e)}"
    else:
        return generate_smart_mock(query, year)

@tool
def fact_check_db_lookup(claim: str) -> str:
    """Queries fact-checking databases."""
    if tavily_tool:
        try:
            # ✅ FIX 2: Changed from .search() to .invoke()
            results = tavily_tool.invoke({"query": f"fact check {claim} 2026"})
            if results and isinstance(results, list):
                return f"Fact-check: {results[0].get('content', 'No clear debunking.')}"
            return "Fact-check: No specific debunking found."
        except Exception as e:
            return f"Fact-check service unavailable: {e}"
    else:
        return "🔍 MOCK Fact-check: This claim is now challenged by new 2026 evidence."

@tool
def confidence_scorer(analysis: str) -> str:
    """Scores the volatility/contradiction level."""
    score = 50
    for word in ["however", "but", "contradict", "reversed", "changed", "shift", "debate"]:
        if word in analysis.lower():
            score += 10
    return f"📊 Volatility Score: {min(100, score)}/100."

@tool
def export_report(content: str, filename: str = "chronos_latest_report.md") -> str:
    """Exports the final analysis to a Markdown file."""
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"# 📄 Chronos-Nexus Report\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(content)
    return f"✅ Report exported to '{filename}'"

all_tools = [web_search_temporal, fact_check_db_lookup, confidence_scorer, export_report]
tool_node = ToolNode(all_tools)

# ==================== 5. LANGGRAPH STATE ====================
class ChronosState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    query: str
    years_data: Dict[int, str]
    rag_context: str
    contradictions: List[str]
    memory_alert: str
    final_report: str
    iteration_count: int

# ==================== 6. LANGGRAPH NODES ====================
def search_all_years(state: ChronosState) -> Dict:
    yearly_results = {}
    for year in YEARS_TO_CHECK:
        yearly_results[year] = web_search_temporal.invoke({"query": state["query"], "year": year})
    combined = "\n".join([f"--- YEAR {y} ---\n{d}" for y, d in yearly_results.items()])
    return {"years_data": yearly_results, "rag_context": combined, "messages": [AIMessage(content="Data retrieved.")]}

def build_and_retrieve_rag(state: ChronosState) -> Dict:
    raw_docs = []
    for year, content in state["years_data"].items():
        for idx, chunk in enumerate(content.split("\n\n")):
            if len(chunk.strip()) > 20:
                raw_docs.append(Document(page_content=chunk, metadata={"year": year}))
    if not raw_docs:
        return {"rag_context": "No documents found."}
    try:
        vectorstore = Chroma.from_documents(raw_docs, EMBEDDINGS, collection_name=f"temp_{datetime.now().timestamp()}")
        docs = vectorstore.as_retriever(search_kwargs={"k": 5}).invoke(state["query"])
        context = "🔍 **RAG RETRIEVAL RESULTS**:\n" + "\n".join([f"[Year: {d.metadata.get('year')}] {d.page_content}" for d in docs])
        return {"rag_context": context, "messages": [AIMessage(content="RAG done.")]}
    except Exception as e:
        return {"rag_context": f"RAG fallback: {str(e)}", "messages": [AIMessage(content="RAG fallback.")]}

def detect_contradictions(state: ChronosState) -> Dict:
    try:
        prompt = ChatPromptTemplate.from_template(
            "Analyze this data from {years}. Look for contradictions or reversals.\n"
            "If you see clear disagreements, output 'CONTRADICTION_FOUND'. Else 'NO_CONTRADICTION'.\n\nData: {context}"
        )
        result = (prompt | LLM).invoke({"years": str(YEARS_TO_CHECK), "context": state["rag_context"][:2000]})
        is_contra = "CONTRADICTION_FOUND" in result.content
        return {"contradictions": [result.content] if is_contra else [], "messages": [result]}
    except Exception as e:
        print(f"⚠️ Contradiction detection skipped: {e}")
        return {"contradictions": [], "messages": [AIMessage(content="Analysis skipped.")]}

def execute_additional_tools(state: ChronosState) -> Dict:
    text = state["rag_context"][:1000]
    return {
        "messages": [
            ToolMessage(content=fact_check_db_lookup.invoke({"claim": text}), tool_call_id="fc"),
            ToolMessage(content=confidence_scorer.invoke({"analysis": text}), tool_call_id="cs")
        ]
    }

def update_long_term_memory(state: ChronosState) -> Dict:
    namespace = ("memories", "global_user")
    existing = long_term_store.get(namespace, "history")
    history = existing.value if existing else []
    history.append({
        "timestamp": datetime.now().isoformat(),
        "query": state["query"],
        "contradiction": len(state["contradictions"]) > 0
    })
    long_term_store.put(namespace, "history", history)
    
    alert = ""
    if len(history) > 1:
        if history[-2]["query"].lower() in state["query"].lower() or state["query"].lower() in history[-2]["query"].lower():
            if history[-2]["contradiction"] != history[-1]["contradiction"]:
                alert = f"🧠 MEMORY ALERT: On {history[-2]['timestamp']} you asked about '{history[-2]['query']}'. New findings overturn the old narrative!"
    
    return {"memory_alert": alert, "messages": [AIMessage(content=alert or "Memory saved.")]}

def generate_final_report(state: ChronosState) -> Dict:
    try:
        raw_data = ""
        for year, data in state.get("years_data", {}).items():
            raw_data += f"**{year}**:\n{data}\n\n"
        
        full_context = f"RAW DATA:\n{raw_data}\n\nRAG RESULTS:\n{state.get('rag_context', '')}"
        
        prompt = ChatPromptTemplate.from_template(
            "You are a financial/historical analyst. Write a detailed 'Temporal Evolution Brief' for: '{query}'.\n"
            "Cover the timeline from {years}.\n"
            "Structure your report:\n"
            "1. **Overview**\n"
            "2. **2024 Baseline**\n"
            "3. **2025 Transition**\n"
            "4. **2026 Current Reality**\n"
            "5. **Contradictions Found**\n"
            "6. **Final Verdict**\n\n"
            "DATA:\n{context}"
        )
        result = (prompt | LLM).invoke({
            "query": state["query"],
            "years": str(YEARS_TO_CHECK),
            "context": full_context[:3000]
        })
        report_content = result.content
    except Exception as e:
        print(f"⚠️ LLM generation failed ({e}). Using raw data fallback.")
        report_content = f"# 🗂️ Chronos-Nexus Report\n\n**Topic:** {state['query']}\n\n"
        for year, data in state.get("years_data", {}).items():
            report_content += f"## 📌 {year}\n{data}\n\n"
    
    export_report.invoke({"content": report_content})
    return {"final_report": report_content, "messages": [AIMessage(content="✅ Report exported.")]}

# ==================== 7. BUILD GRAPH ====================
def should_loop(state: ChronosState) -> Literal["execute_tools", "update_memory"]:
    return "execute_tools" if state.get("contradictions") and state.get("iteration_count", 0) < 1 else "update_memory"

workflow = StateGraph(ChronosState)
workflow.add_node("search_years", search_all_years)
workflow.add_node("build_rag", build_and_retrieve_rag)
workflow.add_node("detect_contradictions", detect_contradictions)
workflow.add_node("execute_tools", execute_additional_tools)
workflow.add_node("update_memory", update_long_term_memory)
workflow.add_node("generate_report", generate_final_report)

workflow.set_entry_point("search_years")
workflow.add_edge("search_years", "build_rag")
workflow.add_edge("build_rag", "detect_contradictions")
workflow.add_conditional_edges("detect_contradictions", should_loop, {
    "execute_tools": "execute_tools",
    "update_memory": "update_memory"
})
workflow.add_edge("execute_tools", "build_rag")
workflow.add_edge("update_memory", "generate_report")
workflow.add_edge("generate_report", END)

app = workflow.compile(checkpointer=checkpointer, store=long_term_store)

# ==================== 8. CHATBOT LOOP ====================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 CHRONOS-NEXUS CHATBOT (2024 → 2026)")
    print("="*60)
    print("💡 Type any topic (politics, tech, finance, etc.).")
    print("💡 Type 'exit' to quit.")
    print("="*60 + "\n")
    
    config = {"configurable": {"thread_id": "chatbot_thread_001"}}
    
    while True:
        try:
            user_input = input("🧑 You: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit", "bye"]:
                print("👋 Goodbye! Check 'chronos_latest_report.md' for the last analysis.")
                break
            
            print("\n⏳ Analyzing across 2024, 2025, 2026...\n")
            
            final_state = app.invoke(
                {"messages": [HumanMessage(content=user_input)], "query": user_input, "iteration_count": 0},
                config=config
            )
            
            print("\n" + "="*60)
            print("📊 CHRONOS-NEXUS REPORT:")
            print("="*60)
            print(final_state["final_report"])
            
            if final_state.get("memory_alert"):
                print("\n" + "="*60)
                print("🧠 " + final_state["memory_alert"])
                print("="*60)
            
            print("\n" + "-"*60)
            print("💡 Ask another topic or type 'exit'.\n")
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            print("Please try a different topic.\n")