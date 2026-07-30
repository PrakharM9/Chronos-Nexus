import streamlit as st
import os
import sys
from datetime import datetime

# Add the current directory to path so we can import code.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the workflow app and required types from code.py
# Ensure your code.py defines 'app', 'ChronosState', etc.
from code import app, ChronosState, HumanMessage, long_term_store  # Ensure your code.py defines 'app', 'ChronosState', etc.

# Set page config
st.set_page_config(
    page_title="Chronos-Nexus 2026",
    page_icon="🕒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better visuals
st.markdown("""
<style>
    .report-title {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 20px;
    }
    .year-badge {
        background-color: #ff7f0e;
        color: white;
        padding: 5px 12px;
        border-radius: 20px;
        font-weight: bold;
    }
    .contradiction-box {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 10px 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .memory-alert {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
        padding: 10px 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .footer {
        text-align: center;
        margin-top: 40px;
        color: #6c757d;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/time-machine.png", width=80)
    st.title("🧠 Chronos-Nexus")
    st.markdown("### Temporal Contradiction Engine")
    st.markdown("---")
    st.markdown("**Analyze facts across 2024 → 2026**")
    st.markdown("Detects reversals, contradictions, and evolutions.")
    st.markdown("---")
    st.caption(f"Current Year: {datetime.now().year}")
    
    # Option to clear memory (just for testing)
    if st.button("🗑️ Clear Long-Term Memory"):
        # Reset the store (optional)
        try:
            from code import long_term_store
            long_term_store.delete(("memories", "global_user"))
            st.success("Long-term memory cleared!")
        except:
            st.warning("Could not clear memory.")

# Main area
st.markdown('<div class="report-title">🕒 Chronos-Nexus 2026</div>', unsafe_allow_html=True)
st.markdown("### Ask about any topic – we'll show how it evolved across 2024, 2025, and 2026.")

# User input
user_query = st.text_input("💬 Enter your question:", placeholder="e.g., Is the Rupee falling?", key="query_input")

if st.button("🔍 Analyze", type="primary") or user_query:
    if not user_query.strip():
        st.warning("Please enter a question.")
        st.stop()
    
    with st.spinner("⏳ Retrieving data and analyzing contradictions across time..."):
        try:
            # Invoke the workflow (same as before)
            final_state = app.invoke(
                {
                    "messages": [HumanMessage(content=user_query)],
                    "query": user_query,
                    "iteration_count": 0
                },
                config={"configurable": {"thread_id": "streamlit_thread"}}
            )
            
            # Extract data
            report = final_state.get("final_report", "No report generated.")
            memory_alert = final_state.get("memory_alert", "")
            contradictions = final_state.get("contradictions", [])
            years_data = final_state.get("years_data", {})
            
            # Display the report in a nice format
            st.markdown("---")
            st.markdown("## 📊 Temporal Evolution Brief")
            
            # Use tabs for better organization
            tab1, tab2, tab3 = st.tabs(["📄 Report", "📈 Raw Data", "🧠 Memory"])
            
            with tab1:
                # Render markdown with proper formatting
                st.markdown(report)
                
                # If contradictions found, highlight them
                if contradictions:
                    st.markdown("---")
                    st.markdown("#### ⚠️ Contradictions Detected")
                    for contra in contradictions:
                        st.markdown(f'<div class="contradiction-box">🔍 {contra}</div>', unsafe_allow_html=True)
                
                # Memory alert
                if memory_alert:
                    st.markdown(f'<div class="memory-alert">🧠 {memory_alert}</div>', unsafe_allow_html=True)
            
            with tab2:
                st.subheader("Raw Data Retrieved by Year")
                for year, data in years_data.items():
                    with st.expander(f"📅 {year}"):
                        st.code(data, language="text")
            
            with tab3:
                st.subheader("Long-Term Memory (Past Queries)")
                # Access the store from code.py
                from code import long_term_store
                stored = long_term_store.get(("memories", "global_user"), "history")
                if stored and stored.value:
                    import json
                    st.json(stored.value)
                else:
                    st.info("No past queries stored yet.")
            
            # Download button for the report
            st.download_button(
                label="📥 Download Report (Markdown)",
                data=report,
                file_name=f"chronos_report_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
                mime="text/markdown"
            )
            
        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")
            st.stop()

# Footer
st.markdown('<div class="footer">Built with LangGraph, Gemini, and Streamlit | © 2026</div>', unsafe_allow_html=True)