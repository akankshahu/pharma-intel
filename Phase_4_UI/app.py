# Phase 4: Streamlit Web Interface
# Interactive UI for Pharma-Intellect RAG Engine
# ==================================================

import streamlit as st
import sys
import os
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Phase_3_RAG_Engine.rag_engine import PharmaIntellectRAG
from config import APP_NAME, APP_VERSION, APP_DESCRIPTION

# ============================================
# Page Configuration
# ============================================

st.set_page_config(
    page_title=f"{APP_NAME} | AI Research Assistant",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# Custom CSS
# ============================================

st.markdown("""
<style>
    /* Main styling */
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        margin-bottom: 10px;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #555;
        margin-bottom: 30px;
    }
    .source-box {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
        border-left: 4px solid #1f77b4;
    }
    .answer-box {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 5px;
        margin: 15px 0;
        border-left: 4px solid #28a745;
    }
    .loading-spinner {
        text-align: center;
        font-size: 1.2rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# Session State Management
# ============================================

if 'rag_engine' not in st.session_state:
    with st.spinner("⏳ Loading RAG Engine (this may take a moment)..."):
        try:
            st.session_state.rag_engine = PharmaIntellectRAG()
            st.session_state.engine_ready = True
        except Exception as e:
            st.session_state.engine_ready = False
            st.session_state.error = str(e)

if 'query_history' not in st.session_state:
    st.session_state.query_history = []

# ============================================
# Header Section
# ============================================

col1, col2 = st.columns([3, 1])

with col1:
    st.markdown(f"""
<div class='main-header'>💊 {APP_NAME}</div>
<div class='sub-header'>{APP_DESCRIPTION}</div>
""", unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div style='text-align: right; color: #888; font-size: 0.9rem;'>
        <p><strong>v{APP_VERSION}</strong></p>
        <p>Powered by LlamaIndex & OpenAI</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================
# Information Box
# ============================================

st.markdown("""
---
**🔬 About Pharma-Intellect:**

Pharma-Intellect is an AI-powered research assistant designed to accelerate pharmaceutical R&D. It combines the power of **Retrieval-Augmented Generation (RAG)** with biomedical literature and clinical trial data to help researchers:

- 🎯 Identify promising drug targets
- 🔬 Discover drug repurposing opportunities
- 📚 Streamline literature reviews
- 🔗 Access source citations for every answer

**Data Sources:** PubMed + ClinicalTrials.gov | **Model:** GPT-3.5-Turbo | **Vector Database:** ChromaDB

---
""")

# ============================================
# Sidebar Configuration
# ============================================

with st.sidebar:
    st.header("⚙️ Settings & Examples")

    # Check if engine is ready
    if not st.session_state.engine_ready:
        st.error(f"❌ Engine Error: {st.session_state.error}")
        st.stop()

    # Number of sources slider
    num_sources = st.slider(
        "📊 Number of sources to retrieve:",
        min_value=3,
        max_value=10,
        value=5,
        step=1,
        help="More sources = more context but slower responses"
    )

    st.markdown("---")

    # Example queries
    st.subheader("📖 Quick Start Examples")
    st.markdown("Click any example to search:")

    examples = [
        "What are recent advances in oncology drug discovery?",
        "Summarize phase 2 clinical trials for autoimmune disorders",
        "What adverse events are associated with risperidone?",
        "Explain drug repurposing strategies in modern medicine",
        "What is the current status of hypertension clinical trials?"
    ]

    for example in examples:
        if st.button(f"🔍 {example[:45]}...", key=example, use_container_width=True):
            st.session_state.selected_example = example

    st.markdown("---")

    # Query history
    if st.session_state.query_history:
        st.subheader("📋 Recent Searches")
        for i, q in enumerate(st.session_state.query_history[-5:], 1):
            if st.button(f"{i}. {q[:40]}...", key=f"history_{i}", use_container_width=True):
                st.session_state.selected_history = q

    st.markdown("---")

    # Info
    st.info(
        "💡 **Tip:** Ask specific, detailed questions for better results. "
        "Include medical terms, drug names, or conditions for best performance."
    )

# ============================================
# Main Query Interface
# ============================================

st.header("🔍 Ask Your Research Question")

# Check if example or history was selected
if 'selected_example' in st.session_state:
    query_text = st.session_state.selected_example
    del st.session_state.selected_example
elif 'selected_history' in st.session_state:
    query_text = st.session_state.selected_history
    del st.session_state.selected_history
else:
    query_text = ""

# Query input
user_query = st.text_area(
    "Enter your pharmaceutical research question:",
    value=query_text,
    height=100,
    placeholder="Example: What are the latest findings on risperidone efficacy in treating autoimmune disorders based on phase 2 clinical trials?",
    help="Be specific and include relevant medical terms for best results"
)

# Query buttons
col1, col2, col3 = st.columns([1, 1, 2])

with col1:
    search_button = st.button("🚀 Search", use_container_width=True)

with col2:
    clear_button = st.button("🗑️ Clear", use_container_width=True)

with col3:
    st.markdown("*or select an example from the sidebar →*")

# ============================================
# Query Processing
# ============================================

if clear_button:
    st.rerun()

if search_button and user_query:
    # Add to history
    if user_query not in st.session_state.query_history:
        st.session_state.query_history.append(user_query)

    with st.spinner("🔄 Searching millions of documents..."):
        try:
            # Execute RAG query
            result = st.session_state.rag_engine.query(user_query, num_results=num_sources)

            # Display success message
            st.success("✅ Analysis Complete!", icon="✅")

            # ====== Display Answer ======
            st.subheader("📄 Answer")
            st.markdown(f"""
<div class='answer-box'>
{result['answer']}
</div>
""", unsafe_allow_html=True)

            # ====== Display Sources ======
            st.subheader(f"📚 Sources Retrieved ({len(result['sources'])} total)")

            if result['sources']:
                # Create columns for sources
                for i, source in enumerate(result['sources'], 1):
                    with st.expander(
                        f"📖 Source {i}: {source['title'][:70]}{'...' if len(source['title']) > 70 else ''}",
                        expanded=(i == 1)
                    ):
                        col1, col2 = st.columns([3, 1])

                        with col1:
                            st.write(f"**Title:** {source['title']}")
                            st.write(f"**Source:** {source['source']}")
                            st.write(f"**Relevance Score:** {source['relevance']:.2%}")

                        with col2:
                            if source['url'] != '#':
                                st.markdown(f"[🔗 View Original]({source['url']})")
            else:
                st.info("No sources found for this query.")

            # ====== Metadata ======
            st.markdown("---")
            st.markdown(f"""
            <div style='text-align: center; color: #888; font-size: 0.85rem;'>
                <p>Query processed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} |
                   {result['num_sources_retrieved']} documents retrieved</p>
            </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"❌ Error processing query: {str(e)}")
            st.info("Please ensure your OpenAI API key is configured and valid.")

elif search_button and not user_query:
    st.warning("⚠️ Please enter a research question first.")

# ============================================
# Footer
# ============================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888; font-size: 0.85rem;'>
    <p>
    💡 <strong>Citation:</strong> Every answer is backed by citations to original sources for scientific rigor.
    </p>
    <p>
    🔐 <strong>Privacy:</strong> Your queries are processed locally. No data is stored or sold.
    </p>
    <p>
    📊 Pharma-Intellect v""" + APP_VERSION + """ | Powered by LlamaIndex, ChromaDB & OpenAI
    </p>
</div>
""", unsafe_allow_html=True)
