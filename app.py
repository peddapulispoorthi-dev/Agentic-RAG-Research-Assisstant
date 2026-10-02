import os
import time
from pathlib import Path
import streamlit as st

from src.config import settings
from src.ingest import ingest_pdfs
from src.service import run_agent_query
from utils.exporter import export_to_markdown, export_to_pdf

st.set_page_config(
    page_title="Agentic RAG Research Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling for enterprise polish
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #0EA5E9;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.0rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    .subquery-box {
        background-color: #F0F9FF;
        border-left: 4px solid #0284C7;
        padding: 0.8rem;
        margin-bottom: 0.8rem;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🔬 Agentic RAG Research Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Production-grade Multi-Perspective Deep Research & Autonomous Agent System</div>', unsafe_allow_html=True)

# --- SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.header("⚙️ System Control Panel")
    
    # Model & Tracing status
    st.subheader("Model & Observability")
    st.info(f"**LLM Provider:** `{settings.LLM_PROVIDER.upper()}`\n\n**Model:** `{settings.LLM_MODEL}`")
    
    if settings.LANGCHAIN_TRACING_V2:
        st.success(f"🟢 **LangSmith Tracing:** Active\n\nProject: `{settings.LANGCHAIN_PROJECT}`")
    else:
        st.warning("⚪ **LangSmith Tracing:** Inactive")

    st.divider()
    
    # Document Ingestion UI & Index Status
    st.subheader("📁 Document Library & Indexing")
    
    raw_dir = Path(settings.RAW_DIR)
    raw_dir.mkdir(parents=True, exist_ok=True)
    existing_pdfs = list(raw_dir.glob("*.pdf"))
    index_exists = os.path.exists(os.path.join(settings.INDEX_DIR, "index.faiss"))
    
    st.caption(f"**Index Status:** {'✅ FAISS + BM25 Ready' if index_exists else '⚠️ No Index Found'}")
    st.caption(f"**Stored PDFs:** {len(existing_pdfs)} file(s)")

    uploaded_files = st.file_uploader(
        "Upload PDF Research Papers / Docs", 
        type=["pdf"], 
        accept_multiple_files=True
    )
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📥 Ingest PDFs", use_container_width=True):
            if uploaded_files:
                for file in uploaded_files:
                    target_path = raw_dir / file.name
                    with open(target_path, "wb") as f:
                        f.write(file.getbuffer())
                st.toast(f"Saved {len(uploaded_files)} PDF(s) to `{settings.RAW_DIR}`", icon="✅")
            
            with st.spinner("Indexing PDFs into FAISS vectorstore & BM25..."):
                ingest_pdfs(reset=False)
            st.success("Indexing complete!")
            st.rerun()

    with col2:
        if st.button("🗑️ Reset Index", use_container_width=True):
            with st.spinner("Clearing vectorstore index..."):
                ingest_pdfs(reset=True)
            st.warning("Index cleared!")
            st.rerun()

    st.divider()

    # Research Mode Selection
    st.subheader("🎯 Agent Synthesis Mode")
    research_mode = st.radio(
        "Select Workflow Mode:",
        options=["Standard Agentic QA", "Deep Research Synthesis Mode"],
        index=1,
        help="Deep Research Mode decomposes broad prompts into 3 distinct sub-queries (Technical, Comparative, Future Outlook) for academic synthesis."
    )
    is_deep_research = research_mode == "Deep Research Synthesis Mode"

# --- MAIN WORKFLOW INTERFACE ---
query = st.text_area(
    "Enter your research prompt or query:",
    placeholder="e.g. Compare LangGraph and AutoGen for stateful multi-agent orchestration, highlighting architectural mechanisms, comparative trade-offs, and scaling implications.",
    height=110
)

if st.button("🚀 Run Agentic Research", type="primary", use_container_width=True):
    if not query.strip():
        st.error("Please enter a valid research query.")
    else:
        with st.spinner("Executing Agent Graph (Decomposition ➔ Parallel Retrieval ➔ Grading ➔ Synthesis ➔ Grounding)..."):
            results = run_agent_query(
                question=query.strip(),
                is_deep_research=is_deep_research
            )

        st.session_state["last_results"] = results
        st.session_state["last_query"] = query

if "last_results" in st.session_state:
    results = st.session_state["last_results"]
    last_query = st.session_state.get("last_query", "Research Topic")
    
    st.divider()
    
    # Latency and Status Metrics Bar
    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        st.metric("Total Latency", f"{results.get('latency_seconds', 0.0)}s")
    with col_b:
        st.metric("Web Fallback Used", "Yes" if results.get("used_web_search") else "No")
    with col_c:
        st.metric("Grounding Verified", "Passed" if results.get("grounded", True) else "Regenerated")
    with col_d:
        st.metric("Retrieved Sources", len(results.get("sources", [])))

    # Output Tabs
    tab_report, tab_subqueries, tab_path, tab_sources = st.tabs([
        "📜 Generated Report / Synthesis",
        "🔀 Multi-Query Decomposition",
        "🗺️ Agent Execution Path",
        "📚 Source References"
    ])

    with tab_report:
        report_content = results.get("report") or results.get("answer")
        st.markdown(report_content)
        
        st.divider()
        st.subheader("📥 Export Options")
        exp_col1, exp_col2 = st.columns(2)
        
        with exp_col1:
            md_data = export_to_markdown(report_content)
            st.download_button(
                label="📄 Download Markdown (.md)",
                data=md_data,
                file_name="Deep_Research_Report.md",
                mime="text/markdown",
                use_container_width=True
            )
            
        with exp_col2:
            pdf_bytes = export_to_pdf(report_content)
            st.download_button(
                label="📕 Download PDF Report (.pdf)",
                data=pdf_bytes,
                file_name="Deep_Research_Report.pdf",
                mime="application/pdf",
                use_container_width=True
            )

    with tab_subqueries:
        sub_queries = results.get("sub_queries", [])
        if sub_queries:
            st.markdown("### Multi-Perspective Decomposed Sub-Queries:")
            perspectives = [
                "1️⃣ Technical Architecture & Core Principles", 
                "2️⃣ Comparative Analysis & Trade-offs", 
                "3️⃣ Future Implications & Strategic Outlook"
            ]
            for i, sq in enumerate(sub_queries):
                label = perspectives[i] if i < len(perspectives) else f"Sub-query {i+1}"
                st.markdown(f'<div class="subquery-box"><strong>{label}:</strong> {sq}</div>', unsafe_allow_html=True)
        else:
            st.info("Standard single-query retrieval was executed.")

    with tab_path:
        path = results.get("path", [])
        st.write("**Ordered Agent Node Visited Sequence:**")
        st.code(" ➔ ".join(path), language="text")

    with tab_sources:
        sources = results.get("sources", [])
        if sources:
            for idx, src in enumerate(sources, 1):
                with st.expander(f"[{idx}] {src.get('source')} (Page {src.get('page') or 'N/A'})"):
                    st.write(src.get("snippet"))
        else:
            st.write("No source documents were retrieved.")
