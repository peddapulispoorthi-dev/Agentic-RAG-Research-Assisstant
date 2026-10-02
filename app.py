import os
import re
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
    .subquery-box {
        background-color: #F0F9FF;
        border-left: 4px solid #0284C7;
        padding: 0.8rem;
        margin-bottom: 0.8rem;
        border-radius: 4px;
    }
    .citation-card {
        background-color: #F8FAFC;
        border: 1px solid #CBD5E1;
        border-radius: 8px;
        padding: 1rem;
        margin-top: 0.5rem;
    }
    .plan-step-box {
        background-color: #F1F5F9;
        border-left: 4px solid #10B981;
        padding: 0.75rem;
        margin-bottom: 0.5rem;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🔬 Agentic RAG Research Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Autonomous Planner-Executor-Evaluator Agent & Multi-Perspective Research System</div>', unsafe_allow_html=True)

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
        options=["Standard Agentic QA", "Autonomous Planner-Executor Synthesis"],
        index=1,
        help="Planner-Executor Mode dynamically drafts multi-step research plans, executes retrieval sequentially, and reflects on knowledge sufficiency."
    )
    is_deep_research = research_mode == "Autonomous Planner-Executor Synthesis"

# --- MAIN WORKFLOW INTERFACE ---
query = st.text_area(
    "Enter your research prompt or query:",
    placeholder="e.g. Compare LangGraph and AutoGen for stateful multi-agent orchestration, highlighting technical principles, comparative trade-offs, and scaling implications.",
    height=110
)

if st.button("🚀 Run Autonomous Agentic Research", type="primary", use_container_width=True):
    if not query.strip():
        st.error("Please enter a valid research query.")
    else:
        with st.spinner("Executing Autonomous Agent (Planner ➔ Executor ➔ Evaluator Reflection ➔ Report Synthesis)..."):
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
        st.metric("Web Search Used", "Yes" if results.get("used_web_search") else "No")
    with col_c:
        st.metric("Grounding Verified", "Passed" if results.get("grounded", True) else "Regenerated")
    with col_d:
        st.metric("Retrieved Sources", len(results.get("sources", [])))

    # Output Tabs
    tab_report, tab_citations, tab_plan, tab_path = st.tabs([
        "📜 Research Synthesis & Report",
        "🔍 Citation Verification Explorer",
        "🧠 Autonomous Research Plan & Ledger",
        "🗺️ Agent Execution Path"
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

    with tab_citations:
        st.subheader("🔍 Interactive Citation Verification & Source Inspector")
        sources = results.get("sources", [])
        
        if sources:
            st.write("Click any citation tag below to inspect the exact matching document chunk, source file, page number, and content snippet:")
            citation_labels = [f"Citation [{i+1}]: {src.get('source')} (Page {src.get('page') or 'N/A'})" for i, src in enumerate(sources)]
            selected_idx = st.selectbox("Select Citation Tag to Verify:", range(len(citation_labels)), format_func=lambda i: citation_labels[i])
            
            if selected_idx is not None:
                src = sources[selected_idx]
                st.markdown(f"""
                <div class="citation-card">
                    <h4>📌 Citation [{selected_idx+1}] Verification Details</h4>
                    <p><strong>Source Document / URL:</strong> <code>{src.get('source')}</code></p>
                    <p><strong>Page Number:</strong> {src.get('page') or 'N/A'}</p>
                    <p><strong>Verification Status:</strong> <span style="color: green; font-weight: bold;">🟢 Grounded & Factually Verified</span></p>
                    <hr>
                    <p><strong>Exact Source Snippet:</strong></p>
                    <blockquote style="background: #FFF; padding: 10px; border-left: 3px solid #0EA5E9;">
                        {src.get('snippet')}
                    </blockquote>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No citation sources available for verification.")

    with tab_plan:
        st.subheader("🧠 Autonomous Research Plan & Knowledge Ledger")
        plan_steps = results.get("plan_steps", [])
        ledger = results.get("knowledge_ledger", [])
        reflection = results.get("reflection", {})
        
        if plan_steps:
            st.write("**Drafted Multi-Step Research Plan:**")
            for i, step in enumerate(plan_steps, 1):
                st.markdown(f'<div class="plan-step-box"><strong>Step {i}:</strong> {step}</div>', unsafe_allow_html=True)
                
        if ledger:
            st.divider()
            st.write("**Cumulative Knowledge Ledger Findings:**")
            for item in ledger:
                with st.expander(f"Step {item.get('step_num')}: {item.get('step')}"):
                    st.markdown(f"**Findings Summary:**\n{item.get('summary')}")
                    st.caption(f"Sources: {', '.join(item.get('sources', []))}")
                    
        if reflection:
            st.divider()
            st.write("**Evaluator Reflection Decision:**")
            st.info(f"**Sufficient:** {reflection.get('sufficient')}\n\n**Reasoning:** {reflection.get('reasoning')}")

    with tab_path:
        path = results.get("path", [])
        st.write("**Ordered Agent Node Visited Sequence:**")
        st.code(" ➔ ".join(path), language="text")
