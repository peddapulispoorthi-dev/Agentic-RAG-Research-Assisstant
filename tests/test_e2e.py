import os
import pytest
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document

from src.schemas import RouteDecision, RelevanceGrade, GroundingGrade, MultiQueryDecomposition
from src.state import GraphState
from src.nodes import (
    route_question,
    rewrite_followup,
    decompose_query,
    parallel_retrieve,
    grade_documents,
    transform_query,
    web_search,
    generate,
    synthesize_deep_research_report,
    check_grounding,
    regenerate_strict,
    chitchat
)
from src.service import run_agent_query
from utils.exporter import export_to_markdown, export_to_pdf

# Sample test documents
DOC_1 = Document(page_content="LangGraph supports state persistence and cyclical multi-agent graph flows.", metadata={"source": "doc1.pdf", "page": 1, "chunk_id": "c1"})
DOC_2 = Document(page_content="AutoGen provides conversational multi-agent frameworks.", metadata={"source": "doc2.pdf", "page": 2, "chunk_id": "c2"})

# --- UNIT & INTEGRATION NODE TESTS ---

@patch("src.nodes.call_llm_with_structured_output")
def test_route_question_vectorstore(mock_call):
    mock_call.return_value = RouteDecision(datasource="vectorstore", reasoning="Document question")
    state = {"question": "What is in doc1.pdf?", "history": [], "path": []}
    res = route_question(state)
    assert res["route"] == "vectorstore"
    assert "route_question" in res["path"]

@patch("src.nodes.call_llm_with_structured_output")
def test_route_question_deep_research(mock_call):
    state = {"question": "Provide a comprehensive report on RAG.", "is_deep_research": True, "history": [], "path": []}
    res = route_question(state)
    assert res["route"] == "deep_research"
    assert res["is_deep_research"] is True

@patch("src.nodes.call_llm_with_structured_output")
def test_decompose_query(mock_call):
    mock_call.return_value = MultiQueryDecomposition(
        technical_definition="Technical definition of RAG architecture",
        comparative_analysis="Comparative analysis of FAISS vs Chroma",
        future_implications="Future implications of graph RAG"
    )
    state = {"question": "Tell me about RAG", "path": []}
    res = decompose_query(state)
    assert len(res["sub_queries"]) == 3
    assert "decompose_query" in res["path"]

@patch("src.nodes.get_retriever")
def test_parallel_retrieve(mock_get_retriever):
    mock_retriever = MagicMock()
    mock_retriever.parallel_hybrid_retrieve.return_value = [DOC_1, DOC_2]
    mock_get_retriever.return_value = mock_retriever

    state = {
        "sub_queries": ["q1", "q2", "q3"],
        "question": "tell me about agents",
        "path": []
    }
    res = parallel_retrieve(state)
    assert len(res["candidate_documents"]) == 2
    assert "parallel_retrieve" in res["path"]

@patch("src.nodes.call_llm_with_structured_output")
def test_grade_documents(mock_call):
    mock_call.return_value = RelevanceGrade(relevant=True, reasoning="Relevant chunk")
    state = {
        "question": "What is LangGraph?",
        "candidate_documents": [DOC_1, DOC_2],
        "path": []
    }
    res = grade_documents(state)
    assert len(res["documents"]) == 2
    assert "grade_documents" in res["path"]

@patch("src.nodes.TavilyClient")
def test_web_search_failure_degradation(mock_tavily_client):
    """Test circuit breaker when web search API fails or throws rate limit / key exception."""
    mock_tavily_client.side_effect = Exception("Tavily API Rate Limit 429 / Key Failure")
    
    with patch("src.config.settings.TAVILY_API_KEY", "dummy_key"):
        state = {"question": "Latest AI news", "path": []}
        res = web_search(state)
        # Verify circuit breaker caught exception and returned empty list gracefully
        assert res["web_results"] == []
        assert "web_search" in res["path"]

@patch("src.nodes.call_llm_with_structured_output")
def test_check_grounding_rejection(mock_call):
    mock_call.return_value = GroundingGrade(grounded=False, unsupported_claims=["Unfounded claim"])
    state = {
        "documents": [DOC_1],
        "answer": "Some hallucinated claim.",
        "path": []
    }
    res = check_grounding(state)
    assert res["grounded"] is False
    assert "check_grounding" in res["path"]

# --- END-TO-END GRAPH WORKFLOW TESTS ---

@patch("src.nodes.get_llm")
@patch("src.nodes.get_retriever")
@patch("src.nodes.call_llm_with_structured_output")
def test_e2e_deep_research_flow(mock_structured_call, mock_get_retriever, mock_get_llm):
    # Mock LLM and Retriever responses
    mock_retriever = MagicMock()
    mock_retriever.parallel_hybrid_retrieve.return_value = [DOC_1, DOC_2]
    mock_get_retriever.return_value = mock_retriever

    def mock_structured_side_effect(prompt, schema, system_instruction=""):
        if schema == MultiQueryDecomposition:
            return MultiQueryDecomposition(
                technical_definition="RAG technical definition",
                comparative_analysis="RAG comparison",
                future_implications="RAG future"
            )
        elif schema == RelevanceGrade:
            return RelevanceGrade(relevant=True, reasoning="Relevant")
        elif schema == GroundingGrade:
            return GroundingGrade(grounded=True, unsupported_claims=[])
        return None

    mock_structured_call.side_effect = mock_structured_side_effect

    mock_llm_instance = MagicMock()
    mock_llm_response = MagicMock()
    mock_llm_response.content = "# Deep Research Report: Agentic RAG\n\n## Executive Summary\nTest report summary."
    mock_llm_instance.invoke.return_value = mock_llm_response
    mock_get_llm.return_value = mock_llm_instance

    result = run_agent_query("Deep research on Agentic RAG", is_deep_research=True)

    assert result["grounded"] is True
    assert "synthesize_deep_research_report" in result["path"]
    assert len(result["path"]) > 0

def test_exporter_utility():
    report_text = "# Sample Report\n\n## Section 1\n- Bullet point detail 1\n- Bullet point detail 2"
    
    md_output = export_to_markdown(report_text)
    assert md_output == report_text

    pdf_bytes = export_to_pdf(report_text)
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 500  # Valid PDF binary file header and structure
