from typing import List, Dict, Any, TypedDict
from langchain_core.documents import Document

class GraphState(TypedDict):
    question: str
    history: List[Dict[str, str]]
    rewritten_question: str
    route: str
    candidate_documents: List[Document]
    documents: List[Document]
    web_results: List[Dict[str, Any]]
    answer: str
    grounded: bool
    retry_count: int
    regen_count: int
    path: List[str]
    is_deep_research: bool
    sub_queries: List[str]
    report: str