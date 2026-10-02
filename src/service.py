import logging
import time
from typing import Dict, Any, List
from src.graph import app_graph

logger = logging.getLogger(__name__)

def run_agent_query(
    question: str, 
    history: List[Dict[str, str]] = None,
    is_deep_research: bool = False
) -> Dict[str, Any]:
    if history is None:
        history = []
        
    initial_state = {
        "question": question,
        "history": history,
        "rewritten_question": "",
        "route": "",
        "candidate_documents": [],
        "documents": [],
        "web_results": [],
        "answer": "",
        "report": "",
        "grounded": True,
        "retry_count": 0,
        "regen_count": 0,
        "path": [],
        "is_deep_research": is_deep_research,
        "sub_queries": [],
        "plan_steps": [],
        "current_step_idx": 0,
        "knowledge_ledger": [],
        "reflection": {}
    }
    
    start_time = time.time()
    logger.info(f"Invoking Agent Graph for query: '{question}' (deep_research={is_deep_research})")
    
    try:
        final_state = app_graph.invoke(initial_state)
        elapsed = round(time.time() - start_time, 3)
        
        sources = []
        for doc in final_state.get("documents", []):
            sources.append({
                "source": doc.metadata.get("source", "Document"),
                "page": doc.metadata.get("page", 1),
                "snippet": doc.page_content[:250],
                "chunk_id": doc.metadata.get("chunk_id", "c0")
            })
        for web in final_state.get("web_results", []):
            sources.append({
                "source": web.get("url", "Web"),
                "page": None,
                "snippet": web.get("content", "")[:250],
                "chunk_id": web.get("url", "web")
            })

        return {
            "answer": final_state.get("answer", "No answer generated."),
            "report": final_state.get("report", ""),
            "sub_queries": final_state.get("sub_queries", []),
            "plan_steps": final_state.get("plan_steps", []),
            "knowledge_ledger": final_state.get("knowledge_ledger", []),
            "reflection": final_state.get("reflection", {}),
            "sources": sources,
            "path": final_state.get("path", []),
            "latency_seconds": elapsed,
            "used_web_search": "web_search" in final_state.get("path", []),
            "grounded": final_state.get("grounded", True)
        }
    except Exception as e:
        logger.error(f"Agent execution failed: {e}")
        return {
            "answer": f"An error occurred while processing your research request: {e}",
            "report": "",
            "sub_queries": [],
            "plan_steps": [],
            "knowledge_ledger": [],
            "reflection": {},
            "sources": [],
            "path": [],
            "latency_seconds": round(time.time() - start_time, 3),
            "used_web_search": False,
            "grounded": False
        }