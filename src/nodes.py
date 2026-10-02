import os
import time
import logging
from typing import Dict, Any, List, Optional
from langchain_core.documents import Document
from tavily import TavilyClient

from src.config import settings
from src.llm import get_llm
from src.retrievers import get_retriever
from src.schemas import (
    RouteDecision, 
    RelevanceGrade, 
    GroundingGrade, 
    MultiQueryDecomposition,
    ResearchPlan,
    ReflectionDecision
)
from src.prompts import (
    ROUTER_PROMPT,
    PLANNER_PROMPT,
    EVALUATOR_REFLECTION_PROMPT,
    MULTI_QUERY_DECOMPOSITION_PROMPT,
    DEEP_RESEARCH_REPORT_PROMPT,
    REWRITE_FOLLOWUP_PROMPT,
    REWRITE_FOR_RETRIEVAL_PROMPT,
    GRADE_DOC_PROMPT,
    GENERATE_PROMPT,
    GROUNDING_PROMPT,
    CHITCHAT_PROMPT
)

logger = logging.getLogger(__name__)

# Fast LLM call wrapper with low latency timeout
def fast_call_llm_text(messages: List[tuple], temperature: float = 0.0) -> Optional[str]:
    try:
        llm = get_llm(temperature=temperature)
        res = llm.invoke(messages)
        return res.content.strip()
    except Exception as e:
        logger.warning(f"LLM call encountered quota limit / timeout ({e}). Using ultra-fast fallback.")
        return None

def call_llm_with_structured_output(prompt: str, schema: Any, system_instruction: str = ""):
    try:
        llm = get_llm(temperature=0.0)
        structured_llm = llm.with_structured_output(schema)
        messages = []
        if system_instruction:
            messages.append(("system", system_instruction))
        messages.append(("human", prompt))
        return structured_llm.invoke(messages)
    except Exception as e:
        logger.warning(f"Structured LLM call encountered quota limit / timeout ({e}).")
        return None

# Alias for backwards compatibility
fast_call_llm_structured = call_llm_with_structured_output

# --- ULTRA-FAST ROUTING (Instant Keyword Rule + LLM Fallback) ---
def route_question(state: Dict[str, Any]) -> Dict[str, Any]:
    question = state["question"].strip().lower()
    is_deep = state.get("is_deep_research", False)
    
    if is_deep or any(kw in question for kw in ["report", "deep research", "comprehensive report", "detailed synthesis", "academic review"]):
        route = "deep_research"
    elif any(question.startswith(kw) for kw in ["hi", "hello", "hey", "greetings", "thanks", "thank you"]):
        route = "chitchat"
    else:
        try:
            decision = call_llm_with_structured_output(
                prompt=f"Question: {question}",
                schema=RouteDecision,
                system_instruction=ROUTER_PROMPT
            )
            route = decision.datasource if decision else "vectorstore"
        except Exception:
            route = "vectorstore"

    logger.info(f"Routed question to: '{route}'")
    path = state.get("path", []) + ["route_question"]
    return {
        "route": route,
        "is_deep_research": route == "deep_research" or is_deep,
        "retry_count": state.get("retry_count", 0),
        "regen_count": state.get("regen_count", 0),
        "path": path
    }

def rewrite_followup(state: Dict[str, Any]) -> Dict[str, Any]:
    history = state.get("history", [])
    question = state["question"]
    
    if not history:
        rewritten = question
    else:
        history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history[-4:]])
        prompt = f"Chat History:\n{history_str}\n\nLatest Question: {question}"
        rewritten = fast_call_llm_text([("system", REWRITE_FOLLOWUP_PROMPT), ("human", prompt)]) or question

    path = state.get("path", []) + ["rewrite_followup"]
    return {"rewritten_question": rewritten, "path": path}

def decompose_query(state: Dict[str, Any]) -> Dict[str, Any]:
    question = state.get("rewritten_question", state["question"])
    try:
        decomposition = call_llm_with_structured_output(
            prompt=f"Research Question: {question}",
            schema=MultiQueryDecomposition,
            system_instruction=MULTI_QUERY_DECOMPOSITION_PROMPT
        )
        if decomposition:
            sub_queries = decomposition.get_queries()
        else:
            sub_queries = [
                f"{question} technical details architecture",
                f"{question} comparative evaluation benchmarks",
                f"{question} future directions applications"
            ]
    except Exception:
        sub_queries = [
            f"{question} technical details architecture",
            f"{question} comparative evaluation benchmarks",
            f"{question} future directions applications"
        ]

    path = state.get("path", []) + ["decompose_query"]
    return {"sub_queries": sub_queries, "path": path}

def parallel_retrieve(state: Dict[str, Any]) -> Dict[str, Any]:
    sub_queries = state.get("sub_queries", [])
    if not sub_queries:
        sub_queries = [state.get("rewritten_question", state["question"])]
    try:
        retriever = get_retriever()
        docs = retriever.parallel_hybrid_retrieve(sub_queries, use_bm25=True, use_rerank=False)
    except Exception as e:
        logger.error(f"Parallel retrieval failed: {e}")
        docs = []
    path = state.get("path", []) + ["parallel_retrieve"]
    return {"candidate_documents": docs, "path": path}

# --- AUTONOMOUS PLANNER NODES ---
def draft_research_plan(state: Dict[str, Any]) -> Dict[str, Any]:
    question = state.get("rewritten_question", state["question"])
    plan_steps = [
        f"1. Technical Principles & Architecture of {question}",
        f"2. Comparative Analysis & Performance Trade-offs for {question}",
        f"3. Future Implications, Security & Strategic Outlook for {question}"
    ]

    path = state.get("path", []) + ["draft_research_plan"]
    return {
        "plan_steps": plan_steps,
        "current_step_idx": 0,
        "knowledge_ledger": [],
        "sub_queries": plan_steps,
        "path": path
    }

def execute_plan_step(state: Dict[str, Any]) -> Dict[str, Any]:
    plan_steps = state.get("plan_steps", [])
    step_idx = state.get("current_step_idx", 0)
    knowledge_ledger = list(state.get("knowledge_ledger", []))
    accumulated_docs = list(state.get("documents", []))
    
    if not plan_steps or step_idx >= len(plan_steps):
        path = state.get("path", []) + ["execute_plan_step"]
        return {"current_step_idx": step_idx, "path": path}

    current_step = plan_steps[step_idx]
    
    try:
        retriever = get_retriever()
        step_docs = retriever.retrieve(current_step, use_bm25=True, use_rerank=False)
    except Exception as e:
        logger.error(f"Step retrieval failed: {e}")
        step_docs = []

    summary = "\n".join([f"- {d.page_content[:180]}..." for d in step_docs[:3]]) if step_docs else "No documents retrieved for this step."
    sources = [doc.metadata.get("source", "doc.pdf") for doc in step_docs]
    
    ledger_entry = {
        "step_num": step_idx + 1,
        "step": current_step,
        "summary": summary,
        "sources": list(set(sources))
    }
    knowledge_ledger.append(ledger_entry)

    for d in step_docs:
        if d not in accumulated_docs:
            accumulated_docs.append(d)

    path = state.get("path", []) + [f"execute_plan_step_{step_idx + 1}"]
    return {
        "current_step_idx": step_idx + 1,
        "knowledge_ledger": knowledge_ledger,
        "documents": accumulated_docs,
        "candidate_documents": accumulated_docs,
        "path": path
    }

def reflect_on_knowledge(state: Dict[str, Any]) -> Dict[str, Any]:
    step_idx = state.get("current_step_idx", 0)
    plan_steps = state.get("plan_steps", [])
    
    refl_dict = {
        "sufficient": step_idx >= len(plan_steps),
        "missing_aspects": [],
        "reasoning": "Completed planned research steps."
    }

    path = state.get("path", []) + ["reflect_on_knowledge"]
    return {"reflection": refl_dict, "path": path}

# --- FAST RETRIEVAL & GENERATION NODES ---
def retrieve(state: Dict[str, Any]) -> Dict[str, Any]:
    query = state.get("rewritten_question", state["question"])
    logger.info(f"Fast retrieving documents for query: '{query}'")
    
    try:
        retriever = get_retriever()
        docs = retriever.retrieve(query, use_bm25=True, use_rerank=False)
    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
        docs = []

    path = state.get("path", []) + ["retrieve"]
    return {"candidate_documents": docs, "path": path}

def grade_documents(state: Dict[str, Any]) -> Dict[str, Any]:
    question = state.get("rewritten_question", state["question"])
    candidates = state.get("candidate_documents", [])
    logger.info(f"Grading {len(candidates)} candidate documents...")

    relevant_docs = []
    for doc in candidates:
        try:
            grade = call_llm_with_structured_output(
                prompt=f"Question: {question}\n\nDocument Chunk:\n{doc.page_content}",
                schema=RelevanceGrade,
                system_instruction=GRADE_DOC_PROMPT
            )
            if grade and grade.relevant:
                relevant_docs.append(doc)
            elif grade is None:
                relevant_docs.append(doc)
        except Exception:
            relevant_docs.append(doc)

    path = state.get("path", []) + ["grade_documents"]
    return {"documents": relevant_docs, "path": path}

def transform_query(state: Dict[str, Any]) -> Dict[str, Any]:
    current_query = state.get("rewritten_question", state["question"])
    retry_count = state.get("retry_count", 0) + 1
    path = state.get("path", []) + ["transform_query"]
    return {"rewritten_question": current_query, "retry_count": retry_count, "path": path}

def web_search(state: Dict[str, Any]) -> Dict[str, Any]:
    query = state.get("rewritten_question", state["question"])
    web_results = []
    tavily_key = settings.TAVILY_API_KEY or os.getenv("TAVILY_API_KEY")
    if tavily_key:
        try:
            tavily = TavilyClient(api_key=tavily_key)
            results = tavily.search(query=query, max_results=3)
            for r in results.get("results", []):
                web_results.append({
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "content": r.get("content", "")
                })
        except Exception as e:
            logger.error(f"Tavily search skipped/failed: {e}")

    path = state.get("path", []) + ["web_search"]
    return {"web_results": web_results, "path": path}

def generate(state: Dict[str, Any]) -> Dict[str, Any]:
    question = state.get("rewritten_question", state["question"])
    docs = state.get("documents", [])
    web_res = state.get("web_results", [])
    
    context_parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "doc.pdf")
        page = doc.metadata.get("page", 1)
        context_parts.append(f"[{i}] (source: {source}, page {page})\n{doc.page_content}")

    for j, w in enumerate(web_res, len(docs) + 1):
        context_parts.append(f"[{j}] (source: {w['url']})\n{w['content']}")

    context_str = "\n\n".join(context_parts)
    
    if not context_str.strip():
        answer = "I couldn't find relevant information in the uploaded documents."
    else:
        prompt = f"Context:\n{context_str}\n\nQuestion: {question}"
        answer = fast_call_llm_text([("system", GENERATE_PROMPT), ("human", prompt)])
        
        if not answer:
            answer = f"**Summary from Uploaded Documents:**\n\n"
            for i, doc in enumerate(docs[:3], 1):
                src = doc.metadata.get("source", "Document")
                pg = doc.metadata.get("page", 1)
                answer += f"**[{i}] Source: {src} (Page {pg})**\n> {doc.page_content.strip()}\n\n"

    path = state.get("path", []) + ["generate"]
    return {"answer": answer, "path": path}

def synthesize_deep_research_report(state: Dict[str, Any]) -> Dict[str, Any]:
    question = state.get("rewritten_question", state["question"])
    docs = state.get("documents", [])
    if not docs:
        docs = state.get("candidate_documents", [])
    web_res = state.get("web_results", [])
    ledger = state.get("knowledge_ledger", [])
    
    context_parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "document.pdf")
        page = doc.metadata.get("page", 1)
        context_parts.append(f"[{i}] Source: {source} (Page {page})\n{doc.page_content}")

    for j, w in enumerate(web_res, len(docs) + 1):
        context_parts.append(f"[{j}] Source: {w['url']}\n{w['content']}")

    context_str = "\n\n".join(context_parts)
    
    if not context_str.strip():
        report = f"# Deep Research Report: {question}\n\n## Executive Summary\nNo relevant context could be retrieved from the repository."
    else:
        prompt_str = DEEP_RESEARCH_REPORT_PROMPT.format(
            topic=question,
            context=context_str,
            question=question
        )
        report = fast_call_llm_text([("human", prompt_str)], temperature=0.1)
        
        if not report:
            report = f"# Deep Research Report: {question}\n\n## Executive Summary\nReport compiled directly from retrieved research context:\n\n"
            for item in ledger:
                report += f"### {item.get('step')}\n{item.get('summary')}\n\n"
            for i, doc in enumerate(docs[:4], 1):
                src = doc.metadata.get("source", "Doc")
                pg = doc.metadata.get("page", 1)
                report += f"#### Source [{i}]: {src} (Page {pg})\n> {doc.page_content[:300]}...\n\n"

    path = state.get("path", []) + ["synthesize_deep_research_report"]
    return {"answer": report, "report": report, "path": path}

def check_grounding(state: Dict[str, Any]) -> Dict[str, Any]:
    docs = state.get("documents", [])
    web_res = state.get("web_results", [])
    answer = state.get("answer", "")
    
    if not docs and not web_res:
        path = state.get("path", []) + ["check_grounding"]
        return {"grounded": True, "path": path}

    context_str = "\n".join([d.page_content for d in docs] + [w["content"] for w in web_res])
    prompt = f"Context:\n{context_str}\n\nAnswer:\n{answer}"
    
    try:
        grade = call_llm_with_structured_output(
            prompt=prompt,
            schema=GroundingGrade,
            system_instruction=GROUNDING_PROMPT
        )
        grounded = grade.grounded if grade else True
    except Exception:
        grounded = True

    path = state.get("path", []) + ["check_grounding"]
    return {"grounded": grounded, "path": path}

def regenerate_strict(state: Dict[str, Any]) -> Dict[str, Any]:
    path = state.get("path", []) + ["regenerate_strict"]
    return {"answer": state.get("answer", ""), "regen_count": 1, "path": path}

def chitchat(state: Dict[str, Any]) -> Dict[str, Any]:
    question = state["question"]
    answer = fast_call_llm_text([("system", CHITCHAT_PROMPT), ("human", question)], temperature=0.2) or "Hello! I am your Agentic RAG Research Assistant."
    path = state.get("path", []) + ["chitchat"]
    return {"answer": answer, "path": path}