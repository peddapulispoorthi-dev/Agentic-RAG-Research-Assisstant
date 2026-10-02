import os
import time
import logging
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_core.documents import Document
from tavily import TavilyClient
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import settings
from src.llm import get_llm
from src.retrievers import get_retriever
from src.schemas import RouteDecision, RelevanceGrade, GroundingGrade, MultiQueryDecomposition
from src.prompts import (
    ROUTER_PROMPT,
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

# Resilient LLM structured output invoker with exponential backoff for 429 rate limit protection
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=False
)
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
        logger.warning(f"LLM structured call encountered exception: {e}. Retrying via tenacity...")
        raise e

def route_question(state: Dict[str, Any]) -> Dict[str, Any]:
    question = state["question"]
    is_deep = state.get("is_deep_research", False)
    logger.info(f"Routing question: '{question}' (explicit deep_research={is_deep})")
    
    if is_deep:
        route = "deep_research"
    else:
        try:
            decision = call_llm_with_structured_output(
                prompt=f"Question: {question}",
                schema=RouteDecision,
                system_instruction=ROUTER_PROMPT
            )
            route = decision.datasource if decision else "vectorstore"
        except Exception as e:
            logger.error(f"Routing failed: {e}. Defaulting to vectorstore route.")
            route = "vectorstore"

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
        try:
            llm = get_llm(temperature=0.0)
            res = llm.invoke([("system", REWRITE_FOLLOWUP_PROMPT), ("human", prompt)])
            rewritten = res.content.strip()
        except Exception as e:
            logger.error(f"Follow-up rewrite failed: {e}. Using original question.")
            rewritten = question

    path = state.get("path", []) + ["rewrite_followup"]
    return {"rewritten_question": rewritten, "path": path}

def decompose_query(state: Dict[str, Any]) -> Dict[str, Any]:
    question = state.get("rewritten_question", state["question"])
    logger.info(f"Decomposing broad research prompt into sub-queries: '{question}'")
    
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
                f"{question} technical definition core principles",
                f"{question} comparative analysis pros cons alternatives",
                f"{question} future implications scaling security trends"
            ]
    except Exception as e:
        logger.error(f"Multi-query decomposition failed: {e}. Using fallback sub-query heuristics.")
        sub_queries = [
            f"{question} technical details architecture",
            f"{question} comparative evaluation benchmarks",
            f"{question} future directions applications"
        ]

    path = state.get("path", []) + ["decompose_query"]
    return {"sub_queries": sub_queries, "path": path}

def retrieve(state: Dict[str, Any]) -> Dict[str, Any]:
    query = state.get("rewritten_question", state["question"])
    logger.info(f"Retrieving documents for query: '{query}'")
    
    try:
        retriever = get_retriever()
        docs = retriever.retrieve(query, use_bm25=True, use_rerank=True)
    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
        docs = []

    path = state.get("path", []) + ["retrieve"]
    return {"candidate_documents": docs, "path": path}

def parallel_retrieve(state: Dict[str, Any]) -> Dict[str, Any]:
    sub_queries = state.get("sub_queries", [])
    if not sub_queries:
        sub_queries = [state.get("rewritten_question", state["question"])]
        
    logger.info(f"Executing parallel hybrid retrieval for {len(sub_queries)} sub-queries...")
    
    try:
        retriever = get_retriever()
        docs = retriever.parallel_hybrid_retrieve(sub_queries, use_bm25=True, use_rerank=True)
    except Exception as e:
        logger.error(f"Parallel hybrid retrieval failed: {e}")
        docs = []

    path = state.get("path", []) + ["parallel_retrieve"]
    return {"candidate_documents": docs, "path": path}

def grade_documents(state: Dict[str, Any]) -> Dict[str, Any]:
    question = state.get("rewritten_question", state["question"])
    candidates = state.get("candidate_documents", [])
    logger.info(f"Grading {len(candidates)} candidate documents concurrently for question: '{question}'...")

    if not candidates:
        path = state.get("path", []) + ["grade_documents"]
        return {"documents": [], "path": path}

    def _grade_single_doc(doc: Document) -> Optional[Document]:
        prompt = f"Question: {question}\n\nDocument Chunk:\n{doc.page_content}"
        try:
            grade = call_llm_with_structured_output(
                prompt=prompt,
                schema=RelevanceGrade,
                system_instruction=GRADE_DOC_PROMPT
            )
            if grade and grade.relevant:
                return doc
            elif grade is None:
                return doc
            return None
        except Exception as e:
            logger.warning(f"Document grading failed for chunk: {e}. Preserving chunk for resilience.")
            return doc

    relevant_docs = []
    workers = min(len(candidates), 5)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_grade_single_doc, doc) for doc in candidates]
        for future in as_completed(futures):
            res = future.result()
            if res is not None:
                relevant_docs.append(res)

    logger.info(f"Grading complete: {len(relevant_docs)} / {len(candidates)} documents marked relevant.")
    path = state.get("path", []) + ["grade_documents"]
    return {"documents": relevant_docs, "path": path}

def transform_query(state: Dict[str, Any]) -> Dict[str, Any]:
    current_query = state.get("rewritten_question", state["question"])
    logger.info(f"Transforming query for better retrieval: '{current_query}'")
    
    try:
        llm = get_llm(temperature=0.0)
        res = llm.invoke([("system", REWRITE_FOR_RETRIEVAL_PROMPT), ("human", current_query)])
        new_query = res.content.strip()
    except Exception as e:
        logger.error(f"Query transformation failed: {e}")
        new_query = current_query

    retry_count = state.get("retry_count", 0) + 1
    path = state.get("path", []) + ["transform_query"]
    return {"rewritten_question": new_query, "retry_count": retry_count, "path": path}

def web_search(state: Dict[str, Any]) -> Dict[str, Any]:
    query = state.get("rewritten_question", state["question"])
    logger.info(f"Performing resilient Tavily web search for: '{query}'")
    
    web_results = []
    tavily_key = settings.TAVILY_API_KEY or os.getenv("TAVILY_API_KEY")
    if tavily_key:
        try:
            tavily = TavilyClient(api_key=tavily_key)
            results = tavily.search(query=query, max_results=5)
            for r in results.get("results", []):
                web_results.append({
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "content": r.get("content", "")
                })
        except Exception as e:
            logger.error(f"Tavily web search circuit breaker triggered: {e}")
    else:
        logger.warning("TAVILY_API_KEY is missing. Skipping web search execution.")

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
        answer = "I couldn't find relevant information in the uploaded documents or web results."
    else:
        prompt = f"Context:\n{context_str}\n\nQuestion: {question}"
        try:
            llm = get_llm(temperature=0.0)
            res = llm.invoke([("system", GENERATE_PROMPT), ("human", prompt)])
            answer = res.content
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            answer = "Error generating answer due to an unexpected system exception."

    path = state.get("path", []) + ["generate"]
    return {"answer": answer, "path": path}

def synthesize_deep_research_report(state: Dict[str, Any]) -> Dict[str, Any]:
    question = state.get("rewritten_question", state["question"])
    docs = state.get("documents", [])
    if not docs:
        docs = state.get("candidate_documents", [])
    web_res = state.get("web_results", [])
    
    logger.info(f"Synthesizing Deep Research Report across {len(docs)} documents and {len(web_res)} web results...")
    
    context_parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "document.pdf")
        page = doc.metadata.get("page", 1)
        context_parts.append(f"[{i}] Source: {source} (Page {page})\n{doc.page_content}")

    for j, w in enumerate(web_res, len(docs) + 1):
        context_parts.append(f"[{j}] Source: {w['url']}\n{w['content']}")

    context_str = "\n\n".join(context_parts)
    
    if not context_str.strip():
        report = f"# Deep Research Report: {question}\n\n## Executive Summary\nNo relevant context could be retrieved from the repository or web sources to generate this report."
    else:
        prompt_str = DEEP_RESEARCH_REPORT_PROMPT.format(
            topic=question,
            context=context_str,
            question=question
        )
        try:
            llm = get_llm(temperature=0.1)
            res = llm.invoke([("human", prompt_str)])
            report = res.content
        except Exception as e:
            logger.error(f"Deep Research Synthesis failed: {e}")
            report = f"# Deep Research Report: {question}\n\n## Executive Summary\nError synthesizing report: {e}"

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
    except Exception as e:
        logger.error(f"Grounding check failed: {e}. Defaulting to grounded=True for fault tolerance.")
        grounded = True

    path = state.get("path", []) + ["check_grounding"]
    return {"grounded": grounded, "path": path}

def regenerate_strict(state: Dict[str, Any]) -> Dict[str, Any]:
    logger.info("Regenerating answer with strict grounding rules...")
    question = state.get("rewritten_question", state["question"])
    docs = state.get("documents", [])
    web_res = state.get("web_results", [])
    
    context_str = "\n".join([d.page_content for d in docs] + [w["content"] for w in web_res])
    prompt = f"Context:\n{context_str}\n\nQuestion: {question}\n\nInstruction: Remove any claim not directly supported by the context."
    
    try:
        llm = get_llm(temperature=0.0)
        res = llm.invoke([("system", GENERATE_PROMPT), ("human", prompt)])
        answer = res.content
    except Exception as e:
        logger.error(f"Strict regeneration failed: {e}")
        answer = state.get("answer", "")

    regen_count = state.get("regen_count", 0) + 1
    path = state.get("path", []) + ["regenerate_strict"]
    return {"answer": answer, "regen_count": regen_count, "path": path}

def chitchat(state: Dict[str, Any]) -> Dict[str, Any]:
    question = state["question"]
    try:
        llm = get_llm(temperature=0.2)
        res = llm.invoke([("system", CHITCHAT_PROMPT), ("human", question)])
        answer = res.content
    except Exception as e:
        logger.error(f"Chitchat failed: {e}")
        answer = "Hello! I am your Agentic RAG Research Assistant. I can help you answer questions or generate Deep Research Reports."

    path = state.get("path", []) + ["chitchat"]
    return {"answer": answer, "path": path}