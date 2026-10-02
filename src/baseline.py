import logging
from typing import Tuple, List
from langchain_core.documents import Document
from src.retrievers import get_retriever
from src.llm import get_llm
from src.prompts import GENERATE_PROMPT # We will use prompts or raw string if needed, let's define inline or import

logger = logging.getLogger(__name__)

def run_baseline(question: str, mode: str = "hybrid_rerank") -> Tuple[str, List[Document]]:
    """
    Runs plain RAG comparison baseline.
    mode options: 'dense', 'hybrid', 'hybrid_rerank'
    """
    retriever = get_retriever()
    
    if mode == "dense":
        docs = retriever.retrieve(question, use_bm25=False, use_rerank=False)
    elif mode == "hybrid":
        docs = retriever.retrieve(question, use_bm25=True, use_rerank=False)
    elif mode == "hybrid_rerank":
        docs = retriever.retrieve(question, use_bm25=True, use_rerank=True)
    else:
        raise ValueError(f"Unknown baseline mode: {mode}")

    # Build context string
    context_parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", 1)
        context_parts.append(f"[{i}] (source: {source}, page {page})\n{doc.page_content}")
    
    context_str = "\n\n".join(context_parts)
    
    prompt = f"""Answer using ONLY the context below. Cite sources inline as [n] where n is the context item number. If the context does not contain the answer, say you could not find it in the provided sources. Be concise and accurate. Do not use outside knowledge.

Context:
{context_str}

Question: {question}
Answer:"""

    llm = get_llm(temperature=0.0)
    response = llm.invoke(prompt)
    
    return response.content, docs