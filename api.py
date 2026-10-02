import os
import time
from typing import List, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import Response, JSONResponse, StreamingResponse
import io

from src.config import settings
from src.schemas import AskRequest, AnswerResult
from src.service import run_agent_query
from src.ingest import ingest_pdfs
from utils.exporter import export_to_markdown, export_to_pdf

app = FastAPI(
    title="Agentic RAG Research Assistant API",
    description="Production-grade REST API for Autonomous Agentic RAG and Multi-Perspective Research Synthesis",
    version="1.0.0"
)

@app.get("/")
def read_root():
    return {
        "status": "online",
        "service": "Agentic RAG Research Assistant API",
        "llm_provider": settings.LLM_PROVIDER,
        "llm_model": settings.LLM_MODEL,
        "langsmith_tracing": settings.LANGCHAIN_TRACING_V2
    }

@app.get("/health")
def health_check():
    index_exists = os.path.exists(os.path.join(settings.INDEX_DIR, "index.faiss"))
    return {
        "status": "healthy",
        "vectorstore_indexed": index_exists,
        "raw_dir": settings.RAW_DIR,
        "index_dir": settings.INDEX_DIR
    }

@app.post("/api/v1/query", response_model=AnswerResult)
def query_agent(request: AskRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question prompt cannot be empty.")
    
    start_time = time.time()
    res = run_agent_query(
        question=request.question,
        history=request.history,
        is_deep_research=request.is_deep_research
    )
    elapsed = round(time.time() - start_time, 3)

    return AnswerResult(
        answer=res.get("answer", ""),
        report=res.get("report", ""),
        citations=res.get("sources", []),
        path=res.get("path", []),
        used_web_search=res.get("used_web_search", False),
        sub_queries=res.get("sub_queries", []),
        latency_seconds=elapsed
    )

@app.post("/api/v1/ingest")
def upload_and_ingest(
    files: List[UploadFile] = File(...),
    reset: bool = Query(False, description="Reset existing index before ingesting")
):
    raw_dir = settings.RAW_DIR
    os.makedirs(raw_dir, exist_ok=True)
    
    saved_files = []
    for file in files:
        if not file.filename.endswith(".pdf"):
            continue
        file_path = os.path.join(raw_dir, file.filename)
        with open(file_path, "wb") as f:
            f.write(file.file.read())
        saved_files.append(file.filename)
        
    if not saved_files:
        raise HTTPException(status_code=400, detail="No valid PDF files provided for ingestion.")
        
    try:
        ingest_pdfs(reset=reset)
        return {
            "status": "success",
            "message": f"Successfully ingested {len(saved_files)} PDF document(s).",
            "files": saved_files
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

@app.post("/api/v1/export/pdf")
def export_pdf(report_text: str):
    if not report_text.strip():
        raise HTTPException(status_code=400, detail="Report content cannot be empty.")
    
    pdf_bytes = export_to_pdf(report_text)
    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=Deep_Research_Report.pdf"}
    )

@app.post("/api/v1/export/markdown")
def export_md(report_text: str):
    if not report_text.strip():
        raise HTTPException(status_code=400, detail="Report content cannot be empty.")
    
    md_content = export_to_markdown(report_text)
    return Response(
        content=md_content,
        media_type="text/markdown",
        headers={"Content-Disposition": "attachment; filename=Deep_Research_Report.md"}
    )
