import os
from pathlib import Path
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()

class Settings(BaseSettings):
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "google")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gemini-2.5-flash")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.1")
    TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")

    # LangSmith Tracing & Observability Configuration
    LANGCHAIN_TRACING_V2: bool = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
    LANGCHAIN_ENDPOINT: str = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
    LANGCHAIN_API_KEY: str = os.getenv("LANGCHAIN_API_KEY", "")
    LANGCHAIN_PROJECT: str = os.getenv("LANGCHAIN_PROJECT", "agentic-rag-research-assistant")
    
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 120
    EMBED_MODEL: str = "BAAI/bge-small-en-v1.5"
    RERANK_MODEL: str = "BAAI/bge-reranker-base"
    
    TOP_K_DENSE: int = 10
    TOP_K_BM25: int = 10
    TOP_K_FUSED: int = 15
    TOP_K_FINAL: int = 5
    RRF_K: int = 60

    # Multi-Query Parallel Retrieval & Synthesis Settings
    DEDUPLICATION_SIMILARITY_THRESHOLD: float = 0.85
    PARALLEL_RETRIEVAL_WORKERS: int = 3
    
    MIN_RELEVANT_DOCS: int = 2
    MAX_QUERY_RETRIES: int = 2
    MAX_REGENERATIONS: int = 1
    
    INDEX_DIR: str = "data/index"
    RAW_DIR: str = "data/raw"

settings = Settings()

# Automatically configure LangSmith environment variables if tracing is enabled
if settings.LANGCHAIN_TRACING_V2 and settings.LANGCHAIN_API_KEY:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"] = settings.LANGCHAIN_ENDPOINT
    os.environ["LANGCHAIN_API_KEY"] = settings.LANGCHAIN_API_KEY
    os.environ["LANGCHAIN_PROJECT"] = settings.LANGCHAIN_PROJECT