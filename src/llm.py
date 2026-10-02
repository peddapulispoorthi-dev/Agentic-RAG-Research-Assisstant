import functools
import logging
import os
from langchain_core.language_models import BaseChatModel
from langchain_huggingface import HuggingFaceEmbeddings
from src.config import settings

logger = logging.getLogger(__name__)

@functools.lru_cache(maxsize=1)
def get_llm(temperature: float = 0.0) -> BaseChatModel:
    provider = settings.LLM_PROVIDER.lower()
    logger.info(f"Initializing LLM with provider: {provider}, model: {settings.LLM_MODEL}")
    
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=settings.LLM_MODEL, temperature=temperature)
    elif provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        api_key = os.getenv("GOOGLE_API_KEY")
        if api_key:
            os.environ["GOOGLE_API_KEY"] = api_key
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=temperature
        )
    elif provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=settings.OLLAMA_MODEL, temperature=temperature)
    else:
        raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")

@functools.lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    logger.info(f"Loading HuggingFace embeddings model: {settings.EMBED_MODEL}")
    return HuggingFaceEmbeddings(
        model_name=settings.EMBED_MODEL,
        encode_kwargs={"normalize_embeddings": True}
    )