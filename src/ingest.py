import os
import pickle
import glob
import logging
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from src.config import settings
from src.llm import get_embeddings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def ingest_pdfs(reset: bool = False):
    raw_dir = Path(settings.RAW_DIR)
    index_dir = Path(settings.INDEX_DIR)
    
    if not raw_dir.exists():
        raw_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory {raw_dir}. Please put some PDF files in there.")
        return

    pdf_files = list(raw_dir.glob("*.pdf"))
    if not pdf_files:
        logger.warning(f"No PDF files found in {raw_dir}.")
        return

    index_dir.mkdir(parents=True, exist_ok=True)
    
    if reset:
        logger.info("Reset flag set. Clearing old index files.")
        for f in index_dir.glob("*"):
            if f.name != ".gitkeep":
                f.unlink()

    all_docs = []
    file_count = 0
    total_pages = 0

    for pdf_path in pdf_files:
        file_count += 1
        logger.info(f"Loading PDF: {pdf_path.name}")
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()
        
        filename = pdf_path.name
        for page_idx, page in enumerate(pages):
            total_pages += 1
            text = page.page_content.strip()
            if len(text) < 30:
                continue  # Skip nearly empty pages
            
            page_num = page_idx + 1
            page.metadata["source"] = filename
            page.metadata["page"] = page_num
            all_docs.append(page)

    if not all_docs:
        logger.warning("No valid text extracted from the provided PDFs.")
        return

    logger.info(f"Splitting text from {total_pages} pages into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = text_splitter.split_documents(all_docs)
    
    for i, chunk in enumerate(chunks):
        source = chunk.metadata.get("source", "unknown")
        page = chunk.metadata.get("page", 1)
        chunk.metadata["chunk_id"] = f"{source}::p{page}::c{i}"

    logger.info(f"Generated {len(chunks)} chunks. Building FAISS index...")
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    vectorstore.save_local(str(index_dir))
    
    chunks_path = index_dir / "chunks.pkl"
    with open(chunks_path, "wb") as f:
        pickle.dump(chunks, f)

    # Force clear retriever cache so newly ingested PDFs are immediately searchable
    try:
        from src.retrievers import reload_retriever
        reload_retriever()
    except Exception as e:
        logger.warning(f"Could not automatically clear retriever cache: {e}")

    logger.info(f"Ingestion complete! Processed {file_count} files, {total_pages} pages, resulting in {len(chunks)} chunks.")
    logger.info(f"Index saved successfully to {index_dir}")

if __name__ == "__main__":
    import sys
    reset_flag = "--reset" in sys.argv
    ingest_pdfs(reset=reset_flag)