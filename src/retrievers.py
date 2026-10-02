import os
import pickle
import logging
import re
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import functools

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from src.config import settings
from src.llm import get_embeddings

logger = logging.getLogger(__name__)

class HybridRetriever:
    def __init__(self):
        self._load_index()

    def _load_index(self):
        index_dir = settings.INDEX_DIR
        faiss_path = os.path.join(index_dir, "index.faiss")
        chunks_path = os.path.join(index_dir, "chunks.pkl")

        if not os.path.exists(faiss_path) or not os.path.exists(chunks_path):
            logger.warning(
                f"Index files not found in {index_dir}. Initializing empty retriever placeholder."
            )
            self.vectorstore = None
            self.chunks = []
            self.chunk_map = {}
            self.bm25 = None
            self.cross_encoder = None
            return

        logger.info("Loading FAISS index...")
        embeddings = get_embeddings()
        self.vectorstore = FAISS.load_local(
            index_dir, 
            embeddings, 
            allow_dangerous_deserialization=True
        )

        logger.info("Loading chunks for BM25...")
        with open(chunks_path, "rb") as f:
            self.chunks: List[Document] = pickle.load(f)

        self.chunk_map = {
            chunk.metadata.get("chunk_id", str(i)): chunk 
            for i, chunk in enumerate(self.chunks)
        }

        logger.info("Initializing BM25 Okapi index...")
        tokenized_corpus = [self._tokenize(doc.page_content) for doc in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

        try:
            self.cross_encoder = CrossEncoder(settings.RERANK_MODEL)
        except Exception as e:
            logger.warning(f"Skipping CrossEncoder initialization: {e}")
            self.cross_encoder = None

    def _reload_if_needed(self):
        """Dynamically check and reload index if files were added after initialization."""
        if self.vectorstore is None:
            index_dir = settings.INDEX_DIR
            faiss_path = os.path.join(index_dir, "index.faiss")
            chunks_path = os.path.join(index_dir, "chunks.pkl")
            if os.path.exists(faiss_path) and os.path.exists(chunks_path):
                logger.info("New index files detected. Dynamically reloading FAISS + BM25...")
                self._load_index()

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\w+", text.lower())

    def dense_search(self, query: str, k: int) -> List[Document]:
        self._reload_if_needed()
        if not self.vectorstore:
            return []
        try:
            docs_and_scores = self.vectorstore.similarity_search_with_score(query, k=k)
            return [doc for doc, _ in docs_and_scores]
        except Exception as e:
            logger.error(f"Dense vector search failed: {e}")
            return []

    def bm25_search(self, query: str, k: int) -> List[Document]:
        self._reload_if_needed()
        if not self.bm25 or not self.chunks:
            return []
        try:
            tokenized_query = self._tokenize(query)
            scores = self.bm25.get_scores(tokenized_query)
            top_n_indices = scores.argsort()[::-1][:k]
            return [self.chunks[i] for i in top_n_indices if scores[i] > 0]
        except Exception as e:
            logger.error(f"BM25 search failed: {e}")
            return []

    def fuse(self, lists: List[List[Document]], k: int) -> List[Document]:
        """Reciprocal Rank Fusion (RRF)"""
        rrf_scores: Dict[str, float] = {}
        doc_store: Dict[str, Document] = {}

        for doc_list in lists:
            for rank, doc in enumerate(doc_list, start=1):
                chunk_id = doc.metadata.get("chunk_id")
                if not chunk_id:
                    chunk_id = f"{doc.metadata.get('source')}::p{doc.metadata.get('page')}::{hash(doc.page_content[:50])}"
                    doc.metadata["chunk_id"] = chunk_id

                doc_store[chunk_id] = doc
                if chunk_id not in rrf_scores:
                    rrf_scores[chunk_id] = 0.0
                
                rrf_scores[chunk_id] += 1.0 / (settings.RRF_K + rank)

        sorted_chunk_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        
        fused_docs = []
        for cid in sorted_chunk_ids[:k]:
            if cid in doc_store:
                fused_docs.append(doc_store[cid])
        return fused_docs

    def rerank(self, query: str, docs: List[Document], k: int) -> List[Document]:
        if not docs:
            return []
        if not self.cross_encoder:
            return docs[:k]
        
        try:
            pairs = [(query, doc.page_content) for doc in docs]
            scores = self.cross_encoder.predict(pairs)

            scored_docs = list(zip(docs, scores))
            scored_docs.sort(key=lambda x: x[1], reverse=True)

            final_docs = []
            for doc, score in scored_docs[:k]:
                doc.metadata["rerank_score"] = float(score)
                final_docs.append(doc)

            return final_docs
        except Exception as e:
            logger.error(f"Reranking failed: {e}. Returning un-reranked top candidate documents.")
            return docs[:k]

    def retrieve(self, query: str, use_bm25: bool = True, use_rerank: bool = True) -> List[Document]:
        self._reload_if_needed()
        dense_docs = self.dense_search(query, k=settings.TOP_K_DENSE)
        
        if not use_bm25:
            candidate_docs = dense_docs
        else:
            bm25_docs = self.bm25_search(query, k=settings.TOP_K_BM25)
            candidate_docs = self.fuse([dense_docs, bm25_docs], k=settings.TOP_K_FUSED)

        if not use_rerank:
            return candidate_docs[:settings.TOP_K_FINAL]
        
        return self.rerank(query, candidate_docs, k=settings.TOP_K_FINAL)

    def _is_duplicate(self, doc: Document, existing_docs: List[Document], threshold: float = 0.85) -> bool:
        doc_id = doc.metadata.get("chunk_id")
        content1 = doc.page_content.strip()
        words1 = set(self._tokenize(content1))
        
        if not words1:
            return True

        for existing in existing_docs:
            if doc_id and doc_id == existing.metadata.get("chunk_id"):
                return True
            content2 = existing.page_content.strip()
            words2 = set(self._tokenize(content2))
            if not words2:
                continue
            
            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))
            jaccard = intersection / union if union > 0 else 0.0
            
            if jaccard >= threshold:
                return True
        return False

    def parallel_hybrid_retrieve(
        self, 
        queries: List[str], 
        use_bm25: bool = True, 
        use_rerank: bool = True,
        similarity_threshold: Optional[float] = None
    ) -> List[Document]:
        self._reload_if_needed()
        if not queries:
            return []

        if similarity_threshold is None:
            similarity_threshold = settings.DEDUPLICATION_SIMILARITY_THRESHOLD

        results_per_query: List[List[Document]] = []
        workers = min(len(queries), settings.PARALLEL_RETRIEVAL_WORKERS)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_query = {
                executor.submit(self.retrieve, q, use_bm25, use_rerank): q for q in queries
            }
            for future in as_completed(future_to_query):
                query = future_to_query[future]
                try:
                    docs = future.result()
                    results_per_query.append(docs)
                except Exception as e:
                    logger.error(f"Retrieval failed for sub-query '{query}': {e}")
                    results_per_query.append([])

        consolidated_docs: List[Document] = []
        for docs in results_per_query:
            for doc in docs:
                if not self._is_duplicate(doc, consolidated_docs, threshold=similarity_threshold):
                    consolidated_docs.append(doc)

        return consolidated_docs

@functools.lru_cache(maxsize=1)
def get_retriever() -> HybridRetriever:
    return HybridRetriever()

def reload_retriever():
    get_retriever.cache_clear()
    return get_retriever()