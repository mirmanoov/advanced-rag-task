"""
Reranker module using cross-encoder models for improved relevance scoring.
"""
from typing import List, Tuple
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder


class Reranker:
    """
    Reranks retrieved documents using a cross-encoder model.
    Cross-encoders provide more accurate relevance scoring than bi-encoders
    by encoding query-document pairs together.
    """
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize reranker with a cross-encoder model.
        
        Args:
            model_name: HuggingFace model name for cross-encoder
                       Default: ms-marco-MiniLM-L-6-v2 (fast and effective)
        """
        print(f"Loading reranker model: {model_name}")
        self.model = CrossEncoder(model_name, max_length=512)
        self.model_name = model_name
    
    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int = 4
    ) -> List[Document]:
        """
        Rerank documents based on relevance to query.
        
        Args:
            query: Search query
            documents: List of documents to rerank
            top_k: Number of top documents to return
        
        Returns:
            Reranked list of documents (most relevant first)
        """
        if not documents:
            return []
        
        if len(documents) <= top_k:
            # If we have fewer docs than top_k, still rerank for ordering
            top_k = len(documents)
        
        # Create query-document pairs
        pairs = [(query, doc.page_content) for doc in documents]
        
        # Get relevance scores
        scores = self.model.predict(pairs)
        
        # Sort by score (descending)
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # Return top-k
        reranked = [doc for doc, score in scored_docs[:top_k]]
        
        return reranked
    
    def rerank_with_scores(
        self,
        query: str,
        documents: List[Document],
        top_k: int = 4
    ) -> List[Tuple[Document, float]]:
        """
        Rerank documents and return with scores.
        
        Args:
            query: Search query
            documents: List of documents to rerank
            top_k: Number of top documents to return
        
        Returns:
            List of (document, score) tuples
        """
        if not documents:
            return []
        
        if len(documents) <= top_k:
            top_k = len(documents)
        
        # Create query-document pairs
        pairs = [(query, doc.page_content) for doc in documents]
        
        # Get relevance scores
        scores = self.model.predict(pairs)
        
        # Sort by score (descending)
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        return scored_docs[:top_k]


class LLMReranker:
    """
    Alternative reranker using LLM for relevance scoring.
    More expensive but can provide better understanding of relevance.
    """
    
    def __init__(self, llm=None):
        """Initialize with optional LLM."""
        if llm is None:
            from langchain_openai import ChatOpenAI
            from src.config import OPENAI_API_KEY, LLM_MODEL
            self.llm = ChatOpenAI(
                model=LLM_MODEL,
                openai_api_key=OPENAI_API_KEY,
                temperature=0
            )
        else:
            self.llm = llm
    
    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int = 4
    ) -> List[Document]:
        """
        Rerank using LLM relevance scoring.
        Note: This is slower and more expensive than cross-encoder.
        """
        if len(documents) <= top_k:
            return documents
        
        # For efficiency, use cross-encoder for now
        # LLM reranking can be added as an option later
        print("LLM reranking not implemented - using document order")
        return documents[:top_k]

