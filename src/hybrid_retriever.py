"""
Hybrid retriever combining semantic search with BM25 keyword search.
"""
from typing import List, Dict, Any, Optional
from collections import defaultdict
import math

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from src.vector_store import VectorStoreManager
from src.document_processor import DocumentProcessor


class HybridRetriever:
    """
    Hybrid retriever that combines:
    1. Semantic search (vector similarity)
    2. BM25 keyword search
    
    Uses Reciprocal Rank Fusion (RRF) to combine results.
    """
    
    def __init__(
        self,
        vector_store: VectorStoreManager,
        documents: Optional[List[Document]] = None,
        semantic_weight: float = 0.5,
        bm25_weight: float = 0.5,
        rrf_k: int = 60
    ):
        """
        Initialize hybrid retriever.
        
        Args:
            vector_store: Vector store manager for semantic search
            documents: Documents for BM25 index (if None, loads from processor)
            semantic_weight: Weight for semantic search results
            bm25_weight: Weight for BM25 results
            rrf_k: RRF constant (default 60)
        """
        self.vector_store = vector_store
        self.semantic_weight = semantic_weight
        self.bm25_weight = bm25_weight
        self.rrf_k = rrf_k
        
        # Load documents if not provided
        if documents is None:
            processor = DocumentProcessor()
            documents = processor.load_documents()
            documents = processor.chunk_documents(documents, use_advanced=True)
        
        self.documents = documents
        
        # Build BM25 index
        self._build_bm25_index()
    
    def _build_bm25_index(self):
        """Build BM25 index from documents."""
        # Tokenize documents (simple whitespace tokenization)
        self.tokenized_docs = [
            self._tokenize(doc.page_content) 
            for doc in self.documents
        ]
        
        self.bm25 = BM25Okapi(self.tokenized_docs)
        print(f"Built BM25 index with {len(self.documents)} documents")
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization for BM25."""
        # Lowercase and split on whitespace/punctuation
        text = text.lower()
        # Replace common punctuation with spaces
        for char in '.,;:!?()[]{}"\'-_/\\':
            text = text.replace(char, ' ')
        return text.split()
    
    def _semantic_search(self, query: str, k: int) -> List[Dict[str, Any]]:
        """Perform semantic search."""
        results = self.vector_store.similarity_search(query, k=k)
        return [
            {"content": doc.page_content, "metadata": doc.metadata, "source": "semantic"}
            for doc in results
        ]
    
    def _bm25_search(self, query: str, k: int) -> List[Dict[str, Any]]:
        """Perform BM25 keyword search."""
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k indices
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include if there's some match
                results.append({
                    "content": self.documents[idx].page_content,
                    "metadata": self.documents[idx].metadata,
                    "source": "bm25",
                    "bm25_score": scores[idx]
                })
        
        return results
    
    def _reciprocal_rank_fusion(
        self,
        semantic_results: List[Dict],
        bm25_results: List[Dict],
        k: int
    ) -> List[Dict[str, Any]]:
        """
        Combine results using Reciprocal Rank Fusion.
        
        RRF score = sum(1 / (k + rank)) for each result list
        """
        # Create content -> score mapping
        rrf_scores = defaultdict(float)
        content_to_result = {}
        
        # Score semantic results
        for rank, result in enumerate(semantic_results):
            content = result["content"]
            rrf_scores[content] += self.semantic_weight * (1 / (self.rrf_k + rank + 1))
            content_to_result[content] = result
        
        # Score BM25 results
        for rank, result in enumerate(bm25_results):
            content = result["content"]
            rrf_scores[content] += self.bm25_weight * (1 / (self.rrf_k + rank + 1))
            if content not in content_to_result:
                content_to_result[content] = result
        
        # Sort by RRF score
        sorted_contents = sorted(rrf_scores.keys(), key=lambda c: rrf_scores[c], reverse=True)
        
        # Return top-k results
        results = []
        for content in sorted_contents[:k]:
            result = content_to_result[content]
            result["rrf_score"] = rrf_scores[content]
            results.append(result)
        
        return results
    
    def retrieve(self, query: str, k: int = 4) -> List[Document]:
        """
        Retrieve documents using hybrid search.
        
        Args:
            query: Search query
            k: Number of documents to return
        
        Returns:
            List of retrieved documents
        """
        # Get more results from each source for fusion
        fetch_k = k * 2
        
        # Semantic search
        semantic_results = self._semantic_search(query, fetch_k)
        
        # BM25 search
        bm25_results = self._bm25_search(query, fetch_k)
        
        # Fuse results
        fused_results = self._reciprocal_rank_fusion(semantic_results, bm25_results, k)
        
        # Convert to Documents
        return [
            Document(page_content=r["content"], metadata=r.get("metadata", {}))
            for r in fused_results
        ]

