"""
Vector store module for RAG system.
Handles embedding and storage of document chunks.
"""
import os
from typing import List, Optional
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from src.config import (
    OPENAI_API_KEY,
    EMBEDDING_MODEL,
    CHROMA_PERSIST_DIR,
    BASELINE_COLLECTION,
    ADVANCED_COLLECTION,
    BASELINE_TOP_K,
    ADVANCED_TOP_K,
)
from src.document_processor import get_baseline_chunks, get_advanced_chunks


def get_embeddings() -> OpenAIEmbeddings:
    """Get the embedding model (same for baseline and advanced)."""
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        openai_api_key=OPENAI_API_KEY
    )


def create_vector_store(
    documents: List[Document],
    collection_name: str,
    persist_directory: str = CHROMA_PERSIST_DIR
) -> Chroma:
    """Create a new vector store from documents."""
    embeddings = get_embeddings()
    
    # Ensure directory exists
    os.makedirs(persist_directory, exist_ok=True)
    
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=persist_directory
    )
    
    return vector_store


def load_vector_store(
    collection_name: str,
    persist_directory: str = CHROMA_PERSIST_DIR
) -> Chroma:
    """Load an existing vector store."""
    embeddings = get_embeddings()
    
    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_directory
    )


def get_baseline_vector_store(force_recreate: bool = False) -> Chroma:
    """Get or create the baseline vector store."""
    store_path = os.path.join(CHROMA_PERSIST_DIR, BASELINE_COLLECTION)
    
    if force_recreate or not os.path.exists(store_path):
        print("Creating baseline vector store...")
        chunks = get_baseline_chunks()
        return create_vector_store(chunks, BASELINE_COLLECTION)
    
    print("Loading existing baseline vector store...")
    return load_vector_store(BASELINE_COLLECTION)


def get_advanced_vector_store(force_recreate: bool = False) -> Chroma:
    """Get or create the advanced vector store."""
    store_path = os.path.join(CHROMA_PERSIST_DIR, ADVANCED_COLLECTION)
    
    if force_recreate or not os.path.exists(store_path):
        print("Creating advanced vector store...")
        chunks = get_advanced_chunks()
        return create_vector_store(chunks, ADVANCED_COLLECTION)
    
    print("Loading existing advanced vector store...")
    return load_vector_store(ADVANCED_COLLECTION)


def get_baseline_retriever(vector_store: Optional[Chroma] = None):
    """Get baseline retriever with simple similarity search."""
    if vector_store is None:
        vector_store = get_baseline_vector_store()
    
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": BASELINE_TOP_K}
    )


def get_advanced_retriever(vector_store: Optional[Chroma] = None):
    """Get advanced retriever with more results."""
    if vector_store is None:
        vector_store = get_advanced_vector_store()

    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": ADVANCED_TOP_K}
    )


class VectorStoreManager:
    """Manager class for vector store operations."""

    def __init__(self, collection_name: str = ADVANCED_COLLECTION):
        self.collection_name = collection_name
        self.embeddings = get_embeddings()
        self.vector_store = None

    def create_vector_store(self, documents: List[Document]) -> Chroma:
        """Create vector store from documents."""
        os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)

        self.vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            collection_name=self.collection_name,
            persist_directory=CHROMA_PERSIST_DIR
        )
        return self.vector_store

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """Perform similarity search."""
        if self.vector_store is None:
            raise ValueError("Vector store not initialized")
        return self.vector_store.similarity_search(query, k=k)

    def as_retriever(self, k: int = 4):
        """Get as retriever."""
        if self.vector_store is None:
            raise ValueError("Vector store not initialized")
        return self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )


if __name__ == "__main__":
    print("Testing vector store creation...")
    
    # Create baseline store
    store = get_baseline_vector_store(force_recreate=True)
    print(f"Baseline store created")
    
    # Test retrieval
    retriever = get_baseline_retriever(store)
    results = retriever.invoke("How do I create an LLM chain?")
    print(f"\nRetrieved {len(results)} documents for test query")
    print(f"Top result: {results[0].page_content[:200]}...")

