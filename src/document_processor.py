"""
Document processing module for RAG system.
Handles loading, cleaning, and chunking documents.
"""
import json
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import (
    RAW_DOCS_PATH,
    BASELINE_CHUNK_SIZE,
    BASELINE_CHUNK_OVERLAP,
    ADVANCED_CHUNK_SIZE,
    ADVANCED_CHUNK_OVERLAP,
)


def load_raw_documents(path: str = RAW_DOCS_PATH) -> List[Document]:
    """Load raw documents from JSON file."""
    with open(path, 'r') as f:
        raw_docs = json.load(f)
    
    documents = []
    for doc in raw_docs:
        # Create Document with metadata
        documents.append(Document(
            page_content=doc['content'],
            metadata={
                'source': doc['url'],
                'title': doc['title']
            }
        ))
    
    return documents


def create_text_splitter(
    chunk_size: int = BASELINE_CHUNK_SIZE,
    chunk_overlap: int = BASELINE_CHUNK_OVERLAP
) -> RecursiveCharacterTextSplitter:
    """Create a text splitter with specified parameters."""
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )


def chunk_documents(
    documents: List[Document],
    chunk_size: int = BASELINE_CHUNK_SIZE,
    chunk_overlap: int = BASELINE_CHUNK_OVERLAP
) -> List[Document]:
    """Split documents into chunks."""
    splitter = create_text_splitter(chunk_size, chunk_overlap)
    chunks = splitter.split_documents(documents)
    
    # Add chunk index to metadata
    for i, chunk in enumerate(chunks):
        chunk.metadata['chunk_id'] = i
    
    return chunks


def get_baseline_chunks() -> List[Document]:
    """Get documents chunked with baseline configuration."""
    documents = load_raw_documents()
    return chunk_documents(
        documents,
        chunk_size=BASELINE_CHUNK_SIZE,
        chunk_overlap=BASELINE_CHUNK_OVERLAP
    )


def get_advanced_chunks() -> List[Document]:
    """Get documents chunked with advanced configuration."""
    documents = load_raw_documents()
    return chunk_documents(
        documents,
        chunk_size=ADVANCED_CHUNK_SIZE,
        chunk_overlap=ADVANCED_CHUNK_OVERLAP
    )


class DocumentProcessor:
    """Document processor class for OOP-style usage."""

    def __init__(self, docs_path: str = RAW_DOCS_PATH):
        self.docs_path = docs_path

    def load_documents(self) -> List[Document]:
        """Load raw documents."""
        return load_raw_documents(self.docs_path)

    def chunk_documents(
        self,
        documents: List[Document],
        use_advanced: bool = False
    ) -> List[Document]:
        """Chunk documents with baseline or advanced config."""
        if use_advanced:
            return chunk_documents(
                documents,
                chunk_size=ADVANCED_CHUNK_SIZE,
                chunk_overlap=ADVANCED_CHUNK_OVERLAP
            )
        else:
            return chunk_documents(
                documents,
                chunk_size=BASELINE_CHUNK_SIZE,
                chunk_overlap=BASELINE_CHUNK_OVERLAP
            )


if __name__ == "__main__":
    # Test document processing
    print("Loading documents...")
    docs = load_raw_documents()
    print(f"Loaded {len(docs)} documents")
    
    print("\nChunking with baseline config...")
    baseline_chunks = get_baseline_chunks()
    print(f"Created {len(baseline_chunks)} baseline chunks")
    
    print("\nChunking with advanced config...")
    advanced_chunks = get_advanced_chunks()
    print(f"Created {len(advanced_chunks)} advanced chunks")
    
    print("\nSample chunk:")
    print(f"Content: {baseline_chunks[0].page_content[:200]}...")
    print(f"Metadata: {baseline_chunks[0].metadata}")

