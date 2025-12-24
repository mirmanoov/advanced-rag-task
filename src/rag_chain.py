"""
RAG Chain module for baseline and advanced implementations.
"""
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.documents import Document
from src.config import OPENAI_API_KEY, LLM_MODEL
from src.vector_store import get_baseline_retriever, get_advanced_retriever


def get_llm(temperature: float = 0.1) -> ChatOpenAI:
    """Get the LLM (same for baseline and advanced)."""
    return ChatOpenAI(
        model=LLM_MODEL,
        openai_api_key=OPENAI_API_KEY,
        temperature=temperature,
        max_tokens=1024
    )


def format_docs(docs: List[Document]) -> str:
    """Format documents into a single string for context."""
    return "\n\n---\n\n".join(
        f"Source: {doc.metadata.get('title', 'Unknown')}\n{doc.page_content}"
        for doc in docs
    )


# Baseline prompt - simple and direct
BASELINE_PROMPT = ChatPromptTemplate.from_template("""
You are a helpful assistant for LangChain documentation. 
Answer the question based on the provided context.

Context:
{context}

Question: {question}

Answer:
""")


def create_baseline_chain():
    """Create the baseline RAG chain with simple retrieval."""
    retriever = get_baseline_retriever()
    llm = get_llm()
    
    chain = (
        RunnableParallel(
            context=retriever | format_docs,
            question=RunnablePassthrough()
        )
        | BASELINE_PROMPT
        | llm
        | StrOutputParser()
    )
    
    return chain


def create_baseline_chain_with_sources():
    """Create baseline chain that also returns source documents."""
    retriever = get_baseline_retriever()
    llm = get_llm()
    
    def get_response_with_sources(inputs: Dict[str, Any]) -> Dict[str, Any]:
        question = inputs["question"]
        docs = retriever.invoke(question)
        context = format_docs(docs)
        
        prompt = BASELINE_PROMPT.format(context=context, question=question)
        response = llm.invoke(prompt)
        
        return {
            "question": question,
            "answer": response.content,
            "contexts": [doc.page_content for doc in docs],
            "source_documents": docs
        }
    
    return get_response_with_sources


class BaselineRAG:
    """Baseline RAG system wrapper for evaluation."""
    
    def __init__(self):
        self.retriever = get_baseline_retriever()
        self.llm = get_llm()
        self.prompt = BASELINE_PROMPT
    
    def query(self, question: str) -> Dict[str, Any]:
        """Query the RAG system and return answer with contexts."""
        # Retrieve documents
        docs = self.retriever.invoke(question)
        contexts = [doc.page_content for doc in docs]
        context_str = format_docs(docs)
        
        # Generate answer
        prompt = self.prompt.format(context=context_str, question=question)
        response = self.llm.invoke(prompt)
        
        return {
            "question": question,
            "answer": response.content,
            "contexts": contexts
        }


# Advanced prompt - explicit grounding instructions while being helpful
ADVANCED_PROMPT = ChatPromptTemplate.from_template("""You are a helpful assistant for LangChain documentation.
Your task is to answer questions based on the provided context from the documentation.

INSTRUCTIONS:
1. Base your answer primarily on the information in the context below
2. Be specific - mention class names, method names, and include code examples when available in the context
3. If the context mentions relevant concepts, explain them clearly
4. Synthesize information from multiple parts of the context when helpful
5. Provide a complete, helpful answer that addresses the user's question

Context from LangChain Documentation:
{context}

Question: {question}

Provide a helpful, accurate answer based on the documentation above:""")


class AdvancedRAG:
    """
    Advanced RAG system with multiple enhancements:
    1. Query expansion for better retrieval
    2. Hybrid search (semantic + BM25)
    3. Cross-encoder reranking
    4. Improved prompting for faithfulness
    """

    def __init__(self):
        from src.vector_store import VectorStoreManager
        from src.document_processor import DocumentProcessor
        from src.query_expansion import QueryExpander
        from src.hybrid_retriever import HybridRetriever
        from src.reranker import Reranker

        print("Initializing Advanced RAG system...")

        # Load and process documents with advanced settings
        self.processor = DocumentProcessor()
        docs = self.processor.load_documents()
        self.chunks = self.processor.chunk_documents(docs, use_advanced=True)

        # Vector store with advanced settings
        self.vector_store = VectorStoreManager(collection_name="langchain_advanced")
        self.vector_store.create_vector_store(self.chunks)

        # Query expander
        self.query_expander = QueryExpander()

        # Hybrid retriever
        self.hybrid_retriever = HybridRetriever(
            vector_store=self.vector_store,
            documents=self.chunks,
            semantic_weight=0.6,
            bm25_weight=0.4
        )

        # Reranker
        self.reranker = Reranker()

        # LLM and prompt
        self.llm = get_llm(temperature=0.1)
        self.prompt = ADVANCED_PROMPT

        print("Advanced RAG system initialized!")

    def query(self, question: str) -> Dict[str, Any]:
        """
        Query with all advanced enhancements.

        Pipeline:
        1. Expand query into multiple variations
        2. Retrieve using hybrid search for each query
        3. Deduplicate and rerank results
        4. Generate answer with improved prompt
        """
        # Step 1: Query expansion
        expanded_queries = self.query_expander.expand_query(question, num_queries=2)

        # Step 2: Hybrid retrieval for each query
        all_docs = []
        seen_contents = set()

        for query in expanded_queries:
            docs = self.hybrid_retriever.retrieve(query, k=4)
            for doc in docs:
                # Deduplicate by content
                content_hash = hash(doc.page_content[:200])
                if content_hash not in seen_contents:
                    seen_contents.add(content_hash)
                    all_docs.append(doc)

        # Step 3: Rerank all retrieved documents
        reranked_docs = self.reranker.rerank(question, all_docs, top_k=4)

        # Step 4: Generate answer
        contexts = [doc.page_content for doc in reranked_docs]
        context_str = format_docs(reranked_docs)

        prompt = self.prompt.format(context=context_str, question=question)
        response = self.llm.invoke(prompt)

        return {
            "question": question,
            "answer": response.content,
            "contexts": contexts
        }


if __name__ == "__main__":
    print("Testing RAG Systems...")

    test_questions = [
        "How do I create an LLM chain in LangChain?",
        "What is LCEL?",
        "How do I use a vector store?"
    ]

    print("\n" + "="*70)
    print("BASELINE RAG")
    print("="*70)
    rag = BaselineRAG()
    for q in test_questions[:1]:
        print(f"\nQuestion: {q}")
        result = rag.query(q)
        print(f"Answer: {result['answer'][:400]}...")

    print("\n" + "="*70)
    print("ADVANCED RAG")
    print("="*70)
    adv_rag = AdvancedRAG()
    for q in test_questions[:1]:
        print(f"\nQuestion: {q}")
        result = adv_rag.query(q)
        print(f"Answer: {result['answer'][:400]}...")

