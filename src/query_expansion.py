"""
Query expansion module for improved retrieval.
Uses LLM to generate multiple query variations.
"""
from typing import List
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.config import OPENAI_API_KEY, LLM_MODEL


QUERY_EXPANSION_PROMPT = """You are a helpful assistant that generates search queries.
Given a user question about LangChain, generate {num_queries} different search queries that would help find relevant documentation.

Each query should:
1. Focus on different aspects or keywords from the original question
2. Use alternative terminology that might appear in documentation
3. Be specific and targeted for documentation search

Original question: {question}

Generate exactly {num_queries} search queries, one per line. Output ONLY the queries, no numbering or explanations."""


QUERY_DECOMPOSITION_PROMPT = """You are a helpful assistant that breaks down complex questions.
Given a question about LangChain, identify the key concepts that need to be looked up.

Original question: {question}

List the key technical concepts or terms that should be searched for, one per line.
Focus on LangChain-specific terminology. Output ONLY the concepts, no explanations."""


class QueryExpander:
    """Expands queries for improved retrieval."""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model=LLM_MODEL,
            openai_api_key=OPENAI_API_KEY,
            temperature=0.3  # Slight creativity for variations
        )
        
        self.expansion_chain = (
            ChatPromptTemplate.from_template(QUERY_EXPANSION_PROMPT)
            | self.llm
            | StrOutputParser()
        )
        
        self.decomposition_chain = (
            ChatPromptTemplate.from_template(QUERY_DECOMPOSITION_PROMPT)
            | self.llm
            | StrOutputParser()
        )
    
    def expand_query(self, question: str, num_queries: int = 3) -> List[str]:
        """
        Generate multiple query variations for a question.
        
        Args:
            question: Original user question
            num_queries: Number of query variations to generate
        
        Returns:
            List of query strings including original
        """
        try:
            response = self.expansion_chain.invoke({
                "question": question,
                "num_queries": num_queries
            })
            
            # Parse response into queries
            queries = [q.strip() for q in response.strip().split('\n') if q.strip()]
            
            # Always include original question
            all_queries = [question] + queries[:num_queries]
            
            return all_queries
            
        except Exception as e:
            print(f"Query expansion failed: {e}")
            return [question]
    
    def decompose_query(self, question: str) -> List[str]:
        """
        Extract key concepts from a question for targeted search.
        
        Args:
            question: Original user question
        
        Returns:
            List of key concepts/terms
        """
        try:
            response = self.decomposition_chain.invoke({"question": question})
            
            # Parse concepts
            concepts = [c.strip() for c in response.strip().split('\n') if c.strip()]
            
            return concepts
            
        except Exception as e:
            print(f"Query decomposition failed: {e}")
            return []
    
    def get_all_queries(self, question: str, num_expansions: int = 2) -> List[str]:
        """
        Get all query variations: original + expansions + key concepts.
        
        Args:
            question: Original user question
            num_expansions: Number of expanded queries
        
        Returns:
            List of all queries to search
        """
        queries = set()
        
        # Original question
        queries.add(question)
        
        # Expanded variations
        expanded = self.expand_query(question, num_expansions)
        queries.update(expanded)
        
        # Key concepts (shorter, more targeted)
        concepts = self.decompose_query(question)
        queries.update(concepts)
        
        return list(queries)

