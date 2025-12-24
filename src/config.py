"""
Configuration for RAG system.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Model Configuration - SAME FOR BASELINE AND ADVANCED
# Using OpenAI gpt-3.5-turbo - older model that benefits from RAG improvements
LLM_MODEL = "gpt-3.5-turbo"  # Chosen for improvement potential with RAG techniques
LLM_PROVIDER = "openai"
EMBEDDING_MODEL = "text-embedding-3-small"  # OpenAI's embedding model

# Chunking Configuration (Baseline)
BASELINE_CHUNK_SIZE = 1000
BASELINE_CHUNK_OVERLAP = 200

# Advanced Chunking Configuration
ADVANCED_CHUNK_SIZE = 500
ADVANCED_CHUNK_OVERLAP = 100

# Retrieval Configuration
BASELINE_TOP_K = 4
ADVANCED_TOP_K = 6

# Vector Store
CHROMA_PERSIST_DIR = "data/chroma_db"
BASELINE_COLLECTION = "langchain_docs_baseline"
ADVANCED_COLLECTION = "langchain_docs_advanced"

# Data Paths
RAW_DOCS_PATH = "data/raw/langchain_docs.json"
PROCESSED_DOCS_PATH = "data/processed/"
EVAL_RESULTS_PATH = "data/eval_results/"

# Evaluation Configuration
EVAL_NUM_QUESTIONS = 20
EVAL_RUNS = 3  # Number of runs for averaging results

