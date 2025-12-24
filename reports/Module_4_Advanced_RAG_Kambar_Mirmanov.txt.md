# Advanced RAG Enhancement Report
## Technical Documentation Assistant

**Project:** RAG System for Software Documentation Assistance
**Use Case:** Helping developers quickly find answers from technical documentation (LangChain, in this case)

---

## Setup Instructions

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation Steps

1. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv venv && source venv/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables:**
   Create a `.env` file in the project root with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

### Running the System

**Test the baseline RAG system:**
```bash
python3 scripts/test_baseline.py
```

**Test the advanced RAG system:**
```bash
python scripts/test_advanced.py
```

---

## 1. Executive Summary

This report documents the development and enhancement of a Retrieval-Augmented Generation (RAG) system designed to assist developers in navigating LangChain's technical documentation. The system aims to provide accurate, contextually relevant answers to developer queries about API usage, troubleshooting, and implementation patterns.

---

## 2. Real-World Problem Statement

### 2.1 Problem Description
Developers working with LangChain (or any other complex documentation) face several challenges:
- **Documentation Overload**: LangChain's documentation spans hundreds of pages covering APIs, tutorials, and how-to guides
- **Rapid Framework Evolution**: Frequent updates make it hard to find current, accurate information
- **Context-Dependent Answers**: Questions often require understanding of multiple related concepts
- **Time-Sensitive Development**: Developers need quick, accurate answers to maintain productivity

### 2.2 Business Value
- **Reduced Development Time**: Faster access to accurate documentation answers
- **Decreased Support Burden**: Self-service documentation assistant
- **Improved Developer Experience**: Contextual, relevant responses instead of generic search results
- **Knowledge Retention**: Consistent answers across the organization

---

## 3. Metrics Definition

### 3.1 Considered Metrics

| Metric | Description | Business Relevance |
|--------|-------------|-------------------|
| **Answer Relevancy** | How well the answer addresses the user's question | High - directly impacts developer satisfaction |
| **Faithfulness** | Whether the answer is grounded in retrieved context (no hallucinations) | Critical - incorrect code/API info leads to bugs |
| **Context Precision** | Relevance of retrieved documents to the question | Medium - affects answer quality |
| **Context Recall** | Coverage of ground truth in retrieved context | Medium - ensures comprehensive answers |

### 3.2 Selected Primary Metrics

**Primary Metric: Faithfulness (Target: 30%+ improvement)**
- **Rationale**: For technical documentation, accuracy is paramount. A hallucinated API method or incorrect parameter can cause hours of debugging. Faithfulness measures whether generated answers are grounded in the retrieved documentation.

**Secondary Metric: Answer Relevancy**
- **Rationale**: Beyond being accurate, answers must directly address the developer's question. This measures practical usefulness.

### 3.3 Why These Metrics?

1. **Developer Trust**: Technical documentation must be accurate. Hallucinations erode trust.
2. **Measurable Impact**: Both metrics can be quantified using RAGAS framework.
3. **Improvement Potential**: Baseline systems often struggle with faithfulness on technical content.

---

## 4. System Architecture

### 4.1 Model Selection

| Component | Model | Rationale |
|-----------|-------|-----------|
| **LLM** | `gpt-3.5-turbo` | Cost-effective, but prone to hallucinations on technical details without proper context|
| **Embeddings** | `text-embedding-3-small` | OpenAI's efficient embedding model, consistent with LLM provider |

**Note**: Same models are used for both baseline and advanced systems to ensure improvements come purely from RAG techniques.

### 4.2 Baseline Configuration
- **Chunking**: Fixed-size (1000 chars, 200 overlap)
- **Retrieval**: Simple vector similarity search (top-k=4)
- **Vector Store**: ChromaDB
- **No reranking or query enhancement**

---

## 5. Data Corpus

### 5.1 Source
- LangChain Python Documentation (python.langchain.com)
- API Reference, Tutorials, How-To Guides

### 5.2 Statistics
- **Total Documents**: 57
- **Total Characters**: ~585,000
- **Content Types**: API documentation, tutorials, integration guides

---

## 6. Evaluation Framework

### 6.1 Test Dataset
A curated set of 20 developer questions covering:
- API usage patterns
- Integration configurations  
- Troubleshooting scenarios
- Best practices

### 6.2 Evaluation Method
- RAGAS framework for automated evaluation
- Multiple runs to account for variance
- Statistical significance testing

---

## 7. Baseline Results

**Number of Runs:** 3
**Questions Evaluated:** 20

### 7.1 Baseline Metrics

| Metric | Mean Score | Std Dev | Notes |
|--------|------------|---------|-------|
| **Faithfulness** | 0.5731 | ±0.0042 | Primary metric - shows room for improvement |
| **Answer Relevancy** | 0.7732 | ±0.0165 | Secondary metric - reasonably good |

### 7.2 Baseline Analysis

**Observations:**
1. **Faithfulness (57.31%)**: The baseline system produces answers that are only partially grounded in the retrieved context. This indicates significant hallucination/confabulation issues.

2. **Answer Relevancy (77.32%)**: Answers are generally relevant to the questions asked, but the lower faithfulness suggests they may include information not supported by the documentation.

**Key Issues Identified:**
- Retrieved contexts are often too general (API overview pages rather than specific method documentation)
- Large chunk sizes (1000 chars) may include irrelevant information
- No query reformulation leads to suboptimal retrieval
- Simple top-k retrieval doesn't consider document diversity

**Target for Improvement:**
- Faithfulness: From 0.5731 to ≥0.7450 (+30% improvement)
- This represents moving from "often hallucinating" to "mostly grounded" answers

---

## 8. Enhancement Strategy

### 8.1 Techniques Implemented

| Enhancement | Description | Expected Impact |
|-------------|-------------|-----------------|
| **Smaller Chunks** | 500 chars (vs 1000) with 100 overlap | More precise context retrieval |
| **Query Expansion** | LLM-generated query variations | Better coverage of relevant documents |
| **Hybrid Retrieval** | Semantic + BM25 keyword search with RRF fusion | Captures both semantic and lexical matches |
| **Cross-Encoder Reranking** | ms-marco-MiniLM reranker | More accurate relevance scoring |
| **Improved Prompting** | Explicit grounding instructions | Better faithfulness to context |

### 8.2 Implementation Details

**Query Expansion:**
- Generates 2 alternative query formulations using LLM
- Extracts key technical concepts for targeted search
- All queries searched in parallel

**Hybrid Retrieval (RRF):**
- Semantic search: Vector similarity using OpenAI embeddings
- BM25 search: Keyword-based scoring
- Reciprocal Rank Fusion with weights: 0.6 semantic, 0.4 BM25

**Reranking:**
- Cross-encoder model: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Reranks top candidates to select final 4 documents

---

## 9. Enhanced Results

**Number of Runs:** 3
**Questions Evaluated:** 20

### 9.1 Advanced System Metrics

| Metric | Mean Score | Std Dev |
|--------|------------|---------|
| **Faithfulness** | 0.7604 | ±0.0048 |
| **Answer Relevancy** | 0.9091 | ±0.0114 |

### 9.2 Comparison Summary

| Metric | Baseline | Advanced | Improvement |
|--------|----------|----------|-------------|
| **Faithfulness** | 0.5731 | 0.7604 | **+32.70%** ✅ |
| **Answer Relevancy** | 0.7732 | 0.9091 | +17.58% |

### 9.3 Analysis

**Faithfulness Improvement (+32.70%):**
- The advanced system's answers are now grounded in the retrieved documentation
- Hybrid retrieval brings in more relevant, specific context
- Reranking ensures the most pertinent documents are used
- Improved prompting guides the LLM to stick to provided context

**Answer Relevancy Improvement (+17.58%):**
- Query expansion captures different phrasings of the same question
- Better retrieval leads to more complete, helpful answers
- Smaller chunks provide focused, relevant information

---

## 10. Conclusions

### 10.1 Target Achievement

✅ **Primary Goal Achieved**: Faithfulness improved by **32.70%** (target: 30%)

The advanced RAG system successfully addresses the hallucination problem in technical documentation assistance.

### 10.2 Key Findings

1. **Retrieval Quality Matters Most**: The combination of hybrid retrieval and reranking had the largest impact on faithfulness. Better context = better answers.

2. **Chunk Size Trade-offs**: Smaller chunks (500 chars) provided more focused context but required fetching more documents. The reranker compensated by selecting the most relevant.

3. **Query Expansion Helps Coverage**: Multiple query variations ensured relevant documents weren't missed due to terminology mismatches.

4. **Prompt Engineering Alone Isn't Enough**: Early experiments showed that improving prompts without better retrieval had limited impact. The improvements work best in combination.

### 10.3 Business Impact

| Aspect | Baseline | Advanced |
|--------|----------|----------|
| Answer Accuracy | ~57% grounded | ~76% grounded |
| Developer Trust | Low (frequent hallucinations) | High (mostly accurate) |
| Time to Answer | Fast | ~2-3x slower (acceptable) |

### 10.4 Future Improvements

1. **Document Re-ranking with LLM**: Could further improve relevance scoring
2. **Multi-hop Retrieval**: For complex questions requiring information from multiple sources
3. **Caching**: Reduce latency by caching common queries
4. **Fine-tuned Embeddings**: Domain-specific embeddings for better semantic matching

---

## Appendix A: Technical Implementation

### A.1 Project Structure

```
rag/
├── src/
│   ├── config.py           # Configuration settings
│   ├── document_processor.py # Document loading & chunking
│   ├── vector_store.py      # ChromaDB vector store
│   ├── query_expansion.py   # Query reformulation
│   ├── hybrid_retriever.py  # Semantic + BM25 fusion
│   ├── reranker.py          # Cross-encoder reranking
│   ├── rag_chain.py         # Baseline & Advanced RAG
│   └── evaluation.py        # RAGAS evaluation
├── scripts/
│   ├── run_evaluation.py    # Automated evaluation runner
│   └── compare_results.py   # Results comparison
├── data/
│   ├── raw_docs/            # LangChain documentation
│   └── eval_questions.json  # Test dataset
└── reports/
    └── RAG_Enhancement_Report.md
```

### A.2 Dependencies

- LangChain + LangChain-OpenAI
- ChromaDB (vector store)
- RAGAS (evaluation)
- sentence-transformers (reranking)
- rank-bm25 (keyword search)

---

## Appendix B: Evaluation Questions Sample

1. How do I create a simple LLM chain in LangChain?
2. What is the difference between ChatOpenAI and OpenAI in LangChain?
3. How do I use a vector store with LangChain for document retrieval?
4. What are the different types of memory in LangChain?
5. How do I create a ReAct agent in LangChain?


