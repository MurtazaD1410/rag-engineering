# RAG Engineering

A hands-on exploration of Retrieval-Augmented Generation (RAG), progressing from fundamental RAG pipelines to advanced document processing and retrieval optimisation. This project demonstrates how LLMs can retrieve relevant information from external knowledge sources and generate context-aware responses using locally hosted models and free-tier AI services.

## ✨ Features

- **Document Ingestion:** Load and process documents to prepare them for retrieval and knowledge extraction.
- **Basic RAG Pipeline:** Implement document ingestion, embedding generation, vector storage, retrieval, and answer generation.
- **Conversational RAG:** Maintain conversation history and generate context-aware responses using history-aware retrieval.
- **Chunking Strategies:** Explore recursive character splitting, semantic chunking, and agentic chunking to improve document segmentation.
- **Multimodal RAG:** Process PDFs containing text, tables, and images using document partitioning and AI-enhanced content summaries.
- **Retrieval Methods:** Explore similarity search, similarity score thresholds, and MMR (Maximum Marginal Relevance).
- **Multi-Query Retrieval:** Generate alternative query variations to improve retrieval coverage.
- **Reciprocal Rank Fusion (RRF):** Combine multiple retrieval result rankings into a unified ranking.
- **Hybrid Search:** Combine BM25 keyword retrieval with dense vector search to leverage both lexical and semantic matching.
- **Reranking:** Integrate Cohere Rerank to reorder retrieved documents based on query relevance before passing context to the LLM.

## 🚀 Tech Stack

This project is built using Python and the LangChain ecosystem.

### Core Technologies

- **Python** - Primary programming language.
- **LangChain** - Framework for building LLM-powered applications and RAG pipelines.
- **ChromaDB** - Vector database for storing document embeddings and performing similarity searches.
- **Hugging Face** - Provides embedding models for semantic representation of documents and queries.
- **Ollama** - Locally hosted LLM inference for document understanding and answer generation.
- **Cohere Rerank** - Reranking retrieved documents based on semantic relevance.

### Additional Technologies

- **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2`
- **LLM:** Qwen3 8B (`qwen3:8b`) via Ollama.
- **Vision LLM:** Qwen3-VL 8B (`qwen3-vl:8b`) via Ollama.
- **Reranking Model:** Cohere `rerank-english-v3.0`.
- **Retrieval:** Dense Vector Search, BM25, Hybrid Search, Multi-Query Retrieval, Reciprocal Rank Fusion.
- **Document Processing:** Unstructured.
- **Configuration:** Python Dotenv.

## 📦 Installation

### Prerequisites

- Python 3.10+
- pip
- [Ollama](https://ollama.com/) installed and running.
- Cohere API key for reranking functionality.

### Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/MurtazaD1410/rag-engineering.git
   cd rag-engineering
   ```

2. **Create a virtual environment**

   ```bash
   python -m venv venv
   ```

   Activate the environment:

   ```bash
   # macOS / Linux
   source venv/bin/activate

   # Windows
   venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -U pip

   pip install \
     python-dotenv \
     langchain \
     langchain-core \
     langchain-community \
     langchain-classic \
     langchain-chroma \
     langchain-huggingface \
     langchain-ollama \
     langchain-text-splitters \
     langchain-cohere \
     sentence-transformers \
     rank-bm25 \
     unstructured[pdf]
   ```

   Depending on your operating system, PDF processing may also require additional system dependencies such as Poppler.

4. **Configure environment variables**

   Create a `.env` file in the project root:

   ```env
   CO_API_KEY=your_cohere_api_key
   ```

   Replace the placeholder with your Cohere API key.

## ⚙️ Service Configuration

### Ollama Setup

Ollama runs the language models locally, allowing you to generate responses without relying on paid hosted LLM APIs.

1. Install Ollama from [ollama.com](https://ollama.com/).
2. Pull the required models:

   ```bash
   ollama pull qwen3:8b
   ollama pull qwen3-vl:8b
   ```

3. Ensure the Ollama service is running:

   ```bash
   ollama serve
   ```

   If Ollama is already running as a background service, you do not need to start it again.

### Hugging Face Embeddings

The project uses `sentence-transformers/all-MiniLM-L6-v2` for generating document and query embeddings.

The model is downloaded automatically when the embedding pipeline is initialized for the first time.

No API key is required for this public embedding model.

### Cohere Reranking

Cohere is used to improve retrieval relevance by reranking the documents returned by the initial retrieval stage.

1. Create an account at [Cohere](https://cohere.com/).
2. Generate an API key.
3. Add the key to your `.env` file:

   ```env
   CO_API_KEY=your_cohere_api_key
   ```

## 🧪 Running the Project

The repository contains numbered Python scripts that progressively explore different RAG concepts and implementations.

Run the scripts individually from the project root.

### 1. Basic RAG Pipeline

```bash
python 1_ingestion_pipeline.py
python 2_retrieval_pipeline.py
python 3_answer_generation.py
python 4_history_aware_generation.py
```

- Document loading and ingestion.
- Text splitting and embedding generation.
- Persistent vector storage with ChromaDB.
- Similarity-based document retrieval.
- Context-aware answer generation using Ollama.
- Conversation history-aware retrieval.

### 2. Chunking Strategies

```bash
python 5_recursive_character_text_splitter.py
python 6_semantic_chunking.py
python 7_agentic_chunking.py
```

Explore different approaches to splitting documents into meaningful chunks:

- Recursive Character Text Splitting.
- Semantic Chunking.
- Agentic Chunking.

### 3. Multimodal RAG

```bash
python 8_multi_modal_rag.py
```

Explore document processing beyond plain text:

- PDF document partitioning.
- Text, table, and image extraction.
- Title-based document chunking.
- AI-enhanced summaries using Qwen3-VL.
- Multimodal content storage and retrieval.

### 4. Retrieval Optimisation

```bash
python 9_retrieval_methods.py
python 10_multi_query_retrieval.py
python 11_reciprocal_rank_fusion.py
python 12_hybrid_search.py
python 13_reranker.py
```

Explore advanced retrieval techniques:

- Similarity Search and MMR.
- Multi-Query Retrieval.
- Reciprocal Rank Fusion (RRF).
- Hybrid Search using BM25 and dense vector retrieval.
- Cohere-based document reranking.

**Note:** Some scripts depend on previously generated ChromaDB collections or input documents. Run the ingestion pipeline before executing retrieval scripts that use its persistent vector store. The multimodal pipeline has its own vector store and PDF input requirements.

## 📁 Repository Structure

```text
rag-engineering/
│
├── docs/                                  # Source documents
│
├── 1_ingestion_pipeline.py                # Document ingestion and vector storage
├── 2_retrieval_pipeline.py                # Similarity-based retrieval
├── 3_answer_generation.py                 # RAG answer generation
├── 4_history_aware_generation.py          # Conversational RAG
│
├── 5_recursive_character_text_splitter.py # Recursive chunking
├── 6_semantic_chunking.py                 # Semantic chunking
├── 7_agentic_chunking.py                  # Agentic chunking
├── 8_multi_modal_rag.py                   # Multimodal document processing
│
├── 9_retrieval_methods.py                 # Retrieval strategies
├── 10_multi_query_retrieval.py            # Multi-query retrieval
├── 11_reciprocal_rank_fusion.py           # Reciprocal Rank Fusion
├── 12_hybrid_search.py                    # BM25 + vector search
├── 13_reranker.py                         # Cohere document reranking
│
├── chunks_export.json                     # Exported document chunks
├── rag_results.json                       # Retrieved results
├── synthetic_questions.pdf                # Question dataset
│
├── .env                                   # Environment variables (not committed)
├── .gitignore
└── README.md
```

⭐ If you find this project helpful, consider giving it a star on GitHub!
