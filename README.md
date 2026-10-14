# Football Tactical Analysis RAG System

A Retrieval-Augmented Generation (RAG) application for querying UEFA Champions League technical reports and retrieving evidence-backed football tactical insights using natural language.

## Overview

Football analysts, coaches, and performance staff often need to search lengthy technical reports to find tactical observations, match analysis, and team performance insights. This project simplifies that process through a conversational interface that retrieves relevant passages from PDF reports and uses a large language model (LLM) to generate context-grounded answers.

The system displays retrieved document excerpts with source filenames and page numbers to make answers easier to verify.

## Key Features

- **PDF ingestion:** Load and process football technical reports.
- **Text chunking:** Split documents into overlapping chunks using `RecursiveCharacterTextSplitter`.
- **Semantic embeddings:** Generate document embeddings with `sentence-transformers/all-MiniLM-L6-v2`.
- **Persistent vector storage:** Store and retrieve document chunks using ChromaDB.
- **Conversational question answering:** Generate answers using LangChain and a Groq-hosted LLM.
- **Source references:** Display retrieved passages with document filenames and page numbers.
- **Retrieval evaluation:** Compare BM25, similarity search, hybrid search, and Maximal Marginal Relevance (MMR).
- **Streamlit interface:** Ask questions about indexed reports through a chat interface.

## Architecture

```text
Football Technical Reports (PDF)
              |
              v
        PDF Ingestion
              |
              v
      Text Chunking
  (1000 characters, 200 overlap)
              |
              v
       Hugging Face
       Embeddings
              |
              v
        ChromaDB
     Persistent Storage
              |
User Question -> Retriever
              |
              v
     Retrieved Context
              |
              v
       LLM + Prompt
              |
              v
   Answer + Source References
              |
              v
       Streamlit UI
```

## Technology Stack

| Component | Technology |
|---|---|
| Language | Python |
| RAG framework | LangChain |
| LLM provider | Groq |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector database | ChromaDB |
| PDF processing | PyPDFLoader |
| Text splitting | RecursiveCharacterTextSplitter |
| Frontend | Streamlit |
| Environment configuration | python-dotenv |

## Project Structure

```text
rag_system/
├── data/                  # Source PDF reports
├── embedding/
│   └── embeddings.py       # Embedding model
├── evaluations/           # Retrieval evaluation
├── ingestions/
│   ├── pdf_loader.py      # PDF loading
│   └── chunker.py          # Text chunking
├── llmprompt/
│   └── llmprompt.py       # LLM and prompt configuration
├── retrival/
│   └── retrivals.py       # Retrieval configuration
├── vectorstores/
│   └── vectorstore.py     # ChromaDB operations
├── app.py                 # Original application
├── app_persistence.py     # Streamlit chat application
├── ingest.py              # Document ingestion entry point
└── rag_pipleline.py       # RAG pipeline
```

## Installation

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd _LANGCHAIN_LEARN
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
```

Use the environment variable name expected by your LLM configuration. Never commit your actual API key.

### 5. Add source documents

Place the football technical-report PDFs inside:

```text
rag_system/data/
```

Source
