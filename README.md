# DocuMind AI

DocuMind AI is a Retrieval-Augmented Generation (RAG) based PDF Question Answering system built with FastAPI, LangChain, ChromaDB, Hugging Face embeddings, and Google Gemini. Users can upload PDF documents, ask questions about their content, and receive answers grounded in the uploaded document.

## Features

* PDF upload endpoint (`POST /upload`)
* Question answering endpoint (`POST /ask`)
* Automatic document chunking
* Semantic search using ChromaDB
* Embeddings generated with `sentence-transformers/all-MiniLM-L6-v2`
* Persistent vector storage
* Hallucination guard for low-confidence retrieval
* FastAPI-based REST API

## Tech Stack

* Python
* FastAPI
* LangChain
* ChromaDB
* Hugging Face Embeddings
* Google Gemini
* PyPDFLoader

## Setup Instructions

### 1. Create Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Create a `.env` file:

```env
GOOGLE_API_KEY=<YOUR_API_KEY>
```

### 4. Run the Application

```bash
uvicorn app:app --reload
```

The application will start at:

```text
http://127.0.0.1:8000
```

## API Usage

### Upload a PDF

```bash
curl -X POST http://127.0.0.1:8000/upload \
-F "file=@/path/to/document.pdf"
```

### Ask a Question

```bash
curl -X POST http://127.0.0.1:8000/ask \
-H "Content-Type: application/json" \
-d '{"question":"What is this document about?"}'
```

## Submission Note

### Why I Selected This Chunk Size and Overlap

I selected a chunk size of 700 characters with a 100-character overlap. A chunk size of 700 preserves enough surrounding context for meaningful semantic retrieval while remaining small enough to maintain retrieval precision. The 100-character overlap helps retain information that spans chunk boundaries, reducing the possibility of losing important context when text is split into multiple chunks.

### How the Implementation Mitigates Hallucinations

The application uses a Retrieval-Augmented Generation (RAG) approach. Before generating an answer, relevant chunks are retrieved from ChromaDB using semantic similarity search. The answer is generated only from the retrieved context rather than relying solely on the language model's internal knowledge. A relevance threshold is also applied. If retrieval confidence is too low, the system returns: "I could not find relevant information in the document." This prevents the model from generating unsupported or fabricated answers.

### Supporting Multiple Users

To support multiple users, the application would require authentication and user-specific document management. Each uploaded document should be associated with a unique user ID, and embeddings should be stored in separate collections or filtered using metadata. The retrieval pipeline would search only the authenticated user's documents. For a production environment, a relational database such as PostgreSQL could be added to manage users, document metadata, permissions, and audit logs securely.
