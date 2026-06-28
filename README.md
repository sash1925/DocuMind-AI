# DocuMind AI

DocuMind AI is a FastAPI application for asking questions about an uploaded PDF. It extracts text from the PDF, chunks the document, creates embeddings with `sentence-transformers/all-MiniLM-L6-v2`, stores them in a persistent ChromaDB index, and returns a grounded answer with the top 3 retrieved chunks.

## Features

- PDF upload endpoint: `POST /upload`
- Question endpoint: `POST /ask`
- Persistent ChromaDB storage in `chroma_db/`
- `all-MiniLM-L6-v2` embeddings through LangChain Hugging Face integration
- A polished browser UI at `/`
- Hallucination guard that returns `I could not find relevant information in the document.` when retrieval confidence is too low

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --reload
```

Open the app at [http://127.0.0.1:9000](http://127.0.0.1:9000).

## API Usage

Upload a PDF:

```bash
curl -X POST http://127.0.0.1:8000/upload \
  -F "file=@/path/to/document.pdf"
```

Ask a question:

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"What is this document about?"}'
```

Example response:

```json
{
  "answer": "The answer generated from retrieved document context.",
  "chunks": [
    {
      "content": "Retrieved text chunk...",
      "page": 0,
      "score": 0.72
    }
  ]
}
```

## Submission Note

I selected a chunk size of 700 characters with 100 characters of overlap because PDF text often contains dense paragraphs, headings, and page-boundary artifacts. A 700-character chunk is large enough to preserve local meaning for factual questions, while still small enough for precise retrieval. The 100-character overlap helps retain context when an answer spans the end of one chunk and the beginning of another, reducing the chance that related sentences are separated during retrieval.

The implementation mitigates hallucinations by grounding every answer in the top retrieved ChromaDB chunks. The `/ask` endpoint returns those top 3 chunks alongside the answer so the user can inspect the evidence. It also applies a minimum relevance score threshold. If the highest retrieved chunk is not relevant enough, the API returns `I could not find relevant information in the document.` instead of inventing an answer. The answer generator is extractive and uses sentences from retrieved context, so it avoids adding unsupported outside knowledge.

To support multiple users, the app would need document ownership and separate vector collections or metadata filters per user and document. Uploads should be stored under user-specific paths, and Chroma records should include `user_id`, `document_id`, filename, and page metadata. Authentication would be required before upload or question requests. The `/ask` endpoint would then search only the authenticated user’s selected document or document set. For production, background indexing, file size limits, rate limits, and cleanup jobs should also be added.

## GitHub Repository

This local folder is not currently initialized as a git repository. After creating a GitHub repository, run:

```bash
git init
git add app.py README.md requirements.txt .gitignore
git commit -m "Build PDF question answering API"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/DocuMind-AI.git
git push -u origin main
```
