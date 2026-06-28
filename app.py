from __future__ import annotations

import os
import re
import shutil
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field


APP_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = APP_DIR / "uploads"
CHROMA_DIR = APP_DIR / "chroma_db"
COLLECTION_NAME = "uploaded_pdf"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 700
CHUNK_OVERLAP = 100
MIN_RELEVANCE_SCORE = 0.3
MIN_SUMMARY_SCORE = 0.18
FALLBACK_MESSAGE = "I could not find relevant information in the document."

UPLOAD_DIR.mkdir(exist_ok=True)
CHROMA_DIR.mkdir(exist_ok=True)

embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)

app = FastAPI(title="DocuMind AI", version="1.0.0")


class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=3, examples=["What is the main idea of this document?"])


class ChunkResponse(BaseModel):
    content: str
    page: int | None = None
    score: float
    source: str | None = None


class QuestionResponse(BaseModel):
    answer: str
    chunks: list[ChunkResponse]


def get_vectorstore() -> Chroma:
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def split_sentences(text: str) -> list[str]:
    cleaned = normalize_text(text)
    return [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", cleaned) if sentence.strip()]


def tokenize(text: str) -> set[str]:
    stopwords = {
        "a",
        "about",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "how",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "was",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "with",
        "me",
        "tell",
        "brief",
        "short",
        "give",
        "define",
        "explain",
    }
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9-]+", text.lower())
    return {word for word in words if word not in stopwords and len(word) > 2}


def asks_for_summary(question: str) -> bool:
    summary_words = {"about", "brief", "overview", "summarize", "summary", "describe"}
    lowered = question.lower()
    return any(word in lowered for word in summary_words) or "tell me" in lowered


def compact_chunk_summary(text: str, max_chars: int = 520) -> str:
    cleaned = normalize_text(text)
    sentences = split_sentences(cleaned)
    summary = " ".join(sentences[:3]) if sentences else cleaned
    if len(summary) <= max_chars:
        return summary
    return summary[:max_chars].rsplit(" ", 1)[0] + "..."


def build_grounded_answer(question: str, chunks: list[ChunkResponse]) -> str:
    if not chunks:
        return FALLBACK_MESSAGE

    top_chunk = chunks[0]
    question_terms = tokenize(question)
    top_chunk_terms = tokenize(top_chunk.content)
    source_terms = tokenize(top_chunk.source or "")
    has_text_overlap = bool(question_terms & top_chunk_terms)
    has_source_overlap = bool(question_terms & source_terms)

    if top_chunk.score < MIN_RELEVANCE_SCORE and not (
        asks_for_summary(question)
        and top_chunk.score >= MIN_SUMMARY_SCORE
        and (has_text_overlap or has_source_overlap)
    ):
        return FALLBACK_MESSAGE

    if asks_for_summary(question) and top_chunk.score >= MIN_SUMMARY_SCORE:
        summary = compact_chunk_summary(top_chunk.content)
        if summary:
            return f"Based on the document, it appears to be about: {summary}"

    candidate_sentences: list[tuple[float, str]] = []

    for chunk_index, chunk in enumerate(chunks):
        for sentence in split_sentences(chunk.content):
            sentence_terms = tokenize(sentence)
            if not sentence_terms:
                continue

            overlap = len(question_terms & sentence_terms)
            coverage = overlap / max(len(question_terms), 1)
            score = coverage + (chunk.score * 0.7) - (chunk_index * 0.05)
            if overlap > 0:
                candidate_sentences.append((score, sentence))

    candidate_sentences.sort(key=lambda item: item[0], reverse=True)
    selected: list[str] = []

    for score, sentence in candidate_sentences:
        if score < MIN_RELEVANCE_SCORE and selected:
            continue
        if sentence not in selected:
            selected.append(sentence)
        if len(selected) == 3:
            break

    if not selected:
        return FALLBACK_MESSAGE

    return " ".join(selected)


def distance_to_score(distance: float) -> float:
    return 1 / (1 + max(distance, 0))


def rerank_chunks(question: str, chunks: list[ChunkResponse]) -> list[ChunkResponse]:
    question_terms = tokenize(question)

    def rank(chunk: ChunkResponse) -> tuple[float, float]:
        chunk_terms = tokenize(chunk.content)
        source_terms = tokenize(chunk.source or "")
        overlap = len(question_terms & (chunk_terms | source_terms))
        coverage = overlap / max(len(question_terms), 1)
        return (coverage + chunk.score, chunk.score)

    return sorted(chunks, key=rank, reverse=True)


def clear_previous_index() -> None:
    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)
    CHROMA_DIR.mkdir(exist_ok=True)


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return HTML


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "embedding_model": MODEL_NAME}


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)) -> dict[str, Any]:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a PDF file.")

    safe_filename = Path(file.filename).name
    file_path = UPLOAD_DIR / safe_filename

    with file_path.open("wb") as destination:
        destination.write(await file.read())

    loader = PyPDFLoader(str(file_path))
    documents = loader.load()
    if not documents:
        raise HTTPException(status_code=400, detail="No readable text was found in the PDF.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    clear_previous_index()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
    )

    return {
        "message": "PDF indexed successfully",
        "filename": safe_filename,
        "chunks_created": len(chunks),
        "embedding_model": MODEL_NAME,
    }


@app.post("/ask", response_model=QuestionResponse)
def ask_question(request: QuestionRequest) -> QuestionResponse:
    vectorstore = get_vectorstore()

    try:
        results = vectorstore.similarity_search_with_score(request.question, k=8)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Upload and index a PDF before asking questions.") from exc

    retrieved_chunks = [
        ChunkResponse(
            content=normalize_text(document.page_content),
            page=document.metadata.get("page"),
            score=round(distance_to_score(float(distance)), 4),
            source=Path(document.metadata.get("source", "")).name or None,
        )
        for document, distance in results
    ]

    ranked_chunks = rerank_chunks(request.question, retrieved_chunks)[:3]
    answer = build_grounded_answer(request.question, ranked_chunks)
    return QuestionResponse(answer=answer, chunks=ranked_chunks)


HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>DocuMind AI</title>
  <style>
    :root {
      color-scheme: light;
      --ink: #172126;
      --muted: #5d6b73;
      --line: #d9e3e7;
      --panel: #ffffff;
      --panel-soft: #f5f8f7;
      --teal: #087b83;
      --teal-dark: #055b61;
      --coral: #d85f49;
      --gold: #c58a1a;
      --shadow: 0 24px 70px rgba(18, 34, 39, 0.16);
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      min-height: 100vh;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background:
        linear-gradient(120deg, rgba(8, 123, 131, 0.13), transparent 36%),
        linear-gradient(210deg, rgba(216, 95, 73, 0.14), transparent 42%),
        #eef4f2;
    }

    .shell {
      width: min(1180px, calc(100% - 32px));
      margin: 0 auto;
      padding: 32px 0;
    }

    header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 24px;
      padding: 18px 0 28px;
    }

    .brand {
      display: flex;
      align-items: center;
      gap: 12px;
      font-weight: 800;
      font-size: 1.1rem;
    }

    .mark {
      display: grid;
      place-items: center;
      width: 42px;
      height: 42px;
      border-radius: 8px;
      background: var(--ink);
      color: #fff;
      letter-spacing: 0;
    }

    .status {
      color: var(--muted);
      font-size: 0.92rem;
    }

    .hero {
      display: grid;
      grid-template-columns: 0.95fr 1.4fr;
      gap: 22px;
      align-items: start;
    }

    .intro {
      padding: 18px 0 8px;
    }

    h1 {
      margin: 0;
      max-width: 520px;
      font-size: clamp(2.4rem, 7vw, 5.6rem);
      line-height: 0.92;
      letter-spacing: 0;
    }

    .lead {
      max-width: 460px;
      margin: 22px 0 0;
      color: var(--muted);
      font-size: 1.05rem;
      line-height: 1.65;
    }

    .panel {
      background: rgba(255, 255, 255, 0.88);
      border: 1px solid rgba(217, 227, 231, 0.95);
      border-radius: 8px;
      box-shadow: var(--shadow);
      overflow: hidden;
    }

    .toolbar {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 12px;
      align-items: center;
      padding: 16px;
      border-bottom: 1px solid var(--line);
      background: rgba(245, 248, 247, 0.86);
    }

    .filebox {
      position: relative;
      display: flex;
      align-items: center;
      min-height: 48px;
      padding: 0 14px;
      border: 1px dashed #9fb1b8;
      border-radius: 8px;
      background: #fff;
      color: var(--muted);
      overflow: hidden;
    }

    .filebox input {
      position: absolute;
      inset: 0;
      opacity: 0;
      cursor: pointer;
    }

    button {
      min-height: 48px;
      border: 0;
      border-radius: 8px;
      padding: 0 18px;
      background: var(--teal);
      color: #fff;
      font: inherit;
      font-weight: 750;
      cursor: pointer;
      transition: transform 140ms ease, background 140ms ease, opacity 140ms ease;
    }

    button:hover { background: var(--teal-dark); transform: translateY(-1px); }
    button:disabled { cursor: not-allowed; opacity: 0.58; transform: none; }

    .workspace {
      display: grid;
      grid-template-rows: minmax(360px, 1fr) auto;
      min-height: 640px;
    }

    .conversation {
      display: flex;
      flex-direction: column;
      gap: 14px;
      padding: 18px;
      overflow: auto;
    }

    .empty {
      display: grid;
      place-items: center;
      min-height: 340px;
      text-align: center;
      color: var(--muted);
      border: 1px solid var(--line);
      border-radius: 8px;
      background:
        linear-gradient(135deg, rgba(8, 123, 131, 0.08), transparent),
        #fff;
    }

    .message {
      max-width: 82%;
      padding: 14px 16px;
      border-radius: 8px;
      line-height: 1.55;
      white-space: pre-wrap;
    }

    .message.user {
      align-self: flex-end;
      background: var(--ink);
      color: #fff;
    }

    .message.answer {
      align-self: flex-start;
      background: #fff;
      border: 1px solid var(--line);
    }

    .chunks {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
      margin-top: 12px;
    }

    .chunk {
      padding: 12px;
      border-radius: 8px;
      background: var(--panel-soft);
      border: 1px solid var(--line);
      color: #34434a;
      font-size: 0.86rem;
      max-height: 210px;
      overflow: auto;
    }

    .chunk strong {
      display: block;
      margin-bottom: 8px;
      color: var(--teal-dark);
      font-size: 0.78rem;
      text-transform: uppercase;
      letter-spacing: 0;
    }

    form.askbar {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 12px;
      padding: 16px;
      border-top: 1px solid var(--line);
      background: #fff;
    }

    textarea {
      width: 100%;
      min-height: 52px;
      max-height: 140px;
      resize: vertical;
      border: 1px solid #b8c8cd;
      border-radius: 8px;
      padding: 14px;
      color: var(--ink);
      font: inherit;
      line-height: 1.4;
    }

    textarea:focus, .filebox:focus-within {
      outline: 3px solid rgba(8, 123, 131, 0.18);
      border-color: var(--teal);
    }

    .metrics {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
      margin-top: 26px;
    }

    .metric {
      border-left: 3px solid var(--coral);
      padding: 10px 12px;
      background: rgba(255, 255, 255, 0.65);
    }

    .metric:nth-child(2) { border-color: var(--teal); }
    .metric:nth-child(3) { border-color: var(--gold); }
    .metric b { display: block; font-size: 1.1rem; }
    .metric span { color: var(--muted); font-size: 0.82rem; }

    .toast {
      min-height: 22px;
      margin-top: 14px;
      color: var(--teal-dark);
      font-weight: 700;
    }

    @media (max-width: 920px) {
      .hero { grid-template-columns: 1fr; }
      h1 { max-width: 760px; }
      .lead { max-width: 680px; }
      .message { max-width: 100%; }
    }

    @media (max-width: 680px) {
      .shell { width: min(100% - 20px, 1180px); padding: 14px 0; }
      header, .toolbar, form.askbar { grid-template-columns: 1fr; }
      header { align-items: flex-start; }
      .chunks, .metrics { grid-template-columns: 1fr; }
      .workspace { min-height: 560px; }
      button { width: 100%; }
    }
  </style>
</head>
<body>
  <main class="shell">
    <header>
      <div class="brand"><div class="mark">DM</div><span>DocuMind AI</span></div>
      <div class="status" id="status">FastAPI + ChromaDB + MiniLM embeddings</div>
    </header>

    <section class="hero">
      <div class="intro">
        <h1>Ask sharper questions of your PDFs.</h1>
        <p class="lead">Upload one document, build a persistent vector index, then ask grounded questions with the top retrieved chunks shown beside every answer.</p>
        <div class="metrics">
          <div class="metric"><b>700</b><span>chunk size</span></div>
          <div class="metric"><b>100</b><span>overlap</span></div>
          <div class="metric"><b>Top 3</b><span>retrieved chunks</span></div>
        </div>
        <div class="toast" id="toast"></div>
      </div>

      <section class="panel" aria-label="PDF question answering workspace">
        <div class="toolbar">
          <label class="filebox">
            <span id="fileLabel">Choose a PDF to index</span>
            <input id="pdfInput" type="file" accept="application/pdf" />
          </label>
          <button id="uploadBtn" type="button">Upload PDF</button>
        </div>

        <div class="workspace">
          <div class="conversation" id="conversation">
            <div class="empty">Upload a PDF, then ask a question about its contents.</div>
          </div>

          <form class="askbar" id="askForm">
            <textarea id="questionInput" placeholder="Ask about the uploaded document" required></textarea>
            <button id="askBtn" type="submit">Ask</button>
          </form>
        </div>
      </section>
    </section>
  </main>

  <script>
    const pdfInput = document.querySelector("#pdfInput");
    const fileLabel = document.querySelector("#fileLabel");
    const uploadBtn = document.querySelector("#uploadBtn");
    const askForm = document.querySelector("#askForm");
    const askBtn = document.querySelector("#askBtn");
    const questionInput = document.querySelector("#questionInput");
    const conversation = document.querySelector("#conversation");
    const toast = document.querySelector("#toast");

    let indexed = false;

    function setToast(message, isError = false) {
      toast.textContent = message;
      toast.style.color = isError ? "#b33d2c" : "#055b61";
    }

    function clearEmptyState() {
      const empty = conversation.querySelector(".empty");
      if (empty) empty.remove();
    }

    function addMessage(text, type, chunks = []) {
      clearEmptyState();
      const node = document.createElement("article");
      node.className = `message ${type}`;
      const body = document.createElement("div");
      body.textContent = text;
      node.appendChild(body);

      if (chunks.length) {
        const chunkGrid = document.createElement("div");
        chunkGrid.className = "chunks";
        chunks.forEach((chunk, index) => {
          const card = document.createElement("div");
          card.className = "chunk";
          const page = Number.isInteger(chunk.page) ? `Page ${chunk.page + 1}` : "Page unknown";
          card.innerHTML = `<strong>Chunk ${index + 1} · ${page} · score ${chunk.score}</strong>`;
          const text = document.createElement("span");
          text.textContent = chunk.content;
          card.appendChild(text);
          chunkGrid.appendChild(card);
        });
        node.appendChild(chunkGrid);
      }

      conversation.appendChild(node);
      conversation.scrollTop = conversation.scrollHeight;
    }

    pdfInput.addEventListener("change", () => {
      fileLabel.textContent = pdfInput.files[0]?.name || "Choose a PDF to index";
      indexed = false;
      if (pdfInput.files.length) setToast("PDF selected. Uploading will build a fresh index.");
    });

    async function uploadCurrentPdf() {
      if (!pdfInput.files.length) {
        setToast("Select a PDF first.", true);
        return false;
      }

      const formData = new FormData();
      formData.append("file", pdfInput.files[0]);
      uploadBtn.disabled = true;
      setToast("Indexing PDF...");

      try {
        const response = await fetch("/upload", { method: "POST", body: formData });
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || "Upload failed.");
        indexed = true;
        setToast(`${data.filename} indexed with ${data.chunks_created} chunks.`);
        return true;
      } catch (error) {
        indexed = false;
        setToast(error.message, true);
        return false;
      } finally {
        uploadBtn.disabled = false;
      }
    }

    uploadBtn.addEventListener("click", async () => {
      await uploadCurrentPdf();
    });

    askForm.addEventListener("submit", async (event) => {
      event.preventDefault();
      const question = questionInput.value.trim();
      if (!question) return;
      if (!indexed && pdfInput.files.length) {
        const uploaded = await uploadCurrentPdf();
        if (!uploaded) return;
      }
      if (!indexed) {
        setToast("Upload a PDF before asking a question.", true);
        return;
      }

      addMessage(question, "user");
      questionInput.value = "";
      askBtn.disabled = true;

      try {
        const response = await fetch("/ask", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ question })
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || "Question failed.");
        addMessage(data.answer, "answer", data.chunks);
      } catch (error) {
        addMessage(error.message, "answer");
      } finally {
        askBtn.disabled = false;
        questionInput.focus();
      }
    });
  </script>
</body>
</html>
"""
