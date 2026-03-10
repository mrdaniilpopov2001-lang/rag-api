import json
import os
import re
from typing import List, Dict, Any

from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from rank_bm25 import BM25Okapi

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyMuPDFLoader

app = FastAPI()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set")

DATA_DIR = "/data"
FILES_DIR = os.path.join(DATA_DIR, "files")
DB_DIR = os.path.join(DATA_DIR, "db")
PAGES_JSON = os.path.join(DATA_DIR, "pages.json")


class Question(BaseModel):
    query: str


@app.get("/")
def root():
    return {"status": "ATLAS-DAEDALUS RAG API running"}


# ===============================
# HELPERS
# ===============================

def normalize_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_for_bm25(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\.\-\+\(\)/ ]", " ", text)
    return text.split()


def load_pages_corpus() -> List[Dict[str, Any]]:
    if not os.path.exists(PAGES_JSON):
        return []

    with open(PAGES_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


def save_pages_corpus(corpus: List[Dict[str, Any]]) -> None:
    with open(PAGES_JSON, "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False, indent=2)


def unique_hits(hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    result = []

    for hit in hits:
        key = (
            hit.get("source", ""),
            hit.get("page", ""),
            hit.get("content", "")[:500]
        )
        if key not in seen:
            seen.add(key)
            result.append(hit)

    return result


def bm25_search(query: str, corpus: List[Dict[str, Any]], k: int = 10) -> List[Dict[str, Any]]:
    if not corpus:
        return []

    tokenized_corpus = [tokenize_for_bm25(item["content"]) for item in corpus]
    bm25 = BM25Okapi(tokenized_corpus)

    query_tokens = tokenize_for_bm25(query)
    scores = bm25.get_scores(query_tokens)

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )[:k]

    results = []
    for idx in ranked_indices:
        item = corpus[idx]
        results.append(item)

    return results


def expand_query(query: str, llm: ChatOpenAI) -> List[str]:
    prompt = f"""
Rewrite the engineering question into 4 alternative search queries
that could appear in a technical standard.

Rules:
- Keep technical meaning unchanged.
- Prefer wording used in standards.
- Return only the queries, one per line.
- Do not number them.

Question:
{query}
"""
    try:
        response = llm.invoke(prompt)
        alternatives = [
            line.strip()
            for line in response.content.split("\n")
            if line.strip()
        ]
        return [query] + alternatives[:4]
    except Exception:
        return [query]


def build_context(hits: List[Dict[str, Any]], max_docs: int = 8) -> str:
    selected = hits[:max_docs]
    parts = []

    for i, hit in enumerate(selected, start=1):
        parts.append(
            f"[Document {i} | page={hit['page']} | source={hit['source']}]\n{hit['content']}"
        )

    return "\n\n".join(parts)


def vector_search(db: Chroma, query: str, k: int = 10, fetch_k: int = 40) -> List[Dict[str, Any]]:
    docs = db.max_marginal_relevance_search(
        query,
        k=k,
        fetch_k=fetch_k
    )

    results = []
    for doc in docs:
        results.append(
            {
                "page": doc.metadata.get("page", "unknown"),
                "source": doc.metadata.get("source", "unknown"),
                "content": doc.page_content
            }
        )

    return results


# ===============================
# ASK
# ===============================

@app.post("/ask")
def ask(question: Question):
    if not os.path.exists(DB_DIR):
        return {"error": "Vector database not initialized yet. Upload a PDF first."}

    corpus = load_pages_corpus()
    if not corpus:
        return {"error": "Page corpus not initialized yet. Upload a PDF first."}

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    db = Chroma(
        persist_directory=DB_DIR,
        embedding_function=embeddings
    )

    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        openai_api_key=OPENAI_API_KEY
    )

    expanded_queries = expand_query(question.query, llm)

    all_hits: List[Dict[str, Any]] = []

    for q in expanded_queries:
        try:
            all_hits.extend(vector_search(db, q, k=10, fetch_k=40))
        except Exception:
            pass

        try:
            all_hits.extend(bm25_search(q, corpus, k=10))
        except Exception:
            pass

    merged_hits = unique_hits(all_hits)

    if not merged_hits:
        return {
            "answer": "No documents retrieved from database.",
            "sources": [],
            "retrieved_chunks_preview": []
        }

    context = build_context(merged_hits, max_docs=8)

    prompt = f"""
You are an engineering assistant working with technical standards.

Use ONLY the provided context to answer the question.
Do not invent formulas, values, limits, section numbers, or requirements.

If the answer is explicitly present in the context:
- answer clearly
- include the exact value/formula if available
- include the section or page if visible in the context

If the answer is NOT explicitly supported by the context, say exactly:
Not found in the provided standard excerpt.

Question:
{question.query}

Context:
{context}
"""

    response = llm.invoke(prompt)
    answer_text = response.content.strip()

    return {
        "answer": answer_text,
        "sources": [
            {
                "page": hit["page"],
                "source": hit["source"]
            }
            for hit in merged_hits[:8]
        ],
        "retrieved_chunks_preview": [
            hit["content"][:700] for hit in merged_hits[:8]
        ]
    }


# ===============================
# UPLOAD
# ===============================

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    os.makedirs(FILES_DIR, exist_ok=True)
    os.makedirs(DB_DIR, exist_ok=True)

    file_path = os.path.join(FILES_DIR, file.filename)

    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    if os.path.getsize(file_path) == 0:
        return {"error": "Uploaded file is empty"}

    loader = PyMuPDFLoader(file_path)
    documents = loader.load()

    if not documents:
        return {"error": "PDF parsing failed or document is empty"}

    cleaned_documents = []
    pages_corpus = []

    for doc in documents:
        cleaned_text = normalize_text(doc.page_content)

        if not cleaned_text:
            continue

        metadata = {
            "source": doc.metadata.get("source", file_path),
            "page": doc.metadata.get("page", "unknown")
        }

        doc.page_content = cleaned_text
        doc.metadata = metadata
        cleaned_documents.append(doc)

        pages_corpus.append(
            {
                "page": metadata["page"],
                "source": metadata["source"],
                "content": cleaned_text
            }
        )

    if not cleaned_documents:
        return {"error": "No valid text extracted from PDF"}

    save_pages_corpus(pages_corpus)

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    Chroma.from_documents(
        cleaned_documents,
        embeddings,
        persist_directory=DB_DIR
    )

    return {
        "status": "PDF uploaded and indexed successfully",
        "pages_indexed": len(cleaned_documents),
        "file": file.filename
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)