import json
import os
import re
from typing import List, Dict, Any, Tuple

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


def doc_key(hit: Dict[str, Any]) -> Tuple[str, Any, str]:
    return (
        hit.get("source", ""),
        hit.get("page", ""),
        hit.get("content", "")[:500]
    )


def unique_hits(hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    result = []

    for hit in hits:
        key = doc_key(hit)
        if key not in seen:
            seen.add(key)
            result.append(hit)

    return result


def build_context(hits: List[Dict[str, Any]], max_docs: int = 10) -> str:
    selected = hits[:max_docs]
    parts = []

    for i, hit in enumerate(selected, start=1):
        parts.append(
            f"[Document {i} | page={hit['page']} | source={hit['source']}]\n{hit['content']}"
        )

    return "\n\n".join(parts)


# ===============================
# RETRIEVAL
# ===============================

def bm25_search(query: str, corpus: List[Dict[str, Any]], k: int = 12) -> List[Dict[str, Any]]:
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


def vector_search(db: Chroma, query: str, k: int = 12, fetch_k: int = 50) -> List[Dict[str, Any]]:
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


def rrf_fuse(result_lists: List[List[Dict[str, Any]]], k: int = 60) -> List[Dict[str, Any]]:
    score_map: Dict[Tuple[str, Any, str], float] = {}
    obj_map: Dict[Tuple[str, Any, str], Dict[str, Any]] = {}

    for result_list in result_lists:
        for rank, hit in enumerate(result_list, start=1):
            key = doc_key(hit)
            score_map[key] = score_map.get(key, 0.0) + 1.0 / (k + rank)
            obj_map[key] = hit

    ranked_keys = sorted(score_map.keys(), key=lambda x: score_map[x], reverse=True)
    return [obj_map[key] for key in ranked_keys]


# ===============================
# QUERY EXPANSION
# ===============================

def heuristic_expansions(query: str) -> List[str]:
    q = query.lower()
    expansions = [query]

    if "external thrust" in q and "gear coupling" in q:
        expansions.extend([
            "API 617 external thrust force gear coupling formula",
            "API 617 gear coupling external force equation",
            "API 617 annex k external forces and moments gear couplings",
            "external thrust force allowable gear coupling API 617",
            "formula external thrust gear couplings rated power speed shaft diameter"
        ])

    if "shaft vibration" in q and "mechanical running test" in q:
        expansions.extend([
            "API 617 mechanical running test shaft vibration limit",
            "API 617 allowable shaft vibration mechanical running test",
            "API 617 peak to peak amplitude unfiltered shaft vibration",
            "API 617 equation 13 shaft vibration",
            "API 617 25.4 um 1.0 mil shaft vibration",
            "API 617 6.8.9 shaft vibration mechanical running test",
            "mechanical running test maximum allowable shaft vibration peak to peak"
        ])

    return list(dict.fromkeys(expansions))


def llm_expand_query(query: str, llm: ChatOpenAI) -> List[str]:
    prompt = f"""
Rewrite the engineering question into 5 alternative search queries
that could appear in a technical standard.

Rules:
- Keep the meaning unchanged.
- Prefer standard wording.
- Include likely keyword-heavy phrasing.
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
        return alternatives[:5]
    except Exception:
        return []


def build_query_set(query: str, llm: ChatOpenAI) -> List[str]:
    queries = []
    queries.extend(heuristic_expansions(query))
    queries.extend(llm_expand_query(query, llm))
    return list(dict.fromkeys([q.strip() for q in queries if q.strip()]))


# ===============================
# ANSWER GENERATION
# ===============================

def answer_from_context(question: str, hits: List[Dict[str, Any]], llm: ChatOpenAI) -> str:
    context = build_context(hits, max_docs=10)

    prompt = f"""
You are an engineering assistant working with technical standards.

Use ONLY the provided context to answer the question.
Do not invent formulas, values, limits, section numbers, or requirements.

If the answer is explicitly present in the context:
- answer clearly
- include the exact value or formula if available
- include the page number if it is visible in the context

If the answer is NOT explicitly supported by the context, say exactly:
Not found in the provided standard excerpt.

Question:
{question}

Context:
{context}
"""
    response = llm.invoke(prompt)
    return response.content.strip()


def second_pass_queries(question: str, llm: ChatOpenAI) -> List[str]:
    prompt = f"""
Generate up to 6 very short keyword-heavy retrieval queries
for finding the exact answer in a technical standard.

Rules:
- Focus on section-style wording, equations, limits, and exact terms.
- Return only the queries, one per line.
- Do not explain anything.

Question:
{question}
"""
    try:
        response = llm.invoke(prompt)
        queries = [line.strip() for line in response.content.split("\n") if line.strip()]
        return queries[:6]
    except Exception:
        return []


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

    query_set = build_query_set(question.query, llm)

    vector_result_lists: List[List[Dict[str, Any]]] = []
    bm25_result_lists: List[List[Dict[str, Any]]] = []

    for q in query_set:
        try:
            vector_result_lists.append(vector_search(db, q, k=12, fetch_k=50))
        except Exception:
            pass

        try:
            bm25_result_lists.append(bm25_search(q, corpus, k=12))
        except Exception:
            pass

    fused_hits = rrf_fuse(vector_result_lists + bm25_result_lists)
    fused_hits = unique_hits(fused_hits)

    if not fused_hits:
        return {
            "answer": "No documents retrieved from database.",
            "sources": [],
            "retrieved_chunks_preview": [],
            "queries_used": query_set
        }

    answer_text = answer_from_context(question.query, fused_hits, llm)

    # Second pass if the first pass did not find the answer
    if answer_text == "Not found in the provided standard excerpt.":
        extra_queries = second_pass_queries(question.query, llm)
        extra_lists: List[List[Dict[str, Any]]] = []

        for q in extra_queries:
            try:
                extra_lists.append(vector_search(db, q, k=12, fetch_k=50))
            except Exception:
                pass

            try:
                extra_lists.append(bm25_search(q, corpus, k=12))
            except Exception:
                pass

        if extra_lists:
            second_pass_hits = rrf_fuse([fused_hits] + extra_lists)
            second_pass_hits = unique_hits(second_pass_hits)
            second_answer = answer_from_context(question.query, second_pass_hits, llm)

            if second_answer != "Not found in the provided standard excerpt.":
                fused_hits = second_pass_hits
                answer_text = second_answer
                query_set = list(dict.fromkeys(query_set + extra_queries))

    return {
        "answer": answer_text,
        "sources": [
            {
                "page": hit["page"],
                "source": hit["source"]
            }
            for hit in fused_hits[:10]
        ],
        "retrieved_chunks_preview": [
            hit["content"][:900] for hit in fused_hits[:10]
        ],
        "queries_used": query_set
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