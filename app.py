import os
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever

app = FastAPI()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set")

DATA_DIR = "/data"
FILES_DIR = os.path.join(DATA_DIR, "files")
DB_DIR = os.path.join(DATA_DIR, "db")


class Question(BaseModel):
    query: str


@app.get("/")
def root():
    return {"status": "ATLAS-DAEDALUS RAG API running"}


# ===============================
# QUERY EXPANSION
# ===============================

def expand_query(query: str, llm):
    prompt = f"""
Rewrite the engineering question into 4 alternative search queries
that could appear in a technical standard.

Return only the queries, one per line.

Question:
{query}
"""
    response = llm.invoke(prompt)

    alternatives = [
        q.strip()
        for q in response.content.split("\n")
        if q.strip()
    ]

    return [query] + alternatives


# ===============================
# HELPERS
# ===============================

def unique_docs(docs):
    seen = set()
    result = []

    for doc in docs:
        metadata_items = tuple(sorted(doc.metadata.items())) if doc.metadata else ()
        key = (doc.page_content[:500], metadata_items)

        if key not in seen:
            seen.add(key)
            result.append(doc)

    return result


def build_context(docs, max_docs=8):
    selected = docs[:max_docs]
    parts = []

    for i, doc in enumerate(selected, start=1):
        page = doc.metadata.get("page", "unknown")
        source = doc.metadata.get("source", "unknown")
        parts.append(
            f"[Document {i} | page={page} | source={source}]\n{doc.page_content}"
        )

    return "\n\n".join(parts)


# ===============================
# ASK
# ===============================

@app.post("/ask")
def ask(question: Question):

    if not os.path.exists(DB_DIR):
        return {"error": "Vector database not initialized yet. Upload a PDF first."}

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    db = Chroma(
        persist_directory=DB_DIR,
        embedding_function=embeddings
    )

    vector_retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 12,
            "fetch_k": 30
        }
    )

    # Набор документов для BM25
    bm25_docs = db.similarity_search(
        "engineering standard compressor vibration thrust coupling shaft test limit equation section",
        k=80
    )

    bm25_retriever = BM25Retriever.from_documents(bm25_docs)
    bm25_retriever.k = 8

    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        openai_api_key=OPENAI_API_KEY
    )

    queries = expand_query(question.query, llm)
    answers = []

    for q in queries:
        vector_docs = vector_retriever.invoke(q)
        keyword_docs = bm25_retriever.invoke(q)

        merged_docs = unique_docs(vector_docs + keyword_docs)

        if not merged_docs:
            continue

        context = build_context(merged_docs, max_docs=8)

        prompt = f"""
You are an engineering assistant working with technical standards.

Use ONLY the provided context to answer the question.
Do not invent values, formulas, limits, or section numbers.
If the answer is not explicitly supported by the context, say exactly:
Not found in the provided standard excerpt.

Question:
{q}

Context:
{context}
"""

        response = llm.invoke(prompt)
        answer_text = response.content.strip()

        if answer_text != "Not found in the provided standard excerpt.":
            answers.append(answer_text)

    if answers:
        return {"answer": answers[0]}

    return {"answer": "Not found in the provided standard excerpt."}


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

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=300,
        separators=[
            "\nSECTION ",
            "\nSection ",
            "\nCHAPTER ",
            "\nChapter ",
            "\n\n",
            "\n",
            ". ",
            " "
        ]
    )

    splits = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    Chroma.from_documents(
        splits,
        embeddings,
        persist_directory=DB_DIR
    )

    return {"status": "PDF uploaded and indexed successfully"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)