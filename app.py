import os
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

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


@app.post("/ask")
def ask(question: Question):
    if not os.path.exists(DB_DIR):
        return {"error": "Vector database not initialized yet. Upload a PDF first."}

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    db = Chroma(
        persist_directory=DB_DIR,
        embedding_function=embeddings
    )

    docs = db.similarity_search(question.query, k=6)

    if not docs:
        return {
            "answer": "No documents retrieved from vector database.",
            "sources": [],
            "retrieved_chunks_preview": []
        }

    context_parts = []
    sources = []

    for i, doc in enumerate(docs, start=1):
        page = doc.metadata.get("page", "unknown")
        source = doc.metadata.get("source", "unknown")

        context_parts.append(
            f"[Document {i} | page={page} | source={source}]\n{doc.page_content}"
        )

        sources.append({
            "page": page,
            "source": source
        })

    context = "\n\n".join(context_parts)

    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        openai_api_key=OPENAI_API_KEY
    )

    prompt = f"""
You are an engineering assistant working with technical standards.

Use ONLY the provided context to answer the question.
If the answer is not explicitly present in the context, say exactly:
Not found in the provided standard excerpt.

Question:
{question.query}

Context:
{context}
"""

    response = llm.invoke(prompt)

    return {
        "answer": response.content,
        "sources": sources,
        "retrieved_chunks_preview": [doc.page_content[:500] for doc in docs]
    }


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