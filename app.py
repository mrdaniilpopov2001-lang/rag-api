import os
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.retrievers import BM25Retriever, EnsembleRetriever

from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

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

    # VECTOR RETRIEVER
    vector_retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 12,
            "fetch_k": 30
        }
    )

    # Получаем документы для BM25
    docs = db.similarity_search("engineering", k=50)

    # BM25 RETRIEVER
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = 6

    # HYBRID RETRIEVER
    retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.4, 0.6]
    )

    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        openai_api_key=OPENAI_API_KEY
    )

    prompt = ChatPromptTemplate.from_template(
        """
You are an engineering assistant working with technical standards.

Use ONLY the provided context to answer the question.

If the answer is not in the context, say:
"Not found in the provided standard excerpt."

Context:
{context}

Question:
{input}
"""
    )

    document_chain = create_stuff_documents_chain(
        llm,
        prompt
    )

    qa_chain = create_retrieval_chain(
        retriever,
        document_chain
    )

    result = qa_chain.invoke({
        "input": question.query
    })

    return {
        "answer": result["answer"]
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

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=500
    )

    splits = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    Chroma.from_documents(
        splits,
        embeddings,
        persist_directory=DB_DIR
    )

    return {"status": "PDF uploaded and indexed successfully"}