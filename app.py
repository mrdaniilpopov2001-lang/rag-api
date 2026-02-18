import os
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA

app = FastAPI()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class Question(BaseModel):
    query: str

@app.get("/")
def root():
    return {"status": "RAG API running"}

@app.post("/ask")
def ask(question: Question):

    if not os.path.exists("db"):
        return {"error": "Vector database not initialized yet."}

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    db = Chroma(
        persist_directory="db",
        embedding_function=embeddings
    )

    retriever = db.as_retriever(search_kwargs={"k": 5})

    llm = ChatOpenAI(
        model="gpt-4o",
        openai_api_key=OPENAI_API_KEY
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    result = qa(question.query)

    return {
        "answer": result["result"],
        "sources": [doc.metadata for doc in result["source_documents"]]
    }
