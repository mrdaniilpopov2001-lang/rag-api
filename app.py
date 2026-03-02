import os
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

app = FastAPI()

# ===============================
#           ENVIRONMENT
# ===============================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set")

# ===============================
#      PERSISTENT STORAGE
# ===============================

DATA_DIR = "/data"
FILES_DIR = os.path.join(DATA_DIR, "files")
DB_DIR = os.path.join(DATA_DIR, "db")


class Question(BaseModel):
    query: str


@app.get("/")
def root():
    return {"status": "THE ATLAS-DAEDALUS PROJECT running"}


# ===============================
#         STRICT PROMPT
# ===============================

STRICT_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a senior rotating equipment engineer answering strictly from API standard text.

Rules:
- Answer ONLY using the provided context.
- If exact value exists, state it precisely.
- Always include:
  - Section number
  - Exact quotation
- If not found in context, respond: "NOT FOUND IN PROVIDED EXCERPT."

Context:
{context}

Question:
{question}

Answer:
"""
)

# ===============================
#            ASK
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

    retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 25,
            "fetch_k": 60
        }
    )

    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        openai_api_key=OPENAI_API_KEY
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="map_reduce",
        retriever=retriever,
        chain_type_kwargs={
            "prompt": STRICT_PROMPT
        },
        return_source_documents=True
    )

    result = qa.invoke({"query": question.query})

    return {
        "answer": result["result"],
        "sources": [
            {
                "page": doc.metadata.get("page"),
                "file": doc.metadata.get("source")
            }
            for doc in result["source_documents"]
        ]
    }


# ===============================
#           UPLOAD
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
        chunk_overlap=300
    )

    splits = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    Chroma.from_documents(
        splits,
        embeddings,
        persist_directory=DB_DIR
    )

    return {"status": "PDF uploaded and indexed successfully"}