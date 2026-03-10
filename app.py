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
        return {"answer": "No documents retrieved from vector database."}

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
        "retrieved_chunks_preview": [
            doc.page_content[:500] for doc in docs
        ]
    }