from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
import os
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allow frontend (HTML/JS) to call API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # you can restrict to ["http://127.0.0.1:5500"] if serving HTML from VSCode/LiveServer
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

qa_chain = None  # global but initialized later

class QueryRequest(BaseModel):
    query: str

@app.on_event("startup")
def load_pipeline():
    global qa_chain
    print("🚀 Initializing vectorstore + LLM...")
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma(
        collection_name="law",
        embedding_function=embeddings,
        persist_directory="./chroma_db"
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    api_key = os.getenv("GROQ_API_KEY")
    llm = ChatGroq(
        api_key="",
        model_name="llama-3.1-8b-instant",  # smaller model
        temperature=0,
        max_tokens=512,
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True
    )
    print("✅ Pipeline ready!")

@app.post("/ask")
async def ask_question(req: QueryRequest):
    global qa_chain
    if qa_chain is None:
        return {"error": "Pipeline not initialized yet"}
    
    result = qa_chain.invoke({"query": req.query})
    sources = [
        {"page": doc.metadata.get("page"), "source": doc.metadata.get("source")}
        for doc in result["source_documents"]
    ]
    return {"answer": result["result"], "sources": sources}


