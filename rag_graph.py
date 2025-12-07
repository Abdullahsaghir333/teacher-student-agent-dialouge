# rag_graph.py ----------------------------------------------------

from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


# -------------------------------
# Graph State Definition
# -------------------------------
class RAGState(TypedDict):
    query: str
    context: str


# -------------------------------
# Load FAISS + Embeddings
# -------------------------------
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

try:
    vectorstore = FAISS.load_local(
        "faiss_index_fast",
        embedding,
        allow_dangerous_deserialization=True
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
except:
    retriever = None
    print("⚠️ WARNING: FAISS index missing → RAG disabled")


# -------------------------------
# Retrieval Node
# -------------------------------
def retrieve(state: RAGState):
    query = state["query"]

    if retriever:
        docs = retriever.invoke(query)
        context = "\n".join([d.page_content for d in docs])
    else:
        context = ""

    return {"query": query, "context": context}


# -------------------------------
# Build Retrieval LangGraph
# -------------------------------
workflow = StateGraph(RAGState)
workflow.add_node("retrieve", retrieve)
workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", END)

rag_graph = workflow.compile()


# -------------------------------
# External API
# -------------------------------
def run_rag(query: str):
    """Call this from TeacherAgent to retrieve context."""
    return rag_graph.invoke({"query": query})
