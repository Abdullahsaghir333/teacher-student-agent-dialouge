# rag/retriever.py

class RAGRetriever:

    def __init__(self, vectorstore):
        self.vectorstore = vectorstore

    def retrieve(self, query, k=2):
        docs = self.vectorstore.similarity_search(query, k=k)
        return "\n\n".join([d.page_content for d in docs])
