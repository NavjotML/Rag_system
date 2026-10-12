from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever



def create_retrievers(vectorstore):
    vector_retriever=vectorstore.as_retriever(search_type='mmr',search_kwargs={"k":3})

    # bm25_retriever=BM25Retriever.from_documents(chunks, k=5)

    # Hybrid_retriever=EnsembleRetriever(retrievers=[vector_retriever,bm25_retriever],weights=[0.5,0.5])

    return vector_retriever
