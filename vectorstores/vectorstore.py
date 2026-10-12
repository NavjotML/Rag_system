from langchain_community.vectorstores import Chroma

def create_vectorstore(documents,embeddings):
    vectorstore=Chroma.from_documents(documents, embeddings, collection_name="langchain-documents",persist_directory="rag_system/chroma_db")
    return vectorstore

def load_vectorstore(embeddings):
    vectorstore=Chroma(
        collection_name="langchain-documents",
        persist_directory="rag_system/chroma_db",
        embedding_function=embeddings

    )
    return vectorstore