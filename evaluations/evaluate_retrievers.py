from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever


def create_eval_retrievers(vectorstore, chunks):

    similarity_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )

    mmr_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5}
    )

    bm25_retriever = BM25Retriever.from_documents(
        chunks,
        k=5
    )

    hybrid_retriever = EnsembleRetriever(
        retrievers=[
            mmr_retriever,
            bm25_retriever
        ],
        weights=[0.5, 0.5]
    )

    return {
        "similarity": similarity_retriever,
        "mmr": mmr_retriever,
        "bm25": bm25_retriever,
        "hybrid": hybrid_retriever
    }


def hit_at_k(retrieved_pages, relevant_pages, k=5):

    top_k = retrieved_pages[:k]

    return int(
        any(page in relevant_pages for page in top_k)
    )


def mrr(retrieved_pages, relevant_pages):

    for rank, page in enumerate(retrieved_pages, start=1):

        if page in relevant_pages:
            return 1 / rank

    return 0.0

def unique_pages(retrieved_pages):
    seen=set()
    unique=[]

    for page in retrieved_pages:
        if page not in seen:
            seen.add(page)
            unique.append(page)
    return unique
