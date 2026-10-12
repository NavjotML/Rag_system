from rag_system.embedding.embeddings import embed_documents
from rag_system.vectorstores.vectorstore import load_vectorstore

embeddings = embed_documents()

vectorstore = load_vectorstore(embeddings)

print("Chroma loaded successfully!")

results = vectorstore.similarity_search(
    "How did Juventus carve space against Benfica?",
    k=3
)

for i, doc in enumerate(results, start=1):
    print(f"\n--- Result {i} ---")
    print("Page:", doc.metadata.get("page"))
    print(doc.page_content[:300])