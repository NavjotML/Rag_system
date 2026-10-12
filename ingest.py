from pathlib import Path

from rag_system.ingestions.pdf_loader import load_pdf
from rag_system.ingestions.chunker import split_documents
from rag_system.embedding.embeddings import embed_documents
from rag_system.vectorstores.vectorstore import create_vectorstore


DATA_DIR = Path("rag_system/data")


def ingest_pdfs():

    all_documents = []

    pdf_files = list(DATA_DIR.glob("*.pdf"))

    print(f"Found {len(pdf_files)} PDF(s).")

    for pdf_path in pdf_files:

        print(f"\nLoading: {pdf_path.name}")

        documents = load_pdf(str(pdf_path))

        print(f"Pages loaded: {len(documents)}")

        chunks = split_documents(documents)

        print(f"Chunks created: {len(chunks)}")

        all_documents.extend(chunks)

    if not all_documents:
        print("No PDF files found.")
        return

    print(f"\nTotal chunks: {len(all_documents)}")

    embeddings = embed_documents()

    print("Creating persistent Chroma database...")

    vectorstore = create_vectorstore(
        all_documents,
        embeddings
    )

    print("\nIngestion completed successfully.")
    print("Vector database: rag_system/chroma_db")


if __name__ == "__main__":
    ingest_pdfs()