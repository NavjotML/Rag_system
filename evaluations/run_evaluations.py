from rag_system.ingestions.pdf_loader import load_pdf
from rag_system.evaluations.eval_questions import evaluation_questions
from rag_system.ingestions.chunker import split_documents
from rag_system.embedding.embeddings import embed_documents
from rag_system.vectorstores.vectorstore import create_vectorstore
from rag_system.evaluations.evaluate_retrievers import (
    create_eval_retrievers,
    hit_at_k,
    mrr,
    unique_pages
)


PDF_PATH = "rag_system/data/UCL_Technical-Report_2025-26_DIGITAL.pdf"


# 1. Load PDF
documents = load_pdf(PDF_PATH)

# 2. Create chunks
chunks = split_documents(documents)

# 3. Create embeddings
embeddings = embed_documents()

# 4. Create vectorstore
vectorstore = create_vectorstore(
    chunks,
    embeddings
)

# 5. Create all evaluation retrievers
retrievers = create_eval_retrievers(
    vectorstore,
    chunks
)


# 6. Test first question
for question_data in evaluation_questions:

    question = question_data["question"]
    relevant_pages = question_data["relevant_pages"]

    print("\nQUESTION:")
    print(question)

    print("\nGROUND TRUTH:")
    print(relevant_pages)
# question='How can player movement disrupt a well-organised defence?'
# relevant_pages = [25, 37, 59, 60]


        # 7. Run question through every retriever
    for name, retriever in retrievers.items():
        docs = retriever.invoke(question)
        retrieved_pages = [
                doc.metadata.get("page")
                for doc in docs
            ]
        retrieved_pages=unique_pages(retrieved_pages)

        print(f"\n===== {name} =====")

        print("Retrieved pages:")
        print(retrieved_pages)

        print(
                "Hit@5:",
                hit_at_k(
                    retrieved_pages,
                    relevant_pages,
                    k=5
                )
            )

        print(
                "MRR:",
                mrr(
                    retrieved_pages,
                    relevant_pages
                )
            )
results = []

for q in evaluation_questions:

    question = q["question"]
    relevant_pages = q["relevant_pages"]
    q_type = q["type"]

    for name, retriever in retrievers.items():

        docs = retriever.invoke(question)

        retrieved_pages = [
            doc.metadata.get("page")
            for doc in docs
        ]

        retrieved_pages = unique_pages(retrieved_pages)

        hit = hit_at_k(
            retrieved_pages,
            relevant_pages,
            k=5
        )

        reciprocal_rank = mrr(
            retrieved_pages,
            relevant_pages
        )

        results.append({
            "id": q["id"],
            "type": q_type,
            "retriever": name,
            "hit": hit,
            "mrr": reciprocal_rank
        })
import pandas as pd

df = pd.DataFrame(results)

print("\n===== PERFORMANCE BY QUESTION TYPE =====")

summary = (
    df.groupby(["type", "retriever"])
      .agg(
          Hit_at_5=("hit", "mean"),
          MRR=("mrr", "mean")
      )
      .reset_index()
)

print(summary)
overall = (
    df.groupby("retriever")
      .agg(
          Hit_at_5=("hit", "mean"),
          MRR=("mrr", "mean")
      )
      .reset_index()
)

overall["Hit_at_5"] *= 100

print("\n===== OVERALL PERFORMANCE =====")
print(overall.round(3))
# for doc in chunks:
#     page = doc.metadata.get("page")

#     if page in target_pages:
#         print("\n" + "=" * 80)
#         print("PAGE:", page)
#         print("=" * 80)
#         print(doc.page_content)

#python -m rag_system.evaluations.run_evaluations