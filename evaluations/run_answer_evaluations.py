from rag_system.ingestions.pdf_loader import load_pdf
from rag_system.ingestions.chunker import split_documents
from rag_system.embedding.embeddings import embed_documents
from rag_system.vectorstores.vectorstore import create_vectorstore
from rag_system.retrival.retrivals import create_retrievers

from rag_system.evaluations.eval_questions import evaluation_questions
from rag_system.evaluations.evaluate_answers import generate_answer

from rag_system.llmprompt.llmprompt import create_llm, create_prompt

from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()

PDF_PATH = "rag_system/data/UCL_Technical-Report_2025-26_DIGITAL.pdf"

documents = load_pdf(PDF_PATH)

chunks = split_documents(documents)

embeddings = embed_documents()

vectorstore = create_vectorstore(
    chunks,
    embeddings
)

retriever = create_retrievers(vectorstore)

llm = create_llm(
    model="qwen/qwen3.8-27b",
    tempreture=0.1
)

prompt = create_prompt()

chain = prompt | llm | StrOutputParser()

answer_eval_ids = {
    "Q01",
    "Q07",
    "Q11",
    "Q16",
    "Q20"
}
for q in evaluation_questions:

    if q["id"] not in answer_eval_ids:
        continue

    result = generate_answer(
        retriever,
        chain,
        q["question"]
    )

    print("\n" + "=" * 80)
    print(q["id"])
    print("=" * 80)

    print("\nQUESTION:")
    print(q["question"])

    print("\nREFERENCE ANSWER:")
    print(q["reference_answer"])

    print("\nGENERATED ANSWER:")
    print(result["answer"])

    print("\nRETRIEVED PAGES:")
    print([
        doc.metadata.get("page")
        for doc in result["documents"]
    ])


