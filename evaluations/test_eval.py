from rag_system.evaluations.eval_questions import evaluation_questions

for i,docs in enumerate(evaluation_questions):
    print(f"Question {i + 1}:")
    print(docs)
