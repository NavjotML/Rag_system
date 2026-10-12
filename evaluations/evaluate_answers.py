def build_context(docs):
    context="\n\n".join(
        f"[Source {i} | Page {doc.metadata.get('page','Unknown')}]\n"
        f"{doc.page_content}"
        for i,doc in enumerate(docs,start=1)
    )
    return context
def generate_answer(retriever,chain,question):
    docs=retriever.invoke(question)
    context=build_context(docs)
    inputs = {
        "context": context,
        "question": question
    }

    answer = chain.invoke(inputs)

    return {
        "answer": answer,
        "context": context,
        "documents": docs
    }