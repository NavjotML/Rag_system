import streamlit as st
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
import os

from rag_system.embedding.embeddings import embed_documents
from rag_system.vectorstores.vectorstore import load_vectorstore
from rag_system.retrival.retrivals import create_retrievers
from rag_system.llmprompt.llmprompt import create_llm, create_prompt

os.environ['LANGCHAIN_PROJECTS']='RAG CHATBOT'
load_dotenv()

st.title('PDF QUERY RETRIEVER CHATBOT')
st.caption("Ask questions across your indexed reports.")

@st.cache_resource
def load_rag_system():
    embeddings=embed_documents()
    vectorstore=load_vectorstore(embeddings)
    retriever=create_retrievers(vectorstore)

    return retriever
retriever=load_rag_system()

llm=create_llm(model='qwen/qwen3.8-27b',tempreture=0.5)
prompt = create_prompt()
output=StrOutputParser()
chain=prompt|llm|output

if 'messages' not in st.session_state:
    st.session_state.messages=[]

for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.write(message['content'])

query = st.chat_input("Ask about your reports...")

if query:

    st.session_state.messages.append({
        'role':'user',
        'content':query
    })

    with st.chat_message("user"):
        st.write(query)

    # Generate the assistant's answer
    with st.chat_message("assistant"):
        with st.spinner("Searching reports..."):
            docs = retriever.invoke(query)

        context='\n\n'.join(
            f"Source {i} | File: {doc.metadata.get('source','Unknown')}"
            f"Page : {doc.metadata.get('page','Unknown')}\n"
            f"{doc.page_content}"
            for i,doc in enumerate(docs,start=1)
        )

        answer_input={
            "context":context,
            "question":query
        }
        response=chain.invoke(answer_input)
        st.write(response)
    st.session_state.messages.append({
        'role':'assistant',
        'content':response
    })

  

    with st.expander("Retrieved sources"):

        for i, doc in enumerate(docs, start=1):
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "Unknown")

            st.markdown(f"**Source {i}**")
            st.caption(f"File: {source} | Page: {page}")
            st.write(doc.page_content)
            st.divider()