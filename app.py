import streamlit as st
import tempfile
from ingestions.chunker import split_documents
from ingestions.pdf_loader import load_pdf
from langchain_huggingface import HuggingFaceEmbeddings
from vectorstores.vectorstore import create_vectorstore
from llmprompt.llmprompt import create_llm,create_prompt
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableLambda , RunnablePassthrough
from retrival.retrivals import create_retrievers
load_dotenv()  # Load environment variables from .env file

st.title("PDF RAG Query System")

uploaded_file=st.file_uploader("Upload a PDF file", type=["pdf"])
@st.cache_resource
def embed_documents():
    embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings

if uploaded_file is not None:
    @st.cache_resource
    def process_pdf(pdf_bytes):

        with tempfile.NamedTemporaryFile(delete=False,suffix=".pdf") as temp_file:
            temp_file.write(pdf_bytes)
            pdf_path=temp_file.name
        embeddings=embed_documents()
        loader=load_pdf(pdf_path)
        documents=split_documents(loader)
        vectorstore=create_vectorstore(documents,embeddings)
        retriever=create_retrievers(vectorstore)
        return retriever,documents,vectorstore
    pdf_bytes=uploaded_file.getvalue()
    retriever,documents,vectorstore=process_pdf(pdf_bytes)
    llm=create_llm(model='qwen/qwen3.8-27b',tempreture=0.3)
    prompt=create_prompt()
    output_parser=StrOutputParser()


    chain = prompt | llm | output_parser
        
    query=st.text_input("Ask a question about the PDF:")

    if st.button("Submit") and query:

        st.subheader("Answer:")
        docs=retriever.invoke(query)
        context='\n\n'.join(f"[Source {i} | Page {docs.metadata.get('page','Unknown')}]\n"
                            f"{docs.page_content}"
                            for i,docs in enumerate(docs,start=1))

        input={
            'context':context,
            'question':query
        }

        with st.spinner("Generating answer..."):
            response=st.write_stream(chain.stream(input))

        
        st.subheader("Retrieved Documents:")
        for i,doc in enumerate(docs,start=1):
            st.write(f"{i} **Source for the answer:**")
            st.write(f'Page: {doc.metadata.get("page","Unknown")}')
            st.write(doc.page_content[:100])