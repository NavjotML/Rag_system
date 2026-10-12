from ingestions.chunker import split_documents
from ingestions.pdf_loader import load_pdf
from embedding.embeddings import embed_documents
from langchain_huggingface import HuggingFaceEmbeddings
from vectorstores.vectorstore import create_vectorstore
from retrival.retrivals import retriver
from llmprompt.llmprompt import create_llm,create_prompt
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableLambda , RunnablePassthrough
load_dotenv()  # Load environment variables from .env file

loader=load_pdf('pb_pdfgen_September_28_1790590404463wkKFDHvMIi.pdf')
documents=split_documents(loader)
vectorstore=create_vectorstore(documents)
retriver=retriver(vectorstore)
llm=create_llm(model='qwen/qwen3.8-27b',tempreture=0.1)
prompt=create_prompt()
output_parser=StrOutputParser()


def get_context(query):
    docs=retriver.invoke(query)
    context_text='\n\n'.join(doc.page_content for doc in docs)
    return context_text

runnable={'context':RunnableLambda(get_context),
     'question':RunnablePassthrough()}




chain = runnable | prompt | llm | output_parser

response = chain.invoke("what is the main topic of the document?")
print("Response:", response)
