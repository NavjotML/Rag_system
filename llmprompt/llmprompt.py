from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
import streamlit as st

@st.cache_resource
def create_llm(model,tempreture):
    llm=ChatGroq(
        model=model,
        temperature=tempreture,
    )
    return llm
def create_prompt():
    
    prompt_template=PromptTemplate(
        template='''
    you are a helpful assistant. Answer only using the provided context.
    Each Context section has source label such as:
    [Source 1 | Page 42]
    When making a factual claim, cite the relevant source using:
    [Source 2 | Page 56] like this means you will get the context about retrieved document and you will answer the question based on the context.
    Do not invent citations.
    Do not use outside knowledge.
    If the context does not contain enough information, say: "Given context is not sufficient.".

    ========context========
    {context}
    ========question========
    {question}
    ''',
        input_variables=['context','question']
    )
    return prompt_template





