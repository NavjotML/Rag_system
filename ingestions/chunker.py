from langchain_text_splitters import RecursiveCharacterTextSplitter as RCT

def split_documents(text):
    text_splitter=RCT(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_documents(text)
