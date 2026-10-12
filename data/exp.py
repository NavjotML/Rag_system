from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter as RCT
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
loading=PyPDFLoader('C:\\Users\\Asus\\Desktop\\ML\\_LANGCHAIN_LEARN\\rag_system\\data\\UCL_Technical-Report_2025_DIGITAL.pdf')
docs=loading.load()  # Use lazy_load to load documents lazily
#split docs
splitter=RCT(chunk_size=1000,chunk_overlap=200,length_function=len)
chunks=splitter.split_documents(docs)
print(f"Number of documents loaded: {len(docs)}")
print(f'Number of chunks docs: {len(chunks)}')
embeded_docs=embeddings.embed_documents([doc.page_content for doc in chunks])
print(f"Number of embedded documents: {len(embeded_docs)}")
vectorstore=Chroma.from_documents(chunks,embeddings,collection_name="uefa-documents")
print(f"Number of documents in vectorstore: {vectorstore._collection.count()}")
vectorstoremmr=vectorstore.as_retriever(search_type='mmr',search_kwargs={"k":5,})
bm25_retriever=BM25Retriever.from_documents(chunks,k=5)
retriever = EnsembleRetriever(retrievers=[vectorstoremmr,bm25_retriever],weights=[0.5,0.5])
query='how is paris saint germain use positional fluidity ?'
results=retriever.invoke(query)


questions = [
    "How did Paris Saint-Germain use positional fluidity?",
    "How did PSG create overloads against Inter?",
    "What role did Achraf Hakimi play in PSG's attacking structure?",
    "How did PSG's movement differ from Inter's approach?"
]

for question in questions:
    print("\n" + "=" * 80)
    print("QUESTION:", question)

    results = retriever.invoke(question)

    for i, doc in enumerate(results, 1):
        print(f"\n--- Result {i} for hybrid retriever ---")
        print(doc.page_content[:1000])
        print("PAGE:", doc.metadata.get("page_label"))











# print(chunks[0].page_content[:1000]) 

# print(chunks[0].metadata)

# for i, chunk in enumerate(chunks):
#     if 'fluid' in chunk.page_content.lower():
#         print('loop started')
#         print(chunk.page_content)
#         print(chunk.metadata)

# for i in [0,1,5,20,50,100]:
#     print(f"\n---Document {i+1} content---\n")
#     print(docs[i].page_content[:1500])  # Print the first 100 characters of the document content
#     print(f"\n---Document {i+1} metadata---\n")
#     print(docs[i].metadata)

# # print('\n---First document content---\n')
# # print(docs[0].page_content[:100])

# # print('\n---First document metadata---\n')
# # print(docs[0].metadata)