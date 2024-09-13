from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import faiss
import numpy as np

api_key = st.secrets["OPENAI_API_KEY"]
def Create_DB():
    # Step 1 - load and split documents
    pdf_loader = PyPDFDirectoryLoader("/workspaces/Thilaks-AI-Assistant/Data")
    loaders = [pdf_loader]

    documents = []
    for loader in loaders:
        documents.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
    all_documents = text_splitter.split_documents(documents)

    # Step 2 - Create embedding endpoint
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small",openai_api_key=api_key")

    # create vector db
    vector_db=FAISS.from_documents(all_documents, embedding=embeddings)
    # save vector db
    vector_db.save_local("index_test3")


    return 
