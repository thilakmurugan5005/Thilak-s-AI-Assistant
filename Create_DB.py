from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import faiss
import numpy as np
import streamlit as st
import PyPDF2

api_key = st.secrets["OPENAI_API_KEY"]


def Create_DB():
    file_path = 'Personal_data.pdf'

    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page in reader.pages:
            text += page.extract_text()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
    all_text = text_splitter.split_text(text)

        # Step 2 - Create embedding endpoint
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002",openai_api_key=api_key)

        # create vector db
    vector_db=FAISS.from_texts(all_text, embedding=embeddings)
        # save vector db
    vector_db.save_local("index_te123")

 

    return 
