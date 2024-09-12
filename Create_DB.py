from langchain_community.vectorstores.chroma import Chroma
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
from chromadb.config import Settings

api_key = st.secrets["OPENAI_API_KEY"]
def embeddings():
    # Step 1 - load and split documents
    pdf_loader = PyPDFDirectoryLoader("D:/code/Data/Personal_chatbot_data")
    loaders = [pdf_loader]
    print(loaders)

    documents = []
    for loader in loaders:
        documents.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
    all_documents = text_splitter.split_documents(documents)



    # Set the batch size
    batch_size = 96

    # Calculate the number of batches
    num_batches = len(all_documents) // batch_size + (len(all_documents) % batch_size > 0)
    print(num_batches)

    #Step 2 - Create embedding endpoint
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)

    #Setting up the Chroma db path
    db = Chroma(embedding_function=embeddings , persist_directory="./chromadb")
    retv = db.as_retriever()




    # Iterate over batches
    for batch_num in range(num_batches):
        # Calculate start and end indices for the current batch
        start_index = batch_num * batch_size
        end_index = (batch_num + 1) * batch_size
        # Extract documents for the current batch
        batch_documents = all_documents[start_index:end_index]
        # Your code to process each document goes here
        retv.add_documents(batch_documents)
        print(start_index, end_index)

    #Step 4 - here we persist the collection
    #Since Chroma 0.4.x the manual persistence method is no longer supported as docs are automatically persisted.
    db.persist()

