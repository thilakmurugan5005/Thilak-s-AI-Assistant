import chromadb
from langchain_community.vectorstores import Chroma
from chromadb.config import Settings

from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from Create_DB import embeddings
from pydantic_settings import BaseSettings
api_key = st.secrets["OPENAI_API_KEY"]

#In this demo we will explore using RetirvalQA chain to retrieve relevant documents and send these as a context in a query.
# We will use Chroma vectorstore.




#Step 1 - this will set up chain , to be called later

def create_chain():
    embeddings()
    client = chromadb.HttpClient(host="127.0.0.1",settings=Settings(allow_reset=True))

    embeddings = OpenAIEmbeddings(openai_api_key=api_key,
                                  model="text-embedding-3-small")

    db = Chroma(client=client, embedding_function=embeddings)
    #retv = db.as_retriever(search_type="mmr", search_kwargs={"k": 7})
    retv = db.as_retriever(search_kwargs={"k": 3})

    llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=api_key,
                 temperature=0)

    memory = ConversationBufferMemory(llm=llm, memory_key="chat_history", return_messages=True, output_key='answer')

    qa = ConversationalRetrievalChain.from_llm(llm, retriever=retv , memory=memory,
                                               return_source_documents=True)
    return qa

#Step 2 - create chain, here we create a ConversationalRetrievalChain.

chain = create_chain()

#Step 3 - here we declare a chat function
def chat(user_message):
    # generate a prediction for a prompt
    bot_json = chain.invoke({"question": user_message})
    print(bot_json)
    return {"bot_response": bot_json}

#Step 4 - here we setup Streamlit text input and pass input text to chat function.
# chat function returns the response and we print it.

if __name__ == "__main__":
    import streamlit as st

    # Center both the main title and subheader
    st.markdown(
        """
        <div style="text-align: center;">
            <h2>Thilak's AI Assistant</h2>
            <h3>------Ask me a question about Thilak------</h3>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Create two columns
    col1, col2 = st.columns([4, 1])

    # User input with chat functionality
    user_input = st.chat_input()

    with col1:

        # Check if session state for messages exists
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # If user input is provided, generate a bot response
        if user_input:
            bot_response = chat(user_input)
            st.session_state.messages.append({"role": "chatbot", "content": bot_response})

            # Loop through messages and display them
            for message in st.session_state.messages:
                st.chat_message("user")
                st.write("Question: ", message['content']['bot_response']['question'])

                st.chat_message("assistant")
                st.write("Answer: ", message['content']['bot_response']['answer'])



