import streamlit as st
import os
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from Create_DB import Create_DB
from uuid import uuid4


#unique_id = uuid4().hex[0:8]
#os.environ["LANGCHAIN_TRACING_V2"] = "true"
#os.environ["LANGCHAIN_PROJECT"] = f"Testi - {unique_id}"
#os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
#os.environ["LANGCHAIN_API_KEY"] = "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"

# Load .env file
##load_dotenv()

# Access API key
#api_key = os.getenv('API_KEY')

api_key = st.secrets["OPENAI_API_KEY"]

st.set_page_config(page_title="Thilak's AI Buddy", layout="wide")

# Add custom CSS for centering the header
st.markdown(
    """
    <style>
    .centered-header {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100px;
        font-size: 3rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize the embeddings and FAISS database
Create_DB()
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=api_key)
new_db = FAISS.load_local("index_test3", embeddings, allow_dangerous_deserialization=True)

def get_conversational_chain(memory):
    prompt_template = """
    You are an AI chatbot, helping to answer users about Thilak's details. Answer the question as detailed as possible from the provided context. If the answer is not in
    the provided context about Thilak, say "Sorry, I dont know. Ask some other question". If they ask about another person please say ,"Sorry, I dont know. Please ask about Thilak\n I am Thilak's AI Buddy.!" Don't provide incorrect information.
    If they ask about work experience give answer in bullet points.\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=api_key, temperature=0)
    #retv = new_db
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = ConversationalRetrievalChain.from_llm(
        llm=model,
        retriever=new_db.as_retriever(),
        memory = memory,
        combine_docs_chain_kwargs={"prompt": prompt}
        )
    return chain

# Main function
def main():
    st.markdown('<div class="centered-header">Thilak\'s AI Chatbot</div>', unsafe_allow_html=True)

    # Initialize memory and vectorstore in session state
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None

    # Initialize chat messages in session state
    if "messages" not in st.session_state:
        # Default first message from the assistant
        st.session_state.messages = [{"role": "assistant", "content": "Hii Buddy.! I am Thilak's AI Assistant. Ask me about Thilak"}]
    
    # Display chat history
    #for message in st.session_state.messages:
     #   with st.chat_message(message["role"]):
     #       st.markdown(message["content"])

    # Display the conversation in the chat format
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.chat_message("user").write(message["content"])
        else:
            st.chat_message("assistant").write(message["content"])

    # Get user input
    if user_question := st.chat_input("Ask a question about Thilak"):
        # Add user message to chat history
                # Display user's input in chat
        with st.chat_message("user"):
            st.markdown(user_question)
        #st.session_state.messages.append({"role": "user", "content": user_question})
        chain = get_conversational_chain(st.session_state.memory)
        response = chain.invoke({"question": user_question})
        print("Responce :",response)
        bot_response = response["answer"]

        if bot_response:  # Check if there's a valid response
            st.session_state.messages.append({"role": "user", "content": user_question})
            st.session_state.messages.append({"role": "assistant", "content": bot_response})

        st.chat_message("assistant").write(bot_response)

# Run the app
if __name__ == "__main__":
    main()
