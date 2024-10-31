import streamlit as st
import os
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import uuid

load_dotenv()

# Embedding vector
def create_vector_embedding(chatbot_type):
    if "vectors" not in st.session_state:
        st.session_state.embeddings = OpenAIEmbeddings()
        data_path = 'store/'
        # Load FAISS vectors
        try:
            if chatbot_type == 'Thu Dang(Credit scoring)':
                data_path = data_path + 'credit'
            else:
                data_path = data_path + 'cooking'
            st.session_state.vectors = FAISS.load_local(
                data_path, 
                st.session_state.embeddings, 
                allow_dangerous_deserialization=True
            )
        except Exception as e:
            st.error(f"Failed to load FAISS index: {e}")
            return

# Load the API keys
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

# App title
st.title("Ga1nang")

chatbot_type = st.sidebar.selectbox("Select a chatbot type", ['Credit Papa(Credit scoring)', 'Cooking Mama(Cooking master)'])

# Create vector embeddings
if chatbot_type:
    create_vector_embedding(chatbot_type)

# Initialize sessions in session state if not already done
if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = {}

# Initialize current session ID if not already set
if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = None

# Sidebar options
st.sidebar.header("Chat Sessions")

# Button to start a new chat session
if st.sidebar.button("New Chat Session"):
    # Generate a unique session ID
    session_id = str(uuid.uuid4())
    # Start a new session with a welcome message
    st.session_state.chat_sessions[session_id] = [
        {"role": "assistant", "content": "Hi, I'm a chatbot who can help you. How can I help you?"}
    ]
    # Automatically set the new session as the active session
    st.session_state.current_session_id = session_id
    st.sidebar.write("New chat session started!")

# Session selection dropdown
session_options = list(st.session_state.chat_sessions.keys())
current_session_id = st.sidebar.selectbox("Select a Chat Session", session_options, index=0 if session_options else -1)

# Set the current session ID based on selection
if current_session_id:
    st.session_state.current_session_id = current_session_id

# Set up the active session messages
if st.session_state.current_session_id:
    st.session_state.messages = st.session_state.chat_sessions[st.session_state.current_session_id]

# Sidebar model selection
engine = st.sidebar.selectbox("Select Open AI model", ["gpt-4o", "gpt-4-turbo", "gpt-4"])

# Initialize the language model if an engine is selected
if engine:
    llm = ChatOpenAI(model=engine)

    # Set up the prompt template
    prompt = ChatPromptTemplate.from_template(
        """
        Answer the questions based on the provided context.
        If no specific information in the context is available, answer by yourself.
        <context>
        {context}
        <context>
        Question: {input}
        """
    )

    # Create retrieval chain
    if "vectors" in st.session_state:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        st.session_state.retrieval_chain = create_retrieval_chain(retriever, document_chain)
    else:
        st.write("Vectors not initialized. Please check the FAISS index.")

    # Display messages for the current session
    for msg in st.session_state.messages:
        st.chat_message(msg['role']).write(msg['content'])

    # Handle new user input
    if user_input := st.chat_input(placeholder="What is risk management?"):
        # Append user input to the current session messages
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.chat_message("user").write(user_input)

        # Concatenate previous messages as context
        historical_context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages])

        # Generate response from the retrieval chain
        if "retrieval_chain" in st.session_state:
            with st.chat_message("assistant"):
                try:
                    # Pass the historical context along with the user input to the retrieval chain
                    response = st.session_state.retrieval_chain.invoke({'context': historical_context, 'input': user_input})
                    # Save the assistant's response to the message history
                    assistant_response = response['answer'] if isinstance(response, dict) and 'answer' in response else str(response)
                    st.session_state.messages.append({"role": "assistant", "content": assistant_response})
                    st.write(assistant_response)
                    
                    # Update the current chat session with the new messages
                    st.session_state.chat_sessions[st.session_state.current_session_id] = st.session_state.messages
                except Exception as e:
                    st.error(f"Error generating response: {e}")
        else:
            st.error("Retrieval chain not initialized.")
