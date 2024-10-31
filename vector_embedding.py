import streamlit as st
import os

from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from tempfile import TemporaryDirectory
from dotenv import load_dotenv
load_dotenv()


os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
embeddings = OpenAIEmbeddings()


#set up streamlit
st.title('Update the vector store')
st.write('Upload your pdf')

temp_dir = "RAG_Q&A_Conversation"
os.makedirs(temp_dir, exist_ok=True)


#process uploaded PDF
uploaded_files = st.file_uploader("Choose a PDF file", type='pdf', accept_multiple_files=True)
start = st.button('Generate Embedding')
if start:
    with st.spinner('Generating Embedding....'):
        #load data
        documents = []
        with TemporaryDirectory() as temp_dir:
            for uploaded_file in uploaded_files:
                temp_pdf = os.path.join(temp_dir, f"temp_{uploaded_file.name}")
                with open(temp_pdf, 'wb') as file:
                    file.write(uploaded_file.getvalue())
                    file_name = uploaded_file.name
                
                #load the pdf file and save to documents    
                loader = PyPDFLoader(temp_pdf)
                docs = loader.load()
                documents.extend(docs)
                    
            
                
            #split and create embeddings for the documents
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
            splits = text_splitter.split_documents(documents)
            
            
            
            # Load previous vector store
            vector_store_path = 'store/cooking'
            try:
                #add new data to vector store
                vector_store = FAISS.load_local(vector_store_path, embeddings=embeddings, allow_dangerous_deserialization=True)
                vector_store.add_documents(splits, embeddings=embeddings)
                
            except (FileNotFoundError, RuntimeError) as e:
                #create new vector store
                if "could not open" in str(e) or "No such file or directory" in str(e):
                    st.warning("No existing vector store found. A new vector store will be created.")
                    vector_store = FAISS.from_documents(documents=splits, embedding=embeddings)
                else:
                    raise e
            
            

            # Save the updated vector store
            vector_store.save_local(vector_store_path)
     
os.remove(temp_dir)
st.success("Vector store updated and saved successfully.")


